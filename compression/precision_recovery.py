"""Matched FP32, INT8 QAT and ternary recovery using the original curriculum optimizer."""
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

from compression.int8 import CalibratedMatrix, IntegerMatrix
from compression.int8_model import calibrate_model, calibration_contract, convert_model, validate_plan
from compression.precision_fallback import LAYER_ORDER, PRECISION_ASSIGNMENTS, _validation_inputs
from compression.ternary import TernaryMatrix
from compression.ternary_model import PackedTernary, pack_model, ternarize_model, validate_model
from inference.light_cnn import validate_source
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import _forward, state_hash
from transfer.xuzhou_multiscale import LR, SEED


HEADS = ('fused', 'graph', 'cnn')
ARMS = {'fp32': ('fp32', ()),
        **{f'{mode}-{name}': (mode, paths) for mode in ('qat', 'ternary')
           for name, paths in zip(('first', 'second', 'both'), PRECISION_ASSIGNMENTS, strict=True)}}
TERNARY_RATIO = .05


def transform(source, scene, record, arm, tiles):
    mode, paths = ARMS[arm]
    if mode == 'qat':
        contract = calibration_contract(source, scene, record, paths, tiles, qat=True)
        return calibrate_model(source, scene, record, paths=paths, contract=contract, expected_hash=digest(contract))
    if mode == 'ternary':
        return ternarize_model(source, paths=paths, threshold_ratio=TERNARY_RATIO)
    return deepcopy(source).eval()


def frozen_buffers(model, arm):
    mode, paths = ARMS[arm]
    state = model.state_dict()
    if mode == 'qat':
        names = [k for p in paths for k in state if k.startswith(p + '.') and not k.startswith(p + '.layer.')]
    elif mode == 'ternary':
        names = [p + '.threshold_ratio' for p in paths]
    else:
        names = []
    return {name: tensor_hash(state[name]) for name in names}


def initialization(source, model, arm):
    mode, paths = ARMS[arm]
    old, new = source.state_dict(), model.state_dict()
    before = {name: tensor_hash(value) for name, value in old.items()}
    after = {name: tensor_hash(value) for name, value in new.items()}
    mapping = {name: next((path + '.layer.' + name.removeprefix(path + '.') for path in paths
                           if name.startswith(path + '.')), name) for name in old}
    if (len(set(mapping.values())) != len(mapping)
            or any(after.get(target) != before[key] for key, target in mapping.items())
            or model.export_config() != source.export_config()
            or model.training_config != source.training_config):
        raise ValueError('Precision initialization changed dense masters, architecture or training provenance')
    extra = {name: after[name] for name in sorted(after.keys() - set(mapping.values()))}
    return dict(dense_state=state_hash(source), transformed_state=state_hash(model),
                mode=mode, paths=list(paths), dense_tensor_hashes=before,
                mapped_tensor_hash=digest({name: after[target] for name, target in mapping.items()}),
                added_tensor_hashes=extra, frozen_buffer_hashes=frozen_buffers(model, arm))


def deployment(model, arm):
    mode, _ = ARMS[arm]
    if mode == 'qat':
        result = convert_model(model)
    elif mode == 'ternary':
        result = pack_model(model)
    else:
        result = deepcopy(model).eval()
    result.backbone.adapter.cache_attention = False
    return result.requires_grad_(False)


def _kernel_hook(integer, path, counts):
    def seen(_, inputs, output):
        if integer:
            if not inputs[0].is_quantized or not output.is_quantized:
                raise ValueError('INT8 deployment bypassed quantized activation or convolution')
        elif inputs[0].dtype != torch.float32 or output.dtype != torch.float32:
            raise ValueError('Packed ternary arithmetic must remain FP32')
        counts[path] += 1
    return seen


@torch.no_grad()
def validate_deployment(model, scene, blocks, *, arm):
    """Use the original V5 accumulation order and report every head on each event."""
    mode, paths = ARMS[arm]
    k = scene.classes
    counts = dict.fromkeys(paths, 0)
    matrices = np.zeros((3, k, k), np.int64)
    nlls = [0.] * len(HEADS)
    total = 0
    with ExitStack() as stack:
        for path in paths:
            module = model.get_submodule(path)
            if mode == 'qat' and type(module) is IntegerMatrix:
                target = module.kernel
            elif mode == 'ternary' and type(module) is PackedTernary:
                target = module
            else:
                raise ValueError('Expected a deployed integer or packed module')
            stack.enter_context(target.register_forward_hook(_kernel_hook(mode == 'qat', path, counts)))
        for tile, mask, labels in blocks:
            target = labels - 1
            outputs = _forward(model, scene, tile)
            total += len(target)
            for i, output in enumerate(outputs[:3]):
                probs = output.detach().cpu().numpy()[mask]
                if probs.shape != (len(target), k) or not np.isfinite(probs).all():
                    raise ValueError('Invalid three-head deployment validation probabilities')
                prediction = probs.argmax(1)
                matrices[i] += np.bincount(target * k + prediction, minlength=k ** 2).reshape(k, k)
                nlls[i] -= float(np.log(np.maximum(probs[np.arange(len(target)), target], 1e-8)).sum())
    if (not total or any(count != len(blocks) for count in counts.values())
            or not (matrices.sum(axis=2) == 5).all()):
        raise ValueError('Registered deployment validation support or kernel call counts differ')
    rows = {head: dict(nll=nll / total, oa=float(np.trace(matrix) / total), count=total,
                       confusion=matrix.tolist(), support=matrix.sum(1).tolist(),
                       recall=(np.diag(matrix) / matrix.sum(1)).tolist())
            for head, matrix, nll in zip(HEADS, matrices, nlls, strict=True)}
    return rows, counts


class PrecisionRecoveryRun(CurriculumRun):
    """Train only in the floating master representation, and select on deployed copies."""
    FORMAT = 'vca-precision-recovery-v1'
    FAILURE_FORMAT = 'vca-precision-recovery-failure-v1'

    def __init__(self, source, scene, record, *, trial, selector_seed=SEED, **kwargs):
        if trial not in ARMS or selector_seed != SEED:
            raise ValueError('Unknown precision arm or recovery seed')
        self.arm = trial
        self.mode, self.paths = ARMS[trial]
        self.timings = defaultdict(list)
        self.validation_spec, self.validation_blocks = _validation_inputs(scene, record)
        self.observers = frozen_buffers(source, trial)
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        if self.mode == 'fp32':
            validate_source(model)
        elif self.mode == 'qat':
            plan = validate_plan(model)
            if plan['phase'] != 'qat' or plan['contract']['paths'] != list(self.paths):
                raise ValueError('Expected the registered QAT masters and observers')
        else:
            plan = validate_model(model)
            if (plan['phase'] != 'training' or plan['spec']['paths'] != list(self.paths)
                    or plan['spec']['threshold_ratio'] != float(np.float32(TERNARY_RATIO))):
                raise ValueError('Expected trainable ternary masters and the registered threshold')

    def _contract_extensions(self):
        return dict(precision_arm=self.arm, precision_paths=list(self.paths), frozen_buffers=self.observers,
                    precision_plan=deepcopy(getattr(self.source, '_int8_plan', getattr(self.source, '_tq_plan', None))),
                    validation_scope=self.validation_spec, validation_events=list(range(1, self.plan.updates + 1)),
                    precision_selection='first_min_deployment_fused_nll', observer_policy='initial_devtrain_only')

    def _check_extensions(self):
        if (frozen_buffers(self.model, self.arm) != self.observers
                or frozen_buffers(self.source, self.arm) != self.observers):
            raise ValueError('Frozen QAT observers or ternary threshold changed')
        if any(getattr(self.model, name, None) != getattr(self.source, name, None)
               for name in ('_int8_plan', '_tq_plan')):
            raise ValueError('Precision plan changed after registration')

    def _training_forward(self, tile, step, extra):
        start = perf_counter()
        output = super()._training_forward(tile, step, extra)
        self.timings['training_forward'].append(perf_counter() - start)
        return output

    def _backward(self, loss):
        start = perf_counter()
        super()._backward(loss)
        self.timings['backward'].append(perf_counter() - start)

    def _evaluate(self, model):
        # The master remains the recovery state. Only a private copy sees V5.
        model.eval()
        start = perf_counter()
        before = state_hash(model)
        candidate = deployment(model, self.arm)
        prepared = perf_counter()
        rows, kernels = validate_deployment(candidate, self.scene, self.validation_blocks, arm=self.arm)
        evaluated = perf_counter()
        self.timings['deployment_conversion'].append(prepared - start)
        self.timings['deployment_validation'].append(evaluated - prepared)
        if state_hash(model) != before or frozen_buffers(model, self.arm) != self.observers:
            raise ValueError('Validation mutated a training parameter or observer')
        self._deployment_event = dict(heads=rows, kernel_calls=kernels)
        return rows['fused']

    def _parameter_history(self, before):
        active = {}
        for name, parameter in self.model.named_parameters():
            changed = int((before[name] != parameter.detach()).sum())
            if parameter.grad is None and changed:
                raise ValueError('A parameter without gradient participation changed')
            if any(name.startswith(path + '.') for path in LAYER_ORDER):
                active[name] = dict(nonzero_gradient=0 if parameter.grad is None else int(parameter.grad.count_nonzero()),
                                    changed=changed, elements=parameter.numel())
        return dict(precision_parameters=active, deployment=self._deployment_event)

    def _history_extension_keys(self):
        return {'precision_parameters', 'deployment'}

    def _validate_history_extensions(self, row):
        expected = {key: p.numel() for key, p in self.model.named_parameters()
                    if any(key.startswith(path + '.') for path in LAYER_ORDER)}
        evidence = row['precision_parameters']
        if evidence.keys() != expected.keys() or any(
                set(v) != {'nonzero_gradient', 'changed', 'elements'} or v['elements'] != expected[k]
                or any(type(v[n]) is not int or not 0 <= v[n] <= expected[k] for n in ('changed', 'nonzero_gradient'))
                for k, v in evidence.items()):
            raise ValueError('Registered precision parameter activity evidence differs')
        event = row['deployment']
        if (set(event) != {'heads', 'kernel_calls'} or set(event['heads']) != set(HEADS)
                or event['heads']['fused'] != row['validation']
                or event['kernel_calls'] != dict.fromkeys(self.paths, len(self.validation_blocks))):
            raise ValueError('Deployment event or actual kernel evidence differs')
        for value in event['heads'].values():
            matrix = np.asarray(value['confusion'])
            count = self.scene.classes * 5
            if (set(value) != {'nll', 'oa', 'count', 'confusion', 'support', 'recall'}
                    or matrix.shape != (self.scene.classes, self.scene.classes) or matrix.dtype.kind not in 'iu'
                    or (matrix < 0).any() or not (matrix.sum(1) == 5).all() or value['count'] != count
                    or value['support'] != matrix.sum(1).tolist() or value['recall'] != (np.diag(matrix)/5).tolist()
                    or value['oa'] != float(np.trace(matrix)/count) or not np.isfinite(value['nll']) or value['nll'] < 0):
                raise ValueError('Invalid registered deployment head metrics')

    @classmethod
    def load(cls, source, scene, record, path, *, plan, trial, selector_seed,
             expected_sha256, expected_contract_hash):
        return super().load(source, scene, record, path, plan=plan, policy=None,
                            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(trial=trial, selector_seed=selector_seed))


def new_run(source, scene, record, plan, trial):
    return PrecisionRecoveryRun(source, scene, record, plan=plan, trial=trial, policy=None, lr=LR,
                                torch_seed=SEED, expected_source_state=state_hash(source),
                                expected_plan_hash=plan.plan_hash, expected_policy_hash=None)
