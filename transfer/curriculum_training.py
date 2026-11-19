"""Opt-in CPU curriculum optimization bound to frozen development exposure."""
from copy import deepcopy
from collections import Counter
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from inference.light_cnn import validate_source
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from transfer.curriculum_exposure import ExposurePlan, ExposureLedger
from transfer.development_training import sparse_labels, evaluate_validation, development_counts
from transfer.g3_training import _forward, state_hash
from transfer.observation_policy import scene_identity, ObservationPolicy, evaluate_conditions
from transfer.observation_curriculum import CurriculumPolicy
from transfer.sample_curriculum import SampleCurriculum, curriculum_loss
from transfer.tile_sampling import SupportTileSampler


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _data(scene, record):
    """Hash only permitted development contexts; never touch test cube blocks."""
    identity = scene_identity(scene, record)
    pins = []
    for tile in record['roles']:
        if tile['role'] not in ('devtrain', 'devval'):
            continue
        a = np.ascontiguousarray(scene.cube[tile_slice(tile)])
        if a.dtype.kind != 'f' or not np.isfinite(a).all():
            raise ValueError('Finite floating development contexts required')
        pins.append(dict(tile=tile['id'], role=tile['role'], shape=list(a.shape),
                         dtype=a.dtype.str, sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    return dict(identity=identity, contexts=pins)


def _group(name):
    if name.startswith('backbone.adapter.'): return 'adapter'
    if name.startswith(('backbone.CNN_',)): return 'cnn'
    if name.startswith('backbone.'): return 'graph'
    return 'heads'


class CurriculumRun:
    """Explicit single-task, uncoupled, GroupNorm full-model development runner.

    Fixed Adam and original class-balanced three probability losses; no scheduler,
    model rebuilding, or automatic budget extension. Sample and optional observation
    curricula are separate experiments, never enabled together by this runner.
    The source is unchanged. Save/load authenticates bytes before deserialization.
    Logged execution is not a replacement for independent training acceptance.
    """
    FORMAT = 'vca-curriculum-run-v1'
    FAILURE_FORMAT = 'vca-curriculum-failure-v1'

    def _validate_model(self, model): validate_source(model)
    def _model_hash(self, model): return state_hash(model)
    def _contract_extensions(self): return {}
    def _check_extensions(self): pass
    def _history_extensions(self, outputs): return {}
    def _parameter_history(self, before): return {}
    def _history_extension_keys(self): return set()
    def _validate_history_extensions(self, row): pass
    def _evaluate(self, model): return evaluate_validation(model, self.scene, self.record)
    def _training_forward(self, tile, step, extra): return _forward(self.model, self.scene, tile, **extra)

    def _restore_model(self, state):
        ref = self.model.state_dict()
        if (not isinstance(state, dict) or set(state) != set(ref) or any(not torch.is_tensor(t) or t.shape != ref[k].shape
                or t.dtype != ref[k].dtype or not torch.isfinite(t).all() for k,t in state.items())):
            raise ValueError('Invalid model state')
        self.model.load_state_dict(state)

    def __init__(self, source, scene, record, *, plan, policy, lr, torch_seed,
                 expected_source_state, expected_plan_hash, expected_policy_hash,
                 observation_policy=None, expected_observation_hash=None):
        self.scene = scene
        self._validate_model(source)
        if self._model_hash(source) != expected_source_state:
            raise ValueError('Registered source state mismatch')
        if (list(source.heads) != [scene.dataset] or source.heads[scene.dataset].out_features != scene.classes
                or any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in source.modules())):
            raise ValueError('Single matching task and GroupNorm source required')
        if not _finite(lr) or lr <= 0 or type(torch_seed) is not int or not 0 <= torch_seed < 2**32:
            raise ValueError('Positive explicit Adam lr and uint32 torch seed required')
        if not isinstance(plan, ExposurePlan): raise ValueError('Frozen exposure plan required')
        self.plan = ExposurePlan(scene, record, plan.to_dict(), expected_plan_hash=expected_plan_hash)
        self.plan.require_eligible()
        if getattr(source, 'training_config', {}).get('observation_policy_hash') is not None:
            raise ValueError('Source carrying another observation training contract requires separate migration')
        self.observation_policy = None
        if observation_policy is None:
            if expected_observation_hash is not None: raise ValueError('Disabled observation hash must be null')
        else:
            if policy is not None: raise ValueError('Sample and observation curricula require separate experiments')
            if not isinstance(observation_policy, ObservationPolicy) or observation_policy.policy_hash != expected_observation_hash:
                raise ValueError('Registered observation policy required')
            value = observation_policy.to_dict()
            cls = CurriculumPolicy if 'curriculum' in value else ObservationPolicy
            self.observation_policy = cls(value)
            self.observation_policy.validate(scene, record, self.plan.updates)
            if value['selection_condition'] != 'full' or self.observation_policy.training_application != 'real_subset':
                raise ValueError('Real subsets and unchanged full validation selection required')
        if policy is None:
            if expected_policy_hash is not None: raise ValueError('Disabled policy hash must be null')
        else:
            if not isinstance(policy, SampleCurriculum) or policy.policy_hash != expected_policy_hash:
                raise ValueError('Registered curriculum hash mismatch')
            policy.validate(scene, record)
            if policy.to_dict()['updates'] != self.plan.updates: raise ValueError('Curriculum/exposure budget mismatch')
        self.source, self.scene, self.record = source, scene, deepcopy(record)
        self.policy = None if policy is None else SampleCurriculum(policy.to_dict())
        self.model = deepcopy(source).requires_grad_(True).eval()
        self.model.backbone.adapter.clear_attention_cache()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.9, .999), eps=1e-8,
                                          weight_decay=0., amsgrad=False)
        self.sampler = SupportTileSampler(scene.plan, record['roles'], sparse_labels(record, scene.plan, 'train_support'),
                                          record['classes'], 'class_first')
        self.spec = dict(schema_version=1, source_state=expected_source_state, architecture=source.export_config(),
                         source_training_config=deepcopy(getattr(source, 'training_config', {})),
                         data=_data(scene, record), exposure_hash=expected_plan_hash, policy_hash=expected_policy_hash,
                         updates=self.plan.updates, sampler_seed=plan.to_dict()['seed'], torch_seed=torch_seed, lr=lr,
                         optimizer_config=deepcopy(self.optimizer.state_dict()['param_groups']),
                         parameters=list(dict(self.model.named_parameters())), sampler=self.sampler.metadata,
                         loss='original_three_probability_losses_class_balanced_default_coefficients',
                         selection='first_min_full_devval_fused_NLL_each_update', device='cpu_fp32', scheduler=None)
        # Omit the optional field when off, preserving existing CUR-S3 contracts.
        if self.observation_policy is not None: self.spec['observation_policy_hash'] = expected_observation_hash
        self.spec.update(self._contract_extensions())
        self.contract_hash = digest(self.spec)
        self.rng = np.random.default_rng(self.spec['sampler_seed'])
        self.torch_rng = torch.Generator().manual_seed(torch_seed).get_state()
        self.ledger = ExposureLedger(self.plan)
        self.history, self.selected_step, self.selected_state = [], None, None
        self.failed = False; self.failure = None
        self._seal()

    @property
    def completed(self): return self.ledger.completed

    def _seal(self):
        self._state = self._model_hash(self.model)
        self._history = digest(self.history)

    def _check(self):
        if self.failed: raise ValueError('Failed run must be preserved and diagnosed')
        self._validate_model(self.source)
        self._validate_model(self.model)
        self._check_extensions()
        if (digest(self.spec) != self.contract_hash or self._model_hash(self.source) != self.spec['source_state']
                or self.source.export_config() != self.spec['architecture']
                or getattr(self.source, 'training_config', {}) != self.spec['source_training_config']
                or getattr(self.model, 'training_config', {}) != self.spec['source_training_config']
                or self.model.export_config() != self.spec['architecture']
                or _data(self.scene, self.record) != self.spec['data']
                or self.plan.plan_hash != self.spec['exposure_hash']
                or (None if self.observation_policy is None else self.observation_policy.policy_hash) != self.spec.get('observation_policy_hash')
                or (None if self.policy is None else self.policy.policy_hash) != self.spec['policy_hash']
                or self._model_hash(self.model) != self._state or digest(self.history) != self._history
                or self.completed != len(self.history)):
            raise ValueError('Frozen data/source/contract or live progress changed')
        if any(not p.requires_grad for p in self.model.parameters()): raise ValueError('Full-model trainability changed')
        groups = self.optimizer.param_groups
        if (len(groups) != 1 or [id(p) for p in groups[0]['params']] != [id(p) for p in self.model.parameters()]
                or self.optimizer.state_dict()['param_groups'] != self.spec['optimizer_config']):
            raise ValueError('Registered full-model optimizer changed')
        self._validate_optimizer(self.optimizer.state_dict(), self.history)
        if self.observation_policy is not None:
            value = self.observation_policy.to_dict()
            if digest({k:v for k,v in value.items() if k != 'policy_hash'}) != self.spec['observation_policy_hash']:
                raise ValueError('Frozen observation payload changed')

    def _validate_optimizer(self, opt, history):
        if not isinstance(opt, dict) or set(opt) != {'state', 'param_groups'} or opt['param_groups'] != self.spec['optimizer_config']:
            raise ValueError('Invalid optimizer configuration')
        counts = {name: sum(name in r['gradient_parameters'] for r in history) for name in self.spec['parameters']}
        ids = self.spec['optimizer_config'][0]['params']
        if set(opt['state']) != {i for i, name in zip(ids, counts) if counts[name]}:
            raise ValueError('Adam slots do not match actual gradient participation')
        for i, (name, p) in zip(ids, self.model.named_parameters()):
            if not counts[name]: continue
            slot = opt['state'][i]
            if not isinstance(slot, dict) or set(slot) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Invalid Adam slot')
            n = slot['step']
            if not torch.is_tensor(n) or n.shape != torch.Size([]) or n.item() != counts[name]:
                raise ValueError('Adam step/execution mismatch')
            for k in ('exp_avg', 'exp_avg_sq'):
                v = slot[k]
                if (not torch.is_tensor(v) or v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all()
                        or (k == 'exp_avg_sq' and (v < 0).any())): raise ValueError('Invalid Adam moment')

    def step(self):
        self._check()
        if self.completed == self.plan.updates: raise ValueError('Completed budget; no more updates')
        try:
            n = self.completed + 1
            event = self.plan.event(n)
            tile, anchor = self.sampler.draw(self.rng)
            if (tile['id'], anchor) != (event['tile_id'], event['anchor']):
                raise ValueError('Original sampler differs from frozen exposure')
            before = {k: p.detach().clone() for k, p in self.model.named_parameters()}
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.torch_rng)
                self.model.train(); self.optimizer.zero_grad(set_to_none=True)
                extra = {} if self.observation_policy is None else dict(observation_view=self.observation_policy.view(
                    self.scene, self.record, tile, phase='training', step=n))
                outputs = self._training_forward(tile, n, extra)
                loss = curriculum_loss(outputs, self.scene, self.record, tile, policy=self.policy,
                                       step=None if self.policy is None else n, class_balance=True)['total']
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward()
                names, gradient = [], dict.fromkeys(('adapter', 'cnn', 'graph', 'heads'), 0.)
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): raise ValueError('Nonfinite gradient')
                        names.append(name); gradient[_group(name)] += float(p.grad.abs().sum())
                if not names: raise ValueError('No optimizer gradients')
                self.optimizer.step()
                validation = self._evaluate(self.model)
                self.torch_rng = torch.get_rng_state().clone()
            self._validate_model(self.model)
            if not _finite(validation['nll']): raise ValueError('Nonfinite validation')
            changed = dict.fromkeys(gradient, 0)
            for name, p in self.model.named_parameters(): changed[_group(name)] += int((before[name] != p.detach()).sum())
            ids = event['support_ids']
            weights = np.ones(len(ids)) if self.policy is None else self.policy.weights(ids, step=n)
            row = dict(**event, loss=float(loss.detach()), gradient_parameters=names, gradient_l1=gradient,
                       changed_parameters=changed, weights=weights.tolist(), validation=validation,
                       state_hash=self._model_hash(self.model))
            row.update(self._history_extensions(outputs))
            row.update(self._parameter_history(before))
            if self.observation_policy is not None: row['observation'] = self.observation_policy.training_record(n)
            self.ledger.append(event); self.history.append(row)
            if self.selected_step is None or validation['nll'] < self.history[self.selected_step-1]['validation']['nll']:
                self.selected_step = n; self.selected_state = deepcopy(self.model.state_dict())
            self._seal(); self._check()
            return deepcopy(row)
        except Exception as e:
            self.failed = True; self.failure = dict(type=type(e).__name__, message=str(e), attempted_step=n)
            raise

    def advance(self, to_step):
        self._check()
        if type(to_step) is not int or not self.completed <= to_step <= self.plan.updates: raise ValueError('Invalid absolute target')
        while self.completed < to_step: self.step()
        return deepcopy(self.history)

    def finish(self):
        self._check(); self.ledger.finish()
        result = dict(status='completed_execution_evidence', updates=self.completed, selected_step=self.selected_step,
                    exposure=self.ledger.snapshot(), optimizer_steps_verified=True, independent_acceptance=False,
                    selected_state_hash=self.history[self.selected_step-1]['state_hash'], query=0)
        if self.observation_policy is not None: result['observations'] = self.observation_exposure()
        return result

    def observation_exposure(self):
        """Actual condition-by-tile/support/class exposure, not a compute-cost proxy."""
        self._check()
        if self.observation_policy is None: raise ValueError('No registered observation schedule')
        rows = self.observation_policy.to_dict()['training']; labels = dict(zip(self.record['train_support']['indices'],self.record['train_support']['labels']))
        keys = sorted({digest(r['config']) for r in rows})
        configs = {k:dict(updates=0, retained_bands=0, tiles=Counter(), support_visits=Counter(), class_blocks=Counter(), class_label_visits=Counter()) for k in keys}
        records = []
        for row in self.history:
            n = row['step']; frozen = rows[n-1]
            if row['observation'] != self.observation_policy.training_record(n): raise ValueError('Observed execution record differs')
            item = configs[digest(frozen['config'])]; item['updates'] += 1; item['retained_bands'] += row['observation']['selected_band_count']
            item['tiles'][str(row['tile_id'])] += 1
            item['support_visits'].update(str(i) for i in row['support_ids'])
            item['class_label_visits'].update(str(labels[i]) for i in row['support_ids'])
            item['class_blocks'].update(str(c) for c in {labels[i] for i in row['support_ids']})
            records.append(row['observation']['record_hash'])
        return dict(completed=self.completed, policy_hash=self.observation_policy.policy_hash,
                    record_multiset_hash=digest(sorted(records)), by_config=configs,
                    interpretation='equal marginal records do not imply equal condition-by-tile/class/support exposure or compute cost')

    def review_observation_conditions(self):
        """Read-only review of the selected state, after completion; never reselect."""
        self.finish()
        if self.observation_policy is None: raise ValueError('No registered validation conditions')
        selected = deepcopy(self.model); selected.load_state_dict(self.selected_state); selected.eval()
        before = state_hash(selected)
        with torch.random.fork_rng(devices=[]):
            result = evaluate_conditions(selected, self.scene, self.record, self.observation_policy)
        if state_hash(selected) != before: raise ValueError('Condition review mutated selected state')
        self._check()
        return dict(contract_hash=self.contract_hash, selected_step=self.selected_step,
                    selected_state_hash=before, policy_selection=False, review=result)

    def save(self, path):
        self._check()
        packet = dict(format=self.FORMAT, spec=deepcopy(self.spec), contract_hash=self.contract_hash,
                      state=deepcopy(self.model.state_dict()), optimizer=deepcopy(self.optimizer.state_dict()),
                      rng=deepcopy(self.rng.bit_generator.state), torch_rng=self.torch_rng.clone(),
                      history=deepcopy(self.history), progress=self.ledger.snapshot(),
                      selected_step=self.selected_step, selected_state=deepcopy(self.selected_state))
        return self._write(packet, path)

    def save_failure(self, path):
        """Preserve partial optimizer/model state for diagnosis, never for resume."""
        if not self.failed: raise ValueError('Only a failed run has diagnostic evidence')
        packet = dict(format=self.FAILURE_FORMAT, resumable=False, failure=deepcopy(self.failure),
                      spec=deepcopy(self.spec), contract_hash=self.contract_hash,
                      state=deepcopy(self.model.state_dict()), optimizer=deepcopy(self.optimizer.state_dict()),
                      rng=deepcopy(self.rng.bit_generator.state), torch_rng=self.torch_rng.clone(),
                      history=deepcopy(self.history), progress=self.ledger.snapshot())
        return self._write(packet, path)

    @staticmethod
    def _write(packet, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
                temp = Path(f.name); torch.save(packet, f); f.flush(); os.fsync(f.fileno())
            os.link(temp, path)
        finally:
            if temp is not None: temp.unlink(missing_ok=True)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, source, scene, record, path, *, plan, policy, expected_sha256, expected_contract_hash,
             observation_policy=None, constructor_kwargs=None):
        if cls is CurriculumRun and constructor_kwargs is not None:
            raise ValueError('Base curriculum does not accept an execution extension')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError('Run file hash mismatch')
        v = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        keys = {'format','spec','contract_hash','state','optimizer','rng','torch_rng','history','progress','selected_step','selected_state'}
        if not isinstance(v, dict) or set(v) != keys or v['format'] != cls.FORMAT or v['contract_hash'] != expected_contract_hash:
            raise ValueError('Invalid registered packet')
        s = v['spec']
        if not isinstance(s, dict) or digest(s) != expected_contract_hash: raise ValueError('Invalid contract')
        run = cls(source, scene, record, plan=plan, policy=policy, lr=s['lr'], torch_seed=s['torch_seed'],
                  expected_source_state=s['source_state'], expected_plan_hash=s['exposure_hash'], expected_policy_hash=s['policy_hash'],
                  observation_policy=observation_policy, expected_observation_hash=s.get('observation_policy_hash'),
                  **(constructor_kwargs or {}))
        if run.spec != s: raise ValueError('Resume contract changed')
        run.ledger = ExposureLedger(run.plan, v['progress']); n = run.completed; history = v['history']
        if not isinstance(history, list) or len(history) != n: raise ValueError('Incomplete history')
        rng = np.random.default_rng(s['sampler_seed'])
        for step, row in enumerate(history, 1):
            event = run.plan.event(step); tile, anchor = run.sampler.draw(rng)
            extra = (set() if observation_policy is None else {'observation'}) | run._history_extension_keys()
            if (set(row) != set(event) | {'loss','gradient_parameters','gradient_l1','changed_parameters','weights','validation','state_hash'} | extra
                    or any(row[k] != val or type(row[k]) != type(val) for k, val in event.items())
                    or (tile['id'], anchor) != (row['tile_id'], row['anchor'])
                    or not _finite(row['loss']) or row['loss'] < 0): raise ValueError('Invalid execution history')
            if observation_policy is not None and row['observation'] != observation_policy.training_record(step):
                raise ValueError('Wrong frozen observation execution')
            run._validate_history_extensions(row)
            names = row['gradient_parameters']
            if not isinstance(names, list) or not names or names != [k for k in s['parameters'] if k in names]: raise ValueError('Invalid gradient parameters')
            for k in ('gradient_l1','changed_parameters'):
                if set(row[k]) != {'adapter','cnn','graph','heads'} or any(not _finite(x) or x < 0 for x in row[k].values()): raise ValueError('Invalid gradient/update evidence')
            expected = np.ones(len(event['support_ids'])) if policy is None else policy.weights(event['support_ids'], step=step)
            if row['weights'] != expected.tolist(): raise ValueError('Wrong weighted exposure')
            val = row['validation']; c = np.asarray(val['confusion'])
            _,validation_count = development_counts(record)
            if (set(val) != {'nll','oa','count','confusion','support','recall'} or not _finite(val['nll']) or val['nll'] < 0
                    or c.shape != (scene.classes,scene.classes) or c.dtype.kind not in 'iu' or (c < 0).any()
                    or val['count'] != validation_count or c.sum() != validation_count or not np.array_equal(c.sum(1), np.full(scene.classes,5))
                    or val['support'] != c.sum(1).tolist() or val['recall'] != (np.diag(c)/c.sum(1)).tolist()
                    or val['oa'] != float(np.trace(c)/validation_count)): raise ValueError('Invalid validation history')
        selected = None if n == 0 else min(range(n), key=lambda i:history[i]['validation']['nll']) + 1
        if v['selected_step'] != selected or v['rng'] != rng.bit_generator.state: raise ValueError('Wrong selection/sampler state')
        run._validate_optimizer(v['optimizer'], history)
        with torch.random.fork_rng(devices=[]):
            try: torch.set_rng_state(v['torch_rng'])
            except Exception as e: raise ValueError('Invalid torch RNG') from e
            if n:
                run._restore_model(v['selected_state'])
                if run._model_hash(run.model) != history[selected-1]['state_hash'] or run._evaluate(run.model) != history[selected-1]['validation']:
                    raise ValueError('Selected state/validation mismatch')
            elif v['selected_state'] is not None: raise ValueError('Selection before first update')
            run._restore_model(v['state'])
            if n:
                if run._model_hash(run.model) != history[-1]['state_hash'] or run._evaluate(run.model) != history[-1]['validation']:
                    raise ValueError('Last state/validation mismatch')
            elif run._model_hash(run.model) != s['source_state'] or not torch.equal(v['torch_rng'], run.torch_rng): raise ValueError('Zero-step initialization changed')
        run.optimizer.load_state_dict(v['optimizer']); run.rng.bit_generator.state = v['rng']; run.torch_rng = v['torch_rng'].clone()
        run.history = deepcopy(history); run.selected_step = selected; run.selected_state = deepcopy(v['selected_state'])
        run._seal(); run._check()
        return run
