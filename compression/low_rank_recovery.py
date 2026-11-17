"""Matched FP32 recovery of one explicit two-factor convolution."""

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from compression.low_rank import LowRankMatrix, factorize_model, matrix
from compression.precision_fallback import _validation_inputs
from compression.precision_recovery import HEADS, validate_deployment
from inference.light_cnn import validate_source
from research_runtime import tensor_hash
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import LR, SEED


TARGET = "backbone.CNN_Branch.0.point_conv"
RECOVERY_SEEDS = (64, 512, 2048)
RANK_REQUEST = (8, 16, 32)


def ladder(model):
    kind, weights = matrix(model.get_submodule(TARGET))
    if kind != "conv1x1":
        raise ValueError("The registered first CNN point convolution must be Conv2d")
    m, n = weights.shape
    limit = min(min(m, n), (m*n-1)//(m+n))
    mapped = {str(rank): min(rank, limit) if limit else None for rank in RANK_REQUEST}
    arms = {"dense": dict(rank=None, init=None, allow_expansion=False)}
    for rank in dict.fromkeys(value for value in mapped.values() if value is not None):
        arms[f"svd-r{rank}"] = dict(rank=rank, init="svd", allow_expansion=False)
    arms["full-rank"] = dict(rank=min(m, n), init="svd", allow_expansion=True)
    if mapped["16"] is not None:
        arms[f"random-r{mapped['16']}"] = dict(rank=mapped["16"], init="random", allow_expansion=False)
    return dict(shape=[m, n], requested_mapping=mapped, arms=arms)


def exposure(scene, record, seeds, limit):
    rows, plans = {}, {}
    required = set(record["train_support"]["indices"])
    for seed in seeds:
        trial = ExposurePlan.freeze(scene, record, updates=limit, seed=seed,
                                    minimum_class_blocks={str(c): 1 for c in record["classes"]}, minimum_support_visits=1)
        plans[seed] = trial
        first = {}
        for row in trial.to_dict()["events"]:
            for index in row["support_ids"]:
                first.setdefault(index, row["step"])
        missing = sorted(required - first.keys())
        rows[str(seed)] = dict(shortest=None if missing else max(first.values()), missing=missing,
                               first_visit={str(k): n for k, n in sorted(first.items())}, plan_at_limit=trial.plan_hash)
    return rows, plans


def transform(dense, assignment, seed):
    if assignment["rank"] is None:
        return dense
    return factorize_model(dense, {TARGET: assignment["rank"]}, init=assignment["init"], seed=seed,
                           allow_expansion=assignment["allow_expansion"]).eval()


def initial_evidence(dense, candidate, assignment):
    initial = dense.state_dict()
    actual = candidate.state_dict()
    prefix = TARGET + "."
    shared = {k: tensor_hash(v) for k, v in initial.items() if not k.startswith(prefix)}
    if (candidate.export_config() != dense.export_config() or candidate.training_config != dense.training_config
            or {k: tensor_hash(v) for k, v in actual.items() if not k.startswith(prefix)} != shared):
        raise ValueError("Nonfactor source tensors or architecture changed")
    if assignment["rank"] is None:
        if state_hash(dense) != state_hash(candidate):
            raise ValueError("Dense control differs from the original source")
        error = 0.
    else:
        layer = candidate.get_submodule(TARGET)
        original = dense.get_submodule(TARGET)
        if (type(layer) is not LowRankMatrix or layer.rank != assignment["rank"]
                or (original.bias is None) != (layer.bias is None)
                or (original.bias is not None and not torch.equal(original.bias, layer.bias))):
            raise ValueError("Factor dimensions or second-operator bias differ")
        reconstructed = layer.left.double() @ layer.right.double()
        reference = original.weight.detach().flatten(1).double()
        error = float(torch.linalg.norm(reconstructed-reference) / max(float(torch.linalg.norm(reference)), 1e-30))
    return dict(state=state_hash(candidate), dense_state=state_hash(dense),
                shared_tensors=shared, shared_count=len(shared), initial_relative_weight_error=error,
                parameter_count=sum(p.numel() for p in candidate.parameters()),
                plan=deepcopy(getattr(candidate, "_low_rank_plan", None)))


@contextmanager
def factor_calls(model):
    layer = model.get_submodule(TARGET)
    counts = {}
    if type(layer) is not LowRankMatrix:
        yield counts
        return
    counts.update(right=0, left=0)
    addresses = {layer.right.untyped_storage().data_ptr(): "right", layer.left.untyped_storage().data_ptr(): "left"}
    operator = F.conv2d

    def counted(input, weight, *args, **kwargs):
        name = addresses.get(weight.untyped_storage().data_ptr())
        if name is not None:
            if input.dtype != torch.float32 or weight.dtype != torch.float32:
                raise ValueError("Low-rank FP32 path did not call floating factors")
            counts[name] += 1
        return operator(input, weight, *args, **kwargs)

    with patch.object(F, "conv2d", new=counted):
        yield counts


def factor_counts(model, calls):
    return dict.fromkeys(("right", "left"), calls) if type(model.get_submodule(TARGET)) is LowRankMatrix else {}


class LowRankRecoveryRun(CurriculumRun):
    """Keep optimizer arithmetic and V5 event ordering from the original runner."""

    FORMAT = "vca-wp10-low-rank-recovery-v1"
    FAILURE_FORMAT = "vca-wp10-low-rank-recovery-failure-v1"

    def __init__(self, source, scene, record, *, trial, **kwargs):
        self.trial = trial
        self.factor_plan = deepcopy(getattr(source, "_low_rank_plan", None))
        self.validation_spec, self.validation_blocks = _validation_inputs(scene, record)
        self.timings = defaultdict(list)
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        layer = model.get_submodule(TARGET)
        if self.factor_plan is None:
            validate_source(model)
        elif (model._low_rank_plan != self.factor_plan or type(layer) is not LowRankMatrix
              or layer.kind != "conv1x1" or layer.rank != self.factor_plan["ranks"][TARGET]
              or layer.right.shape != (layer.rank, layer.in_features)
              or layer.left.shape != (layer.out_features, layer.rank)
              or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in layer.parameters())):
            raise ValueError("Registered two-factor recovery topology changed")

    def _contract_extensions(self):
        return dict(low_rank_trial=self.trial, factor_plan=self.factor_plan, validation_scope=self.validation_spec)

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        with factor_calls(self.model) as observed:
            output = super()._training_forward(tile, step, extra)
        self.training_factor_calls = observed
        if observed != factor_counts(self.model, 1):
            raise ValueError("Training forward did not call both registered factors")
        self.timings["training_forward"].append(perf_counter()-started)
        return output

    def _evaluate(self, model):
        model.eval()
        started = perf_counter()
        before = state_hash(model)
        with factor_calls(model) as observed:
            rows, _ = validate_deployment(model, self.scene, self.validation_blocks, arm="fp32")
        if observed != factor_counts(model, len(self.validation_blocks)) or state_hash(model) != before:
            raise ValueError("V5 changed the model or did not execute both factors")
        self.deployment_event = dict(heads=rows, factor_calls=observed)
        self.timings["validation"].append(perf_counter()-started)
        return rows["fused"]

    def _parameter_history(self, before):
        activity = {}
        for name, parameter in self.model.get_submodule(TARGET).named_parameters():
            changed = int((before[f"{TARGET}.{name}"] != parameter.detach()).sum())
            grad = parameter.grad
            if grad is None and changed:
                raise ValueError("Parameter changed without gradient")
            activity[name] = dict(gradient_nonzero=0 if grad is None else int(grad.count_nonzero()),
                                  changed=changed, elements=parameter.numel())
        return dict(target_activity=activity, training_factors=self.training_factor_calls, deployment=self.deployment_event)

    def _history_extension_keys(self):
        return {"target_activity", "training_factors", "deployment"}

    def _validate_history_extensions(self, row):
        parameters = dict(self.model.get_submodule(TARGET).named_parameters())
        activity = row["target_activity"]
        event = row["deployment"]
        if (activity.keys() != parameters.keys() or row["training_factors"] != factor_counts(self.model, 1)
                or set(event) != {"heads", "factor_calls"} or set(event["heads"]) != set(HEADS)
                or event["heads"]["fused"] != row["validation"]
                or event["factor_calls"] != factor_counts(self.model, len(self.validation_blocks))):
            raise ValueError("Factor activity or validation identity differs")
        for name, value in activity.items():
            n = parameters[name].numel()
            if (set(value) != {"gradient_nonzero", "changed", "elements"} or value["elements"] != n
                    or any(type(value[k]) is not int or not 0 <= value[k] <= n for k in ("gradient_nonzero", "changed"))):
                raise ValueError("Invalid factor parameter evidence")
        for quality in event["heads"].values():
            matrix = np.asarray(quality["confusion"])
            size = self.scene.classes
            if (matrix.shape != (size, size) or matrix.dtype.kind not in "iu" or (matrix < 0).any()
                    or not (matrix.sum(1) == 5).all() or quality["count"] != 5*size
                    or quality["support"] != [5]*size or quality["recall"] != (np.diag(matrix)/5).tolist()
                    or quality["oa"] != float(np.trace(matrix)/(5*size)) or not np.isfinite(quality["nll"])
                    or quality["nll"] < 0):
                raise ValueError("Invalid registered three-head validation metrics")

    @classmethod
    def load(cls, source, scene, record, path, *, plan, trial, selector_seed, expected_sha256, expected_contract_hash):
        # The generic stage runner carries its historical selector argument. Seed identity is in the plan and packet.
        if selector_seed != SEED:
            raise ValueError("Generic stage selector differs")
        return super().load(source, scene, record, path, plan=plan, policy=None, expected_sha256=expected_sha256,
                            expected_contract_hash=expected_contract_hash, constructor_kwargs=dict(trial=trial))


def new_run(source, scene, record, plan, trial):
    return LowRankRecoveryRun(source, scene, record, plan=plan, trial=trial, policy=None, lr=LR,
                              torch_seed=plan.to_dict()["seed"], expected_source_state=state_hash(source),
                              expected_plan_hash=plan.plan_hash, expected_policy_hash=None)
