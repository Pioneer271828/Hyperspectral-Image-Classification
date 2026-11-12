"""Registered native learning with fixed step-zero development validation."""

from copy import deepcopy
from math import isfinite
from time import perf_counter

import numpy as np
import torch

from inference.light_cnn import validate_source
from loadData.spatial_tiles import tile_slice
from research_runtime import tensor_hash
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import development_counts, sparse_labels, validate
from transfer.g3_training import state_hash
from transfer.native_support import NativeSupportModel, bank_digest, make_selector, native_forward


def _positive_seed(value):
    return type(value) is int and 0 <= value < 2**32


class NativeSupportRun(CurriculumRun):
    FORMAT = "vca-native-regional-run-v2"
    FAILURE_FORMAT = "vca-native-regional-failure-v2"

    def __init__(self, baseline, scene, record, *, trial, selector_seed, bank, donor=None, **kwargs):
        validate_source(baseline)
        if (any(kwargs.get(name) is not None for name in ("policy", "expected_policy_hash",
                "observation_policy", "expected_observation_hash")) or not _positive_seed(selector_seed)
                or set(trial) != {"mode", "schedule", "support_seed", "fixed"} or
                not _positive_seed(trial["support_seed"]) or type(trial["fixed"]) is not bool or
                trial["fixed"] != (donor is not None)):
            raise ValueError("Uncoupled registered native support trial required")
        self.baseline = baseline
        self.baseline_hash = state_hash(baseline)
        self.trial = deepcopy(trial)
        self.selector_seed = selector_seed
        self.bank_hash = bank_digest(bank)
        self.timings = {}
        self.train = {}
        self.devval = {}
        self.donor = None if donor is None else deepcopy(donor)
        source = NativeSupportModel.initialize(baseline, selector_seed)
        plan = kwargs["plan"]
        expected = set(map(str, plan.to_dict()["sampler"]["tile_ids"])) | {
            str(role["id"]) for role in record["roles"] if role["role"] == "devval"}
        if set(bank) != expected:
            raise ValueError("Context bank differs from the registered block coverage")
        roles = {str(role["id"]): role for role in record["roles"]}
        for key, entry in bank.items():
            identity, context = entry["identity"], entry["context"]
            if (identity["source"] != self.baseline_hash or identity["bands"] != scene.bands.fingerprint
                    or identity["scene"] != scene.dataset or key != context["block"]
                    or context["role"] != roles[key]["role"]):
                raise ValueError("Bank source, observation or role differs")
            selector = make_selector(entry, trial, trial["support_seed"])
            if selector.schedule.updates != plan.updates:
                raise ValueError("Context budget differs from exposure")
            if roles[key]["role"] == "devval":
                self.devval[key] = selector
            elif roles[key]["role"] == "devtrain":
                self.train[key] = selector
            else:
                raise ValueError("Released development roles required")
        self.fixed = {}
        if donor is not None:
            if set(donor["snapshots"]) != self.train.keys():
                raise ValueError("Donor support lacks training contexts")
            for key, selector in self.train.items():
                snapshot = donor["snapshots"][key]
                if snapshot["step"] != plan.updates:
                    raise ValueError("Only a completed donor may supply fixed supports")
                self.fixed[key] = selector.authenticate(snapshot)
        self.validation_labels = sparse_labels(record, scene.plan, "validation_support")
        self.validation_tiles = [scene.plan["tiles"][int(key)] for key in self.devval]
        validate(record, scene.plan)
        self.added = {name: tensor_hash(p) for name, p in source.named_parameters() if name not in baseline.state_dict()}
        kwargs.setdefault("expected_source_state", state_hash(source))
        kwargs.setdefault("policy", None)
        kwargs.setdefault("expected_policy_hash", None)
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        if (type(model) is not NativeSupportModel or any(
                v.device.type != "cpu" or v.dtype != torch.float32 or not torch.isfinite(v).all()
                for v in model.state_dict().values())):
            raise ValueError("Native runner requires an FP32 CPU model")

    def _contract_extensions(self):
        return {"regional_support": dict(baseline=self.baseline_hash, bank=self.bank_hash, trial=self.trial,
            scorer_seed=self.selector_seed, parameters=self.added, validation="fixed_step_zero",
            donor=self.donor, original_values="current_native_a0")}

    def _check_extensions(self):
        if self.baseline_hash != state_hash(self.baseline) or self.spec["regional_support"] != self._contract_extensions()["regional_support"]:
            raise ValueError("Native support initialization or contract changed")

    def selector_at(self, key, step):
        if key in self.fixed:
            return self.fixed[key]
        selector = self.train[key]
        selector.advance(step)
        return selector

    def _training_forward(self, tile, step, extra):
        key = str(tile["id"])
        started = perf_counter()
        active = self.fixed[key] if key in self.fixed else self.train[key]
        previous = active.cost.copy()
        selector = self.selector_at(key, step)
        self.timings.setdefault("schedule_replay", []).append(perf_counter() - started)
        output, frame = native_forward(self.model, self.scene, tile, selector=selector, timing=self.timings)
        for name, total in selector.cost.items():
            self.timings.setdefault(name, []).append(total - previous[name])
        output[-1]["regional_support"] = frame["trace"]
        return output

    def validation_predictions(self, model, *, detail=False):
        model.eval()
        result = []
        with torch.no_grad():
            for tile in self.validation_tiles:
                target = self.validation_labels[tile_slice(tile)].reshape(-1)
                valid = target > 0
                if valid.any():
                    output, frame = native_forward(model, self.scene, tile, selector=self.devval[str(tile["id"])],
                                                   timing=self.timings, detail=detail)
                    result.append((target[valid] - 1, tuple(head[valid].cpu().numpy() for head in output[:3]), frame))
        return result

    def _evaluate(self, model):
        started = perf_counter()
        matrix = np.zeros((self.scene.classes, self.scene.classes), dtype=np.int64)
        nll = 0.
        count = 0
        for target, predictions, _ in self.validation_predictions(model):
            probs = predictions[0]
            prediction = probs.argmax(1)
            matrix += np.bincount(target * self.scene.classes + prediction,
                                  minlength=self.scene.classes**2).reshape(matrix.shape)
            count += len(target)
            nll -= float(np.log(np.maximum(probs[np.arange(len(target)), target], 1e-8)).sum())
        if count != development_counts(self.record)[1]:
            raise ValueError("Fixed validation label budget differs")
        self.timings.setdefault("validation", []).append(perf_counter() - started)
        return dict(nll=nll / count, oa=int(np.trace(matrix)) / count, count=count, confusion=matrix.tolist(),
                    support=matrix.sum(1).tolist(), recall=(np.diag(matrix) / matrix.sum(1)).tolist())

    def _backward(self, loss):
        started = perf_counter()
        super()._backward(loss)
        self.timings.setdefault("backward", []).append(perf_counter() - started)

    def _history_extensions(self, outputs):
        return {"support": outputs[-1]["regional_support"],
                "train_probabilities": [tensor_hash(head) for head in outputs[:3]]}

    def _parameter_history(self, before):
        params = dict(self.model.named_parameters())
        return {"scorers": {name: dict(gradient_l1=None if params[name].grad is None else
            float(params[name].grad.abs().sum()), changed=int((before[name] != params[name].detach()).sum()))
            for name in self.added}}

    def _history_extension_keys(self):
        return {"support", "scorers", "train_probabilities"}

    def _validate_history_extensions(self, row):
        selector = self.selector_at(str(row["tile_id"]), row["step"])
        trace = row["support"]
        if (trace["snapshot"] != selector.snapshot() or trace["stage2_nnz"] != len(selector.selected)
                or trace["q_nnz"] < selector.n or trace["a0_nnz"] < selector.n
                or len(row["train_probabilities"]) != 3 or set(row["scorers"]) != self.added.keys()):
            raise ValueError("Registered support trajectory differs")
        for name, values in row["scorers"].items():
            gradient, changed = values["gradient_l1"], values["changed"]
            if (type(changed) is not int or not 0 <= changed <= dict(self.model.named_parameters())[name].numel()
                    or (gradient is None) != (name not in row["gradient_parameters"])
                    or (gradient is None and changed) or (gradient is not None and (not isfinite(gradient) or gradient < 0))):
                raise ValueError("Invalid scorer evidence")

    @classmethod
    def load(cls, baseline, scene, record, path, *, plan, trial, selector_seed,
             expected_sha256, expected_contract_hash, bank, donor=None):
        return super().load(baseline, scene, record, path, plan=plan, policy=None,
                            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(trial=trial, selector_seed=selector_seed, bank=bank, donor=donor))
