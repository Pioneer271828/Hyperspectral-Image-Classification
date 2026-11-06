"""Twelve-update full-source head-only / VCA-adapter controls; no query release."""
import argparse
import copy
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
import time

import torch
import g4_support_target as bounded
from joint_baseline import NAMES, _item_fingerprint
from joint_training import run_training
from loadData.dataset_registry import DATASETS
from m4_full_transfer import initialize_full_target
from m4_full_target_runtime import TARGET_ARITHMETIC
from m4_ordered_source import OrderedValidation
from model.research_config import digest
from none_full_transfer import initialize_none_full_target
from research_runtime import code_snapshot, require_frozen_code, tensor_hash, write_json
from target_bn_calibration import calibrate_target, POLICY as BN_POLICY, CONTEXT, GRAPH_POLICY
from target_initialization import METHODS, state_id
from target_training import HEAD_ONLY, TargetRuntime, loss_terms, observation, selected_outputs

POLICY = "g4_full_source_light_target_engineering_v1"
ADAPTER_HEADS = "source_pretrained_adapter_heads"
CONTROLS = (HEAD_ONLY, ADAPTER_HEADS)
HEAD_PREFIXES = ("heads.", "hgnn_heads.", "cnn_heads.")


def contract(method):
    if method not in CONTROLS:
        raise ValueError("Explicit head-only or VCA-adapter/heads control required")
    return {"method": method, "trainable_prefixes": list(HEAD_PREFIXES) +
        (["backbone.adapter."] if method == ADAPTER_HEADS else []),
        "backbone_training_mode": "eval_with_gradients_for_declared_parameters",
        "initial_bn": "ordered_unlabeled_target_scene_once_per_initialization",
        "candidate_bn": "unchanged_frozen_backbone_rebind" if method == HEAD_ONLY else "ordered_target_recalibration",
        "frozen_bn_during_optimizer_updates": True, "graph_and_cnn_and_m4_parameters_frozen": True,
        "optimizer_updates": 12, "query_labels_accessed": False}


def allowed(name, method):
    return name.startswith(tuple(contract(method)["trainable_prefixes"]))


def apply_parameters(model, method):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(allowed(name, method))
    model.eval()
    return parameter_binding(model, method)


def frozen_hash(model, method):
    return digest({n: tensor_hash(p) for n, p in model.named_parameters() if not allowed(n, method)})


def parameter_binding(model, method):
    if any(p.requires_grad != allowed(n, method) for n, p in model.named_parameters()):
        raise ValueError("Trainable parameter permission differs")
    names = {n: p.numel() for n, p in model.named_parameters() if allowed(n, method)}
    if not names or (method == ADAPTER_HEADS and not any(n.startswith("backbone.adapter.") for n in names)):
        raise ValueError("Declared trainable parameters missing")
    return {"policy": contract(method), "trainable_parameters": names, "trainable_elements": sum(names.values()),
            "frozen_parameter_sha256": frozen_hash(model, method)}


def initialize(target, method, mode, *, model_seed, head_seed, source_spec, device="cpu"):
    contract(method)
    if mode not in bounded.MODES or not source_spec:
        raise ValueError("Accepted full source and explicit architecture required")
    kwargs = dict(model_seed=model_seed, head_seed=head_seed, source_spec=source_spec, device=device)
    if mode == "none":
        model, record = initialize_none_full_target(target, METHODS[1], **kwargs)
    else:
        model, record = initialize_full_target(target, METHODS[1], mode, **kwargs)
    record = copy.deepcopy(record)
    record.update(method=method, full_source_initializer_method=METHODS[1],
                  head_only_requires_target_bn_calibration=method == HEAD_ONLY,
                  light_target_parameter_policy=contract(method))
    model.training_config = {"target_initialization": record}
    binding = apply_parameters(model, method)
    return model, record, binding


def settings(target, method, mode, *, model_seed, support_seed, per_class, binding, source_spec, device="cpu"):
    policy = contract(method)
    value = bounded.settings(target, METHODS[1], mode, model_seed=model_seed, support_seed=support_seed,
        per_class=per_class, binding=binding, source_spec=source_spec, device=device)
    value.update(protocol=POLICY, method=method, light_target_parameter_policy=policy)
    return value


class LightTargetRuntime(TargetRuntime):
    def validate(self, options):
        expected = settings(options["target"], options["method"], options["config"]["model"]["region_residual_mode"],
            model_seed=options["model_seed"], support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], source_spec=self.source_spec, device=options["config"]["device"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Frozen light target engineering policy differs")
        return True

    def validate_inputs(self, items):
        super().validate_inputs(items)
        if digest(_item_fingerprint(items[self.settings["target"]])) != self.settings["support_context_binding"]["item_sha256"]:
            raise ValueError("Bound support context changed")

    def _cost(self, phase, start):
        # Verification after completion stays readonly, including direct reload probes.
        if (self.directory / "result.json").exists():
            return
        path = self.directory / "adaptation-cost.json"
        binding = digest(self.settings)
        value = bounded.read(path) if path.exists() else {"settings_sha256": binding, "events": [],
            "scope": "initialization_calibration_optimizer_and_validation_with_invariant_checks",
            "excludes": ["input_preparation_guards", "trainer_serialization", "completed_readonly_verification"],
            "deployment_speedup_claim": False}
        if value["settings_sha256"] != binding:
            raise ValueError("Cost record belongs to another adaptation policy")
        value["events"].append({"phase": phase, "seconds": time.perf_counter() - start})
        write_json(path, value)

    def build_model(self, config, counts):
        target, method = self.settings["target"], self.settings["method"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("Target class counts differ")
        start = time.perf_counter()
        model, _, self.parameter_contract = initialize(target, method, config.model.region_residual_mode,
            model_seed=self.settings["model_seed"], head_seed=self.settings["head_seed"],
            source_spec=self.source_spec, device=config.device)
        self._cost("full_source_initialization", start)
        start = time.perf_counter()
        with OrderedValidation():
            model, self.initial_calibration = calibrate_target(model, observation(model, self.items[target]),
                mode=BN_POLICY, context=CONTEXT, graph_policy=GRAPH_POLICY, source_state_sha256=state_id(model))
        self.initial_calibration_seconds = time.perf_counter() - start
        self._cost("initial_ordered_target_calibration", start)
        self.initialization = copy.deepcopy(model.training_config["target_initialization"])
        self.optimizer_buffers = digest({n: tensor_hash(v) for n, v in model.named_buffers()})
        if parameter_binding(model, method) != self.parameter_contract:
            raise ValueError("Calibration changed parameter permissions or values")
        return model

    def optimizer(self, model, config):
        binding = parameter_binding(model, self.settings["method"])
        if binding != self.parameter_contract:
            raise ValueError("Optimizer parameter contract changed")
        groups = []
        if self.settings["method"] == ADAPTER_HEADS:
            groups.append({"params": list(model.backbone.adapter.parameters()), "lr": config.training.lr})
        groups.append({"params": [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIXES)],
                       "lr": config.training.head_lr})
        return torch.optim.Adam(groups, weight_decay=config.training.weight_decay)

    def _bank(self, *args, **kwargs):
        with OrderedValidation():
            return super()._bank(*args, **kwargs)

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        start = time.perf_counter()
        method = self.settings["method"]
        if parameter_binding(model, method) != self.parameter_contract:
            raise ValueError("Frozen parameter contract changed before update")
        if digest({n: tensor_hash(v) for n, v in model.named_buffers()}) != self.optimizer_buffers:
            raise ValueError("Frozen optimizer-phase buffers changed before update")
        before = {n: p.detach().clone() for n, p in model.named_parameters() if allowed(n, method)}
        buffers = {n: p.detach().clone() for n, p in model.named_buffers()}
        if method == HEAD_ONLY:
            row = super().train_step_at(cursor, model, optimizer, name, item, config)
        else:
            model.train(); model.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            outputs = selected_outputs(model, name, item, "train")
            terms = loss_terms(outputs, item["train_labels"].to(outputs[0].device), config)
            if not torch.isfinite(terms["total"]):
                raise ValueError("Nonfinite adapter-heads loss")
            terms["total"].backward()
            grad = {}
            for key, prefix in (("adapter", "backbone.adapter."), ("fused_head", "heads."),
                                ("graph_head", "hgnn_heads."), ("cnn_head", "cnn_heads.")):
                params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise ValueError("Missing/nonfinite declared gradient: " + key)
                grad[key] = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
            optimizer.step()
            row = {"dataset": name, "loss": {k: float(v.detach()) for k, v in terms.items()},
                   "gradient_norms": grad, "learning_rates": [g["lr"] for g in optimizer.param_groups]}
        if (any(p.grad is not None for n, p in model.named_parameters() if not allowed(n, method))
                or parameter_binding(model, method) != self.parameter_contract
                or any(not torch.equal(v, buffers[n]) for n, v in model.named_buffers())):
            raise ValueError("Optimizer changed frozen parameters, buffers or gradients")
        row.update(frozen_parameters_unchanged=True, frozen_buffers_unchanged=True,
            changed_trainable_elements=sum(int(torch.count_nonzero(p.detach() - before[n]))
                for n, p in model.named_parameters() if n in before),
            adapter_changed_elements=sum(int(torch.count_nonzero(p.detach() - before[n]))
                for n, p in model.named_parameters() if n.startswith("backbone.adapter.") and n in before))
        self._cost("optimizer_update_with_invariant_checks", start)
        return row

    def validation_candidate(self, *args):
        start = time.perf_counter()
        with OrderedValidation():
            value = super().validation_candidate(*args)
        self._cost("candidate_calibration_or_frozen_rebind", start)
        return value

    def validation_snapshot(self, *args):
        start = time.perf_counter()
        with OrderedValidation():
            value = super().validation_snapshot(*args)
        self._cost("validation_snapshot", start)
        return value

    def checkpoint_metadata(self, model):
        if parameter_binding(model, self.settings["method"]) != self.parameter_contract:
            raise ValueError("Checkpoint parameter contract changed")
        return {**super().checkpoint_metadata(model), "light_target_contract": self.parameter_contract,
                "support_context_binding": self.settings["support_context_binding"],
                "target_validation_arithmetic": copy.deepcopy(TARGET_ARITHMETIC)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        model = super().load_selected(path, items, raw_source_path, device=device)
        actual = apply_parameters(model, self.settings["method"])
        meta = model.training_config
        if (actual != self.parameter_contract or meta.get("light_target_contract") != actual
                or meta.get("support_context_binding") != self.settings["support_context_binding"]
                or meta.get("target_validation_arithmetic") != TARGET_ARITHMETIC):
            raise ValueError("Selected light target permission/context/arithmetic differs")
        return model


def prepare(manifest, manifest_sha256, out, *, target, method, mode, model_seed, support_seed,
            per_class, source_spec, source_files=None, device="cpu"):
    contract(method)
    options, items, provenance, source, _, guard = bounded.prepare(manifest, manifest_sha256, out,
        target=target, method=METHODS[1], mode=mode, model_seed=model_seed, support_seed=support_seed,
        per_class=per_class, source_spec=source_spec, source_files=source_files, device=device)
    options = settings(target, method, mode, model_seed=model_seed, support_seed=support_seed,
        per_class=per_class, binding=options["support_context_binding"], source_spec=source_spec, device=device)
    provenance = {**provenance, "policy": POLICY, "light_target_parameter_policy": contract(method)}
    runtime = LightTargetRuntime(options, items, out, source_spec=source_spec)
    return options, items, provenance, source, runtime, guard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "manifest", "manifest-sha256", "out", "source-spec", "source-spec-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--target", choices=NAMES, required=True)
    parser.add_argument("--method", choices=CONTROLS, required=True)
    parser.add_argument("--mode", choices=bounded.MODES, required=True)
    parser.add_argument("--model-seed", type=int, choices=(64, 512, 2048), required=True)
    parser.add_argument("--support-seed", type=int, choices=(64, 512, 2048), required=True)
    parser.add_argument("--per-class", type=int, choices=(1, 5, 10), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Expected light target commit differs")
    require_frozen_code(code)
    pins = {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    bounded.check_files(pins)
    options, items, provenance, code, runtime, guard = prepare(args.manifest, args.manifest_sha256, args.out,
        target=args.target, method=args.method, mode=args.mode, model_seed=args.model_seed,
        support_seed=args.support_seed, per_class=args.per_class, device=args.device,
        source_spec=bounded.read(args.source_spec), source_files=pins)
    start = time.perf_counter()
    with redirect_stdout(sys.stderr):
        value = run_training(options, items, args.out, provenance, code, runtime_adapter=runtime,
            finish_guard=guard, device=args.device, resume=args.resume, stop_after=args.stop_after)
    print(json.dumps({**value, "trainer_call_wall_seconds": time.perf_counter() - start}, allow_nan=False))


if __name__ == "__main__":
    main()
