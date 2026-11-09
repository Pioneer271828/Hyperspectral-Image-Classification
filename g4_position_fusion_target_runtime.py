"""Bounded12 position-fusion target updates with strict ordered selected-state restore."""
import argparse
import copy
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys

import torch

from g4_position_fusion_target import (CALIBRATION, CALIBRATION_KEY, CONDITIONS, KEY as BINDING_KEY,
    POLICY as INITIALIZATION_POLICY, calibrate, configuration, initialize,
    verify_calibrated, verify_initialization_binding, PREFIX)
from g4_support_context import check_files, read
from g4_support_target import (NoneSupportTargetRuntime, prepare as support_prepare,
                               settings as support_settings)
from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from g4_fusion_target import KEY as MATCHING_BINDING_KEY
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from target_initialization import METHODS, state_id
from target_training import observation, save_tensor_mapping

POLICY = "g4_m8_position_fusion_target_ordered_engineering_v1"
KEY = "target_position_fusion_policy"


def settings(base, condition):
    declared = configuration(condition)
    config = ResearchConfig.from_dict(base["config"])
    if (base["profile"] != "engineering" or base["optimizer_steps"] != 12 or KEY in base
            or config.model.region_residual_mode != "none" or config.model.coverage_mode != "none"
            or config.interaction.active or config.model.alph != .8
            or config.loss.cnn_loss_weight != 1. or config.loss.graph_branch_loss_weight != 1.
            or config.loss.balanced is not True):
        raise ValueError("Separate bounded position fusion profile over original none/M8 configuration required")
    value = {**copy.deepcopy(base), "protocol": POLICY, "config": config.to_dict()}
    value[KEY] = {"version": POLICY, "condition": condition, "base_settings_sha256": digest(base),
        "initialization_component": INITIALIZATION_POLICY, "calibration_policy": CALIBRATION,
        "source": "accepted_full_none_source_position_fusion_target", "maximum_optimizer_updates": 12,
        "full_experiment": False, "position_fusion": declared,
        "gate_parameter_count": declared["added_trainable_parameters"],
        "query_labels_accessed": False, "observation_augmentation": False,
        "selection": "unchanged_first_min_fused_full_observation_validation_nll"}
    return value


class PositionFusionTargetRuntime(NoneSupportTargetRuntime):
    calibration_metadata_key = CALIBRATION_KEY

    def validate(self, options):
        base = support_settings(options["target"], options["method"], "none", model_seed=options["model_seed"],
            support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], source_spec=self.source_spec, device=options["config"]["device"])
        expected = settings(base, options[KEY]["condition"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Position-fusion condition/source/context/configuration/budget differs")
        return True

    def build_model(self, config, counts):
        target = self.settings["target"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("Position-fusion target class counts differ")
        model, self.position_binding = initialize(target, self.settings["method"], self.settings[KEY]["condition"],
            model_seed=self.settings["model_seed"], head_seed=self.settings["head_seed"],
            source_spec=self.source_spec, device=config.device)
        self.initialization = copy.deepcopy(self.position_binding["base_binding"]["base_initialization"])
        return model

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        verify_initialization_binding(model, self.position_binding)
        if config.to_dict() != self.settings["config"]:
            raise ValueError("Actual position fusion training configuration differs")
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(PREFIX)}
        captured = []
        def check_before_step(*_):
            groups = {}
            for part in ("gate_weight", "gate_bias"):
                params = [p for n, p in model.named_parameters() if n == PREFIX + part]
                if len(params) != 1 or params[0].grad is None or not torch.isfinite(params[0].grad).all():
                    raise ValueError("Missing/nonfinite position fusion gradient before optimizer update")
                groups[part] = {"gradient_norm": float(params[0].grad.double().norm()),
                                "parameters": params[0].numel()}
            captured.append(groups)
        # Keep the original selected-row/loss/Adam path. A zero gradient is legal.
        hook = optimizer.register_step_pre_hook(check_before_step) if before else None
        try:
            row = super().train_step_at(cursor, model, optimizer, name, item, config)
        finally:
            if hook is not None:
                hook.remove()
        if before and len(captured) != 1:
            raise ValueError("Exactly one bounded position fusion optimizer step required")
        groups = captured[0] if captured else {}
        for part, stats in groups.items():
            stats["changed_elements"] = sum(int(torch.count_nonzero(p.detach() - before[n]))
                for n, p in model.named_parameters() if n == PREFIX + part)
        row["position_fusion"] = {"condition": self.settings[KEY]["condition"],
            "parameters": self.position_binding["added_parameter_count"],
            "configuration": configuration(self.settings[KEY]["condition"]), "groups": groups,
            "changed_elements": sum(v["changed_elements"] for v in groups.values()),
            "gradient_norm": math.sqrt(sum(v["gradient_norm"] ** 2 for v in groups.values()))}
        return row

    def validation_candidate(self, model, items, raw_source_path):
        save_tensor_mapping({n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, raw_source_path)
        candidate, receipt = calibrate(model, observation(model, items[self.settings["target"]]),
            expected_binding=self.position_binding, source_state_sha256=state_id(model))
        # The trainer serializes this declared receipt key and rebuilds canonical
        # metadata. Keep the complete hash-bound legacy receipt inside it; do
        # not add the trainer's reserved target_bn_calibration key to extras.
        receipt["legacy_calibration_receipt"] = copy.deepcopy(candidate.training_config["target_bn_calibration"])
        receipt["raw_source_file_sha256"] = file_hash(raw_source_path)
        candidate.training_config[CALIBRATION_KEY] = copy.deepcopy(receipt)
        return candidate, receipt

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), BINDING_KEY: copy.deepcopy(self.position_binding),
                MATCHING_BINDING_KEY: copy.deepcopy(self.position_binding["base_binding"]),
                KEY: copy.deepcopy(self.settings[KEY])}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        path = Path(path).resolve(strict=True)
        sidecar = Path(str(path) + ".json")
        pins = {str(p): file_hash(p) for p in (path, sidecar)}
        metadata = read(sidecar); meta = metadata["training_config"]
        # Initialization metadata records the registered training device. Rebuild
        # that identity before moving to the explicitly requested restore device.
        model = self.build_model(ResearchConfig.from_dict(self.settings["config"]),
                                 {self.settings["target"]: DATASETS[self.settings["target"]][2]})
        if {k: v for k, v in metadata.items() if k != "training_config"} != model.export_config():
            raise ValueError("Selected position-fusion-aware architecture differs")
        self.validate(meta["joint_runtime"])
        if ({k: v for k, v in meta["joint_runtime"].items() if k not in ("executed_optimizer_steps", "selected_step")} != self.settings
                or meta.get("research_config") != self.settings["config"]
                or meta.get("target_initialization") != self.initialization
                or meta.get("support_context_binding") != self.settings["support_context_binding"]
                or meta.get(MATCHING_BINDING_KEY) != self.position_binding["base_binding"]
                or meta.get(KEY) != self.settings[KEY] or meta.get(BINDING_KEY) != self.position_binding):
            raise ValueError("Selected position fusion source/context/condition/configuration differs")
        receipt = meta[CALIBRATION_KEY]
        if file_hash(raw_source_path) != receipt["raw_source_file_sha256"]:
            raise ValueError("Selected position fusion raw source file changed")
        legacy = receipt.get("legacy_calibration_receipt")
        if (not isinstance(legacy, dict) or digest(legacy) != receipt["legacy_receipt_sha256"]
                or receipt.get("legacy_receipt") != legacy
                or ("target_bn_calibration" in meta and meta["target_bn_calibration"] != legacy)):
            raise ValueError("Nested original calibration receipt identity differs")
        # Selected/raw checkpoint tensors decode only after declared guards.
        # The initialization factory above reuses authenticated source weights.
        check_files(pins)
        raw = torch.load(raw_source_path, map_location="cpu", weights_only=True)
        source = digest({n: tensor_hash(v) for n, v in raw.items()})
        model.to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        model.training_config = copy.deepcopy(meta); model.eval()
        model.training_config["target_bn_calibration"] = copy.deepcopy(legacy)
        verify_calibrated(model, observation(model, items[self.settings["target"]]),
                          expected_binding=self.position_binding, source_state_sha256=source)
        check_files(pins)
        return model


def prepare(manifest, manifest_sha256, out, *, condition, **kwargs):
    configuration(condition)
    base, items, provenance, code, _, guard = support_prepare(manifest, manifest_sha256, out, mode="none", **kwargs)
    options = settings(base, condition)
    runtime = PositionFusionTargetRuntime(options, items, out, source_spec=kwargs.get("source_spec"))
    return options, items, {**provenance, "position_fusion_runtime": POLICY, "position_fusion_policy_sha256": digest(options[KEY])}, code, runtime, guard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "manifest", "manifest-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    for field in ("model-seed", "support-seed"):
        parser.add_argument("--" + field, required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--per-class", required=True, type=int, choices=(1, 5, 10))
    parser.add_argument("--source-spec"); parser.add_argument("--source-spec-sha256")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv); code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux bounded position fusion runtime required")
    require_frozen_code(code)
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Source specification/hash pair required")
    pins = {} if not args.source_spec else {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    check_files(pins)
    options, items, provenance, source, runtime, guard = prepare(args.manifest, args.manifest_sha256, args.out,
        target=args.target, method=args.method, condition=args.condition, model_seed=args.model_seed,
        support_seed=args.support_seed, per_class=args.per_class, source_spec=read(args.source_spec) if args.source_spec else None,
        source_files=pins, device=args.device)
    with redirect_stdout(sys.stderr):
        result = run_training(options, items, args.out, provenance, source, runtime_adapter=runtime,
            finish_guard=guard, resume=args.resume, stop_after=args.stop_after, device=args.device)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
