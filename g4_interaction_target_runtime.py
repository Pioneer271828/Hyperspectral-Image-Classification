"""Bounded existing-M6 target updates; no full profile, augmentation or query."""
import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import torch

from g4_interaction_target import (CALIBRATION, CALIBRATION_KEY, CONDITIONS, KEY as BINDING_KEY,
    POLICY as INITIALIZATION_POLICY, PREFIX, calibrate, configuration, initialize,
    verify_calibrated, verify_initialization_binding)
from g4_support_context import check_files, read
from g4_support_target import (NoneSupportTargetRuntime, prepare as support_prepare,
                               settings as support_settings)
from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from model.interaction_config import InteractionConfig
from model.checkpoint_io import load_model_checkpoint
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from target_initialization import METHODS, state_id
from target_training import observation, save_tensor_mapping

POLICY = "g4_m6_target_ordered_engineering_v1"
KEY = "target_interaction_policy"


def settings(base, condition):
    declared = configuration(condition)
    config = ResearchConfig.from_dict(base["config"])
    if (base["profile"] != "engineering" or base["optimizer_steps"] != 12 or KEY in base
            or config.model.region_residual_mode != "none" or config.model.coverage_mode != "none"
            or config.interaction.active):
        raise ValueError("Separate bounded M6 target profile over unchanged none base required")
    config = replace(config, interaction=InteractionConfig(**declared))
    value = {**copy.deepcopy(base), "protocol": POLICY, "config": config.to_dict()}
    value[KEY] = {"version": POLICY, "condition": condition, "base_settings_sha256": digest(base),
        "initialization_component": INITIALIZATION_POLICY, "calibration_policy": CALIBRATION,
        "source": "accepted_full_none_source_target_initialized_M6", "maximum_optimizer_updates": 12,
        "full_experiment": False, "query_labels_accessed": False, "observation_augmentation": False,
        "selection": "unchanged_first_min_fused_full_observation_validation_nll"}
    return value


class InteractionTargetRuntime(NoneSupportTargetRuntime):
    calibration_metadata_key = CALIBRATION_KEY

    def validate(self, options):
        base = support_settings(options["target"], options["method"], "none", model_seed=options["model_seed"],
            support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], source_spec=self.source_spec, device=options["config"]["device"])
        expected = settings(base, options[KEY]["condition"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("M6 condition/source/context/configuration/budget differs")
        return True

    def build_model(self, config, counts):
        target = self.settings["target"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("M6 target class counts differ")
        model, self.interaction_binding = initialize(target, self.settings["method"], self.settings[KEY]["condition"],
            model_seed=self.settings["model_seed"], head_seed=self.settings["head_seed"],
            source_spec=self.source_spec, device=config.device)
        self.initialization = copy.deepcopy(self.interaction_binding["base_initialization"])
        return model

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        verify_initialization_binding(model, self.interaction_binding)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(PREFIX)}
        captured = []
        def check_before_step(*_):
            gradients = {}
            for direction in ("graph_to_cnn", "cnn_to_graph"):
                prefix = PREFIX + direction + "."
                params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
                if not params:
                    gradients[direction] = None
                    continue
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise ValueError("Missing/nonfinite M6 direction gradient before optimizer update")
                gradients[direction] = {"gradient_norm": math.sqrt(sum(float(p.grad.double().square().sum()) for p in params)),
                                        "parameters": sum(p.numel() for p in params)}
            captured.append(gradients)
        # Reuse the exact original selected-row/loss/optimizer update function.
        # Only enabled messages install a non-mutating pre-step gradient check.
        hook = optimizer.register_step_pre_hook(check_before_step) if before else None
        try:
            row = super().train_step_at(cursor, model, optimizer, name, item, config)
        finally:
            if hook is not None:
                hook.remove()
        if before and len(captured) != 1:
            raise ValueError("Exactly one bounded M6 optimizer step required")
        directions = captured[0] if captured else {"graph_to_cnn": None, "cnn_to_graph": None}
        for direction, stats in directions.items():
            if stats is not None:
                prefix = PREFIX + direction + "."
                stats["changed_elements"] = sum(int(torch.count_nonzero(p.detach() - before[n]))
                    for n, p in model.named_parameters() if n.startswith(prefix))
        row["interaction"] = {"condition": self.settings[KEY]["condition"],
            "parameters": self.interaction_binding["message_parameter_count"], "directions": directions,
            "changed_elements": sum(v["changed_elements"] for v in directions.values() if v is not None),
            "gradient_norm": math.sqrt(sum(v["gradient_norm"] ** 2 for v in directions.values() if v is not None))}
        return row

    def validation_candidate(self, model, items, raw_source_path):
        save_tensor_mapping({n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, raw_source_path)
        candidate, receipt = calibrate(model, observation(model, items[self.settings["target"]]),
            expected_binding=self.interaction_binding, source_state_sha256=state_id(model))
        receipt["raw_source_file_sha256"] = file_hash(raw_source_path)
        candidate.training_config[CALIBRATION_KEY] = copy.deepcopy(receipt)
        return candidate, receipt

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), BINDING_KEY: copy.deepcopy(self.interaction_binding),
                KEY: copy.deepcopy(self.settings[KEY])}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        model = load_model_checkpoint(path, device=device)
        if not hasattr(self, "interaction_binding"):
            self.build_model(ResearchConfig.from_dict(self.settings["config"]),
                             {self.settings["target"]: DATASETS[self.settings["target"]][2]})
        meta = model.training_config
        self.validate(meta["joint_runtime"])
        if ({k: v for k, v in meta["joint_runtime"].items() if k not in ("executed_optimizer_steps", "selected_step")} != self.settings
                or meta.get("research_config") != self.settings["config"]
                or meta.get("target_initialization") != self.initialization
                or meta.get("support_context_binding") != self.settings["support_context_binding"]
                or meta.get(KEY) != self.settings[KEY] or meta.get(BINDING_KEY) != self.interaction_binding):
            raise ValueError("Selected M6 source/context/condition/configuration differs")
        receipt = meta[CALIBRATION_KEY]
        if file_hash(raw_source_path) != receipt["raw_source_file_sha256"]:
            raise ValueError("Selected M6 raw source file changed")
        raw = torch.load(raw_source_path, map_location="cpu", weights_only=True)
        source = digest({n: tensor_hash(v) for n, v in raw.items()})
        verify_calibrated(model, observation(model, items[self.settings["target"]]),
                          expected_binding=self.interaction_binding, source_state_sha256=source)
        return model


def prepare(manifest, manifest_sha256, out, *, condition, **kwargs):
    configuration(condition)
    base, items, provenance, code, _, guard = support_prepare(manifest, manifest_sha256, out, mode="none", **kwargs)
    options = settings(base, condition)
    runtime = InteractionTargetRuntime(options, items, out, source_spec=kwargs.get("source_spec"))
    return options, items, {**provenance, "interaction_runtime": POLICY, "interaction_policy_sha256": digest(options[KEY])}, code, runtime, guard


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
        raise RuntimeError("Clean expected Linux bounded M6 runtime required")
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
