"""Bounded E1 target training with accepted frozen real-band views and BN.

Source is the accepted full NONE source; E1 parameters initialize at target.
No full profile, campaign runner, query release or shared default changes.
"""
import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import torch

from g4_coverage_target import (CALIBRATION, CALIBRATION_KEY, CONTROLS, KEY as BINDING_KEY,
    POLICY as INITIALIZATION_POLICY, PREFIX, _metadata, calibrate, initialize, verify_calibrated)
from g4_observation_target import (KEY as OBSERVATION_KEY, ObservationTargetRuntime,
    freeze_settings as observation_settings, prepare as observation_prepare)
from g4_support_context import check_files, read
from g4_support_target import settings as support_settings
from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from loadData.observation_sampling import BandMetadata
from m4_ordered_source import OrderedValidation
from model.checkpoint_io import load_model_checkpoint
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from target_initialization import METHODS, state_id
from target_training import loss_terms, observation, read_rows, save_tensor_mapping

POLICY = "g4_coverage_target_ordered_engineering_v1"
KEY = "target_coverage_policy"


def settings(base, condition):
    if (condition not in CONTROLS or base["profile"] != "engineering" or base["optimizer_steps"] != 12
            or KEY in base or base["config"]["model"]["region_residual_mode"] != "none"
            or base["config"]["model"].get("coverage_mode", "none") != "none"):
        raise ValueError("Separate none-architecture bounded E1 control required")
    value = copy.deepcopy(base)
    config = ResearchConfig.from_dict(base["config"])
    if condition != "none":
        config = replace(config, model=replace(config.model, coverage_mode="affine", coverage_condition=condition))
    value.update(protocol=POLICY, config=config.to_dict())
    value[KEY] = {"version": POLICY, "condition": condition, "base_observation_settings_sha256": digest(base),
        "initialization_component": INITIALIZATION_POLICY, "calibration_policy": CALIBRATION,
        "coverage_parameters": 0 if condition == "none" else 194,
        "source": "accepted_full_none_source_target_initialized_coverage",
        "maximum_optimizer_updates": 12, "full_experiment": False, "query_labels_accessed": False,
        "selection": "unchanged_first_min_fused_full_observation_validation_nll"}
    return value


class CoverageTargetRuntime(ObservationTargetRuntime):
    calibration_metadata_key = CALIBRATION_KEY

    def validate(self, options):
        base = support_settings(options["target"], options["method"], "none", model_seed=options["model_seed"],
            support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], source_spec=self.source_spec, device=options["config"]["device"])
        policy = options[OBSERVATION_KEY]
        frozen = observation_settings(base, self.items, self.metadata,
            preset=policy["preset"], sampling_seed=policy["sampling_seed"])
        expected = settings(frozen, options[KEY]["condition"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Bounded coverage/view/source/budget policy differs")
        return True

    def build_model(self, config, counts):
        target = self.settings["target"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("Coverage target class counts differ")
        model, self.coverage_binding = initialize(target, self.settings["method"], self.settings[KEY]["condition"],
            metadata=self.metadata, model_seed=self.settings["model_seed"], head_seed=self.settings["head_seed"],
            source_spec=self.source_spec, device=config.device)
        self.initialization = copy.deepcopy(self.coverage_binding["base_initialization"])
        return model

    def _features(self, model, name, item, metadata):
        _metadata(model, metadata)
        data = observation(model, item)
        # Preserve old selected-row head GEMMs. Do not compute heads on every
        # scene pixel and mask afterward: that would change training arithmetic.
        values = model.backbone.forward_features(data["net_input"], data["Q"], data["A"],
            wavelengths=data["wavelengths"], fwhm=data["fwhm"], dataset_name=name,
            superpixel_scale=data["superpixel_scale"], coverage_metadata=metadata,
            wavelength_unit=metadata.wavelength_unit, fwhm_unit=metadata.fwhm_unit or metadata.wavelength_unit,
            return_graph_info=True, return_diagnostics=True)
        return values[:3], values[3]["coverage"]

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        condition = self.settings[KEY]["condition"]
        if condition == "none":
            row = super().train_step_at(cursor, model, optimizer, name, item, config)
            row["coverage"] = {"condition": condition, "parameters": 0, "changed_elements": 0,
                "gradient_norm": 0., "fallback_reason": "disabled", "usable_anchors": None}
            return row
        view, metadata, record = self.view(cursor, name, item)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prefixes = ("backbone.HGNN_Branch.", PREFIX)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(prefixes)}
        features, coverage = self._features(model, name, view, metadata)
        ids = view["train_indices"].to(features[0].device)
        outputs = model.predict_features(name, *(v[ids] for v in features))
        terms = loss_terms(outputs, view["train_labels"].to(outputs[0].device), config)
        if not torch.isfinite(terms["total"]):
            raise ValueError("Nonfinite target coverage loss")
        terms["total"].backward(); gradients = {}
        for key, prefix in (("graph", "backbone.HGNN_Branch."), ("adapter", "backbone.adapter."),
                ("fused_head", "heads." + name + "."), ("graph_head", "hgnn_heads." + name + "."),
                ("cnn_head", "cnn_heads." + name + "."), ("coverage", PREFIX)):
            params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            bypass = key == "coverage" and coverage["usable_anchors"] == 0
            if (not params and not bypass) or any(not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Missing/nonfinite target gradient: " + key)
            gradients[key] = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
        optimizer.step()
        changed = {key: sum(int(torch.count_nonzero(p.detach() - before[n])) for n, p in model.named_parameters()
            if n in before and n.startswith(prefix)) for key, prefix in (("graph", prefixes[0]), ("coverage", PREFIX))}
        return {"dataset": name, "loss": {k: float(v.detach()) for k, v in terms.items()}, "gradient_norms": gradients,
            "graph_changed_elements": changed["graph"], "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            "observation": {"record_sha256": record["record_hash"], "bands": view["bands"],
                "source_bands": item["bands"], "metadata_sha256": metadata.fingerprint},
            "coverage": {"condition": condition, "parameters": 194, "changed_elements": changed["coverage"],
                "gradient_norm": gradients["coverage"], "fallback_reason": coverage["fallback_reason"],
                "usable_anchors": coverage["usable_anchors"], "metadata_sha256": metadata.fingerprint}}

    def validation_snapshot(self, model, items):
        if self.settings[KEY]["condition"] == "none":
            return super().validation_snapshot(model, items)
        model.eval(); name = self.settings["target"]; item = items[name]
        metrics, probabilities = {name: {}}, {name: {}}
        with torch.no_grad(), OrderedValidation():
            features, _ = self._features(model, name, item, self.metadata)
            for phase in ("train", "val"):
                ids = item[phase + "_indices"].to(features[0].device)
                outputs = model.predict_features(name, *(v[ids] for v in features))
                metrics[name][phase], probabilities[name][phase] = read_rows(outputs, item[phase + "_labels"], item["class_num"])
        val = metrics[name]["val"]["fused"]
        if not math.isfinite(val["nll"]):
            raise ValueError("Nonfinite coverage validation NLL")
        return {"val_loss": val["nll"], "macro_oa": val["oa"], "legacy_val_oa": {name: val["oa"]},
                "metrics": metrics}, probabilities

    def validation_candidate(self, model, items, raw_source_path):
        save_tensor_mapping({n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, raw_source_path)
        candidate, receipt = calibrate(model, observation(model, items[self.settings["target"]]), self.metadata,
            expected_binding=self.coverage_binding, source_state_sha256=state_id(model))
        receipt["raw_source_file_sha256"] = file_hash(raw_source_path)
        candidate.training_config[CALIBRATION_KEY] = copy.deepcopy(receipt)
        return candidate, receipt

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), BINDING_KEY: copy.deepcopy(self.coverage_binding),
                KEY: copy.deepcopy(self.settings[KEY])}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        model = load_model_checkpoint(path, device=device)
        if not hasattr(self, "coverage_binding"):
            self.build_model(ResearchConfig.from_dict(self.settings["config"]),
                             {self.settings["target"]: DATASETS[self.settings["target"]][2]})
        meta = model.training_config
        self.validate(meta["joint_runtime"])
        actual = {k: v for k, v in meta["joint_runtime"].items() if k not in ("executed_optimizer_steps", "selected_step")}
        if (actual != self.settings or meta.get("research_config") != self.settings["config"]
                or meta.get("target_initialization") != self.initialization
                or meta.get("support_context_binding") != self.settings["support_context_binding"]
                or meta.get(OBSERVATION_KEY) != self.settings[OBSERVATION_KEY] or meta.get(KEY) != self.settings[KEY]):
            raise ValueError("Selected coverage/view/context/config metadata differs")
        receipt = meta[CALIBRATION_KEY]
        if file_hash(raw_source_path) != receipt["raw_source_file_sha256"]:
            raise ValueError("Selected raw coverage source file differs")
        raw = torch.load(raw_source_path, map_location="cpu", weights_only=True)
        source = digest({k: tensor_hash(v) for k, v in raw.items()})
        verify_calibrated(model, observation(model, items[self.settings["target"]]), self.metadata,
                          expected_binding=self.coverage_binding, source_state_sha256=source)
        return model


def prepare(manifest, manifest_sha256, out, *, condition, **kwargs):
    base, items, provenance, code, _, guard = observation_prepare(manifest, manifest_sha256, out, **kwargs)
    options = settings(base, condition)
    runtime = CoverageTargetRuntime(options, items, out, source_spec=kwargs.get("source_spec"))
    provenance = {**provenance, "coverage_runtime": POLICY, "coverage_policy_sha256": digest(options[KEY])}
    return options, items, provenance, code, runtime, guard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "manifest", "manifest-sha256", "metadata", "metadata-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    parser.add_argument("--condition", required=True, choices=CONTROLS)
    for field in ("model-seed", "support-seed"):
        parser.add_argument("--" + field, required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--per-class", required=True, type=int, choices=(1, 5, 10))
    parser.add_argument("--preset", required=True, choices=("full", "alternate75"))
    parser.add_argument("--sampling-seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-spec")
    parser.add_argument("--source-spec-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv)
    source = code_snapshot()
    if source["commit"] != args.expected_commit:
        raise ValueError("Expected complete coverage runtime commit differs")
    require_frozen_code(source)
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Pinned source specification pair required")
    pins = {} if not args.source_spec else {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    metadata_pins = {str(Path(args.metadata).resolve(strict=True)): args.metadata_sha256}
    check_files({**pins, **metadata_pins})
    bound = prepare(args.manifest, args.manifest_sha256, args.out, target=args.target, method=args.method,
        condition=args.condition, model_seed=args.model_seed, support_seed=args.support_seed, per_class=args.per_class,
        metadata=BandMetadata(**read(args.metadata)), metadata_files=metadata_pins, preset=args.preset,
        sampling_seed=args.sampling_seed, source_spec=read(args.source_spec) if args.source_spec else None,
        source_files=pins, device=args.device)
    options, items, provenance, code, runtime, guard = bound
    with redirect_stdout(sys.stderr):
        result = run_training(options, items, args.out, provenance, code, runtime_adapter=runtime,
            finish_guard=guard, resume=args.resume, stop_after=args.stop_after, device=args.device)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
