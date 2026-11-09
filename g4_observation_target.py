"""Frozen real-band views for bounded ordered target engineering (M1).

Reuse the accepted sampler and target update/calibration code. Full-scene graph
and normalization context remains available; no unseen-sensor claim follows.
"""
import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from g4_support_context import check_files, read
from g4_support_target import (NoneSupportTargetRuntime, prepare as support_prepare,
                               settings as support_settings)
from joint_baseline import NAMES, _item_fingerprint
from joint_observation import PRESETS, _metadata_check, _stub
from joint_training import run_training
from loadData.observation_sampling import BandMetadata, replay, sample
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from target_initialization import METHODS

POLICY = "g4_target_real_band_engineering_v1"
KEY = "target_observation_policy"


def freeze_settings(base, items, metadata, *, preset, sampling_seed):
    """Freeze all twelve label-blind records without advancing training RNGs."""
    target = base["target"]
    if (preset not in PRESETS or type(sampling_seed) is not int or sampling_seed < 0
            or KEY in base or base["optimizer_steps"] != 12 or base["profile"] != "engineering"
            or base["config"]["model"]["region_residual_mode"] != "none"
            or base["config"]["model"].get("coverage_mode", "none") != "none"
            or tuple(items) != (target,) or base["method"] not in METHODS[:2]):
        raise ValueError("Single none-mode bounded target and registered M1 preset/seed required")
    if not isinstance(metadata, BandMetadata):
        raise ValueError("Explicit BandMetadata declaration required")
    item = items[target]
    if any(k.startswith(("test_", "query_")) for k in item):
        raise ValueError("No query/test fields permitted")
    _metadata_check(item, metadata)
    # Plan identity is the accepted support-context binding, never label values
    # as a sampling seed. _stub has no actual scene pixels or supervision.
    plan_hash = digest(base["support_context_binding"])
    rng = np.random.default_rng(sampling_seed)
    records = []
    for index in range(12):
        selected, record = sample(_stub(metadata, plan_hash), metadata,
                           PRESETS[preset][index % len(PRESETS[preset])], rng)
        # A subset may make legacy auto units ambiguous even if the full
        # vector was unambiguous (e.g. mixed 1nm/20nm widths). Reject before
        # any outputs/updates; do not silently alter inherited arithmetic.
        sub = selected.bands
        _metadata_check({"net_input": torch.empty(1, 1, len(sub.band_ids) + 2),
            "bands": len(sub.band_ids), "wavelengths": torch.tensor(sub.wavelengths, dtype=item["wavelengths"].dtype),
            "fwhm": None if sub.fwhm is None else torch.tensor(sub.fwhm, dtype=item["fwhm"].dtype)}, sub)
        records.append(record)
    policy = {"version": POLICY, "preset": preset, "sampling_seed": sampling_seed,
        "base_settings_sha256": digest(base), "metadata": asdict(metadata),
        "plan_hash": plan_hash, "item_sha256": digest(_item_fingerprint(item)), "records": records,
        "calibration_and_selection": "unchanged_ordered_full_observation",
        "graph_context": "fixed_full_observation_label_free_target_graph",
        "normalization": "select_existing_full_scene_normalized_columns",
        "physical_calibration_authenticated": False, "unseen_sensor_claim": False}
    policy = json.loads(json.dumps(policy, allow_nan=False))
    policy["policy_sha256"] = digest(policy)
    return {**base, "protocol": POLICY, KEY: policy}


class ObservationTargetRuntime(NoneSupportTargetRuntime):
    def __init__(self, options, *args, **kwargs):
        self.metadata = BandMetadata(**options[KEY]["metadata"])
        super().__init__(options, *args, **kwargs)

    def validate(self, options):
        base = support_settings(options["target"], options["method"], "none",
            model_seed=options["model_seed"], support_seed=options["support_seed"],
            per_class=options["support_per_class"], binding=options["support_context_binding"],
            source_spec=self.source_spec, device=options["config"]["device"])
        policy = options[KEY]
        expected = freeze_settings(base, self.items, self.metadata,
            preset=policy["preset"], sampling_seed=policy["sampling_seed"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Frozen target observation profile or replay identity differs")
        return True

    def validate_inputs(self, items):
        super().validate_inputs(items)
        _metadata_check(items[self.settings["target"]], self.metadata)

    def view(self, cursor, name, item):
        if type(cursor) is not int or not 0 <= cursor < 12 or name != self.settings["target"]:
            raise ValueError("Global observation cursor/target outside frozen schedule")
        _metadata_check(item, self.metadata)
        policy = self.settings[KEY]
        record = policy["records"][cursor]
        meta = replay(_stub(self.metadata, policy["plan_hash"]), self.metadata, record).bands
        positions = record["selected_positions"]
        if positions == list(range(len(self.metadata.band_ids))):
            return item, meta, record
        value = item["net_input"]
        ids = torch.tensor(positions, dtype=torch.long, device=value.device)
        view = {**item, "bands": len(positions),
            "net_input": torch.cat((value[..., :-2].index_select(-1, ids), value[..., -2:]), -1),
            "wavelengths": torch.as_tensor(item["wavelengths"], device=value.device).index_select(0, ids),
            "fwhm": None if item["fwhm"] is None else torch.as_tensor(item["fwhm"], device=value.device).index_select(0, ids)}
        return view, meta, record

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        view, metadata, record = self.view(cursor, name, item)
        # Metadata units were checked against the inherited auto-unit path.
        # Keep original loss/optimizer/training arithmetic and no new RNG calls.
        row = super().train_step_at(cursor, model, optimizer, name, view, config)
        row["observation"] = {"record_sha256": record["record_hash"], "bands": view["bands"],
            "source_bands": item["bands"], "metadata_sha256": metadata.fingerprint}
        return row

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), KEY: copy.deepcopy(self.settings[KEY])}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        if model.training_config.get(KEY) != self.settings[KEY]:
            raise ValueError("Selected target observation metadata differs")
        return model


def prepare(manifest, manifest_sha256, out, *, metadata, preset, sampling_seed, metadata_files=None, **kwargs):
    if kwargs.get("mode", "none") != "none":
        raise ValueError("M1 engineering is a separate none architecture factor")
    pins = dict(metadata_files or {})
    check_files(pins)
    base, items, provenance, code, _, guard = support_prepare(manifest, manifest_sha256, out,
        mode="none", **{k: v for k, v in kwargs.items() if k != "mode"})
    output = Path(out).resolve()
    for path in pins:
        parent = Path(path).resolve(strict=True).parent
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError("Output must be separate from metadata input directories")
    options = freeze_settings(base, items, metadata, preset=preset, sampling_seed=sampling_seed)
    runtime = ObservationTargetRuntime(options, items, out, source_spec=kwargs.get("source_spec"))
    provenance = {**provenance, "observation_policy": POLICY,
                  "metadata_files": pins, "observation_policy_sha256": options[KEY]["policy_sha256"]}
    def finish():
        guard()
        check_files(pins)
    finish()
    return options, items, provenance, code, runtime, finish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "manifest", "manifest-sha256", "out", "metadata", "metadata-sha256"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    for field in ("model-seed", "support-seed"):
        parser.add_argument("--" + field, required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--per-class", required=True, type=int, choices=(1, 5, 10))
    parser.add_argument("--preset", required=True, choices=PRESETS)
    parser.add_argument("--sampling-seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-spec")
    parser.add_argument("--source-spec-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Expected complete engineering commit differs")
    require_frozen_code(code)
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Source specification file and hash must be paired")
    pins = {} if not args.source_spec else {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    metadata_pins = {str(Path(args.metadata).resolve(strict=True)): args.metadata_sha256}
    check_files({**pins, **metadata_pins})
    bound = prepare(args.manifest, args.manifest_sha256, args.out, target=args.target, method=args.method,
        model_seed=args.model_seed, support_seed=args.support_seed, per_class=args.per_class,
        source_spec=read(args.source_spec) if args.source_spec else None, source_files=pins, device=args.device,
        metadata=BandMetadata(**read(args.metadata)), preset=args.preset, sampling_seed=args.sampling_seed,
        metadata_files=metadata_pins)
    options, items, provenance, code, runtime, guard = bound
    with redirect_stdout(sys.stderr):
        result = run_training(options, items, args.out, provenance, code, runtime_adapter=runtime,
            finish_guard=guard, resume=args.resume, stop_after=args.stop_after, device=args.device)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
