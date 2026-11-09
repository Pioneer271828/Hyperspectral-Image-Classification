"""Zero-forward CPU audit of one explicit E1 full-source target initialization."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from g4_coverage_target import CONTROLS, initialize
from g4_observation_target import prepare
from g4_support_context import check_files, read
from joint_baseline import NAMES
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code, write_json
from target_initialization import METHODS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "manifest", "manifest-sha256", "metadata", "metadata-sha256", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    parser.add_argument("--condition", required=True, choices=CONTROLS)
    parser.add_argument("--model-seed", required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--support-seed", required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--per-class", required=True, type=int, choices=(1, 5, 10))
    parser.add_argument("--source-spec")
    parser.add_argument("--source-spec-sha256")
    args = parser.parse_args(argv)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Expected complete initializer commit differs")
    require_frozen_code(code)
    if Path(args.out).exists():
        raise ValueError("Initialization audit output already exists")
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Pinned source specification pair required")
    pins = {} if not args.source_spec else {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    metadata_pins = {str(Path(args.metadata).resolve(strict=True)): args.metadata_sha256}
    check_files({**pins, **metadata_pins})
    metadata = BandMetadata(**read(args.metadata))
    options, items, provenance, _, _, guard = prepare(args.manifest, args.manifest_sha256, args.out,
        target=args.target, method=args.method, model_seed=args.model_seed, support_seed=args.support_seed,
        per_class=args.per_class, source_spec=read(args.source_spec) if args.source_spec else None,
        source_files=pins, metadata=metadata, metadata_files=metadata_pins, preset="full", sampling_seed=8128,
        device="cpu")
    model, binding = initialize(args.target, args.method, args.condition, metadata=metadata,
        model_seed=args.model_seed, head_seed=args.model_seed + 1000,
        source_spec=read(args.source_spec) if args.source_spec else None, device="cpu")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU initialization audit must not initialize CUDA")
    guard()
    out = Path(args.out); out.mkdir()
    write_json(out / "receipt.json", {"status": "passed", "code": code, "binding": binding,
        "support_context": provenance["support_context"], "observation_settings_sha256": digest(options),
        "metadata_files": metadata_pins, "source_files": pins, "optimizer_updates": 0, "model_forwards": 0,
        "bn_calibration_executed": False, "query_labels_accessed": False, "gpu_context_initialized": False,
        "training_executable": False, "total_parameters": sum(p.numel() for p in model.parameters())})
    print(json.dumps({"status": "passed", "out": str(out), "coverage_parameters": binding["coverage_parameter_count"]}))


if __name__ == "__main__":
    main()
