"""CPU-only zero-forward authenticated full-source M7 target initialization."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from g4_edge_target import CONDITIONS, initialize
from g4_support_context import read, check_files
from g4_support_target import prepare
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code, write_json
from target_initialization import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "manifest", "manifest-sha256", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    for key in ("model-seed", "support-seed"):
        parser.add_argument("--" + key, required=True, type=int, choices=(64, 512, 2048))
    parser.add_argument("--per-class", required=True, type=int, choices=(1, 5, 10))
    parser.add_argument("--source-spec"); parser.add_argument("--source-spec-sha256")
    args = parser.parse_args(); code = code_snapshot()
    if code["commit"] != args.expected_commit or Path(args.out).exists():
        raise ValueError("Expected clean initializer and fresh output required")
    require_frozen_code(code)
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Source specification/hash pair required")
    pins = {} if not args.source_spec else {str(Path(args.source_spec).resolve(strict=True)): args.source_spec_sha256}
    check_files(pins); spec = read(args.source_spec) if args.source_spec else None
    options, _, provenance, _, _, guard = prepare(args.manifest, args.manifest_sha256, args.out,
        target=args.target, method=args.method, mode="none", model_seed=args.model_seed, support_seed=args.support_seed,
        per_class=args.per_class, source_spec=spec, source_files=pins, device="cpu")
    model, binding = initialize(args.target, args.method, args.condition, model_seed=args.model_seed,
        head_seed=args.model_seed + 1000, source_spec=spec, device="cpu")
    guard()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU initialization audit cannot initialize CUDA")
    out = Path(args.out); out.mkdir()
    write_json(out / "receipt.json", {"status": "passed", "code": code, "binding": binding,
        "support_context": provenance["support_context"], "source_files": pins, "base_settings_sha256": digest(options),
        "model_forwards": 0, "bn_calibration_executed": False, "optimizer_updates": 0,
        "query_labels_accessed": False, "gpu_context_initialized": False, "training_executable": False})
    print(json.dumps({"status": "passed", "added_parameters": binding["added_parameter_count"]}))


if __name__ == "__main__":
    main()
