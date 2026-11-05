"""Twelve-update target engineering with pinned accepted full-source v2 weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint_baseline import NAMES
from joint_training import run_training
from loadData.dataset_registry import file_hash
from m4_full_target_runtime import FullSourceTargetRuntime, METHODS_ACTIVE, settings
from m4_ordered_source import MODES
from research_runtime import code_snapshot, require_frozen_code
from target_data_context import load_target_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "out-dir", "observation", "observation-record",
                "observation-record-sha256", "label-family", "label-family-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--target", choices=NAMES, required=True)
    parser.add_argument("--method", choices=METHODS_ACTIVE, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source-spec")
    parser.add_argument("--source-spec-sha256")
    parser.add_argument("--initial-state-sha256", required=True)
    parser.add_argument("--ordered-validation-v2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Expected clean remote commit required")
    options = settings(args.target, args.method, args.mode, device="cuda", ordered=args.ordered_validation_v2)
    data, provenance, guard = load_target_context(args.target, args.observation, args.observation_record,
        args.observation_record_sha256, args.label_family, args.label_family_sha256, per_class=10, support_seed=64)
    spec = {}
    if args.method == METHODS_ACTIVE[0]:
        if args.source_spec is not None or args.source_spec_sha256 is not None:
            raise ValueError("Scratch source specification forbidden")
    else:
        if not args.source_spec or file_hash(args.source_spec) != args.source_spec_sha256:
            raise ValueError("Pinned source specification required")
        spec = json.loads(Path(args.source_spec).read_text())
        provenance["protected_files"][args.source_spec] = args.source_spec_sha256
        for field in ("source_path", "source_result", "source_launch", "source_acceptance", "source_independent_audit"):
            provenance["protected_files"][spec[field]] = spec["source_weight_sha256" if field == "source_path" else field + "_sha256"]
        provenance["protected_files"][spec["source_path"] + ".json"] = spec["source_sidecar_sha256"]
    provenance.update(source_spec=spec, m4_mode=args.mode)
    def finish():
        guard()
        require_frozen_code(code)
        if any(file_hash(p) != h for p, h in provenance["protected_files"].items()):
            raise ValueError("Protected source/target input changed")
    finish()
    result = run_training(options, data, args.out_dir, provenance, code, device="cuda",
        resume=args.resume, stop_after=args.stop_after, finish_guard=finish,
        runtime_adapter=FullSourceTargetRuntime(options, data, args.out_dir, source_spec=spec,
                                               initial_state_sha256=args.initial_state_sha256))
    print("FULL_SOURCE_TARGET_SHORT", args.target, args.mode, args.method,
          result["operation"], result["updates_this_call"], flush=True)


if __name__ == "__main__":
    main()

