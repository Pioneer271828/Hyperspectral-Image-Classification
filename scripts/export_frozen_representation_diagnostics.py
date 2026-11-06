"""Explicit post-analysis all54-boundary feature diagnostics; no label reader."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.frozen_representation_capture import export_case
from research_runtime import code_snapshot, require_frozen_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "freeze", "freeze-sha256", "analysis", "analysis-sha256", "case", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-regions", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variance-floor", required=True, type=float)
    args = parser.parse_args(argv); code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux frozen capture checkout required")
    require_frozen_code(code)
    result = export_case(args.freeze, args.freeze_sha256, args.analysis, args.analysis_sha256,
        args.case, args.out, code, device=args.device, maximum_regions=args.maximum_regions,
        seed=args.seed, variance_floor=args.variance_floor)
    print(json.dumps({"case": result["case"], "status": result["status"], "out": args.out}))


if __name__ == "__main__":
    main()
