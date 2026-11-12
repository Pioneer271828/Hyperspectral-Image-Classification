"""Produce a pinned dynamic-vs-fixed matched cost comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.dynamic_cost_pairs import matched_cost_pairs
from evaluation.wp12_cost_primitives import read_pinned
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "curve", "curve-sha256", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux comparison checkout required")
    out = external_directory(args.out)
    curve = read_pinned(args.curve, args.curve_sha256)
    registration = read_pinned(args.registration, args.registration_sha256)
    report = matched_cost_pairs(curve, registration, expected_spec_hash=digest(registration),
                                expected_curve_hash=digest(curve))
    report.update(input_file_sha256={"curve": args.curve_sha256, "registration": args.registration_sha256},
                  consumer_commit=code["commit"])
    require_frozen_code(code)
    if hash_file(args.curve) != args.curve_sha256 or hash_file(args.registration) != args.registration_sha256:
        raise ValueError("Comparison inputs changed")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with out.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


if __name__ == "__main__":
    main()
