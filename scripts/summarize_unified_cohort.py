"""Summarize pinned accuracy and cost records for a single WP12 cohort."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.unified_cohort import summarize
from research_runtime import code_snapshot, external_directory, require_frozen_code


def read_pinned(path, expected_hash):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Input SHA-256 differs: " + str(path))
    return json.loads(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "plan", "plan-sha256", "evidence", "evidence-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux reporting checkout required")
    out = external_directory(args.out)
    plan = read_pinned(args.plan, args.plan_sha256)
    evidence = read_pinned(args.evidence, args.evidence_sha256)
    report = summarize(plan, evidence)
    report.update(consumer_commit=code["commit"], plan_sha256=args.plan_sha256,
                  evidence_sha256=args.evidence_sha256)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    require_frozen_code(code)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


if __name__ == "__main__":
    main()
