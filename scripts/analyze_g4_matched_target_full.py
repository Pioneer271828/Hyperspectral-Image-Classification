"""Analyze original full54 query only after complete independent review and freeze."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g4_matched_target_analysis import export_analysis
from research_runtime import code_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "query", "query-sha256", "freeze", "freeze-sha256", "review", "review-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote analysis commit required")
    print(json.dumps(export_analysis(args.query, args.query_sha256, args.freeze, args.freeze_sha256,
        args.review, args.review_sha256, args.out, code), allow_nan=False))


if __name__ == "__main__":
    main()
