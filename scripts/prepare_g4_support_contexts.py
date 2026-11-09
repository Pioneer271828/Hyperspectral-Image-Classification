"""Prepare27 sanitized support/K overlays; no optimizer, model or query evaluator."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_support_context import prepare_contexts
from research_runtime import code_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "inputs", "inputs-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote support-context producer required")
    prepare_contexts(args.inputs, args.inputs_sha256, args.out, code)


if __name__ == "__main__":
    main()
