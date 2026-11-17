"""Capture the original Issue 184 pre-adaptation states on the authorized CPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from evaluation.frozen_initial_capture import export
from research_runtime import code_snapshot, hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "analysis", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--initialization-only", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or code["commit"] != args.expected_commit:
        raise ValueError("Exact Linux producer required")
    torch.set_num_threads(1)
    path, status = export(args.freeze, args.analysis, args.out, code, initialization_only=args.initialization_only)
    print(json.dumps({"path":str(path), "sha256":hash_file(path), "status":status}))
    return int(status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
