"""Read-only CPU cost for the three frozen WP11 static endpoints."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_static_cost import PATHS, run, startup
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "release", "release-sha256"):
        parser.add_argument("--" + key, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", choices=PATHS)
    mode.add_argument("--out")
    args = parser.parse_args()
    code = code_snapshot()
    if not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux container checkout required")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Only CPU cost is registered")
    pin = dict(path=args.release, sha256=args.release_sha256)
    if args.worker:
        print(json.dumps(startup(pin, args.worker), sort_keys=True))
    else:
        run(pin, code["commit"], external_directory(args.out))
    require_frozen_code(code)


if __name__ == "__main__":
    main()
