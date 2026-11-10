"""Additive cost worker on preserved Issue184 source; no modern model imports."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_cost_primitives import read_pinned
from evaluation.wp12_resident_cost import environment, measure
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code


PRODUCER = "331d718567b1a9726934f445665999df972392b1"


def preserved_runtime(freeze, code):
    original = freeze["producer"]
    if (freeze.get("status") != "all54_frozen_before_query"
            or freeze.get("policy") != "g4_matched_ordered_all54_frozen_query_v2"
            or original.get("commit") != PRODUCER or original.get("dirty") is not False or code.get("dirty") is not False
            or not original.get("files") or digest(original["files"]) != original.get("source_hash")
            or not code.get("files") or digest(code["files"]) != code.get("source_hash")
            or any(code["files"].get(p) != h for p, h in original["files"].items())):
        raise ValueError("Every original frozen Python/config file must be preserved")
    return dict(runtime_commit=PRODUCER, runtime_source_hash=original["source_hash"],
                preserved_original_files=len(original["files"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "freeze", "freeze-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    env = sub.add_parser("environment")
    env.add_argument("--device", choices=("cpu", "cuda"), required=True)
    env.add_argument("--arithmetic", choices=("native", "ordered"), required=True)
    worker = sub.add_parser("measure")
    worker.add_argument("--job", required=True)
    worker.add_argument("--job-sha256", required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux historical worker required")
    identity = preserved_runtime(read_pinned(args.freeze, args.freeze_sha256), code)
    out = external_directory(args.out)
    if out.exists():
        raise ValueError("New external output required")
    if args.operation == "environment":
        result = environment(args.device, args.arithmetic)
    else:
        job = read_pinned(args.job, args.job_sha256)
        if job["arithmetic"] != "ordered":
            raise ValueError("Frozen Issue184 requires ordered arithmetic")
        result = measure(job)
        result.update(identity, job_sha256=args.job_sha256, producer_commit=code["commit"], fresh_process=True)
        if hash_file(args.job) != args.job_sha256:
            raise ValueError("Job changed during measurement")
    require_frozen_code(code)
    if hash_file(args.freeze) != args.freeze_sha256:
        raise ValueError("Frozen runtime source manifest changed")
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
