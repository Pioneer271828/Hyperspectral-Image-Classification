"""Fresh native-model cost worker or complete cohort receipt assembly (no labels)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_resident_cost import assemble, environment, measure
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from scripts.summarize_unified_cohort import read_pinned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    env = sub.add_parser("environment")
    env.add_argument("--device", choices=("cpu", "cuda"), required=True)
    env.add_argument("--arithmetic", choices=("native", "ordered"), required=True)
    worker = sub.add_parser("measure")
    worker.add_argument("--job", required=True)
    worker.add_argument("--job-sha256", required=True)
    join = sub.add_parser("assemble")
    for field in ("plan", "plan-sha256", "measurements", "measurements-sha256"):
        join.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux cost checkout required")
    out = external_directory(args.out)
    if out.exists():
        raise ValueError("New external output required")
    pins = {}

    def read(path, sha):
        resolved = Path(path).resolve(strict=True)
        if out == resolved:
            raise ValueError("Output must not overwrite an input")
        pins[str(resolved)] = sha
        return read_pinned(resolved, sha)

    if args.operation == "environment":
        result = environment(args.device, args.arithmetic)
    elif args.operation == "measure":
        result = measure(read(args.job, args.job_sha256))
        result.update(job_sha256=args.job_sha256, producer_commit=code["commit"], fresh_process=True)
    else:
        plan = read(args.plan, args.plan_sha256)
        manifest = read(args.measurements, args.measurements_sha256)
        if not isinstance(manifest, list) or any(set(s) != {"path", "sha256"} for s in manifest):
            raise ValueError("List of pinned worker files required")
        results = [read(s["path"], s["sha256"]) for s in manifest]
        if any(r.get("producer_commit") != code["commit"] or r.get("fresh_process") is not True for r in results):
            raise ValueError("Fresh workers from this exact producer version required")
        result = assemble(plan, results, plan_sha256=args.plan_sha256)
    require_frozen_code(code)
    if any(hash_file(p) != h for p, h in pins.items()):
        raise ValueError("Pinned input changed during measurement/assembly")
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
