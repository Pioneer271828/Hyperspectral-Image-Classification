"""Authenticate frozen54 inputs and export observation-only hardware workloads."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_cost_preparation import prepare
from research_runtime import code_snapshot, hash_file, require_frozen_code
from scripts.summarize_unified_cohort import read_pinned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux preparation checkout required")
    registration = read_pinned(args.registration, args.registration_sha256)
    if set(registration) != {"analysis", "receipt", "freeze", "launch"}:
        raise ValueError("Four explicit frozen evidence inputs required")
    inputs = {}
    for name, spec in registration.items():
        if set(spec) != {"path", "sha256"} or not Path(spec["path"]).is_absolute():
            raise ValueError("Pinned absolute file specification required")
        inputs[name] = read_pinned(spec["path"], spec["sha256"])
    def finish():
        require_frozen_code(code)
        if (hash_file(args.registration) != args.registration_sha256
                or any(hash_file(s["path"]) != s["sha256"] for s in registration.values())):
            raise ValueError("Prepared input evidence changed")

    prepare(**inputs, code=code, out=args.out, finish=finish,
            **{name + "_sha256": registration[name]["sha256"] for name in ("analysis", "freeze", "launch")})


if __name__ == "__main__":
    main()
