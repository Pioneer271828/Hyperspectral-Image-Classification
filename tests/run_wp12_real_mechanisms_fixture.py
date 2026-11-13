"""Check the real supplement's source gates using actual bounded receipts."""
import argparse
import json
from pathlib import Path
from unittest import TestCase

from evaluation import wp12_mechanism_admission as api
from transfer import registered_development as transport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--legacy-region", type=Path, required=True)
    args = p.parse_args()
    root, out = args.runs, args.out
    out.mkdir()
    prior = api.read(transport._pin(root / "wp12-mechanism-admission01/qualification/authorization.json"), root)
    pinned = {**prior["receipts"], "crash": transport._pin(root / "wp07-dual-crash03/fixture/cpu-qualification.json"),
              "calibration": transport._pin(args.calibration), "legacy-region": transport._pin(args.legacy_region)}
    cal, legacy = (api.read(pinned[name], root) for name in ("calibration", "legacy-region"))
    qualification = api.read(pinned["crash"], root)
    auth = {**prior, "receipts": pinned, "crash_controller": qualification["controller"],
            "real_calibration": cal["producer"], "legacy_region": legacy["producer"], "legacy_worker": legacy["backend"]}
    authority = transport._json_new(out / "authorities.json", auth)
    checks = [
        (api.real_calibration, cal, "zero-private-writes", {"private_buffer_writes": 0}, "fitted writes"),
        (api.real_calibration, cal, "validation-fit", {"validation_statistics_fit": True}, "fitted writes"),
        (api.legacy_region, legacy, "promoted-protocol", {"new_native_protocol": True}, "separate protocol"),
        (api.legacy_region, legacy, "missing-endpoint", {"records": legacy["records"][:-1]}, "separate protocol"),
        (api.crash, qualification, "missing-output-owner-proof", {"alternate_journal_refused": False},
         "single-owner controller"),
    ]
    for call, row, name, changes, reason in checks:
        altered = transport._json_new(out / f"{name}.json", {**row, **changes})
        with TestCase().assertRaisesRegex(ValueError, reason):
            call(altered, root, auth)
    print(json.dumps({"authorities": authority, "receipts": pinned, "passed_negative_cases": [v[2] for v in checks]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
