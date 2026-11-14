"""Read real authorized receipts, then attempt four forbidden promotions."""
import argparse
import json
from pathlib import Path
from unittest import TestCase

from evaluation import wp12_mechanism_admission as api
from research_runtime import code_snapshot
from transfer import registered_development as transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root, out = args.runs, args.out
    code = code_snapshot()
    assert code["commit"] == args.expected_commit and not code["dirty"]
    out.mkdir()
    names = {"crash": "wp07-dual-crash01/fixture/cpu-qualification.json",
             "learning": "wp06-learned-channel02/experiment/accepted.json",
             "bn": "p4-domain-bn01/result/diagnostic.json",
             "region": "wp07-trained-region-analysis02/result/analysis.json"}
    pins = {k: transport._pin(root / path) for k, path in names.items()}
    actual = {name: api.read(p, root) for name, p in pins.items()}
    authority = {"schema": api.SCHEMA, "query_authorized": False, "receipts": pins,
                 "science_worker": actual["crash"]["worker"], "crash_controller": actual["crash"]["controller"],
                 "learned_worker": actual["learning"]["producer"],
                 "learned_review": actual["learning"]["review_repair"]["consumer"],
                 "bn_diagnostic": actual["bn"]["producer"], "p4_resource": actual["bn"]["resource"],
                 "region_adapter": actual["region"]["producer"]}
    approved = transport._json_new(out / "authorization.json", authority)
    hooks = {"crash": api.crash, "learning": api.learning, "bn": api.bn, "region": api.adapter}
    passed = []
    for kind, label, change, reject in (
            ("learning", "synthetic_as_real", {"scope": "real_HSI"}, "synthetic learning"),
            ("learning", "unfrozen_confirmation", {"freeze": actual["learning"]["registration"]}, "frozen"),
            ("crash", "missing_acknowledgement",
             {"continuation": {**actual["crash"]["continuation"], "operation_acks": []}}, "acknowledged attempt"),
            ("crash", "missing_failure_gate", {"failure": {**actual["crash"]["failure"], "first_failure": None}}, "first failure"),
            ("bn", "calibration_as_zero_state_updates", {"statistics_calibration_updates": 4}, "calibration"),
            ("region", "real_prepare_as_measured", {"real_optimizer_updates": 48}, "Invalid prepared")):
        item = transport._json_new(out / f"{label}.json", {**actual[kind], **change})
        with TestCase().assertRaisesRegex(ValueError, reject):
            hooks[kind](item, root, authority)
        passed.append(label)
    print(json.dumps({"authority": approved, "sources": pins, "rejected": passed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
