"""Append hash-bound WP04 engineering and zero-update evidence to the WP12 intake."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.run_wp04_factorial import REAL_CATALOG_SHA, SCHEMA, identity, require, roster, verify
from transfer.p4_runtime_device import CUDA_FP64_WEIGHT, UUID


COUNTS = {"endpoints": 108, "per-class": 972, "paired": 126, "paired-per-class": 1134, "costs": 72}


def load(pin, root):
    path = Path(pin["path"])
    require(path.is_relative_to(root) and path != root, "WP04 evidence escapes authorized runs")
    return json.loads(verify(pin).read_text())


def intake(root, cpu, real, gpu=None):
    evidence = {}
    for label, pin in (("cpu", cpu), ("gpu", gpu)):
        if pin is None:
            continue
        data = load(pin, root)
        require(data["schema"] == SCHEMA and data["scope"] == "synthetic_only"
                and data["reporting_scope"] == "engineering_fixture_only"
                and data["real_updates"] == 0 and data["query_access"] is False
                and data["cases"] == 18 and data["synthetic_updates"] == 432 and data["counts"] == COUNTS,
                "WP04 engineering scope or roster differs")
        require((data["device"] == "cpu") == (label == "cpu"), "WP04 measured hardware differs")
        reg = load(data["registration"], root)
        require(reg["schema"] == SCHEMA and reg["scope"] == "synthetic_only" and reg["cases"] == roster(),
                "WP04 registration is missing paired arms")
        cases = [load(row, root) for row in data["case_pins"]]
        require([row["case"] for row in cases] == roster()
                and all(row["exact_whole_split"] is True and row["resume_completed_updates"] == row["audit_updates"] == 0
                        for row in cases), "WP04 endpoint receipts are incomplete")
        if label == "gpu":
            require(UUID.fullmatch(data["device"]) and data.get("strict_qualification") is True
                    and data.get("arithmetic") == reg.get("arithmetic") == CUDA_FP64_WEIGHT
                    and all(case.get("exact_independent_step_one") is True
                            and case["arithmetic"]["physical_uuid"] == data["device"]
                            and case["arithmetic"]["deterministic_algorithms"] is True
                            and all(phase["arithmetic"]["physical_uuid"] == data["device"]
                                    for phase in case["phases"].values()) for case in cases),
                    "Unqualified or mixed-device CUDA evidence cannot enter WP12")
        for item in data["tables"]:
            path = verify(item)
            require(path.is_relative_to(root) and path.stem in COUNTS, "WP04 table outside intake")
            with path.open(encoding="utf-8", newline="") as stream:
                require(sum(1 for _ in csv.DictReader(stream)) == COUNTS[path.stem], "WP04 table count differs")
        require({Path(item["path"]).stem for item in data["tables"]} == set(COUNTS), "WP04 tables are missing")
        evidence[label] = {"evidence": pin, "device": data["device"], "tables": data["tables"],
                           "scope": data["reporting_scope"], "code": data["code"]}
    released = load(real, root)
    require(released["schema"] == SCHEMA and released["scope"] == "real_zero_update_draft"
            and released["real_updates"] == 0 and released["query_access"] is False
            and released["fixed_graph"] is True
            and [row["case"] for row in released["cases"]] == roster()
            and all(row["optimizer_updates"] == 0 for row in released["cases"]), "Preattached real context differs")
    draft = load(released["registration_draft"], root)
    require(draft["approval"] is None and draft["execution_enabled"] is False
            and draft["candidate_updates_after_separate_release"] == 216
            and draft["published_catalog"] == released["published_catalog"]
            and draft["published_catalog"]["sha256"] == REAL_CATALOG_SHA
            and released["rebound_contexts_verified"] == 27
            and load(draft["catalog"], root)["code"]["commit"] == released["code"]["commit"],
            "WP04 draft is not admissible")
    load(draft["published_catalog"], root)
    return {"schema": "wp12_wp04_engineering_supplement_v1", "engineering": evidence,
            "real": {"evidence": real, "scope": "released_xuzhou_zero_update_only",
                     "registration_draft": released["registration_draft"], "real_updates": 0},
            "production_accuracy_qualified": False, "query_access": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    for key in ("cpu", "gpu", "real"):
        parser.add_argument("--" + key, nargs=2, required=key != "gpu")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    require(args.expected_commit == code["commit"], "Wrong WP12 adapter revision")
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    require(root in out.parents, "Keep WP12 output in the authorized runs tree")
    def pair(value):
        return dict(zip(("path", "sha256"), value, strict=True)) if value else None
    data = intake(root, pair(args.cpu), pair(args.real), pair(args.gpu))
    out.mkdir()
    write_json(out / "supplement.json", {**data, "adapter": identity(code)})


if __name__ == "__main__":
    main()
