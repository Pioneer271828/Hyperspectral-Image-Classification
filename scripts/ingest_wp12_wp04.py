"""Import pinned WP04 engineering and zero-update evidence into WP12."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from evaluation.wp12_edge_weight_evidence import intake as edge_weight_intake
from evaluation.wp12_message_correspondence_evidence import intake as message_intake
from evaluation.wp12_residual_interaction_evidence import (
    historical, intake as residual_interaction_intake, write_paper_tables,
)
from scripts.run_wp04_factorial import (CONTROL_CONTRASTS, HEADS, REAL_CATALOG_SHA, SCHEMA,
                                       identity, require, roster, verify)
from transfer.p4_runtime_device import CUDA_FP64_WEIGHT, UUID


def load(pin, root):
    path = Path(pin["path"])
    require(path.is_relative_to(root) and path != root, "WP04 evidence escapes authorized runs")
    return json.loads(verify(pin).read_text())


def engineering(root, pin, group, *, cuda=False):
    expected = roster(group)
    n = len(expected)
    pairs = 126 if group == "factorial" else 18*len(CONTROL_CONTRASTS[group])
    counts = {"endpoints": n*6, "per-class": n*54, "paired": pairs, "paired-per-class": pairs*9, "costs": n*4}
    data = load(pin, root)
    require(data.get("schema") == SCHEMA and data["scope"] == "synthetic_only"
            and data["reporting_scope"] == "engineering_fixture_only"
            and data["real_updates"] == 0 and data["query_access"] is False
            and data.get("group", "factorial") == group
            and data["cases"] == n and data["synthetic_updates"] == n*24 and data["counts"] == counts,
            "WP04 engineering scope or roster differs")
    require((data["device"] == "cpu") == (not cuda), "WP04 measured hardware differs")
    reg = load(data["registration"], root)
    require(reg["schema"] == SCHEMA and reg["scope"] == "synthetic_only" and reg["cases"] == expected,
            "WP04 registration is missing paired arms")
    cases = [load(row, root) for row in data["case_pins"]]
    require([row["case"] for row in cases] == expected
            and all(row["exact_whole_split"] and row["resume_completed_updates"] == row["audit_updates"] == 0
                    for row in cases), "WP04 endpoint receipts are incomplete")
    if cuda:
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
        require(path.is_relative_to(root) and path.stem in counts, "WP04 table outside intake")
        with path.open(encoding="utf-8", newline="") as stream:
            require(sum(1 for _ in csv.DictReader(stream)) == counts[path.stem], "WP04 table count differs")
    require({Path(item["path"]).stem for item in data["tables"]} == set(counts)
            and len(data["tables"]) == len(counts), "WP04 tables are missing")
    return {"evidence": pin, "device": data["device"], "tables": data["tables"],
            "scope": data["reporting_scope"], "code": data["code"]}


def real_intake(root, pin, group, *, strict=False):
    data = load(pin, root)
    expected = roster(group)
    require(data["schema"] == SCHEMA and data["scope"] == "real_zero_update_draft"
            and data["real_updates"] == 0 and data["query_access"] is False
            and data["fixed_graph"] == (group != "edges")
            and data.get("group", "factorial") == group
            and [row["case"] for row in data["cases"]] == expected
            and all(row["optimizer_updates"] == 0 for row in data["cases"]),
            "Preattached real context differs")
    if strict:
        calls = [row.get("strict_sparse_mm_calls", 0) for row in data["cases"]]
        require(data["arithmetic"]["protocol"] == CUDA_FP64_WEIGHT
                and data["arithmetic"]["physical_uuid"] == data["device"]
                and all(count > 0 for count in calls)
                and all(set(row.get("strict_replay", {})) == set(HEADS) for row in data["cases"])
                and data.get("kernel_calls", {}).get("sparse_mm") == sum(calls),
                "Strict real replay did not use its arithmetic")
    draft = load(data["registration_draft"], root)
    require(draft["approval"] is None and draft["execution_enabled"] is False
            and draft["candidate_updates_after_separate_release"] == 12*len(expected)
            and draft["published_catalog"] == data["published_catalog"]
            and draft["published_catalog"]["sha256"] == REAL_CATALOG_SHA
            and data["rebound_contexts_verified"] == 27
            and load(draft["catalog"], root)["code"]["commit"] == data["code"]["commit"],
            "WP04 draft is not admissible")
    load(draft["published_catalog"], root)
    return {"evidence": pin, "scope": "released_xuzhou_zero_update_only", "device": data["device"],
            "registration_draft": data["registration_draft"], "real_updates": 0}


def intake(root, cpu, real, gpu=None, *, strict_real=None, controls=None):
    observed = {"cpu": engineering(root, cpu, "factorial")}
    if gpu is not None:
        observed["gpu"] = engineering(root, gpu, "factorial", cuda=True)
    result = {"schema": "wp12_wp04_engineering_supplement_v1", "engineering": observed,
              "real": real_intake(root, real, "factorial"), "production_accuracy_qualified": False,
              "query_access": False}
    if controls:
        result["real_controls"] = {}
        for group, (fixture, published) in controls.items():
            observed[group] = engineering(root, fixture, group)
            result["real_controls"][group] = real_intake(root, published, group)
    if strict_real is not None:
        replay = real_intake(root, strict_real, "factorial", strict=True)
        require("gpu" in observed and replay["device"] == observed["gpu"]["device"],
                "Strict real replay must use the qualified device")
        result["strict_real"] = replay
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    for key in ("cpu", "gpu", "real", "strict-real"):
        parser.add_argument("--" + key, nargs=2)
    parser.add_argument("--edge-weight-development", nargs=2, metavar=("EVIDENCE", "SHA256"))
    parser.add_argument("--message-development", nargs=2, metavar=("EVIDENCE", "SHA256"))
    parser.add_argument("--message-frozen", nargs=2, action="append", metavar=("EVIDENCE", "SHA256"))
    parser.add_argument("--qualify-remaining-history", nargs=2, metavar=("PUBLISHED_CATALOG", "SHA256"))
    for item in ("residual-assignment", "direct-joint", "remaining-history"):
        parser.add_argument("--" + item, nargs=2, metavar=("EVIDENCE", "SHA256"))
    for key in CONTROL_CONTRASTS:
        parser.add_argument("--" + key, nargs=4, metavar=("SYNTHETIC", "SHA256", "ZERO_UPDATE", "SHA256"))
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    require(args.expected_commit == code["commit"], "Wrong WP12 adapter revision")
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    require(root in out.parents, "Keep WP12 output in the authorized runs tree")

    def pair(value):
        return dict(zip(("path", "sha256"), value, strict=True)) if value else None

    controls = {key: (pair(value[:2]), pair(value[2:])) for key in CONTROL_CONTRASTS
                if (value := getattr(args, key))}
    has_engineering = args.cpu is not None or args.real is not None
    remaining_pins = (args.residual_assignment, args.direct_joint, args.remaining_history)
    require(not any(remaining_pins) or all(remaining_pins), "Supply both new trials and the history qualification")
    if args.qualify_remaining_history:
        require(not (has_engineering or args.gpu or args.strict_real or controls or args.edge_weight_development
                    or args.message_development or args.message_frozen or any(remaining_pins)),
                "Run historical qualification as a separate read-only command")
        qualification = historical(root, pair(args.qualify_remaining_history))
        out.mkdir()
        write_json(out / "history.json", qualification)
        return
    remaining = residual_interaction_intake(root, *(pair(p) for p in remaining_pins)) if all(remaining_pins) else None
    require(bool(args.message_development) == bool(args.message_frozen),
            "A correspondence intake binds both the trained cases and the frozen diagnostics")
    message = (message_intake(root, pair(args.message_development), [pair(pin) for pin in args.message_frozen],
                              code["commit"]) if args.message_development else None)
    if has_engineering:
        require(args.cpu is not None and args.real is not None, "Both original engineering inputs are required")
        data = intake(root, pair(args.cpu), pair(args.real), pair(args.gpu), strict_real=pair(args.strict_real),
                      controls=controls)
        if args.edge_weight_development:
            data["edge_weight_development"] = edge_weight_intake(root, pair(args.edge_weight_development))
        if message:
            data["message_development"] = message
        if remaining:
            data["residual_interaction_development"] = remaining
    else:
        require((args.edge_weight_development is not None or message or remaining)
                and not (args.gpu or args.strict_real or controls),
                "Supply original engineering inputs or a real development receipt")
        data = remaining or message or edge_weight_intake(root, pair(args.edge_weight_development))
        if remaining and message:
            data["message_development"] = message
        if (remaining or message) and args.edge_weight_development:
            data["edge_weight_development"] = edge_weight_intake(root, pair(args.edge_weight_development))
    out.mkdir()
    write_json(out / "supplement.json", {**data, "adapter": identity(code)})
    if remaining:
        write_paper_tables(remaining, out)


if __name__ == "__main__":
    main()
