"""Append development evidence without changing accepted production tables."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

from research_runtime import hash_file
from transfer import registered_development as dev
from transfer.development_blocks import read_pin
from transfer.native_released_points import LOSS
from transfer.native_released_preflight import BANK_ROOTS, EXPECTED, SCHEMA
from transfer.native_released_training import COMPARISONS
from evaluation.wp12_native_support_evidence import OLD_TABLES


P4_SCIENCE = "a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5"
P4_WORKER = {"commit": "3a6381cb4b0c089a10123292bc831a3a613e6106",
             "source_hash": "1c4251f86d0c725323d0b00df80b516064e2ae32bc4436abe5203cd9dd5a495d"}
P4_CONTROLLER = "72e22231b4889e03ea37c4f3629cfbc9494c74b9"
P4_RESOURCE = "eb9cacef34c0cb5b10d654a52d8a8a6eccb645e4d4bd7908160ab2942cce9e43"
P4_SYNTHETIC_ACCEPTANCE = "a682e8a621a8df85f24b3cc1768f6197107d9fab147e83cb3aee7e6a1f4f3080"
P4_SYNTHETIC_SUMMARY = "927c22ab33d6550f1806508f239a34fe6a56453b578d29d1e77d2c32d96b9533"
FIT_TEMPLATE = "5ec030b6b3934618c1052b14d115fedc0fb1a3b94f6ec84076e174726cefee43"
SPARSE_PRODUCER = "19c5ff2e3916f7bbe91146ba779e6298a057fcce"
SPARSE_PREFLIGHT = "22afb23982d1a8559ee71f5cc197934675485ac58cf1273245ab33267600ada10"
OLD = {
    "wp12-native-supplement-reviewed01": {**OLD_TABLES, "native-real-zero-update.csv":
        "63e7e1e2f493f79fe3022ce4e38666ff7c25957b76cb3dcb00ab15b2edc3883f",
        "native-synthetic.csv": "8092030f8f879a1ca68bb2304b27603eb905d8572871a399a81c5b3757b21a3a"},
    "wp12-native-scale02": {"native-real-scope.csv": "63b08f3b4bf3f69a6f8678640801f4ff2f74ce4087c71f461164e05ad7dd1ad3",
                            "native-synthetic-confirmation.csv":
                                "8900fbad4532995aa5973883b0e29f147a51f419f325d35c16484a070e20dcef"},
    "wp12-native-numerical04": {"native-numerical-cost.csv":
        "d20aa390b45ca8465c7cc7a47201d20a8b07fdb265d8745cd80c67696064b31a"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(pin, root):
    return json.loads(read_pin(pin, root))


def queue_states(resource, status=None, accepted=None):
    """Validate progress totals and derive a visible state for each frozen lane."""
    result = {}
    if status is not None:
        require(status["status"] in {"running", "waiting_for_resources", "failed", "completed"}
                and set(status["slots"]) == set(resource["slots"]), "Incomplete or unknown P4 progress record")
    for slot, planned in resource["slots"].items():
        cases, budget = len(planned["cases"]), planned["updates"]
        actual, completed, state = 0, 0, "planned"
        if status is not None:
            row = status["slots"][slot]
            actual, completed = row["optimizer_updates"], row["accepted_cases"]
            require((row["planned_cases"], row["planned_updates"]) == (cases, budget)
                    and type(actual) is type(completed) is int and 0 <= completed <= cases
                    and 120 * completed <= actual <= budget, "P4 slot progress exceeds registered counts")
            if accepted is not None:
                require((completed, actual) == (cases, budget) and accepted["slots"][slot] ==
                        {"cases": cases, "optimizer_updates": budget}, "Accepted P4 slot lacks its full ledger")
                state = "accepted"
            elif slot in status.get("errors", {}):
                state = "failed"
            elif slot in status.get("waiting", {}):
                state = "waiting"
            elif completed == cases:
                state = "audited_pending_batch_acceptance"
            else:
                state = status["status"]
        result[slot] = {"state": state, "planned_cases": cases, "planned_updates": budget,
                        "audited_cases": completed, "actual_updates": actual}
    if accepted is not None:
        require(accepted["status"] == "passed" and accepted["cases"] == sum(r["planned_cases"] for r in result.values())
                and accepted["optimizer_updates"] == sum(r["planned_updates"] for r in result.values()),
                "Only complete accepted P4 evidence can mark a batch accepted")
    return result


def p4_prepared(resource_pin, template_pin, root):
    row, template = read(resource_pin, root), read(template_pin, root)
    science = read(row["science"], root)
    require(resource_pin["sha256"] == P4_RESOURCE and template_pin["sha256"] == FIT_TEMPLATE
            and row["schema"] == "wp05_p4_independent_slot_scope_v1"
            and row["science"]["sha256"] == P4_SCIENCE and row["controller"]["commit"] == P4_CONTROLLER
            and row["mode"] == "honghu_real_prepared"
            and {key: row["worker"][key] for key in P4_WORKER} == P4_WORKER
            and len(science["cases"]) == 20 and science["budget"]["planned_optimizer_updates"] == 2400
            and {i for slot in row["slots"].values() for i in slot["cases"]} == {c["id"] for c in science["cases"]}
            and not Path(row["execution"]).exists()
            and template["schema"] == "wp05_honghu_fit_input_template_v1" and template["status"] == "NOT_RUN"
            and len(template["cases"]) == 9 and sum(c["planned_updates"] for c in template["cases"]) == 13500
            and all(c["status"] == "NOT_RUN" and c["actual_updates"] == 0 and c["selected_step"] is None
                    for c in template["cases"]), "Expected independently prepared P4 and fitting evidence")
    return {"scope": resource_pin, "science": row["science"], "state": "planned",
            "slots": queue_states(row), "optimizer_updates": 0, "fit_template": template_pin,
            "fit_status": template["status"], "accuracy": None, "cost": None,
            "average_benefit": None, "batch_accepted": False, "query_authorized": False}


def p4_engineering(receipt_pin, root):
    require(receipt_pin["sha256"] == P4_SYNTHETIC_SUMMARY, "Expected the final P4 CPU engineering summary")
    summary = read(receipt_pin, root)
    acceptance_pin = summary["matrix_receipt"]
    accepted = read(acceptance_pin, root)
    require(acceptance_pin["sha256"] == P4_SYNTHETIC_ACCEPTANCE
            and accepted["schema"] == "wp05_p4_independent_slot_acceptance_v1"
            and accepted["synthetic"] is True and accepted["status"] == "passed"
            and accepted["cases"] == 20 and accepted["optimizer_updates"] == 2400
            and len(accepted["case_audits"]) == 20 and accepted["raw_gt_loaded"] is accepted["query_accessed"] is False,
            "P4 engineering acceptance cannot be inferred from partial counts")
    resource = read(accepted["resource_scope"], root)
    failed = read(dev._pin(Path(summary["failed_directory"]) / "status.json"), root)
    failure_owner = read(dev._pin(Path(summary["failed_directory"]) / "resource-owner.json"), root)
    failure_resource = read(failure_owner["resource"], root)
    require(failure_resource["slots"] == resource["slots"]
            and failure_resource["science"] == resource["science"]
            and failed["resource_sha256"] == failure_owner["resource"]["sha256"],
            "Failed controller evidence is unrelated")
    final = queue_states(resource, summary["matrix"], accepted)
    partial = queue_states(resource, summary["partial"])
    drained = queue_states(failure_resource, failed)
    require(sum(r["actual_updates"] for r in final.values()) == 2400
            and sum(r["actual_updates"] for r in partial.values()) == 62
            and sum(r["actual_updates"] for r in drained.values()) == 60,
            "P4 controller and engineering budget differ")
    return {"scope": "synthetic_engineering", "receipt": receipt_pin, "accepted": acceptance_pin,
            "completed": final, "waiting": partial, "failed": drained, "query_authorized": False}


def sparse_prepared(manifest, root):
    value = read(manifest, root)
    require(manifest["sha256"] == SPARSE_PREFLIGHT and value["schema"] == SCHEMA and value["producer"] == SPARSE_PRODUCER
            and value["status"] == "preflight_only" and set(value["scenes"]) == set(BANK_ROOTS)
            and value["blocked"] == {"paviaU": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"}
            and value["optimization_updates"] == 0 and value["raw_gt_loaded"] is value["query_accessed"] is False,
            "Expected authenticated real sparse preflight from the approved producer")
    rows = {}
    for scene, pin in value["scenes"].items():
        row = read(pin, root)
        classes, train, val, labeled = EXPECTED[scene]
        require(row["scene"] == scene and row["producer"] == SPARSE_PRODUCER and row["schema"] == SCHEMA
                and row["bank"]["sha256"] == BANK_ROOTS[scene] and row["status"] == "preflight_only"
                and row["objective"] == LOSS and row["role_blocks"] == {"devtrain": train, "devval": val}
                and row["trainable_block_count"] == labeled and row["complete_purity"] is None
                and row["optimization_updates"] == row["labels_added"] == 0
                and row["raw_gt_loaded"] is row["query_accessed"] is False
                and row["draft"]["status"] == "NOT_RUN" and row["draft"]["authorized_for_training"] is False
                and len(row["draft"]["cases"]) == 3*len(COMPARISONS)
                and len(row["counts"]) == 12
                and all(set(counts) == {str(i) for i in range(1, classes+1)} and set(counts.values()) == {5}
                        for counts in row["counts"].values()), "Sparse scene custody, authority or support differs")
        factors = {}
        for factor in (1, 2, 4):
            per_role = {}
            for role in ("devtrain", "devval"):
                views = [view for view in row["views"] if (view["role"], view["band"], view["factor"])
                         == (role, "full", factor)]
                points = Counter()
                for view in views:
                    for i, support in enumerate(view["released"]["per_class"], 1):
                        points[i] += support
                    require(view["scalar_histogram"] == view["released"]["histogram"]
                            and view["released"]["complete_purity"] is None, "Sparse count evidence differs")
                require(set(points.values()) == {5}, "Sparse factor lost a released class")
                per_role[role] = {"released_per_class": [points[i] for i in range(1, classes+1)],
                                  "conflicting_cells": sum(v["released"]["conflicting_cells"] for v in views)}
            factors[str(factor)] = per_role
        rows[scene] = {"receipt": pin, "bank": row["bank"], "classes": classes, "supervision": LOSS,
                       "released_point_metrics": None, "coarse_known_count_metrics": None,
                       "complete_purity": None, "factors": factors, "planned_updates": row["draft"]["planned_updates"],
                       "minimum_exposure_updates": labeled, "actual_cost": None, "status": "NOT_RUN",
                       "training_authorized": False}
    return {"receipt": manifest, "scenes": rows, "blocked": value["blocked"], "optimizer_updates": 0}


def append(out, root, resource, template, sparse, queue_receipt, adapter):
    p4, released = p4_prepared(resource, template, root), sparse_prepared(sparse, root)
    engineering = p4_engineering(queue_receipt, root)
    tables = {}
    for folder, members in OLD.items():
        for name, checksum in members.items():
            pin = {"path": str(root / folder / name), "sha256": checksum}
            tables[name] = (pin, read_pin(pin, root))
    production = list(csv.DictReader(io.StringIO(tables["production-table.csv"][1].decode())))
    model_rows = {(r["module"], r["case_id"], r["arm"], r["state_sha256"]) for r in production}
    require(len(production) == 193 and len(model_rows) == 65, "Original WP12 production cardinality changed")
    out.mkdir()
    for name, (pin, data) in tables.items():
        (out / name).write_bytes(data)
        require(hash_file(out / name) == pin["sha256"] and read_pin(pin, root) == data,
                "Retained WP12 table changed")
    with (out / "p4-queue-progress.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("physical_uuid", "state", "planned_cases", "planned_updates", "audited_cases", "actual_updates"))
        for slot, row in p4["slots"].items():
            writer.writerow((slot, *[row[key] for key in ("state", "planned_cases", "planned_updates",
                                                         "audited_cases", "actual_updates")]))
    with (out / "native-released-dev-status.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene", "status", "planned_updates", "min_point_exposure_updates", "real_accuracy", "real_cost"))
        for scene, row in released["scenes"].items():
            writer.writerow((scene, row["status"], row["planned_updates"], row["minimum_exposure_updates"], "N/A", "N/A"))
        writer.writerow(("paviaU", released["blocked"]["paviaU"], "", "", "N/A", "N/A"))
    names = ("p4-queue-progress.csv", "native-released-dev-status.csv")
    return dev._json_new(out / "supplement.json", {"schema": "wp12_development_intake_supplement_v1",
        "adapter": adapter, "p4": p4, "engineering_p4": engineering, "native_released": released, "preserved": {
            "real_models": len(model_rows), "real_heads": len(production),
            "tables": {name: pin for name, (pin, _) in tables.items()}},
        "tables": {name: dev._pin(out / name) for name in names}})
