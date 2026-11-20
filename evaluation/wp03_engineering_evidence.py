"""WP12 intake for audited synthetic light batches and adaptation phase costs."""
import csv
from itertools import product
import math
from pathlib import Path

from model.research_config import digest
from research_runtime import external_directory, hash_file, write_json
from transfer.registered_wp11_learning import read

SCOPE = "synthetic_engineering"
REVIEW = "g4_light_synthetic_batch_review_v1"
WORKER = "g4_light_synthetic_worker_proof_v1"
CAMPAIGN = "g4_light_synthetic_campaign_proof_v1"
POLICY = "wp12_wp03_light_engineering_v1"
ADAPTATION = "adaptation_phases_only"
NAMES = ("paviaU","xuzhou","whu-honghu")
SEEDS = (64,512,2048)
METHODS = ("source_pretrained_head_only","source_pretrained_adapter_heads")
MODES = ("none","capacity_control","x_residual")
MEMORY = "eval_vca_value_projection_nonreentrant_v1"
GRID = set(product(NAMES,SEEDS,METHODS,MODES))
HEADS = {"fused","graph","cnn"}


def physical_pins(rows):
    return sum("uuid" in device for device in rows)


def verify_files(pins):
    for path, sha in pins.items():
        if not Path(path).is_absolute() or hash_file(external_directory(path)) != sha:
            raise ValueError("Pinned WP03 engineering artifact changed")


def is_synthetic(doc):
    return (doc.get("scope") == SCOPE and doc.get("real_training_authorized") is False
            and doc.get("real_updates",0) == 0 and doc.get("query_access",False) is False)


def read_report(pin, producer):
    report = read(pin)
    if (report.get("policy")!=REVIEW or not is_synthetic(report) or report["producer"]!=producer
            or report["accuracy_scope"]!="synthetic_support_and_validation_only" or report["cost_scope"]!=ADAPTATION):
        raise ValueError("Synthetic audited WP03 engineering report required")
    reg = read(report["registration"])
    fixture = read(report["fixture"])
    status, accepted = (read(report[k]) for k in ("queue_status","queue_acceptance"))
    if (reg["policy"]!="g4_light_synthetic_full54_registration_v1"
            or fixture["policy"]!="g4_light_synthetic_fixture_v1" or not all(is_synthetic(d) for d in (reg,fixture,status,accepted))
            or any(p!=producer for p in (reg["producer"],fixture["code"],status["producer"],accepted["producer"]))
            or accepted["code"]["commit"]!=producer["commit"] or digest(accepted["code"]["files"])!=producer["source_hash"]
            or accepted["code"]["source_hash"]!=producer["source_hash"] or accepted["code"]["dirty"] is not False
            or reg["fixture"]!=report["fixture"] or reg["devices"]!=report["devices"]
            or reg["roles"]!=report["roles"] or reg["real_training_authorized"] is not False
            or status["status"]!="completed" or status["errors"] or accepted["status"]!="passed"
            or accepted["query_labels_accessed"] is not False
            or any(row["registration_sha256"]!=report["registration"]["sha256"] for row in (status,accepted))):
        raise ValueError("Failed, incomplete or foreign WP03 batch")
    launch = read(fixture["launch"])
    if launch["fixture_scope"]!=SCOPE or launch["execution_enabled"] is not False or launch["optimizer_updates"]!=0:
        raise ValueError("Consumer cannot accept a live reference launch")
    roster=reg["roles"]
    count=report["executed_cases"]
    roles={tuple(row["role"]):row for row in roster if row["scheduled"]}
    if (len(roster)!=54 or {tuple(row["role"]) for row in roster}!=GRID or reg["planned_optimizer_updates"]!=81000
            or report["planned_cases"]!=54 or report["planned_updates"]!=81000
            or report["unexecuted_roles"]!=54-count or count!=len(roles) or count!=len(report["rows"])
            or report["executed_updates"]!=count*1500 or reg["scheduled_optimizer_updates"]!=count*1500
            or status["completed_cases"]!=count or accepted["optimizer_updates"]!=count*1500
            or set(accepted["cases"])!={r["name"] for r in report["rows"]}
            or len(accepted["cases"])!=count or len(accepted["audit_receipts"])!=count
            or {tuple(row["binding"]["role"]) for row in report["rows"]}!=set(roles)):
        raise ValueError("Registered54 and executed engineering counts must agree independently")
    pins=accepted["protected_files"]
    if (any(pins.get(path)!=sha for path,sha in report["input_files"].items())
            or any(report["input_files"].get(path)!=sha for path,sha in fixture["protected_files"].items())):
        raise ValueError("WP03 input pin inventory differs")
    verify_files(pins)
    verify_files(accepted["audit_receipts"])
    results = []
    for row in report["rows"]:
        role=tuple(row["binding"]["role"])
        scene,seed,method,mode=role
        binding=roles[role]
        audit=read(row["audit"])
        summary=row["trainval"]
        if (row["binding"]!=binding or binding["registered_updates"]!=1500
                or binding["strategy"]!=(MEMORY if method==METHODS[1] else None)
                or row["audit"]["sha256"]!=accepted["audit_receipts"].get(row["audit"]["path"])
                or audit["name"]!=row["name"] or audit["role"]!=list(role) or audit["ledger"]!=binding
                or audit["producer"]!=producer or audit["registration_sha256"]!=report["registration"]["sha256"]
                or audit["updates"]!=1500 or audit["validation_events"]!=51 or audit["exact_reload_repeats"]!=3
                or audit["query_labels_accessed"] is not False or audit["new_optimizer_updates"]!=0
                or audit["primary_sha256"]!=row["checkpoint"]["sha256"] or audit["sidecar_sha256"]!=row["metadata"]["sha256"]
                or pins.get(row["checkpoint"]["path"])!=audit["primary_sha256"] or pins.get(row["metadata"]["path"])!=audit["sidecar_sha256"]
                or summary["role"]!=list(role) or summary["selected_step"]!=audit["selected_step"]
                or summary.get("adapter_backward_policy")!=binding["strategy"]
                or summary["recorded_cost_scope"]!="initialization_calibration_optimizer_and_validation_with_invariant_checks"
                or set(summary["heads"])!={"train","val"} or any(set(v)!=HEADS for v in summary["heads"].values())):
            raise ValueError("WP03 role, policy, cost or checkpoint differs")
        for cost in summary["recorded_cost"].values():
            if type(cost["events"]) is not int or cost["events"]<0 or not math.isfinite(cost["seconds"]) or cost["seconds"]<0:
                raise ValueError("Invalid measured WP03 adaptation cost")
        path=str(Path(report["fixture"]["path"]).parent/"observations"/scene/"observation.pt")
        results.append({"scene":scene,"seed":seed,"method":method,"mode":mode,"row_name":row["name"],"role":list(role),
                        "binding":binding,"observation":{"path":path,"sha256":fixture["protected_files"][path]},
                        "audit":row["audit"],"checkpoint":row["checkpoint"],"metadata":row["metadata"],
                        "selected_step":audit["selected_step"],"device":audit["device"],"slot":audit["physical_gpu"],
                        "training_cost":summary["recorded_cost"],"cost_scope":ADAPTATION,
                        "trainval":summary["heads"],"class_recalls":audit["class_recalls"]})
    verify_files(report["input_files"])
    return {"review":pin,"registration":report["registration"],"fixture":report["fixture"],"rows":results,
            "executed_cases":count,"executed_updates":report["executed_updates"],"devices":report["devices"]}


def build(spec):
    if set(spec)!={"worker","campaign","producer"}:
        raise ValueError("Pinned WP03 worker and campaign proofs are required")
    producer=spec["producer"]
    worker,campaign=(read(spec[k]) for k in ("worker","campaign"))
    if (worker.get("policy")!=WORKER or campaign.get("policy")!=CAMPAIGN or any(not is_synthetic(x) for x in (worker,campaign))
            or campaign["worker_proof"]!=spec["worker"] or any(x["producer"]!=producer for x in (worker,campaign))
            or any(worker[k]!=campaign[k] for k in ("reviews","bindings","labels"))
            or worker["labels"]!={"query":False,"real":False} or worker["events_per_case"]!=51
            or worker["updates_per_executed_case"]!=1500 or worker["selection"]!="first_min_fused_validation_nll"
            or worker["exact_selected_reloads"]!=3 or campaign["planned_cases"]!=54 or campaign["planned_updates"]!=81000):
        raise ValueError("Legacy, failed or scope-incompatible WP03 proof")
    runs=[read_report(pin,producer) for pin in campaign["reviews"]]
    bindings=[{k:run[k] for k in ("registration","fixture","devices","executed_cases","executed_updates")} for run in runs]
    if (not runs or campaign["bindings"]!=bindings or len({r["registration"]["sha256"] for r in runs})!=len(runs)
            or any(run["fixture"]!=runs[0]["fixture"] for run in runs)
            or campaign["cpu_verified"]!=any(not physical_pins(run["devices"]) for run in runs)
            or campaign["physical_multigpu_verified"]!=any(physical_pins(run["devices"])>=2 for run in runs)):
        raise ValueError("WP03 proof mixed fixture, registration, hardware or counts")
    return {"policy":POLICY,"status":"passed","scope":SCOPE,"producer":producer,"proofs":spec,"runs":runs,
            "registered_cases_per_run":54,"registered_updates_per_run":81000,
            "total_executed_cases":sum(r["executed_cases"] for r in runs),
            "total_executed_updates":sum(r["executed_updates"] for r in runs),
            "production_accuracy_qualified":False,"real_training_authorized":False,"real_updates":0,
            "query_access":False,"cost_scope":ADAPTATION}


def write_table(out, report):
    out.mkdir()
    write_json(out/"wp03-light-engineering.json",report)
    with (out/"wp03-light-engineering.csv").open("x",newline="",encoding="utf-8") as stream:
        fields=("registration","scene","seed","method","mode","device","cost_scope","selected_step","optimizer_seconds","class_recalls")
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for run in report["runs"]:
            for row in run["rows"]:
                result={key:row[key] for key in fields if key in row}
                result.update(registration=run["registration"]["sha256"],
                              optimizer_seconds=row["training_cost"]["optimizer_update_with_invariant_checks"]["seconds"])
                writer.writerow(result)
