"""Read already accepted WP12 evidence and join it by frozen checkpoint."""

from itertools import product
from pathlib import Path

from evaluation.dynamic_curve import pin, timing_summary
from evaluation.unified_cohort import HEADS, SCHEMA, _fields, _index, summarize
from g4_matched_target_analysis import PLAN_SHA256, POLICY as ANALYSIS_POLICY, validate_rows
from g4_matched_target_evaluation import require_frozen_boundary
from g4_matched_target_study import METHODS, MODES, SEEDS, grid
from model.research_config import digest
from research_runtime import hash_file


COST_POLICY = "wp12-frozen-hardware-cost-v1"
ARMS = tuple(f"{method}/{mode}" for method, mode in product(METHODS[:2], MODES))


def accuracy_records(analysis, receipt, freeze, *, analysis_sha, freeze_sha):
    if (receipt.get("policy") != ANALYSIS_POLICY or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != PLAN_SHA256 or receipt.get("freeze_sha256") != freeze_sha
            or receipt.get("outputs", {}).get("analysis.json") != analysis_sha
            or receipt.get("models_analyzed") != 54 or receipt.get("heads_analyzed") != 162
            or type(receipt.get("new_optimizer_updates")) is not int or receipt["new_optimizer_updates"] != 0
            or any(receipt.get(k) is not False for k in
                   ("new_query_release", "model_inference_executed", "raw_labels_deserialized"))
            or receipt.get("input_query_metrics_consumed") is not True
            or analysis.get("policy") != ANALYSIS_POLICY or analysis.get("plan_sha256") != PLAN_SHA256):
        raise ValueError("Accepted frozen54 accuracy receipt and output required")
    require_frozen_boundary(freeze, full_integrity=False)
    lookup = validate_rows(analysis["models"])
    frozen = {r["name"]: r for r in freeze["models"]}
    # Release digests include checkpoint identity. The frozen family hash carries
    # the sample/label identity common to the models in a paired scene.
    protocol = digest({"analysis_plan": PLAN_SHA256, "freeze": freeze_sha})
    rows = []
    for target, seed, method, mode in grid():
        row = lookup[target, seed, method, mode]
        model = frozen[row["name"]]
        if row["checkpoint_sha256"] != model["primary_sha256"] or not pin(model.get("family_hash")):
            raise ValueError("Accuracy record differs from frozen checkpoint/family")
        evaluation = digest({"protocol": protocol, "scene": target, "family": model["family_hash"]})
        rows.append(dict(scene=target, seed=seed, arm=f"{method}/{mode}",
                         checkpoint_sha256=row["checkpoint_sha256"], evaluation_sha256=evaluation,
                         release_sha256=row["release_sha256"],
                         class_ids=row["metrics"]["fused"]["class_ids"],
                         heads={h: row["metrics"][h]["confusion_matrix"] for h in HEADS},
                         nll={h: row["metrics"][h]["nll"] for h in HEADS}))
    families = {}
    for row in rows:
        key = row["scene"], row["seed"]
        if families.setdefault(key, row["evaluation_sha256"]) != row["evaluation_sha256"]:
            raise ValueError("Accuracy pair uses a different frozen evaluation family")
    return {"protocol_sha256": protocol, "seeds": list(SEEDS), "arms": list(ARMS),
            "preregistered_benefit_conclusion": analysis["benefit_conclusion"], "cases": rows}


def legacy_cpu_cost(plan, receipt, execution, *, plan_sha, receipt_sha):
    roles = {"dense", "compact"}
    if (receipt.get("status") != "passed" or receipt.get("plan_sha256") != plan_sha
            or receipt.get("optimizer_updates") != 0 or receipt.get("query_labels_accessed") is not False
            or execution.get("status") != "passed" or execution.get("receipt_sha256") != receipt_sha
            or plan.get("scope") != "resident_cpu_full_scene_native_three_head_forward"
            or plan.get("threads") != 1 or set(plan["models"]) != roles or not pin(plan["input_sha256"])):
        raise ValueError("Accepted resident CPU benchmark plan and receipt required")
    samples, order = receipt["samples"], plan["order"]
    if (len(samples) != len(order) or len(samples) < 4 or len(samples) % 2
            or any(sequence != (["dense", "compact"] if i % 2 == 0 else ["compact", "dense"])
                   for i, sequence in enumerate(order)) or any(set(row) != roles for row in samples)):
        raise ValueError("Complete alternating benchmark samples required")
    rows = []
    for role in ("dense", "compact"):
        files = [(path, sha) for path, sha in plan["pins"].items()
                 if Path(path).name == "selected-state.pt" and Path(path).parent.name == role]
        if len(files) != 1 or not pin(files[0][1]) or hash_file(files[0][0]) != files[0][1]:
            raise ValueError("Legacy benchmark checkpoint pin differs")
        model = plan["models"][role]
        wall = [row[role]["wall_ms"] for row in samples]
        stats = timing_summary(wall, repeats=len(samples), blocks=1)
        expected = receipt["summary"][role]
        if stats["p50_ms"] != expected["wall_p50_ms"] or stats["p95_ms"] != expected["wall_p95_ms"]:
            raise ValueError("Legacy latency summary differs from samples")
        rows.append(dict(role=role, checkpoint_sha256=files[0][1], wall_ms=wall,
                         p50_ms=stats["p50_ms"], p95_ms=stats["p95_ms"],
                         parameters=model["parameters"], state_tensor_bytes=model["state_tensor_bytes"]))
    return dict(scene=plan["target"], scope="resident_forward", device="cpu", cpu=execution["cpu"],
                input_sha256=plan["input_sha256"], missing=["peak_bytes", "package_bytes"],
                eligible_for_joint_report=False, rows=rows)


def join_hardware(accuracy, cost_plan, receipt, *, cost_plan_sha, baseline):
    _fields(cost_plan, "policy protocol_sha256 timing workloads cases")
    _fields(receipt, "policy status plan_sha256 timing cases")
    if (cost_plan["policy"] != COST_POLICY or receipt["policy"] != COST_POLICY
            or receipt["status"] != "passed" or receipt["plan_sha256"] != cost_plan_sha
            or cost_plan["protocol_sha256"] != accuracy["protocol_sha256"]):
        raise ValueError("Accepted hardware receipt must bind the registered protocol and cost plan")
    rows = accuracy["cases"]
    expected = {(r["scene"], r["seed"], r["arm"]) for r in rows}
    registered = _index(cost_plan["cases"], expected, "scene seed arm evaluation_sha256 checkpoint_sha256")
    measured = _index(receipt["cases"], expected, "scene seed arm cost")
    classes = {r["scene"]: r["class_ids"] for r in rows}
    if not isinstance(cost_plan["workloads"], dict) or cost_plan["workloads"].keys() != classes.keys():
        raise ValueError("Hardware workload coverage differs from accuracy scenes")
    scenes = {}
    for scene, workload in cost_plan["workloads"].items():
        _fields(workload, "workload_sha256 work_items")
        scenes[scene] = {**workload, "class_ids": classes[scene]}
    evidence = []
    for row in rows:
        key = row["scene"], row["seed"], row["arm"]
        planned = registered[key]
        if any(planned[field] != row[field] for field in ("evaluation_sha256", "checkpoint_sha256")):
            raise ValueError("Hardware plan evaluation/checkpoint differs from frozen accuracy")
        evidence.append({**measured[key], "accuracy": {
            field: row[field] for field in ("evaluation_sha256", "checkpoint_sha256", "heads")}})
    plan = dict(schema=SCHEMA, protocol_sha256=accuracy["protocol_sha256"], scenes=scenes,
                seeds=accuracy["seeds"], arms=accuracy["arms"], baseline=baseline,
                timing=cost_plan["timing"], cases=cost_plan["cases"])
    return summarize(plan, {"timing": receipt["timing"], "cases": evidence})
