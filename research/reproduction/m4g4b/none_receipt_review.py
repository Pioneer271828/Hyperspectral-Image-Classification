"""Independent CPU record/selection/LR review; a rejected audit can pass review."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.g4_target import verify_training
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from target_initialization import METHODS


def read(path):
    return json.loads(Path(path).read_text())


def close(old, new):
    if not math.isfinite(old) or not math.isfinite(new):
        raise ValueError("Nonfinite metric")
    return abs(new - old) <= 1e-6 + 1e-5 * abs(old)


def check_classification(metric):
    cm = metric["confusion"]
    if (not cm or any(len(row) != len(cm) for row in cm)
            or any(type(x) is not int or x < 0 for row in cm for x in row)
            or any(sum(row) <= 0 for row in cm)):
        raise ValueError("Complete class confusion counts required")
    recalls = [row[i] / sum(row) for i, row in enumerate(cm)]
    accuracy = sum(row[i] for i, row in enumerate(cm)) / sum(map(sum, cm))
    if (len(recalls) != len(metric["recall"])
            or any(abs(a - b) > 1e-12 for a, b in zip(recalls, metric["recall"]))
            or abs(accuracy - metric["oa"]) > 1e-12
            or abs(sum(recalls) / len(recalls) - metric["aa"]) > 1e-12
            or not math.isfinite(metric["nll"])):
        raise ValueError("Confusion-derived metric differs")


def event_gate(original, event):
    """Recompute the decision from counts/scalars, independent of stored flags."""
    new = event["ordered"]
    failures = []
    if original["step"] != event["step"] or any(event["original_" + k] != original[k]
                                                for k in ("macro_oa", "val_loss")):
        raise ValueError("Original event association differs")
    for key in ("macro_oa", "val_loss"):
        if not close(original[key], new[key]):
            failures.append(key)
    if set(original["metrics"]) != set(new["metrics"]):
        raise ValueError("Dataset coverage differs")
    heads = recalls = 0
    maximum_nll_error = 0.
    for name in original["metrics"]:
        for phase in ("train", "val"):
            for head in ("fused", "cnn", "graph"):
                a, b = original["metrics"][name][phase][head], new["metrics"][name][phase][head]
                check_classification(a)
                check_classification(b)
                key = name + "/" + phase + "/" + head
                if a["confusion"] != b["confusion"] or a["recall"] != b["recall"]:
                    failures.append(key + "/classification_counts")
                if not all(close(a[k], b[k]) for k in ("nll", "oa", "aa")):
                    failures.append(key + "/scalar_metrics")
                maximum_nll_error = max(maximum_nll_error, abs(a["nll"] - b["nll"]))
                heads += 1
                recalls += len(b["recall"])
    if event["passes_metrics_gate"] != (not failures) or sorted(event["failures"]) != sorted(failures):
        raise ValueError("Stored event gate differs from independent review")
    if event.get("max_head_nll_abs_error", maximum_nll_error) != maximum_nll_error:
        raise ValueError("Stored metric error differs")
    return not failures, heads, recalls


def pure_rates(settings, events, trace):
    """Plain Python ReduceLROnPlateau: no optimizer or torch scheduler involved."""
    if (len({e["step"] for e in events}) != len(events)
            or [t["step"] for t in trace] != list(range(1, len(trace) + 1))
            or any(e["step"] not in range(1, len(trace) + 1) for e in events)):
        raise ValueError("Complete chronological trace/events required")
    cfg, policy = settings["config"]["training"], settings["scheduler"]
    current = [cfg["lr"], cfg["head_lr"]]
    best, bad = math.inf, 0
    lookup = {e["step"]: e["val_loss"] for e in events}
    rates = []
    for step in range(1, len(trace) + 1):
        rates.append(list(current))
        if step not in lookup:
            continue
        value = lookup[step]
        if not math.isfinite(value):
            raise ValueError("Nonfinite scheduler metric")
        if value < best * (1 - policy["threshold"]):
            best, bad = value, 0
        else:
            bad += 1
        if bad > policy["patience"]:
            current = [max(lr * policy["factor"], 0.) if lr - max(lr * policy["factor"], 0.) > 1e-8
                       else lr for lr in current]
            bad = 0
    return rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "receipt-sha256", "source-root", "target-root", "selected-receipt", "out", "expected-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    current = code_snapshot()
    root, source_root, target_root = (Path(p).resolve(strict=True)
                                     for p in (args.run, args.source_root, args.target_root))
    selected_path = Path(args.selected_receipt).resolve(strict=True)
    out = external_directory(args.out)
    if (current["dirty"] or current["commit"] != args.expected_commit or out.parent != root.parent
            or out in (root, source_root, target_root, selected_path.parent)):
        raise ValueError("Clean registered reviewer and fresh sibling output required")
    if file_hash(root / "receipt.json") != args.receipt_sha256:
        raise ValueError("Pinned trajectory receipt changed")
    value, registration = read(root / "receipt.json"), read(root / "registration.json")
    producer = value["audit_code"]
    if (value["status"] != "audit_completed" or not value["trajectory_audit_completed"]
            or value["new_optimizer_updates"] != 0 or value["query_labels_accessed"] is not False
            or producer["commit"] != "26830f672ef1e0fddfa54cf3f30e6987dec0f121" or producer["dirty"]
            or producer["source_hash"] != digest(producer["files"])
            or any(current["files"].get(k) != v for k, v in producer["files"].items())
            or registration["audit_code"] != producer or file_hash(selected_path) != value["selected_sha256"]
            or file_hash(selected_path.parent / "registration.json") != registration["selected_registration_sha256"]
            or read(selected_path.parent / "registration.json") != registration["inputs"]):
        raise ValueError("Complete immutable audit provenance required")
    inputs = registration["inputs"]
    stage = verify_training(target_root, inputs["target_launch_sha256"], inputs["target_acceptance_sha256"], current)
    if file_hash(source_root / "launch.json") != inputs["source_launch_sha256"]:
        raise ValueError("Source launch changed")
    source_launch = read(source_root / "launch.json")
    sources = {c["name"]: c for c in source_launch["cases"]}
    targets = {e["case"]["name"]: e["case"] for e in stage["rows"] if e["case"]["method"] in METHODS[:2]}
    bindings = {c["source_result"]["path"]: c["source_result"]["sha256"]
                for c in targets.values() if c["source_result"] is not None}
    if len(sources) != 9 or len(targets) != 18 or len(bindings) != 9 or len(value["cases"]) != 27:
        raise ValueError("All historical roles required")
    roots = (root, source_root, target_root, selected_path.parent)
    before = {str(p): file_hash(p) for r in roots for p in r.rglob("*") if p.is_file()}
    pins = dict(stage["protected_files"])
    seen, failures = set(), []
    event_count = metric_count = recall_count = 0
    for row in value["cases"]:
        role, name = row["role"], row["name"]
        if role not in ("source", "target") or (role, name) in seen:
            raise ValueError("Unique source/target role required")
        seen.add((role, name))
        case = (sources if role == "source" else targets)[name]
        if (row["target"] != case["held_out" if role == "source" else "target"]
                or row["seed"] != case["seed"] or (role == "target" and row["method"] != case["method"])):
            raise ValueError("Role metadata differs from original launch")
        folder = Path(case["out"]).resolve(strict=True)
        rp = folder / "result.json"
        if role == "source":
            if file_hash(rp) != bindings[str(rp)]:
                raise ValueError("Source result differs from accepted target binding")
            pins.update(case["source_context"]["protected_files"])
        result = read(rp)
        for key, sha in result["artifacts"].items():
            if Path(key).name != key or file_hash(folder / key) != sha:
                raise ValueError("Historical artifact changed")
        owned = root / (role + "-" + name)
        if read(owned / "result.json") != row or file_hash(owned / "events.json") != row["events_sha256"]:
            raise ValueError("Audited case/result hash differs")
        old, events, trace = read(folder / "history.json"), read(owned / "events.json"), read(folder / "trace.json")
        if len(events) != 51 or [e["step"] for e in events] != [e["step"] for e in old]:
            raise ValueError("Complete51 event coverage required")
        bad = []
        case_metrics = case_recalls = 0
        ordered = []
        for original, event in zip(old, events):
            if event["raw_sha256"] != result["artifacts"][f'raw-validation-{event["step"]:06d}.pt']:
                raise ValueError("Raw event association differs")
            passed, heads, recalls = event_gate(original, event)
            metric_count += heads
            recall_count += recalls
            case_metrics += heads
            case_recalls += recalls
            if not passed:
                bad.append({"step": event["step"], "failures": event["failures"]})
            ordered.append({"step": event["step"], **event["ordered"]})
        old_rates, new_rates = (pure_rates(result["settings"], e, trace) for e in (old, ordered))
        key = "macro_oa" if role == "source" else "val_loss"
        select = max if role == "source" else min
        original_step, ordered_step = (select(e, key=lambda x: x[key])["step"] for e in (old, ordered))
        mismatches = [i + 1 for i, (a, b) in enumerate(zip(old_rates, new_rates)) if a != b]
        passed = not bad and not mismatches and ordered_step == original_step
        if (old_rates != [e["learning_rates"] for e in trace] or original_step != result["selected_step"]
                or row["original_selected_step"] != original_step or row["ordered_selected_step"] != ordered_step
                or row["lr_mismatch_update_steps"] != mismatches
                or row["selected_step_unchanged"] != (ordered_step == original_step)
                or row["learning_rates_unchanged"] != (not mismatches) or row["original_trace_reproduced"] is not True
                or row["validation_events"] != 51 or row["head_metrics"] != case_metrics
                or row["class_recalls"] != case_recalls
                or row["metrics_gate_failures"] != len(bad) or row["passes_trajectory_gate"] != passed):
            raise ValueError("Independent selection/LR/metric decision differs")
        event_count += len(events)
        if not passed:
            failures.append({"role": role, "name": name, "events": bad, "lr_mismatches": mismatches,
                             "original_selected_step": original_step, "ordered_selected_step": ordered_step})
    expected = {("source", n) for n in sources} | {("target", n) for n in targets}
    eligible = not failures
    if (seen != expected or (event_count, metric_count, recall_count) != (1377, 11016, 146880)
            or (value["validation_events"], value["head_metrics"], value["class_recalls"])
                != (event_count, metric_count, recall_count)
            or value["historical_none_eligible_for_reuse"] != eligible
            or value["conclusion"] != ("historical_none_trainval_compatible" if eligible else "historical_none_reuse_rejected")):
        raise ValueError("Independent coverage/eligibility differs")
    if any(file_hash(p) != h for p, h in pins.items()) or before != {
            str(p): file_hash(p) for r in roots for p in r.rglob("*") if p.is_file()} or code_snapshot() != current:
        raise ValueError("Protected artifact/input/code changed during review")
    out.mkdir(exist_ok=False)
    write_json(out / "receipt.json", {"status": "passed", "review_code": current,
        "reviewed_receipt_sha256": args.receipt_sha256, "historical_none_eligible_for_reuse": eligible,
        "roles": len(seen), "events": event_count, "head_metrics": metric_count, "class_recalls": recall_count,
        "failed_roles": failures, "protected_files": len(before), "additional_input_pins": len(pins),
        "new_optimizer_updates": 0, "query_labels_accessed": False,
        "method": "Independent count arithmetic, pure Python scheduler, first-tie selection and immutable inventory"})
    print("NONE_RECEIPT_REVIEW", eligible, len(failures), flush=True)


if __name__ == "__main__":
    main()
