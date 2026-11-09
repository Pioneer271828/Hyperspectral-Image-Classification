"""Issue249 fitting, selection, class and recorded-cost review without query."""
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys
import time

from g4_light_full_audit import audit_schedule
from g4_light_ordered_query import REVIEW_POLICY, light_inputs
from g4_matched_target_evaluation import new_output
from g4_matched_target_review import HEADS, group_summary, metric_record, plot_case
from g4_support_context import check_files, read
from loadData.dataset_registry import DATASETS, file_hash
from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code, write_json


COST_PHASES = {"optimizer_update_with_invariant_checks", "validation_snapshot",
               "candidate_calibration_or_frozen_rebind", "full_source_initialization",
               "initial_ordered_target_calibration"}


def derive_case(model, result, audit, history, trace, cost):
    target, seed, method, mode = model["role"]
    settings = result["settings"]
    selected, _ = audit_schedule(settings, history, trace)
    if (selected["step"] != model["selected_step"] or selected["step"] != audit["selected_step"]
            or result["selected_validation"] != {k: v for k, v in selected.items()
                                                 if k not in ("epoch", "step", "selection_score")}):
        raise ValueError("Accepted first-minimum validation selection differs")
    curve, heads = [], None
    for event in history:
        phases = {phase: {head: metric_record(event["metrics"][target][phase][head], DATASETS[target][2])
                          for head in HEADS} for phase in ("train", "val")}
        if not math.isclose(event["val_loss"], phases["val"]["fused"]["nll"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Validation NLL differs from the fused head")
        curve.append({"step": event["step"], "epoch": event["epoch"], "validation_fused_nll": event["val_loss"],
                      **{f"{phase}_{head}_{metric}": values[metric]
                         for phase, by_head in phases.items() for head, values in by_head.items()
                         for metric in ("oa", "aa", "nll")}})
        if event["step"] == selected["step"]:
            heads = phases
    if cost["settings_sha256"] != digest(settings) or cost["deployment_speedup_claim"] is not False:
        raise ValueError("Recorded cost settings or scope differs")
    seconds = {phase: [] for phase in sorted(COST_PHASES)}
    for event in cost["events"]:
        phase, duration = event["phase"], event["seconds"]
        if phase not in seconds or type(duration) not in (float, int) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid recorded adaptation event")
        seconds[phase].append(duration)
    changed = sum(r["adapter_changed_elements"] > 0 for r in trace)
    if (len(seconds["optimizer_update_with_invariant_checks"]) < len(trace)
            or len(seconds["candidate_calibration_or_frozen_rebind"]) < len(history)
            or changed != audit["adapter_changed_steps"]):
        raise ValueError("Recorded adaptation events or update evidence incomplete")
    loss = [r["loss"]["total"] for r in trace]
    return {"name": model["name"], "role": model["role"], "target": target, "seed": seed,
            "method": method, "mode": mode, "selected_step": selected["step"],
            "selected_at_budget_end": selected["step"] == len(trace),
            "loss_first30": statistics.mean(loss[:30]), "loss_last30": statistics.mean(loss[-30:]),
            "final_minus_previous_five_events_validation_nll": history[-1]["val_loss"] - history[-6]["val_loss"],
            "selected_train_minus_val_fused_oa": heads["train"]["fused"]["oa"] - heads["val"]["fused"]["oa"],
            "lr_change_steps": [b["step"] for a, b in zip(trace, trace[1:]) if a["learning_rates"] != b["learning_rates"]],
            "adapter_changed_steps": changed, "trainable_changed_steps": sum(r["changed_trainable_elements"] > 0 for r in trace),
            "recorded_cost": {k: {"events": len(v), "seconds": math.fsum(v)} for k, v in seconds.items()},
            "recorded_cost_scope": cost["scope"], "recorded_cost_excludes": cost["excludes"],
            "heads": heads, "curve": curve}


def export_case(out, row, trace):
    name = row["name"]
    plot_case(row, trace, out / (name + ".png"))
    with (out / (name + "-curve.csv")).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row["curve"][0]))
        writer.writeheader()
        writer.writerows(row["curve"])


def export_review(path, sha256, out, code):
    started = time.monotonic()
    path = Path(path).resolve(strict=True)
    check_files({str(path): sha256})
    freeze = read(path)
    root, launch, _ = light_inputs(freeze, code)
    protected = freeze["protected_files"]
    out = new_output(out, launch, (root, path.parent))
    if out.exists() or any(out == Path(p).resolve() or out in Path(p).resolve().parents for p in protected):
        raise ValueError("Fresh review output outside protected inputs required")
    accepted = read(root / "training-acceptance.json")
    elapsed = accepted["elapsed_seconds_this_coordinator"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid coordinator elapsed seconds")
    rows, traces = [], []
    for model in freeze["models"]:
        audit = read(model["audit_path"])
        folder = root / "cases" / model["name"]
        artifacts = [folder / name for name in ("result.json", "history.json", "trace.json", "adaptation-cost.json")]
        if any(str(p) not in audit["artifacts"] for p in artifacts):
            raise ValueError("All review artifacts must be covered by the independent audit")
        result, history, trace, cost = map(read, artifacts)
        rows.append(derive_case(model, result, audit, history, trace, cost))
        traces.append(trace)
    groups = group_summary(rows)
    out.mkdir(exist_ok=False)
    for row, trace in zip(rows, traces):
        export_case(out, row, trace)
    check_files(protected)
    check_files({**freeze["boundary_files"], str(path): sha256})
    require_frozen_code(code)
    summary = {"policy": REVIEW_POLICY, "status": "passed", "code": code, "freeze_sha256": sha256,
               "acceptance_sha256": freeze["acceptance_sha256"], "cases": rows, "groups": groups,
               "optimizer_updates_reviewed": 81000, "new_optimizer_updates": 0, "query_labels_accessed": False,
               "controller_seconds_this_coordinator": elapsed, "review_seconds": time.monotonic() - started,
               "exported_files": {p.name: file_hash(p) for p in out.iterdir() if p.is_file()},
               "limitations": ["Recorded phase costs can include repeated work on resume and exclude listed overheads.",
                               "Coordinator time includes resource waits and is separate from summed phase costs.",
                               "Selected-state checks derive from independent audits. No new probability inference.",
                               "Budget-end selection does not establish convergence. No query benefit claim."]}
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "freeze", "freeze-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux review commit required")
    export_review(args.freeze, args.freeze_sha256, args.out, code)
    print(Path(args.out).resolve() / "summary.json")


if __name__ == "__main__":
    main()
