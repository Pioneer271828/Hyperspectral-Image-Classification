"""Pinned full54 train/validation review; no inference, updates or query release."""
from pathlib import Path
import math
import statistics
import time
import csv

import numpy as np
from g4_matched_target_audit import AUDIT_POLICY, check_files, files, full_acceptance, read
from g4_matched_target_evaluation import new_output
from g4_matched_target_execution import unchanged_producer
from g4_matched_target_study import MODES, SEEDS
from joint_baseline import NAMES
from loadData.dataset_registry import DATASETS, file_hash
from research.reproduction.m4g4b.none_source_report import head_record
from research_runtime import external_directory, require_frozen_code, write_json

POLICY = "g4_matched_full54_trainval_review_v1"
HEADS = ("fused", "graph", "cnn")


def validate_audit(launch, accepted, audit, consumer, launch_sha256, acceptance_sha256):
    """The independently checked rows must match every accepted evidence field."""
    unchanged_producer(launch["code"], audit["code"])
    unchanged_producer(audit["code"], consumer)
    rows = audit["cases"]
    if (audit["policy"] != AUDIT_POLICY or audit["status"] != "passed"
            or audit["producer"] != launch["code"] or audit["launch_sha256"] != launch_sha256
            or audit["acceptance_sha256"] != acceptance_sha256
            or audit["protected_manifest_sha256"] != accepted["protected_manifest_sha256"]
            or audit["new_optimizer_updates"] != 0 or audit["query_labels_accessed"] is not False
            or audit["test_scoring"] is not False or len(rows) != 54
            or len({r["name"] for r in rows}) != 54
            or {r["name"] for r in rows} != {c["name"] for c in launch["cases"]}):
        raise ValueError("Complete pinned independent54 audit required before review")
    by_name = {r["name"]: r for r in accepted["models"]}
    for row in rows:
        expected = {k: v for k, v in by_name[row["name"]].items()
                    if k not in ("elapsed_seconds", "completed_resume_updates")}
        if row != expected:
            raise ValueError("Independent54 review evidence differs from acceptance")


def metric_record(metric, count):
    value = head_record(metric)
    cm = np.asarray(metric["confusion"])
    if cm.shape != (count, count) or not 0 <= metric["oa"] <= 1 or not 0 <= metric["aa"] <= 1 or metric["nll"] < 0:
        raise ValueError("Target metric shape/range differs")
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(count, dtype=float), where=support > 0)
    if (not np.allclose(metric["recall"], recall, rtol=0, atol=1e-12)
            or not math.isclose(metric["oa"], float(cm.trace() / cm.sum()), rel_tol=0, abs_tol=1e-12)
            or not math.isclose(metric["aa"], float(recall.mean()), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Recorded target confusion/OA/AA/recall disagree")
    return value


def derive_case(case, accepted, history, trace):
    target = case["target"]
    count = DATASETS[target][2]
    events = list(range(1, 1472, 30)) + [1500]
    if (accepted["status"] != "passed" or accepted["updates"] != 1500
            or accepted["validation_events"] != 51 or accepted["completed_resume_updates"] != 0
            or [h["step"] for h in history] != events
            or [r["step"] for r in trace] != list(range(1, 1501))
            or any(r["dataset"] != target for r in trace)):
        raise ValueError("Complete accepted1500/51 target curves required")
    loss = [r["loss"]["total"] for r in trace]
    rates = [r["learning_rates"] for r in trace]
    groups = len(rates[0])
    if (not groups or any(len(r) != groups or any(not math.isfinite(v) or v < 0 for v in r) for r in rates)
            or any(not math.isfinite(v) or v < 0 for v in loss)
            or any(not math.isfinite(h["val_loss"]) or h["val_loss"] < 0 for h in history)):
        raise ValueError("Finite nonnegative target loss/LR required")
    selected = min(history, key=lambda h: h["val_loss"])
    if (accepted["selected_step"] != selected["step"]
            or accepted["selected_validation"]["metrics"] != selected["metrics"]
            or any(accepted["selected_validation"][k] != selected[k] for k in ("val_loss", "macro_oa"))
            or accepted["last_validation"] != history[-1]):
        raise ValueError("Accepted first-min target selection must be preserved")
    curve = []
    selected_heads = None
    for event in history:
        phases = {phase: {head: metric_record(event["metrics"][target][phase][head], count) for head in HEADS}
                  for phase in ("train", "val")}
        if not math.isclose(event["val_loss"], phases["val"]["fused"]["nll"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Target first-min score differs from fused validation NLL")
        row = {"step": event["step"], "epoch": event["epoch"], "validation_fused_nll": event["val_loss"]}
        for phase in ("train", "val"):
            for head in HEADS:
                for key in ("oa", "aa", "nll"):
                    row[f"{phase}_{head}_{key}"] = phases[phase][head][key]
        curve.append(row)
        if event["step"] == selected["step"]:
            selected_heads = phases
    positive = {g: sum(r["gradient_norms"][g] > 0 for r in trace) for g in ("adapter", "graph")}
    if case["mode"] != "none":
        positive["m4"] = sum(r["m4_parameter_gradient_norm"] > 0 for r in trace)
    changes = accepted["changed_parameter_elements"]
    required_groups = {"adapter", "cnn", "graph"} | ({"m4"} if case["mode"] != "none" else set())
    lr_changes = [r["step"] for old, r in zip(trace, trace[1:]) if old["learning_rates"] != r["learning_rates"]]
    if (any(not math.isfinite(v) or v < 0 for r in trace for v in r["gradient_norms"].values())
            or (case["mode"] != "none" and any(not math.isfinite(r["m4_parameter_gradient_norm"])
                or r["m4_parameter_gradient_norm"] < 0 for r in trace))
            or positive != accepted["positive_gradient_steps"] or any(v <= 0 for v in positive.values())
            or not required_groups <= set(changes) or any(type(changes[g]) is not int or changes[g] <= 0 for g in required_groups)
            or lr_changes != accepted["lr_change_steps"]
            or statistics.mean(loss[:30]) != accepted["loss_first30"]
            or statistics.mean(loss[-30:]) != accepted["loss_last30"]
            or not math.isfinite(accepted["elapsed_seconds"]) or accepted["elapsed_seconds"] < 0):
        raise ValueError("Accepted target module/loss/LR/cost evidence differs")
    return {**{k: case[k] for k in ("name", "target", "seed", "method", "mode")},
        "updates": 1500, "validation_events": 51, "selected_step": selected["step"],
        "selected_at_budget_end": selected["step"] == 1500,
        "loss_first30": statistics.mean(loss[:30]), "loss_last30": statistics.mean(loss[-30:]),
        "late_validation_window_start_step": history[-6]["step"],
        "late_validation_window_updates": history[-1]["step"] - history[-6]["step"],
        "final_minus_previous_five_events_validation_nll": history[-1]["val_loss"] - history[-6]["val_loss"],
        "selected_train_minus_val_fused_oa": selected_heads["train"]["fused"]["oa"] - selected_heads["val"]["fused"]["oa"],
        "changed_parameter_elements": changes, "positive_gradient_steps": positive,
        "lr_change_steps": lr_changes, "case_training_resume_wait_and_acceptance_seconds": accepted["elapsed_seconds"],
        "heads": selected_heads, "curve": curve}


def plot_case(row, trace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    loss = np.asarray([r["loss"]["total"] for r in trace])
    steps = [r["step"] for r in trace]
    axes[0, 0].plot(steps, loss, alpha=.2, label="per update")
    axes[0, 0].plot(steps, [loss[max(0, i - 29):i + 1].mean() for i in range(len(loss))], label="trailing30 mean")
    axes[0, 0].set(title="Target support training loss", xlabel="Optimizer update", ylabel="Loss")
    curve = row["curve"]
    events = [r["step"] for r in curve]
    for phase in ("train", "val"):
        axes[0, 1].plot(events, [r[phase + "_fused_oa"] for r in curve], label=phase)
    axes[0, 1].set(title="Fused train/validation OA", xlabel="Optimizer update", ylabel="OA", ylim=(0, 1.05))
    for head in HEADS:
        axes[1, 0].plot(events, [r["val_" + head + "_oa"] for r in curve], label=head)
    axes[1, 0].set(title="Three-head validation OA", xlabel="Optimizer update", ylabel="OA", ylim=(0, 1.05))
    axes[1, 1].plot(events, [r["validation_fused_nll"] for r in curve], label="val fused NLL")
    axes[1, 1].set(title="Validation fused NLL (first minimum)", xlabel="Optimizer update", ylabel="NLL")
    for index in range(len(trace[0]["learning_rates"])):
        axes[2, 0].plot(steps, [r["learning_rates"][index] for r in trace], label=f"group{index}")
    axes[2, 0].set(title="LR after update, before same-step scheduling", xlabel="Optimizer update", ylabel="LR")
    count = len(row["heads"]["val"]["fused"]["recall"])
    for index, head in enumerate(HEADS):
        axes[2, 1].bar(np.arange(count) + (index - 1) * .25,
            [np.nan if v is None else v for v in row["heads"]["val"][head]["recall"]], width=.25, label=head)
    axes[2, 1].set(title="Selected validation recall (unsupported = NA)",
        xlabel="Local class ID", ylabel="Recall", ylim=(0, 1.12))
    axes[2, 1].set_xticks(np.arange(count), [str(i + 1) for i in range(count)], fontsize=8)
    for ax in axes.flat:
        ax.legend(fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(row["selected_step"], color="black", linestyle=":")
    fig.suptitle(row["name"] + " | train/validation only; no query benefit claim")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def group_summary(rows):
    groups = []
    for target in NAMES:
        for method in sorted({r["method"] for r in rows}):
            for mode in MODES:
                subset = [r for r in rows if (r["target"], r["method"], r["mode"]) == (target, method, mode)]
                if len(subset) != 3 or {r["seed"] for r in subset} != set(SEEDS):
                    raise ValueError("Exactly three model seeds required in each review group")
                heads = {}
                for head in HEADS:
                    heads[head] = {}
                    for metric in ("oa", "aa", "nll"):
                        values = [r["heads"]["val"][head][metric] for r in subset]
                        heads[head][metric + "_mean"] = statistics.mean(values)
                        heads[head][metric + "_sample_sd"] = statistics.stdev(values)
                groups.append({"target": target, "method": method, "mode": mode, "models": 3, "heads": heads,
                    "budget_end_selections": sum(r["selected_at_budget_end"] for r in subset)})
    return groups


def export_review(root, launch_sha256, acceptance_sha256, audit_path, audit_sha256, out, code):
    started = time.monotonic()
    root, audit_path = external_directory(root), Path(audit_path).resolve(strict=True)
    if root == audit_path.parent or root in audit_path.parents or file_hash(audit_path) != audit_sha256:
        raise ValueError("Pinned separate independent audit required")
    launch, accepted, pins = full_acceptance(root, launch_sha256, acceptance_sha256, code)
    out = new_output(out, launch, (audit_path.parent,))
    if out.exists():
        raise ValueError("Fresh review output required")
    audit = read(audit_path)
    validate_audit(launch, accepted, audit, code, launch_sha256, acceptance_sha256)
    if any(not math.isfinite(v) or v < 0 for v in (accepted["elapsed_seconds"], audit["elapsed_seconds"])):
        raise ValueError("Finite nonnegative controller/audit costs required")
    pins.update(files(root))
    pins[str(audit_path)] = audit_sha256
    check_files(pins)
    # Derive every row before publishing any full-batch report files.
    rows = []
    entries = {r["name"]: r for r in accepted["models"]}
    for case in launch["cases"]:
        folder = Path(case["out"])
        rows.append(derive_case(case, entries[case["name"]], read(folder / "history.json"), read(folder / "trace.json")))
    groups = group_summary(rows)
    out.mkdir(exist_ok=False)
    for case, row in zip(launch["cases"], rows):
        plot_case(row, read(Path(case["out"]) / "trace.json"), out / (row["name"] + ".png"))
        with (out / (row["name"] + "-curve.csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row["curve"][0]))
            writer.writeheader()
            writer.writerows(row["curve"])
    require_frozen_code(code)
    check_files(pins)
    summary = {"policy": POLICY, "status": "passed", "code": code, "producer": launch["code"],
        "cases": rows, "groups": groups, "launch_sha256": launch_sha256, "acceptance_sha256": acceptance_sha256,
        "audit_sha256": audit_sha256, "input_files_verified_unchanged": len(pins),
        "optimizer_updates_reviewed": 81000, "new_optimizer_updates": 0,
        "query_labels_accessed": False, "test_scoring": False, "target_query_benefit_claim": False,
        "controller_seconds_excluding_initial_freeze": accepted["elapsed_seconds"],
        "sum_case_training_resume_wait_and_acceptance_seconds": sum(r["case_training_resume_wait_and_acceptance_seconds"] for r in rows),
        "independent_audit_seconds": audit["elapsed_seconds"], "report_elapsed_seconds": time.monotonic() - started,
        "limitations": ["Recorded train/validation curves; no independent probability recomputation for nonselected events.",
            "Parameter changes and selected-state metrics were independently checked by the pinned prior audit.",
            "Budget-end selection and late validation improvement do not prove sufficient convergence.",
            "Wall times include resource waiting/resume/acceptance and do not establish controlled GPU speed.",
            "Parent launch-to-exit wall time is separate; source27054 and short-target216 updates are separate budgets.",
            "Three model seeds with one support set are not support repeats or statistical significance."]}
    summary["exported_files"] = {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "summary.json", summary)
    return summary
