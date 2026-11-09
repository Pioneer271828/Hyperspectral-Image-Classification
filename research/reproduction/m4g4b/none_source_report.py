"""Train/validation-only curves after all-nine independent source acceptance."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from g4_none_source_launch import select_bound_case
from joint_training import validation_epochs
from loadData.dataset_registry import file_hash
from research.reproduction.m4g4b.none_source_full_audit import acceptance_contract
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json

HEADS = ("fused", "graph", "cnn")


def head_record(metric):
    cm = np.asarray(metric["confusion"])
    recall = np.asarray(metric["recall"], dtype=float)
    if (cm.ndim != 2 or cm.shape[0] != cm.shape[1] or len(recall) != len(cm)
            or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0) or cm.sum() <= 0
            or not np.isfinite(recall).all() or np.any((recall < 0) | (recall > 1))
            or not np.isfinite([metric[k] for k in ("oa", "aa", "nll")]).all()):
        raise ValueError("Finite source metric and class support required")
    support = cm.sum(1)
    known = support > 0
    return {k: metric[k] for k in ("oa", "aa", "nll")} | {
        "support": support.tolist(), "recall": [float(r) if n else None for r, n in zip(recall, support)],
        "minimum_observed_recall": float(recall[known].min()),
        "zero_recall_classes": [int(i + 1) for i in np.flatnonzero(known & (recall == 0))],
        "unsupported_classes": [int(i + 1) for i in np.flatnonzero(~known)]}


def derive_case(case, accepted, history, trace):
    settings = case["settings"]
    steps = [(e + 1) * settings["steps_per_epoch"] for e in validation_epochs(settings)]
    if (settings["optimizer_steps"] != 1002 or len(trace) != 1002
            or [x["step"] for x in trace] != list(range(1, 1003))
            or [x["step"] for x in history] != steps
            or any(not np.isfinite([x["macro_oa"], x["val_loss"]]).all() for x in history)
            or any(not np.isfinite(x["loss"]["total"]) for x in trace)):
        raise ValueError("Complete finite full-source curve required")
    selected = max(history, key=lambda h: h["macro_oa"])
    if selected["step"] != accepted["selected_step"] or selected["macro_oa"] != accepted["source_validation_macro_oa"]:
        raise ValueError("Report must preserve accepted first-max selection")
    names = settings["source_tasks"]
    records = {}
    for name in names:
        records[name] = {phase: {head: head_record(selected["metrics"][name][phase][head]) for head in HEADS}
                         for phase in ("train", "val")}
    gap = {n: records[n]["train"]["fused"]["oa"] - records[n]["val"]["fused"]["oa"] for n in names}
    curve = [{"step": x["step"], "epoch": x["epoch"], "validation_nll": x["val_loss"],
              "source_validation_macro_oa": x["macro_oa"],
              "source_training_macro_fused_oa": statistics.mean(x["metrics"][n]["train"]["fused"]["oa"] for n in names)}
             for x in history]
    return {"name": case["name"], "held_out": case["held_out"], "seed": case["seed"],
            "updates": 1002, "selected_step": selected["step"], "selected_at_budget_end": selected["step"] == 1002,
            "source_validation_macro_oa": selected["macro_oa"], "source_train_minus_val_fused_oa": gap,
            "loss_first30": statistics.mean(x["loss"]["total"] for x in trace[:30]),
            "loss_last30": statistics.mean(x["loss"]["total"] for x in trace[-30:]),
            "last50epoch_validation_gain": history[-1]["macro_oa"] - history[-6]["macro_oa"],
            "training_and_review_elapsed_seconds": accepted["elapsed_seconds"], "heads": records, "curve": curve}


def plot_case(row, trace, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    x = np.asarray([r["step"] for r in trace])
    loss = np.asarray([r["loss"]["total"] for r in trace])
    rolling = [float(loss[max(0, i - 29):i + 1].mean()) for i in range(len(loss))]
    axes[0, 0].plot(x, loss, alpha=.2, label="per update")
    axes[0, 0].plot(x, rolling, label="trailing 30 mean")
    axes[0, 0].set(title="Source training total loss", xlabel="Optimizer update", ylabel="Loss")
    axes[0, 0].legend()
    curves = row["curve"]
    steps = [r["step"] for r in curves]
    axes[0, 1].plot(steps, [r["source_training_macro_fused_oa"] for r in curves], label="train")
    axes[0, 1].plot(steps, [r["source_validation_macro_oa"] for r in curves], label="validation")
    axes[0, 1].axvline(row["selected_step"], color="black", linestyle=":", label="first-max selected")
    axes[0, 1].set(title="Two-source macro fused OA", xlabel="Optimizer update", ylabel="OA", ylim=(0, 1))
    axes[0, 1].legend(loc="lower right")
    axes[1, 0].plot(steps, [r["validation_nll"] for r in curves])
    axes[1, 0].set(title="Source validation fused NLL", xlabel="Optimizer update", ylabel="NLL")
    for group in range(len(trace[0]["learning_rates"])):
        axes[1, 1].plot(x, [r["learning_rates"][group] for r in trace], label="group " + str(group))
    axes[1, 1].set(title="Learning rate after update, before same-step scheduling", xlabel="Optimizer update", ylabel="LR")
    axes[1, 1].legend()
    for ax, (name, phases) in zip(axes[2], row["heads"].items()):
        count = len(phases["val"]["fused"]["recall"])
        for i, head in enumerate(HEADS):
            ax.bar(np.arange(count) + (i - 1) * .25,
                   [np.nan if v is None else v for v in phases["val"][head]["recall"]], width=.25, label=head)
        ax.set(title=name + " selected source validation recall", xlabel="Local class ID (unsupported = NA)", ylabel="Recall", ylim=(0, 1.12))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticks(np.arange(count), [str(i + 1) for i in range(count)], fontsize=8)
        ax.legend(loc="upper center", ncol=3, fontsize=9)
    figure.suptitle(row["name"] + " | source-only; no target/query score")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "run", "launch-sha256", "acceptance-sha256", "audit", "audit-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean remote report producer required")
    root, audit_path = Path(args.run).resolve(strict=True), Path(args.audit).resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != root.parent or out == root or root in audit_path.parents:
        raise ValueError("Independent audit and fresh sibling report output required")
    pins = {str(root / "launch.json"): args.launch_sha256, str(root / "training-acceptance.json"): args.acceptance_sha256,
            str(audit_path): args.audit_sha256}
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Pinned source acceptance/audit changed")
    launch, accepted, audit = [json.loads(p.read_text()) for p in (root / "launch.json", root / "training-acceptance.json", audit_path)]
    if (Path(launch["out_dir"]).resolve() != root or audit["status"] != "passed" or audit["producer"] != launch["code"]
            or audit["launch_sha256"] != args.launch_sha256 or audit["acceptance_sha256"] != args.acceptance_sha256
            or audit["new_optimizer_updates"] != 0 or audit["query_labels_accessed"] is not False
            or audit["test_scoring"] is not False or len(audit["cases"]) != 9):
        raise ValueError("Complete independent zero-update source acceptance required before reporting")
    entries = acceptance_contract(launch, accepted, code, launch["code"]["commit"], args.launch_sha256)
    if {x["name"] for x in audit["cases"]} != set(entries):
        raise ValueError("Independent report coverage differs")
    out.mkdir(exist_ok=False)
    rows = []
    for case in launch["cases"]:
        select_bound_case(launch, launch["code"], case["held_out"], case["seed"], full=True)
        folder, entry = Path(case["out"]), entries[case["name"]]
        if file_hash(folder / "result.json") != entry["result_sha256"]:
            raise ValueError("Accepted report source result changed")
        result = json.loads((folder / "result.json").read_text())
        pins[str(folder / "result.json")] = entry["result_sha256"]
        for name, sha in result["artifacts"].items():
            if Path(name).name != name or file_hash(folder / name) != sha:
                raise ValueError("Accepted report source artifact changed")
            pins[str(folder / name)] = sha
        history, trace = [json.loads((folder / name).read_text()) for name in ("history.json", "trace.json")]
        row = derive_case(case, entry, history, trace)
        rows.append(row)
        plot_case(row, trace, out / (case["name"] + ".png"))
        with (out / (case["name"] + "-curve.csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row["curve"][0]))
            writer.writeheader()
            writer.writerows(row["curve"])
    groups = {}
    for held in launch["study"]["held_targets"]:
        subset = [r for r in rows if r["held_out"] == held]
        scores = [r["source_validation_macro_oa"] for r in subset]
        groups[held] = {"models": len(subset), "source_validation_macro_oa_mean": statistics.mean(scores),
                        "source_validation_macro_oa_sample_sd": statistics.stdev(scores)}
    require_frozen_code(code)
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Accepted source/audit input changed during report")
    write_json(out / "summary.json", {"status": "passed", "code": code, "cases": rows, "groups": groups,
        "launch_sha256": args.launch_sha256, "acceptance_sha256": args.acceptance_sha256,
        "audit_sha256": args.audit_sha256, "input_files_verified_unchanged": len(pins),
        "source_training_and_review_seconds": accepted["elapsed_seconds"], "query_labels_accessed": False,
        "new_optimizer_updates": 0, "target_accuracy_claim": False,
        "limitations": ["Source validation only; no target accuracy or controlled speed claim.",
                        "Budget-end selections and remaining validation gains do not prove convergence."]})


if __name__ == "__main__":
    main()
