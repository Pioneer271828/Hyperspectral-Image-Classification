"""Freeze, execute and independently verify six real full-none-source short targets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import independent_trainval
from joint_training import _tree_fingerprint
from loadData.dataset_registry import file_hash
from none_full_target_launch import load_case_data, prepare_launch, select_case
from none_full_target_runtime import NoneFullSourceTargetRuntime
from none_full_transfer import initialize_none_full_target
from research.reproduction.m4g4b.none_source_campaign import wait_gpu
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from target_initialization import state_id


def files(root):
    return {str(p): file_hash(p) for p in Path(root).rglob("*") if p.is_file()}


def audit_case(case, code, *, device="cuda"):
    folder = Path(case["out"])
    before = files(folder)
    result, progress, history, trace = [json.loads((folder / name).read_text()) for name in
        ("result.json", "progress.json", "history.json", "trace.json")]
    if (result["status"] != "completed" or progress["status"] != "completed"
            or result["settings"] != case["settings"] or result["start_code"] != code
            or result["optimizer_steps"] != 12 or len(trace) != 12
            or [r["step"] for r in trace] != list(range(1, 13))
            or [r["step"] for r in history] != [1, 4, 7, 10, 12]
            or result["selected_step"] != min(history, key=lambda h: h["val_loss"])["step"]
            or result["test_scoring"] is not False):
        raise ValueError("Short target budget, first-min selection or completion differs")
    for name, sha in result["artifacts"].items():
        if Path(name).name != name or file_hash(folder / name) != sha:
            raise ValueError("Completed target artifact changed")
    state = torch.load(folder / progress["state_file"], map_location="cpu", weights_only=True)
    saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    if (state["cursor"] != 12 or state["identity"] != result["identity"] or state["trace"] != trace
            or state["history"] != history or state["best_step"] != result["selected_step"]
            or _tree_fingerprint(last) != _tree_fingerprint(state["model"])
            or _tree_fingerprint(saved) != _tree_fingerprint(state["best_probabilities"])
            or any(_tree_fingerprint(state[k]) != state["state_fingerprints"][k]
                   for k in ("model", "optimizer", "scheduler"))):
        raise ValueError("Authoritative full target state/optimizer/selection differs")
    data, _, guard = load_case_data(case)
    target = case["target"]
    independent = independent_trainval(saved[target], data[target], result["selected_validation"]["metrics"][target])
    moved = {n: {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()} for n, item in data.items()}
    runtime = NoneFullSourceTargetRuntime(case["settings"], data, folder, source_spec=case["source_spec"],
                                         initial_state_sha256=case["initial_state_sha256"])
    fingerprints = []
    for _ in range(3):
        model = runtime.load_selected(folder / "primary.pt", moved,
            folder / f'raw-validation-{result["selected_step"]:06d}.pt', device=device)
        model_before = state_id(model)
        _, values = runtime.validation_snapshot(model, moved)
        for phase in ("train", "val"):
            for head in ("fused", "graph", "cnn"):
                old, new = saved[target][phase][head], values[target][phase][head]
                torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                if not torch.equal(old.argmax(1), new.argmax(1)):
                    raise ValueError("Target decisions changed")
        fingerprints.append(_tree_fingerprint(values))
        if state_id(model) != model_before:
            raise ValueError("Selected target mutated during inference")
        del model
    if set(fingerprints) != {_tree_fingerprint(saved)}:
        raise ValueError("Three ordered reloads must be elementwise exact")
    initial, receipt = initialize_none_full_target(target, case["method"], model_seed=64, head_seed=1064,
        source_spec=case["source_spec"], device="cpu")
    if receipt != case["initialization"]:
        raise ValueError("Full source target initialization identity changed")
    changed = {}
    for group, prefix in {"adapter": "backbone.adapter.", "cnn": "backbone.CNN_Branch.",
                          "graph": "backbone.HGNN_Branch."}.items():
        changed[group] = sum(int(torch.count_nonzero(last[n] - p.detach()))
            for n, p in initial.named_parameters() if n.startswith(prefix))
        if changed[group] <= 0:
            raise ValueError("Missing actual target parameter change: " + group)
    if any(r["gradient_norms"]["graph"] <= 0 or r["gradient_norms"]["adapter"] <= 0
           or r["graph_changed_elements"] <= 0 for r in trace):
        raise ValueError("Missing graph/adapter participation at an update")
    guard()
    if before != files(folder):
        raise ValueError("Completed target audit mutated original files")
    return {"name": case["name"], "updates": 12, "selected_step": result["selected_step"], **independent,
        "exact_reload_repeats": 3, "completed_resume_updates": 0, "changed_parameter_elements": changed,
        "positive_graph_gradient_steps": 12, "positive_adapter_gradient_steps": 12,
        "loss_first": trace[0]["loss"]["total"], "loss_last": trace[-1]["loss"]["total"],
        "selected_validation": result["selected_validation"], "initial_state_sha256": case["initial_state_sha256"],
        "result_sha256": file_hash(folder / "result.json"), "primary_sha256": file_hash(folder / "primary.pt"),
        "sidecar_sha256": file_hash(folder / "primary.pt.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "source-root", "source-audit", "launch-sha256", "acceptance-sha256",
                 "protected-sha256", "audit-sha256", "observation-root", "family-root", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--protected-root", action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote implementation required")
    source, audit, out = Path(args.source_root).resolve(strict=True), Path(args.source_audit).resolve(strict=True), external_directory(args.out_dir)
    for path, sha in ((source / "launch.json", args.launch_sha256), (source / "training-acceptance.json", args.acceptance_sha256),
                      (source / "protected-files.json", args.protected_sha256), (audit, args.audit_sha256)):
        if file_hash(path) != sha:
            raise ValueError("Registered accepted source evidence changed")
    roots = [source, audit.parent, *[Path(x).resolve(strict=True) for x in args.protected_root]]
    if any(r == out or r in out.parents or out in r.parents for r in roots):
        raise ValueError("Independent output outside all protected trees required")
    protected = {p: h for r in roots for p, h in files(r).items()}
    protected.update(json.loads((source / "protected-files.json").read_text()))
    if any(file_hash(p) != h for p, h in protected.items()):
        raise ValueError("Historical source/input changed before target initialization")
    started = time.monotonic()
    launch = prepare_launch(source, audit, args.observation_root, args.family_root, out, code)
    lp, rows = out / "launch.json", []
    sha = file_hash(lp)
    protected[str(lp)] = sha
    for case in launch["cases"]:
        protected.update(case["target_context"]["protected_files"])
    write_json(out / "protected-files.json", protected)
    try:
        for case in launch["cases"]:
            select_case(launch, code, case["target"], case["method"])
            command = [sys.executable, "scripts/train_none_full_source_target_short.py", "--expected-commit", args.expected_commit,
                "--launch-plan", str(lp), "--launch-sha256", sha, "--target", case["target"], "--method", case["method"]]
            folder, start = Path(case["out"]), time.monotonic()
            for suffix, extra in (("first2", ["--stop-after", "2"]), ("continue", ["--resume"]), ("readonly", ["--resume"])):
                wait_gpu(args.gpu, out)
                write_json(out / "status.json", {"status": "running", "case": case["name"], "operation": suffix, "completed_cases": len(rows)})
                before = files(folder) if suffix == "readonly" else None
                log_path = out / "logs" / (case["name"] + "-" + suffix + ".log")
                with log_path.open("x") as log:
                    subprocess.run(command + extra, check=True, stdout=log, stderr=subprocess.STDOUT)
                value = json.loads(log_path.read_text().splitlines()[-1])
                if (value["operation"], value["updates_this_call"]) != {"first2": ("interrupted", 2),
                        "continue": ("completed", 10), "readonly": ("verified_completed", 0)}[suffix]:
                    raise ValueError("Target interrupted/resumed/completed budget differs")
                if before is not None and before != files(folder):
                    raise ValueError("Completed target recovery mutated files")
            wait_gpu(args.gpu, out)
            row = audit_case(case, code)
            row["elapsed_seconds"] = time.monotonic() - start
            rows.append(row)
            write_json(out / "progress.json", rows)
            print("NONE_FULL_TARGET_CASE", case["name"], row["selected_step"], row["elapsed_seconds"], flush=True)
            torch.cuda.empty_cache()
        require_frozen_code(code)
        if (len(rows) != 6 or sum(r["updates"] for r in rows) != 72 or sum(r["head_metrics"] for r in rows) != 36
                or sum(r["class_recalls"] for r in rows) != 480
                or any(file_hash(p) != h for p, h in protected.items())):
            raise ValueError("All-six coverage or protected source/target/input differs")
        write_json(out / "receipt.json", {"status": "passed", "code": code, "launch_sha256": sha, "updates": 72,
            "cases": rows, "protected_files_and_inputs": len(protected), "query_labels_accessed": False,
            "test_scoring": False, "elapsed_seconds": time.monotonic() - started})
        write_json(out / "status.json", {"status": "completed", "completed_cases": 6, "optimizer_updates": 72})
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "completed_cases": len(rows),
            "error": type(error).__name__ + ": " + str(error), "retry_policy": "preserve_and_diagnose"})
        raise


if __name__ == "__main__":
    main()
