"""Independent zero-update audit of all nine accepted ordered none sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from g4_none_source_launch import select_bound_case
from joint_training import schedule_for
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research.reproduction.m4g4b.none_source_evidence import audit_case
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.train_joint_m4 import graph_binding
from source_data_context import load_source_context


def acceptance_contract(launch, accepted, consumer, producer_commit, launch_sha256):
    producer = launch["code"]
    if (producer["commit"] != producer_commit or producer["dirty"] or consumer["dirty"]
            or producer["source_hash"] != digest(producer["files"])
            or consumer["source_hash"] != digest(consumer["files"])
            or any(consumer["files"].get(k) != v for k, v in producer["files"].items())
            or accepted["status"] != "passed" or accepted["code"] != producer
            or accepted["optimizer_updates"] != 9018 or accepted["launch_sha256"] != launch_sha256
            or accepted["query_labels_accessed"] is not False or accepted["test_scoring"] is not False
            or len(accepted["cases"]) != 9 or len({r["name"] for r in accepted["cases"]}) != 9
            or sum(r["metrics"] for r in accepted["cases"]) != 108
            or sum(r["recalls"] for r in accepted["cases"]) != 1440
            or any(r["updates"] != 1002 or r["exact_reload_repeats"] != 3 or r["completed_resume_updates"] != 0
                   for r in accepted["cases"])):
        raise ValueError("All-nine full acceptance and unchanged additive audit implementation required")
    return {r["name"]: r for r in accepted["cases"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "producer", "run", "launch-sha256", "acceptance-sha256", "protected-sha256", "out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    current = code_snapshot()
    if sys.platform != "linux" or not __debug__ or current["dirty"] or current["commit"] != args.expected_commit:
        raise RuntimeError("Clean independent remote expected commit required")
    root = Path(args.run).resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != root.parent or out == root:
        raise ValueError("Fresh sibling audit output required")
    for name, sha in (("launch.json", args.launch_sha256), ("training-acceptance.json", args.acceptance_sha256),
                      ("protected-files.json", args.protected_sha256)):
        if file_hash(root / name) != sha:
            raise ValueError("Pinned full campaign evidence changed")
    launch, accepted = [json.loads((root / name).read_text()) for name in ("launch.json", "training-acceptance.json")]
    entries = acceptance_contract(launch, accepted, current, args.producer, args.launch_sha256)
    if Path(launch["out_dir"]).resolve() != root or json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Completed owned full campaign required")
    pins = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    pins.update(json.loads((root / "protected-files.json").read_text()))
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Frozen historical/full source/input changed before audit")
    out.mkdir(exist_ok=False)
    started, rows = time.monotonic(), []
    producer = launch["code"]
    for case in launch["cases"]:
        select_bound_case(launch, producer, case["held_out"], case["seed"], full=True)
        context = load_source_context(launch["prepared"], launch["data_root"], held_out=case["held_out"],
            consumer=producer, manifest_sha256=launch["prepared_manifest_sha256"])
        if context.record != case["source_context"] or graph_binding(context.items) != case["graph_binding"]:
            raise ValueError("Actual independent source context/graph differs")
        pins.update(context.record["protected_files"])
        folder, entry = Path(case["out"]), entries[case["name"]]
        for name, field in (("result.json", "result_sha256"), ("primary.pt", "primary_sha256"), ("primary.pt.json", "sidecar_sha256")):
            if file_hash(folder / name) != entry[field]:
                raise ValueError("Accepted full source file pair/result changed")
        result = json.loads((folder / "result.json").read_text())
        history = json.loads((folder / "history.json").read_text())
        selected = max(history, key=lambda h: h["macro_oa"])
        if result["per_dataset_updates"] != dict(Counter(schedule_for(case["settings"], task_names=context.sources))):
            raise ValueError("Full source exposure differs")
        for key in ("val_loss", "macro_oa"):
            np.testing.assert_allclose(result["selected_validation"][key], selected[key], rtol=1e-5, atol=1e-6)
        if result["selected_validation"]["metrics"] != selected["metrics"]:
            raise ValueError("Selected source metric record differs from frozen trajectory")
        row = audit_case(launch, case, context, args.launch_sha256)
        if any(entry[k] != value for k, value in row.items()):
            raise ValueError("Independent full source evidence differs from acceptance")
        rows.append(row)
        write_json(out / "progress.json", rows)
        print("NONE_FULL_AUDIT_CASE", case["name"], row["selected_step"], flush=True)
        del context
        torch.cuda.empty_cache()
    require_frozen_code(current)
    if any(file_hash(p) != h for p, h in pins.items()):
        raise ValueError("Protected historical/full source/input changed during audit")
    write_json(out / "receipt.json", {"status": "passed", "producer": producer, "audit_code": current,
        "launch_sha256": args.launch_sha256, "acceptance_sha256": args.acceptance_sha256,
        "protected_manifest_sha256": args.protected_sha256, "cases": rows, "protected_files_and_inputs": len(pins),
        "new_optimizer_updates": 0, "query_labels_accessed": False, "test_scoring": False,
        "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
