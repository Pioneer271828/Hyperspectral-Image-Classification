"""Six real source-only v2 engineering cases, 48 updates including resume."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import independent_trainval
from joint_baseline import NAMES
from joint_training import load_plan, run_training, _tree_fingerprint
from loadData.dataset_registry import file_hash
from m4_ordered_source import MODES, short_settings, M4SourceOrderedShortRuntime
from m4_transfer import graph_binding
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from source_data_context import load_source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "prepared-dir", "data-root", "out-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Expected clean remote commit required")
    out = external_directory(args.out_dir)
    out.mkdir(exist_ok=False)
    manifest = load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"]
    start = time.monotonic()
    rows = []
    for target in NAMES:
        context = load_source_context(args.prepared_dir, args.data_root, held_out=target,
                                      consumer=code, manifest_sha256=manifest)
        data = context.items
        gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
               for n, item in data.items()}
        for mode in MODES:
            settings = short_settings(target, 64, mode)
            folder = out / (target + "-" + mode)
            provenance = {**context.record, "m4_mode": mode, "m4_graph_binding": graph_binding(data)}
            runtime = M4SourceOrderedShortRuntime()
            def guard():
                require_frozen_code(code)
                if any(file_hash(p) != h for p, h in context.record["protected_files"].items()):
                    raise ValueError("Source input changed")
            common = dict(finish_guard=guard, runtime_adapter=runtime, device="cuda")
            tick = time.monotonic()
            run_training(settings, data, folder, provenance, code, stop_after=2, **common)
            result = run_training(settings, data, folder, provenance, code, resume=True, **common)
            if result["updates_this_call"] != 6:
                raise ValueError("Short resume budget changed")
            before = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
            if run_training(settings, data, folder, provenance, code, resume=True, **common)["updates_this_call"]:
                raise ValueError("Completed recovery performed updates")
            state = torch.load(folder / "state-000008.pt", map_location="cpu", weights_only=True)
            saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
            raw = folder / f'raw-validation-{state["best_step"]:06d}.pt'
            repeats = []
            for _ in range(3):
                model = runtime.load_selected(folder / "primary.pt", gpu, raw, device="cuda")
                repeats.append(_tree_fingerprint(runtime.validation_snapshot(model, gpu)[1]))
                del model
            if len(set(repeats)) != 1 or repeats[0] != _tree_fingerprint(saved):
                raise ValueError("Ordered probability reload is not exact")
            event = next(h for h in state["history"] if h["step"] == state["best_step"])
            audits = [independent_trainval(saved[n], {
                "class_num": item["class_num"],
                **{phase + "_labels": item[phase + "_samples_gt"][item[phase + "_samples_gt"] > 0].long() - 1
                   for phase in ("train", "val")}}, event["metrics"][n]) for n, item in data.items()]
            if not all(t["m4_parameter_gradient_norm"] > 0 for t in state["trace"]):
                raise ValueError("M4 gradients missing")
            if before != {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}:
                raise ValueError("Completed files changed")
            row = {"target": target, "mode": mode, "updates": 8, "selected_step": state["best_step"],
                   "elapsed_seconds": time.monotonic() - tick, "exact_reload_repeats": 3,
                   "metrics": sum(a["head_metrics"] for a in audits),
                   "recalls": sum(a["class_recalls"] for a in audits),
                   "protected_files": len(before), "loss_first": state["trace"][0]["loss"]["total"],
                   "loss_last": state["trace"][-1]["loss"]["total"]}
            rows.append(row)
            write_json(out / "progress.json", rows)
            print("ORDERED_PRECHECK_CASE", row, flush=True)
        guard()
        del gpu, context
    write_json(out / "receipt.json", {"status": "passed", "code": code, "updates": 48,
        "target_payload_accessed": False, "test_scoring": False, "cases": rows,
        "elapsed_seconds": time.monotonic() - start, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})


if __name__ == "__main__":
    main()

