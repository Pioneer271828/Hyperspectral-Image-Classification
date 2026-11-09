"""Independent train/validation and immutable-artifact audit of all 18 v2 sources."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from evaluation.g4_target import independent_trainval
from joint_training import _tree_fingerprint, load_plan, schedule_for
from loadData.dataset_registry import file_hash
from m4_ordered_source import M4SourceOrderedRuntime, select_case
from m4_transfer import graph_binding
from research_runtime import code_snapshot, external_directory, write_json
from source_data_context import load_source_context
from target_initialization import state_id
from model.research_config import ResearchConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "launch-sha256", "acceptance-sha256", "producer", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(args.run).resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != root.parent or out == root:
        raise ValueError("Fresh sibling audit output required")
    if file_hash(root / "launch.json") != args.launch_sha256 or file_hash(root / "training-acceptance.json") != args.acceptance_sha256:
        raise ValueError("Pinned batch identity changed")
    launch = json.loads((root / "launch.json").read_text())
    accepted = json.loads((root / "training-acceptance.json").read_text())
    producer = launch["code"]
    current = code_snapshot()
    if (producer["commit"] != args.producer or producer["dirty"] or current["dirty"]
            or any(current["files"].get(k) != v for k, v in producer["files"].items())
            or accepted["status"] != "passed" or accepted["optimizer_updates"] != 18036
            or accepted["code"] != producer or len(accepted["cases"]) != 18
            or accepted["launch_sha256"] != args.launch_sha256):
        raise ValueError("Accepted batch or additive verifier differs")
    pins = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = []
    for target in launch["study"]["held_targets"]:
        context = load_source_context(launch["prepared"], launch["data_root"], held_out=target,
            consumer=producer, manifest_sha256=load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"])
        pins.update(context.record["protected_files"])
        gpu = {n: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
               for n, item in context.items.items()}
        for case in [c for c in launch["cases"] if c["held_out"] == target]:
            select_case(launch, producer, target, case["seed"], case["mode"])
            if case["source_context"] != context.record or case["graph_binding"] != graph_binding(context.items):
                raise ValueError("Frozen source context differs")
            folder = Path(case["out"])
            result = json.loads((folder / "result.json").read_text())
            entry = next(c for c in accepted["cases"] if c["name"] == case["name"])
            if file_hash(folder / "result.json") != entry["result_sha256"] or file_hash(root / "reviews" / (case["name"] + ".json")) != entry["review_sha256"]:
                raise ValueError("Accepted result/review changed")
            if (result["optimizer_steps"] != 1002 or result["start_code"] != producer
                    or result["settings"] != case["settings"] or result["status"] != "completed"
                    or result["identity"]["prepared"] != case["provenance"]):
                raise ValueError("Case execution identity differs")
            for name, digest in result["artifacts"].items():
                if Path(name).name != name or file_hash(folder / name) != digest:
                    raise ValueError("Case artifact changed")
            progress = json.loads((folder / "progress.json").read_text())
            state = torch.load(folder / progress["state_file"], map_location="cpu", weights_only=True)
            saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
            trace, history = state["trace"], state["history"]
            if (state["cursor"] != 1002 or len(trace) != 1002 or len(history) != 51
                    or state["start_code"] != producer or progress["status"] != "completed"
                    or [r["dataset"] for r in trace] != schedule_for(case["settings"], task_names=context.sources)
                    or state["best_step"] != max(history, key=lambda h: h["macro_oa"])["step"]
                    or state["best_step"] != result["selected_step"]
                    or _tree_fingerprint(saved) != _tree_fingerprint(state["best_probabilities"])):
                raise ValueError("State, selection or exposure differs")
            for name in ("model", "optimizer", "scheduler"):
                if _tree_fingerprint(state[name]) != state["state_fingerprints"][name]:
                    raise ValueError("Authoritative state fingerprint differs")
            selected = next(h for h in history if h["step"] == state["best_step"])
            audits = []
            for name, item in context.items.items():
                reference = {"class_num": item["class_num"]}
                for phase in ("train", "val"):
                    gt = item[phase + "_samples_gt"]
                    reference[phase + "_labels"] = gt[gt > 0].long() - 1
                audits.append(independent_trainval(saved[name], reference, selected["metrics"][name]))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(case["seed"])
                initial = ResearchConfig.from_dict(case["settings"]["config"]).build_model(context.class_counts)
            if state_id(initial) != case["initial_state_sha256"]:
                raise ValueError("Initial model differs")
            changed = {}
            for group, prefix in {"adapter": "backbone.adapter.", "cnn": "backbone.CNN_Branch.",
                                  "graph": "backbone.HGNN_Branch.", "m4": "backbone.region_residual."}.items():
                changed[group] = sum(int(torch.count_nonzero(state["model"][n] - p.detach()))
                                     for n, p in initial.named_parameters() if n.startswith(prefix))
                if changed[group] <= 0:
                    raise ValueError("Module has no actual updates")
            del initial
            for _ in range(3):
                model = M4SourceOrderedRuntime().load_selected(folder / "primary.pt", gpu,
                    folder / f'raw-validation-{state["best_step"]:06d}.pt', device="cuda")
                if _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"]):
                    raise ValueError("Selected model identity differs")
                if _tree_fingerprint(M4SourceOrderedRuntime.validation_snapshot(model, gpu)[1]) != _tree_fingerprint(saved):
                    raise ValueError("Exact independent reload failed")
                del model
            row = {"name": case["name"], "target": target, "seed": case["seed"], "mode": case["mode"],
                   "selected_step": state["best_step"], "source_validation_macro_oa": selected["macro_oa"],
                   "train_fused_oa": {n: selected["metrics"][n]["train"]["fused"]["oa"] for n in context.sources},
                   "source_val_fused_oa": {n: selected["metrics"][n]["val"]["fused"]["oa"] for n in context.sources},
                   "loss_first30": sum(r["loss"]["total"] for r in trace[:30]) / 30,
                   "loss_last30": sum(r["loss"]["total"] for r in trace[-30:]) / 30,
                   "last50epoch_validation_gain": history[-1]["macro_oa"] - history[-6]["macro_oa"],
                   "positive_m4_gradient_steps": sum(r["m4_parameter_gradient_norm"] > 0 for r in trace),
                   "changed_parameters": changed, "exact_reload_repeats": 3,
                   "metrics": sum(a["head_metrics"] for a in audits),
                   "recalls": sum(a["class_recalls"] for a in audits)}
            rows.append(row)
            write_json(out / "progress.json", rows)
            print("V2_SOURCE_AUDIT_CASE", row, flush=True)
            del state, saved
        del gpu, context
    if any(file_hash(p) != h for p, h in pins.items()) or code_snapshot() != current:
        raise ValueError("Protected artifact/input/code changed during audit")
    write_json(out / "receipt.json", {"status": "passed", "producer": producer, "audit_code": current,
        "launch_sha256": args.launch_sha256, "acceptance_sha256": args.acceptance_sha256,
        "cases": rows, "protected_files_and_inputs": len(pins), "new_optimizer_updates": 0,
        "target_payload_accessed": False, "test_scoring": False, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()

