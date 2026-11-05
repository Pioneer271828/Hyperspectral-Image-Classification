"""Zero-update repeatability experiment for the two frozen Issue 168 source models.

Run in the clean production checkout with this external driver. Report native
probabilities separately from instrumented forwards. No training or finalization.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

PRODUCER = "5b4be2aebc528c0442264a18426bb68adceeef5d"


def differences(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Comparisons require matching shapes and dtypes")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite comparison")
    delta = (candidate - reference).abs()
    return {
        "max_abs": float(delta.max()),
        "unequal": int(torch.count_nonzero(delta)),
        "over_tolerance": int((~torch.isclose(candidate, reference, rtol=1e-5, atol=1e-6)).sum()),
    }


def trace_forward(model, task, item):
    from joint_baseline import forward
    from research_runtime import tensor_hash

    stages, sparse_replays = [], []
    def observe(label, tensor):
        stages.append((label, tensor_hash(tensor)))

    def watch(label):
        def hook(module, args, output):
            observe(label, output)
        return hook

    native_mm = torch.sparse.mm
    def replay(matrix, features):
        # The first (native) result continues down the model. The second is
        # discarded after comparison; inputs and operands are exactly shared.
        result = native_mm(matrix, features)
        caller = sys._getframe(1).f_code.co_name
        label = f"{caller}.{len(sparse_replays)}"
        sparse_replays.append({"op": label, **differences(result, native_mm(matrix, features))})
        observe(label, result)
        return result

    backbone = model.backbone
    with torch.no_grad(), ExitStack() as stack:
        for name, layer in backbone.named_modules():
            if (name == "adapter" or name.startswith(("CNN_denoise.", "CNN_Branch."))
                    and name.count(".") == 1 or name == "region_residual.projection"):
                stack.callback(layer.register_forward_hook(watch(name)).remove)
        for name in ("heads", "hgnn_heads", "cnn_heads"):
            layer = getattr(model, name)[task]
            stack.callback(layer.register_forward_hook(watch(name)).remove)
        stack.enter_context(patch.object(torch.sparse, "mm", replay))
        output = forward(model, task, item)
    return output, stages, sparse_replays


def compare_readout(saved, current):
    result = {}
    for split, heads in current.items():
        for head, candidate in heads.items():
            old = saved[split][head].cpu()
            new = candidate.cpu()
            result[split + "/" + head] = {
                **differences(old, new),
                "decision_changes": int((old.argmax(1) != new.argmax(1)).sum()),
            }
    return result


def csr_intervention(model, items, saved):
    """Four fixed FP32 evaluation paths, three repeats each, same frozen weights."""
    from joint_baseline import forward, readout
    from target_initialization import state_id

    model.eval()
    initial = state_id(model)
    native_mm = torch.sparse.mm
    rows = []
    for scope in ("native", "pool", "graph", "pool_and_graph"):
        # Keep broadcasts native in all four paths. Their same-input replay was stable.
        for task, item in items.items():
            first = None
            for repeat in range(3):
                calls = []
                def multiply(matrix, features):
                    caller = sys._getframe(1).f_code.co_name
                    if (caller == "pool_regions" and scope in ("pool", "pool_and_graph")
                            or caller == "_graph_conv" and scope in ("graph", "pool_and_graph")):
                        calls.append(caller)
                        matrix = matrix.coalesce().to_sparse_csr()
                    return native_mm(matrix, features)

                with torch.no_grad(), ExitStack() as stack:
                    if scope != "native":
                        stack.enter_context(patch.object(torch.sparse, "mm", multiply))
                    outputs = forward(model, task, item)
                    current = {s: readout(outputs, item, s)[1] for s in ("train", "val")}
                expected = {"native": [], "pool": ["pool_regions"],
                            "graph": ["_graph_conv"] * 2,
                            "pool_and_graph": ["pool_regions", "_graph_conv", "_graph_conv"]}
                if calls != expected[scope]:
                    raise ValueError("Unexpected sparse operator coverage")
                if first is None:
                    first = current
                rows.append({"scope": scope, "task": task, "repeat": repeat,
                             "vs_saved": compare_readout(saved[task], current),
                             "vs_first": compare_readout(first, current)})
                del outputs
    if state_id(model) != initial:
        raise ValueError("Frozen model changed during intervention")
    return rows


def study_case(model, items, saved):
    from joint_baseline import forward, readout
    from target_initialization import state_id

    model.eval()
    initial = state_id(model)
    rows = []
    with torch.no_grad():
        for task, item in items.items():
            first = None
            for phase in ("native", "traced"):
                for repeat in range(3):
                    row = {"task": task, "phase": phase, "repeat": repeat}
                    if phase == "native":
                        outputs = forward(model, task, item)
                    else:
                        outputs, stages, row["same_input_sparse_replays"] = trace_forward(model, task, item)
                        if first is None:
                            first = stages
                        if [k for k, _ in first] != [k for k, _ in stages]:
                            raise ValueError("Forward trace structure changed")
                        row["different_stages_from_first_trace"] = [
                            key for (key, old), (_, new) in zip(first, stages) if old != new]
                    current = {s: readout(outputs, item, s)[1] for s in ("train", "val")}
                    row["probabilities"] = compare_readout(saved[task], current)
                    rows.append(row)
                    del outputs
    if state_id(model) != initial:
        raise ValueError("Frozen model changed during tracing")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csr-intervention", action="store_true",
                        help="Run the four frozen evaluation paths instead of layer tracing")
    args = parser.parse_args()
    # Import only the production checkout's model, loader and validation code.
    sys.path.insert(0, str(Path.cwd()))
    from joint_training import _tree_fingerprint, load_plan
    from loadData.dataset_registry import file_hash
    from m4_source_study import M4SourceFullRuntime, MODES, select_case
    from research_runtime import code_snapshot, external_directory, write_json
    from source_data_context import load_source_context

    run = args.run.resolve(strict=True)
    out = external_directory(args.out)
    if out.parent != run.parent or out == run:
        raise ValueError("Use a fresh sibling of the frozen run directory")
    code = code_snapshot()
    status = json.loads((run / "status.json").read_text())
    launch = json.loads((run / "launch.json").read_text())
    if (code["commit"] != PRODUCER or code["dirty"] or status["status"] != "failed"
            or status["launch_sha256"] != file_hash(run / "launch.json")
            or Path(launch["out_dir"]).resolve() != run):
        raise ValueError("Frozen checkout, launch or failed batch identity differs")

    cases = [select_case(launch, code, "paviaU", 64, mode) for mode in MODES]
    context = load_source_context(launch["prepared"], launch["data_root"], held_out="paviaU",
        consumer=code, manifest_sha256=load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"])
    if any(c["source_context"] != context.record for c in cases):
        raise ValueError("Source identities differ")
    def protected():
        return {str(p): file_hash(p) for p in run.rglob("*") if p.is_file()}

    before = protected()
    out.mkdir(exist_ok=False)
    items = {name: {k: v.cuda() if torch.is_tensor(v) else v for k, v in item.items()}
             for name, item in context.items.items()}
    results = {}
    for case in cases:
        folder = Path(case["out"])
        state = torch.load(folder / "state-001002.pt", map_location="cpu", weights_only=True)
        saved = torch.load(folder / "selected-probabilities.pt", map_location="cpu", weights_only=True)
        model = M4SourceFullRuntime().load_selected(folder / "primary.pt", items,
            folder / "raw-validation-001002.pt", device="cuda")
        if (state["start_code"] != code or state["best_step"] != 1002 or state["cursor"] != 1002
                or model.training_config["provenance"] != case["provenance"]
                or _tree_fingerprint(model.state_dict()) != _tree_fingerprint(state["best_model"])
                or _tree_fingerprint(saved) != _tree_fingerprint(state["best_probabilities"])):
            raise ValueError("Frozen state or selected probabilities differ")
        del state
        experiment = csr_intervention if args.csr_intervention else study_case
        results[case["mode"]] = experiment(model, items, saved)
        del model
    if protected() != before or code_snapshot() != code or any(
            file_hash(p) != h for p, h in context.record["protected_files"].items()):
        raise ValueError("Frozen inputs, artifacts or production code changed")
    write_json(out / "summary.json", {"producer": PRODUCER, "driver_sha256": file_hash(__file__),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "protected_files": len(before),
        "protected_inputs": len(context.record["protected_files"]), "updates": 0,
        "experiment": experiment.__name__, "target_access": False,
        "original_failure_retained": True, "cases": results})


if __name__ == "__main__":
    main()

