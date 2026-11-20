"""Zero-update representatives on the original G4 graph and ordered validation."""

from pathlib import Path
from time import perf_counter

import torch

from evaluation.frozen_stage_archives import admission, aggregate_metrics
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, paired, pinned, require
from evaluation.g4_target import independent_trainval
from g4_matched_target_execution import case_data
from g4_matched_target_study import initialize, runtime
from m4_ordered_source import OrderedValidation
from model.research_config import digest
from target_bn_calibration import CONTEXT, GRAPH_POLICY, POLICY, calibrate_target
from target_initialization import state_id
from target_training import observation
from transfer.registered_development import _json_new


def fresh(scene, case):
    model, witness = initialize(scene, case["seed"], case["method"], case["mode"], source_spec=case["source_spec"])
    require(witness == case["initialization"] and state_id(model) == case["initial_state_sha256"],
            "Historical seed, source or untrained task head differs")
    return model


def rows(values, item):
    ids = item["val_indices"].tolist()
    truth = item["val_labels"].tolist()
    return [dict(id=[index], truth=label+1, probabilities={head: values[head][j].tolist() for head in HEADS})
            for j, (index, label) in enumerate(zip(ids, truth, strict=True))]


def replay(root, plan, folder, device):
    require(plan == admission(root, plan["pins"]), "Historical preregistered plan differs")
    launch = json_pin(root, plan["pins"]["launch"])
    picked = [row for row in launch["cases"] if (row["seed"], row["mode"], row["method"]) ==
              (64, "none", "target_scratch_full")]
    require(len(picked) == len({row["target"] for row in picked}) == 3, "One fixed scratch representative per scene")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folder.mkdir(exist_ok=True)
    result = {}
    for case in picked:
        scene, name = case["target"], case["name"]
        identity = plan["cases"][name]
        place = Path(case["out"])
        document = json_pin(root, {"path": str(place/"result.json"), "sha256": identity["result"]})
        hashes = document["artifacts"]
        history = json_pin(root, {"path": str(place/"history.json"), "sha256": hashes["history.json"]})
        selected = next(event for event in history if event["step"] == identity["selected_step"])
        proof = {}
        previous = None
        for role, step in (("initial", 0), ("first_update", 1), ("mid_751", 751), ("last", 1500)):
            filename = f"raw-validation-{step:06d}.pt"
            sha = identity["init"] if step == 0 else hashes[filename]
            dest = folder/f"{name}-{role}.json"
            if dest.exists():
                receipt = json_pin(root, file_pin(dest))
                require(receipt["identity"] == identity and receipt["step"] == step and receipt["state_pin"] == sha
                        and receipt["device"] == str(device), "Historical representative restart differs")
            else:
                if previous is None:
                    previous = case_data(case)
                items, _, guard = previous
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
                model = fresh(scene, case).to(device).eval()
                if step:
                    state = torch.load(pinned(root, {"path": str(place/filename), "sha256": sha}),
                                       map_location="cpu", weights_only=True)
                    model.load_state_dict(state, strict=True)
                before = state_id(model)
                with OrderedValidation():
                    calibrated, use = calibrate_target(model, observation(model, items[scene]), mode=POLICY,
                                    context=CONTEXT, graph_policy=GRAPH_POLICY, source_state_sha256=before)
                require(state_id(model) == before and use["optimizer_updates"] == 0, "Original raw state changed")
                del model
                event, probabilities = runtime(case, items).validation_snapshot(calibrated, items)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                elapsed = perf_counter()-started
                actual = event["metrics"][scene]
                aggregate_metrics(actual, items[scene]["class_num"])
                if step:
                    expected = next(value for value in history if value["step"] == step)["metrics"][scene]
                    independent_trainval(probabilities[scene], items[scene], expected)
                receipt = dict(identity=identity, step=step, role=role, state_pin=sha, raw_state=before,
                               recalibrated_state=use["state_sha256"], metrics=actual,
                               val_points=rows(probabilities[scene]["val"], items[scene]), seconds=elapsed,
                               peak_cuda_bytes=peak, device=str(device), optimizer_updates=0, query_accessed=False)
                guard()
                _json_new(dest, receipt)
                del calibrated
            proof[role] = receipt
        rivals = {5: 8, 6: 3} if scene == "xuzhou" else {}
        base = proof["initial"]["val_points"]
        result[name] = dict(identity=identity, selected_archived={"step": selected["step"], "metrics": selected["metrics"][scene]},
                            stages={role: {**item, "initial_to_full": paired(base, item["val_points"],
                                                    len(item["metrics"]["val"]["fused"]["recall"]), rivals)}
                                    for role, item in proof.items()})
    return dict(scope=plan["scope"], plan_sha256=digest(plan), device=str(device), cases=result,
                optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)
