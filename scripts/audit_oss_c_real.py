"""Recompute the frozen OSS-C short-run results without optimizer or held-out access."""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.classification_metrics import metrics_from_confusion
from joint_calibrated_selection import observations
from joint_training import _tree_fingerprint, schedule_for
from model.bn_calibration import CONTEXT, POLICY, observation_id
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, hash_file, require_frozen_code, tensor_hash
from scripts.run_oss_c_pair import ARMS, CLASSES, GATES, GPU, SEEDS, campaign, release
from scripts.train_joint_m4 import graph_binding
from transfer.container_runtime import training_container_kind
from transfer.oss_c_oracles import HEADS, confusion, torchmetrics_oracle, validate_indices
from transfer.oss_c_vca import GlobalReferenceRuntime


PRODUCER = "a36a389d1ff55a77f921cf7e829f42f5abd1beaa"
PRODUCER_SOURCE = "cfde802fc22458c36e6a42698ccbfa1d55724f096f2c4f9fa4999d8a2e07cc44"
REGISTRATION = "8186b1fa5f822247a4f2fa5dd39f1825f45d9d553d0962c6f9b075dea8fef57e"
ROLES = ("train", "val")
STEPS = (3, 6, 9, 12)
KEYS = tuple(f"{arm}-{seed}" for arm in ARMS for seed in SEEDS)
FIXED_FILES = {"identity.json", "progress.json", "primary.pt", "primary.pt.json", "last.pt",
               "last.pt.json", "selected-probabilities.pt", "trace.json", "history.json"}
ARTIFACTS = FIXED_FILES | {f"{prefix}-{step:06d}.pt" for step in STEPS
                           for prefix in ("state", "raw-validation")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def close(actual, expected, message, tolerance=1e-12):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), message)


def artifact_hashes(directory, expected):
    require(not directory.is_symlink() and directory.is_dir(), "Invalid case directory")
    entries = list(directory.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in entries)
            and {path.name for path in entries} == expected, "Missing or unexpected artifacts")
    return {path.name: hash_file(path) for path in sorted(entries)}


def parse_snapshot(path):
    rows = [line.split("  ./", 1) for line in path.read_text().splitlines()]
    require(all(len(row) == 2 and len(row[0]) == 64 for row in rows), "Invalid completed-resume manifest")
    manifest = {name: sha for sha, name in rows}
    require(len(manifest) == len(rows), "Duplicate completed-resume manifest entry")
    return manifest


def verify_artifacts(directory, execution, key):
    hashes = artifact_hashes(directory, ARTIFACTS | {"result.json"})
    result = read(directory / "result.json")
    require(result["artifacts"] == {name: hashes[name] for name in ARTIFACTS},
            "Artifact hash or roster changed")
    before, after = (parse_snapshot(execution / f"{key}.{phase}.sha256")
                     for phase in ("before", "after"))
    require(hashes == before == after, "Completed-resume artifact hashes differ")
    return result, hashes


def readouts(items, metadata):
    result = {}
    for scene, item in items.items():
        result[scene] = {}
        for role in ROLES:
            labels = item[f"{role}_samples_gt"].flatten()
            require(not labels.is_floating_point() or torch.equal(labels, labels.trunc()),
                    "Nonintegral frozen labels")
            labels = labels.long()
            ids = (labels > 0).nonzero().flatten()
            validate_indices(ids, torch.sort(ids).values)
            mask = item[f"{role}_label_mask"].bool()
            require(mask.shape == (len(labels), CLASSES[scene]) and
                    torch.equal(mask, (labels > 0)[:, None].expand_as(mask)), "Label mask or order differs")
            require(torch.equal(item[f"{role}_gt_onehot"][ids].argmax(1) + 1, labels[ids]),
                    "Label one-hot order differs")
            support = torch.bincount(labels[ids], minlength=CLASSES[scene] + 1)[1:].tolist()
            require(support == metadata[scene]["support"][role] and all(n > 0 for n in support),
                    "Frozen role support differs")
            result[scene][role] = (labels[ids], support, tensor_hash(ids))
    return result


def validated_metrics(record, labels, scores=None, crosscheck=None):
    require(tuple(record) == tuple(labels), "Metric scene order differs")
    result = {}
    legacy = {}
    for scene, roles in labels.items():
        require(tuple(record[scene]) == ROLES, "Metric role order differs")
        result[scene] = {}
        for role, (truth, support, row_hash) in roles.items():
            heads = record[scene][role]
            require(tuple(heads) == HEADS, "Metric head order differs")
            computed = None
            if scores is not None:
                values = scores[scene][role]
                for probability in values.values():
                    require(probability.ndim == 2 and bool(torch.isfinite(probability).all())
                            and bool(((0 <= probability) & (probability <= 1)).all()),
                            "Nonfinite or out-of-range probability")
                    torch.testing.assert_close(probability.sum(1), torch.ones_like(probability[:, 0]),
                                               rtol=1e-5, atol=1e-6)
                computed = confusion(truth, values, len(support), role=role, active_heads=HEADS,
                                     crosscheck=crosscheck)
            result[scene][role] = {}
            for head, row in heads.items():
                matrix = row["confusion"]
                if computed is not None:
                    require(matrix == computed[head]["confusion"], "Probability confusion differs")
                metrics = metrics_from_confusion(matrix, range(1, len(support) + 1))
                require([item["support"] for item in metrics["per_class"]] == support, "Confusion support differs")
                require([item["recall"] for item in metrics["per_class"]] == row["recall"],
                        "Recall differs from confusion")
                for name in ("oa", "aa"):
                    close(row[name], metrics[name], f"{name} differs from confusion")
                if computed is not None:
                    picked = values[head].gather(1, (truth - 1)[:, None]).clamp_min(1e-8)
                    nll = float(-picked.log().mean())
                    close(row["nll"], nll, "NLL differs from probabilities", 3e-6)
                    metrics["nll"] = nll
                metrics["row_identity_sha256"] = row_hash
                result[scene][role][head] = metrics
        matrix = result[scene]["val"]["fused"]["confusion_matrix"]
        # The selection producer computes per-scene OA by dividing float32 counts.
        legacy[scene] = float(np.float32(np.trace(matrix)) / np.float32(np.sum(matrix)))
    return result, legacy


def evaluate_record(record, labels, *, probabilities=None, crosscheck=None):
    if probabilities is not None:
        require(tuple(probabilities) == tuple(labels) and
                all(tuple(roles) == ROLES for roles in probabilities.values()), "Probability scene or role order differs")
    result, legacy = validated_metrics(record["metrics"], labels, probabilities, crosscheck)
    require(record["legacy_val_oa"] == legacy, "Legacy per-scene OA differs from confusion")
    close(record["macro_oa"], mean(legacy.values()), "Macro OA differs from confusion")
    exact = mean(row["val"]["fused"]["oa"] for row in result.values())
    if probabilities is not None:
        close(record["val_loss"], mean(row["val"]["fused"]["nll"] for row in result.values()),
              "Validation loss differs from probabilities", 3e-6)
    return {"macro_oa": exact, "selection_macro_oa": mean(legacy.values()), "metrics": result}


def verify_history(events, labels, selected_step):
    require(tuple(event["step"] for event in events) == STEPS
            and tuple(event["epoch"] for event in events) == (0, 1, 2, 3), "Validation budget or chronology differs")
    evaluated = [evaluate_record(event, labels) for event in events]
    chosen = max(zip(STEPS, evaluated, strict=True), key=lambda row: row[1]["selection_macro_oa"])[0]
    require(selected_step == chosen, "Selected step is not the first maximum")
    return [{"step": event["step"], "val_loss": event["val_loss"], "macro_oa": checked["macro_oa"],
             "selection_macro_oa": checked["selection_macro_oa"]}
            for event, checked in zip(events, evaluated, strict=True)]


def verify_trace(trace, schedule, opts):
    require(len(trace) == 12 and [row["dataset"] for row in trace] == schedule
            and [row["step"] for row in trace] == list(range(1, 13))
            and [row["epoch"] for row in trace] == [i // 3 for i in range(12)], "Training budget or schedule differs")
    keys = ("graph", "adapter", "fused_head", "graph_head", "cnn_head")
    rates = [opts["config"]["training"][name] for name in ("lr", "head_lr")]
    for row in trace:
        require(set(row["gradient_norms"]) == set(keys) and
                all(math.isfinite(value) and value >= 0 for value in row["gradient_norms"].values())
                and all(math.isfinite(value) and value >= 0 for value in row["loss"].values())
                and row["learning_rates"] == rates and row["graph_changed_elements"] >= 0,
                "Invalid training loss, gradient, graph update or learning rate")
    participation = {key: sum(row["gradient_norms"][key] > 0 for row in trace) for key in keys}
    require(all(participation.values()) and any(row["graph_changed_elements"] for row in trace),
            "Required trainable groups did not participate")
    return {"first_loss": trace[0]["loss"], "last_loss": trace[-1]["loss"],
            "positive_gradient_steps": participation, "learning_rates": rates,
            "graph_changed_elements_by_step": [row["graph_changed_elements"] for row in trace]}


def verify_checkpoint(directory, result, hashes, selected_step, opts, observation_binding):
    progress = read(directory / "progress.json")
    require(progress["status"] == "completed" and progress["cursor"] == 12
            and progress["state_file"] == "state-000012.pt"
            and progress["state_sha256"] == hashes["state-000012.pt"]
            and progress["best_step"] == selected_step, "Final progress cursor or checkpoint differs")
    history = read(directory / "history.json")
    trace = read(directory / "trace.json")
    scheduler_best = math.inf
    bad_epochs = 0
    final = None
    for index, step in enumerate(STEPS):
        packet = state(directory / f"state-{step:06d}.pt")
        require(packet["cursor"] == step and packet["identity"] == result["identity"]
                and packet["start_code"] == result["start_code"]
                and packet["trace"] == trace[:step] and packet["history"] == history[:index + 1],
                "Saved training state differs from frozen chronology")
        require(all(_tree_fingerprint(packet[key]) == packet["state_fingerprints"][key]
                    for key in ("model", "optimizer", "scheduler")), "Checkpoint state fingerprint differs")
        raw = state(directory / f"raw-validation-{step:06d}.pt")
        require(_tree_fingerprint(raw) == packet["state_fingerprints"]["model"],
                "Raw calibration input differs from saved training state")
        optimizer = packet["optimizer"]
        groups = optimizer["param_groups"]
        require([group["lr"] for group in groups] == trace[step - 1]["learning_rates"]
                and all(group["weight_decay"] == opts["config"]["training"]["weight_decay"]
                        for group in groups), "Saved optimizer settings differ")
        counts = [int(row["step"]) for row in optimizer["state"].values()]
        require(counts and max(counts) == step and all(0 < count <= step for count in counts),
                "Saved optimizer counters differ from the training cursor")
        best = max(history[:index + 1], key=lambda row: row["macro_oa"])
        loss = history[index]["val_loss"]
        require(math.isfinite(loss) and loss >= 0, "Invalid validation loss")
        if loss < scheduler_best * (1 - opts["scheduler"]["threshold"]):
            scheduler_best = loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler = packet["scheduler"]
        require(packet["best_step"] == best["step"] and packet["best_score"] == best["macro_oa"]
                and scheduler["last_epoch"] == index + 1 and scheduler["best"] == scheduler_best
                and scheduler["num_bad_epochs"] == bad_epochs
                and scheduler["_last_lr"] == trace[step - 1]["learning_rates"]
                and all(scheduler[key] == opts["scheduler"][key]
                        for key in ("factor", "patience", "threshold")),
                "Saved selection or scheduler state differs")
        if step == 12:
            final = packet

    last, primary = (state(directory / f"{name}.pt") for name in ("last", "primary"))
    require(progress["best_score"] == final["best_score"], "Progress selected score differs")
    require(_tree_fingerprint(last) == result["last_model_fingerprint"] == final["state_fingerprints"]["model"],
            "Final checkpoint does not match the authoritative training state")
    require(_tree_fingerprint(primary) == _tree_fingerprint(final["best_model"]),
            "Primary checkpoint is not the selected state")
    calibration = None
    for name, step in (("last", 12), ("primary", selected_step)):
        meta = read(directory / f"{name}.pt.json")
        training = meta["training_config"]
        require(meta["class_counts"] == CLASSES and training["checkpoint_optimizer_step"] == step
                and training["research_config"] == opts["config"]
                and training["training_code"] == result["start_code"]
                and training["provenance"] == result["identity"]["prepared"]
                and training["joint_runtime"] == {**opts, "executed_optimizer_steps": 12,
                                                   "selected_step": selected_step},
                "Checkpoint sidecar is not bound to the case")
        if name == "primary":
            calibration = training["bn_calibration"]
    require(calibration == final["best_calibration"] and calibration["schema_version"] == 1
            and calibration["policy"] == POLICY and calibration["context"] == CONTEXT
            and calibration["scene_identity"] == observation_binding
            and calibration["source_checkpoint_sha256"] == hashes[f"raw-validation-{selected_step:06d}.pt"]
            and calibration["optimizer_updates"] == 0 and calibration["parameters_unchanged"] is True
            and calibration["state_sha256"] == digest({name: tensor_hash(value) for name, value in primary.items()}),
            "Primary calibration binding differs")
    selected_raw = state(directory / f"raw-validation-{selected_step:06d}.pt")
    calibrated = set(calibration["calibrated_buffers"])
    require(selected_raw.keys() == primary.keys() and calibrated <= primary.keys()
            and all(name.endswith((".running_mean", ".running_var")) for name in calibrated)
            and all(torch.equal(value, selected_raw[name]) for name, value in primary.items() if name not in calibrated),
            "Selected calibration changed protected parameters or buffers")
    probabilities = state(directory / "selected-probabilities.pt")
    require(_tree_fingerprint(probabilities) == _tree_fingerprint(final["best_probabilities"]),
            "Selected probability file differs from the training state")
    return last, probabilities


def audit_case(key, spec, registered, directory, execution, labels, graph, observation_binding, crosscheck):
    result, hashes = verify_artifacts(directory, execution, key)
    opts = spec["settings"]
    schedule = schedule_for(opts)
    provenance = dict(campaign_sha256=digest(registered), case=key, m4_graph_binding=graph,
                      prepared=registered["data_manifest"])
    identity = dict(protocol=opts["protocol"], settings_hash=spec["settings_sha256"], settings=opts,
                    execution_device="cuda:0", consumer_source_hash=PRODUCER_SOURCE, prepared=provenance,
                    schedule_sha256=spec["schedule_sha256"])
    source = result["start_code"]
    require(result["status"] == "completed" and result["optimizer_steps"] == spec["updates"] == 12
            and result["test_scoring"] is False and result["settings"] == opts
            and source["commit"] == PRODUCER and source["source_hash"] == PRODUCER_SOURCE
            and source["dirty"] is False and digest(source["files"]) == PRODUCER_SOURCE
            and digest(schedule) == spec["schedule_sha256"] == result["schedule_sha256"]
            and result["identity"] == identity == read(directory / "identity.json")
            and result["per_dataset_updates"] == dict(Counter(schedule))
            and result["execution"]["device"] == "cuda:0"
            and result["execution"]["visible_cuda_devices"] == GPU,
            "Run identity, device, schedule or update budget differs")
    error = result["selected_probability_max_abs_error"]
    require(math.isfinite(error) and 0 <= error <= GATES["checkpoint_probability_max_abs"],
            "Original same-device probability reload contract failed")
    selected_step = result["selected_step"]
    history = read(directory / "history.json")
    events = verify_history(history, labels, selected_step)
    trace = verify_trace(read(directory / "trace.json"), schedule, opts)
    last, probabilities = verify_checkpoint(directory, result, hashes, selected_step, opts, observation_binding)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec["seed"])
        model = GlobalReferenceRuntime(spec["arm"]).build_model(ResearchConfig.from_dict(opts["config"]), CLASSES)
    initial = model.state_dict()
    require(_tree_fingerprint(initial) == spec["initial_state_sha256"] and initial.keys() == last.keys(),
            "Initial model registration differs")
    changed = {}
    for name, parameter in model.named_parameters():
        require(bool(torch.isfinite(last[name]).all()), "Nonfinite trained parameter")
        changed[name] = int(torch.count_nonzero(last[name] != parameter))
    require(sum(n for name, n in changed.items() if name.startswith("backbone.adapter.")) > 0,
            "Trainable adapter did not change")
    selected = evaluate_record(result["selected_validation"], labels, probabilities=probabilities,
                               crosscheck=crosscheck)
    original = next(row for row in history if row["step"] == selected_step)
    from_history = evaluate_record(original, labels, probabilities=probabilities, crosscheck=crosscheck)
    require(selected == from_history, "Selected metrics differ from frozen event")
    return dict(arm=spec["arm"], seed=spec["seed"], updates=12, selected_step=selected_step,
                task_updates=result["per_dataset_updates"], events=events, training=trace,
                changed_parameter_elements=changed, reload_probability_max_abs=error,
                artifact_sha256=hashes, selected=selected)


def summarize(cases):
    arms = {}
    for arm in ARMS:
        cells = [cases[f"{arm}-{seed}"]["selected"] for seed in SEEDS]
        scenes = {}
        for scene in CLASSES:
            scenes[scene] = {}
            for role in ROLES:
                scenes[scene][role] = {}
                for head in HEADS:
                    rows = [cell["metrics"][scene][role][head] for cell in cells]
                    summary = {metric: distribution([row[metric] for row in rows])
                               for metric in ("oa", "aa", "kappa", "macro_f1", "nll")}
                    summary["class_recall"] = [
                        dict(label=group[0]["class_id"], support=group[0]["support"],
                             **distribution([row["recall"] for row in group]))
                        for group in zip(*(row["per_class"] for row in rows), strict=True)]
                    scenes[scene][role][head] = summary
        arms[arm] = dict(macro_oa=distribution([cell["macro_oa"] for cell in cells]), metrics=scenes)
    return arms


def distribution(values):
    if any(value is None for value in values):
        return dict(values=values, mean=None, sample_sd=None)
    return dict(values=values, mean=mean(values), sample_sd=stdev(values))


def compare(cases):
    result = {}
    for control in ("native", "matched_boundary_dropout"):
        pairs = [(cases[f"exclude_out_of_reference-{seed}"]["selected"], cases[f"{control}-{seed}"]["selected"])
                 for seed in SEEDS]
        macro = [(left["macro_oa"] - right["macro_oa"]) * 100 for left, right in pairs]
        scenes = {}
        for scene in CLASSES:
            pairs_by_role = {}
            for role in ROLES:
                pairs_by_head = {}
                for head in HEADS:
                    rows = [(left["metrics"][scene][role][head], right["metrics"][scene][role][head])
                            for left, right in pairs]
                    pair = {metric: distribution([(left[metric] - right[metric]) * 100 for left, right in rows])
                            for metric in ("oa", "aa", "kappa", "macro_f1")
                            if all(left[metric] is not None and right[metric] is not None for left, right in rows)}
                    pair["class_recall"] = [
                        dict(label=i + 1, support=rows[0][0]["per_class"][i]["support"],
                             **distribution([(left["per_class"][i]["recall"] - right["per_class"][i]["recall"]) * 100
                                             for left, right in rows]))
                        for i in range(CLASSES[scene])]
                    pairs_by_head[head] = pair
                pairs_by_role[role] = pairs_by_head
            scenes[scene] = pairs_by_role
        aa = {scene: roles["val"]["fused"]["aa"]["mean"] for scene, roles in scenes.items()}
        honghu = scenes["whu-honghu"]["val"]["fused"]["oa"]["mean"]
        small_failures = []
        for scene, roles in scenes.items():
            training_support = cases[KEYS[0]]["selected"]["metrics"][scene]["train"]["fused"]["per_class"]
            for row in roles["val"]["fused"]["class_recall"]:
                row["train_support"] = training_support[row["label"] - 1]["support"]
                if (row["train_support"] <= GATES["small_class_training_support_at_most"]
                        and row["mean"] < -GATES["all_small_class_mean_recall_drop_max_pp"]):
                    small_failures.append(dict(scene=scene, **row))
        gates = dict(primary=mean(macro) > 0 and sum(value > 0 for value in macro) >= 2,
                     per_scene_aa=all(value >= -GATES["per_scene_aa_drop_max_pp"] for value in aa.values()),
                     honghu_oa=honghu >= -GATES["honghu_oa_drop_max_pp"], all_small_classes=not small_failures)
        result[control] = dict(macro_oa_delta_pp=distribution(macro), positive_seeds=sum(x > 0 for x in macro),
                               all_scene_head_class_deltas_pp=scenes, per_scene_aa_delta_pp=aa,
                               honghu_oa_delta_pp=honghu, small_class_failures=small_failures, gates=gates,
                               development_gate_passed=all(gates.values()))
    return result


def verify_execution(execution, registered):
    authorization = read(execution / "authorization.json")
    identity = dict(registration_sha256=REGISTRATION, campaign_sha256=digest(registered),
                    source_commit=PRODUCER, source_hash=PRODUCER_SOURCE, gpu_uuid=GPU, budget=108,
                    scope="train_val_only", source_chat="01a1aa2e-9fe8-7782-a6e9-327ef5f16c57")
    require(all(authorization.get(key) == value for key, value in identity.items()),
            "Controller authorization identity differs")
    receipts = artifact_hashes(execution / "receipts", {f"{key}.json" for key in KEYS})
    for key in KEYS:
        receipt = read(execution / "receipts" / f"{key}.json")
        require(receipts[f"{key}.json"] == authorization["receipts"][key]
                and receipt == dict(schema="oss-c-explicit-launch-v1", campaign_sha256=digest(registered),
                                    source_commit=PRODUCER, case=key, updates=12, gpu_uuid=GPU,
                                    scope="train_val_only", budget_notice_sent=True), "Launch receipt differs")
    require((execution / "controller.exit").read_text().strip() == "0", "Controller did not complete")
    events = [line.split("\t") for line in (execution / "events.tsv").read_text().splitlines()]
    roster = [(phase, key) for phase in ("fresh", "completed") for key in KEYS]
    require(len(events) == 2 * len(roster), "Controller event roster differs")
    timing = {}
    for i, (phase, key) in enumerate(roster):
        start, end = events[2*i:2*i+2]
        require(start[1:] == [phase, key, "START"] and end[1:5] == [phase, key, "EXIT", "0"]
                and len(end) == 6 and start[0] <= end[0] and (i == 0 or events[2*i-1][0] <= start[0]),
                "Controller chronology or exit differs")
        elapsed = int(end[5])
        require(elapsed >= 0, "Negative controller elapsed time")
        log = (execution / f"{key}.{phase}.log").read_text().splitlines()
        expected = (dict(operation="completed", updates_this_call=12) if phase == "fresh"
                    else dict(operation="verified_completed", updates_this_call=0))
        require(json.loads(log[-1]) == expected, "Worker update receipt differs")
        timing.setdefault(key, {})[phase + "_seconds"] = elapsed
        if phase == "fresh":
            samples = [row.split(",") for row in (execution / f"{key}.gpu.csv").read_text().splitlines()]
            require(samples and all(len(row) == 4 for row in samples), "Missing GPU observations")
            timing[key]["device_peak_memory_used_mib"] = max(int(row[1]) for row in samples)
            timing[key]["gpu_samples"] = len(samples)
    wall = datetime.fromisoformat(events[-1][0]) - datetime.fromisoformat(events[0][0])
    return dict(cases=timing, wall_seconds=wall.total_seconds(),
                fresh_seconds=sum(row["fresh_seconds"] for row in timing.values()),
                completed_resume_updates=0, receipt_sha256=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--torchmetrics-source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = code_snapshot()
    require(training_container_kind() == "lxc" and source["commit"] == args.expected_commit,
            "Authorized LXC and exact analysis source required")
    require_frozen_code(source)
    runs = args.runs.resolve(strict=True)
    out = args.out.resolve()
    require(runs.name == "runs" and out.is_relative_to(runs), "New audit output must be inside runs")
    path = runs / "oss-c-review01/real-registration04/registration.json"
    require(not path.is_symlink() and hash_file(path) == REGISTRATION, "Frozen registration bytes differ")
    registered = read(path)
    loaded = release(runs)
    require(registered == campaign(runs, {"commit": PRODUCER, "source_hash": PRODUCER_SOURCE}, loaded=loaded),
            "Frozen campaign configuration or input identity differs")
    items, _ = loaded
    labels = readouts(items, registered["metadata"])
    graph = graph_binding(items)
    observation_binding = observation_id(observations(items))
    crosscheck = torchmetrics_oracle(args.torchmetrics_source) if args.torchmetrics_source else None
    value = dict(analysis_commit=source["commit"], analysis_source_hash=source["source_hash"],
                 producer_commit=PRODUCER, registration_sha256=REGISTRATION, campaign_sha256=digest(registered),
                 supports={scene: {role: dict(class_support=support, row_identity_sha256=row_hash)
                                   for role, (_, support, row_hash) in roles.items()}
                           for scene, roles in labels.items()},
                 oracle="torch_numpy_torchmetrics" if crosscheck else "torch_numpy",
                 heldout_reads=0, audit_optimizer_updates=0, production_authorized=False)
    if args.execution is not None:
        execution = args.execution.resolve(strict=True)
        require(execution.is_relative_to(runs), "Execution path must be inside runs")
        case_root = path.parent / "cases"
        require(not case_root.is_symlink() and {case.name for case in case_root.iterdir()} == set(KEYS),
                "Exactly nine registered case directories required")
        timing = verify_execution(execution, registered)
        cases = {}
        for key, spec in registered["cases"].items():
            cases[key] = audit_case(key, spec, registered, case_root / key, execution,
                                    labels, graph, observation_binding, crosscheck)
        comparisons = compare(cases)
        value.update(status="audited", real_updates=sum(case["updates"] for case in cases.values()),
                     timing=timing, cases=cases, arm_summary=summarize(cases), comparisons=comparisons,
                     development_gate_passed=all(row["development_gate_passed"] for row in comparisons.values()))
    else:
        value.update(status="registration_verified", case_results_read=0)
    require_frozen_code(source)
    require(hash_file(path) == REGISTRATION, "Registration changed during audit")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("status", "oracle", "audit_optimizer_updates")}))


if __name__ == "__main__":
    main()
