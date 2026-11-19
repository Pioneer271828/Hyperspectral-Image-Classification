"""Frozen same-point spectral comparisons for the released development role."""

import io
import json
import re
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loadData.observation_sampling import sample
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from research_runtime import hash_file
from transfer.g3_training import _forward, state_hash
from transfer.registered_development import _load_source, load_development
from transfer.spatial_coverage_protocol import apply_view, retained_positions
from transfer.wp11_matched_learning import HEADS, count_graphs, validation_blocks

POLICY = "wp12_frozen_development_stages_v1"
VIEWS = ("full", "central_gap", "matched_dispersed")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(root, pin):
    require(isinstance(pin, dict) and set(pin) == {"path", "sha256"}
            and re.fullmatch("[0-9a-f]{64}", pin["sha256"]),
            "Expected an explicit frozen artifact pin")
    path = Path(pin["path"])
    require(path.is_absolute() and path.resolve(strict=True) == path and root in path.parents,
            "Artifact must stay in the authorized runs root")
    require(hash_file(path) == pin["sha256"], "Frozen artifact hash differs")
    return path


def json_pin(root, pin):
    return json.loads(pinned(root, pin).read_bytes())


def file_pin(path):
    return {"path": str(path), "sha256": hash_file(path)}


def probabilities(values, n, classes):
    p = np.asarray(values)
    require(p.shape == (n, classes) and np.isfinite(p).all() and (p >= 0).all()
            and (p <= 1).all() and np.allclose(p.sum(1), 1, rtol=0, atol=2e-6),
            "Invalid aligned three-head probabilities")
    return p


def scores(rows, classes):
    """Regroup stored released-point rows without a model or a label loader."""
    ids = [tuple(row["id"]) for row in rows]
    truth = np.asarray([row["truth"] for row in rows], dtype=np.int64) - 1
    require(len(ids) > 0 and len(set(ids)) == len(ids) and ((truth >= 0) & (truth < classes)).all()
            and all(set(row["probabilities"]) == set(HEADS) for row in rows),
            "Expected unique released devval points with three heads")
    result = {}
    for head in HEADS:
        p = probabilities([row["probabilities"][head] for row in rows], len(rows), classes)
        matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (truth, p.argmax(1)), 1)
        support = matrix.sum(1)
        recall = [float(matrix[i, i] / n) if n else None for i, n in enumerate(support)]
        result[head] = dict(confusion=matrix.tolist(), support=support.tolist(), recall=recall,
                            count=len(rows), oa=float(matrix.trace()/len(rows)),
                            nll=float(-np.log(np.maximum(p[np.arange(len(rows)), truth], 1e-8)).mean()))
    return result


def paired(full, altered, classes, rivals):
    key = lambda row: (row["id"], row["truth"])
    require([key(row) for row in full] == [key(row) for row in altered], "Paired devval roster differs")
    result = {}
    for head in HEADS:
        a, b = (probabilities([row["probabilities"][head] for row in group], len(full), classes)
                for group in (full, altered))
        truth = np.asarray([row["truth"] for row in full], dtype=np.int64) - 1
        before, after = a.argmax(1), b.argmax(1)
        transitions = np.zeros((classes, classes, classes), dtype=np.int64)
        np.add.at(transitions, (truth, before, after), 1)
        sparse = [[int(v) + 1 for v in (y, u, w)] + [int(transitions[y, u, w])]
                  for y, u, w in zip(*np.nonzero(transitions), strict=True)]
        margins = {}
        for actual, rival in rivals.items():
            mask = truth == actual - 1
            u, w = actual - 1, rival - 1
            margins[str(actual)] = dict(rival=rival, n=int(mask.sum()),
                before=(a[mask, u] - a[mask, w]).tolist(), after=(b[mask, u] - b[mask, w]).tolist())
        result[head] = dict(transitions=sparse, locked_probability_margins=margins)
    return result


def roster(record, scene, classes):
    return dict(scene=scene, classes=classes, space=digest(record["roles"]),
                support=digest(record["train_support"]), validation=digest(record["validation_support"]),
                devval_points=len(record["validation_support"]["indices"]))


def stage_plan(root, release_pin):
    """Freeze identities, budget and comparison rules before any prediction readout."""
    release = json_pin(root, release_pin)
    from transfer.registered_wp11_learning import RELEASE

    require(release.get("policy") == RELEASE, "Released matched full trajectory required")
    audit = json_pin(root, release["history_evidence"])
    require(audit.get("policy") == RELEASE and audit.get("status") == "audited"
            and audit.get("exact_continuous_resume_completed") is True
            and audit.get("audit_optimizer_updates") == audit.get("query") == 0
            and audit["label_use"].get("selection_support_hash")
            and audit["registration"] == release["registration"]
            and audit["source_state"] == release["source_state"]
            and audit["full_state"] == release["full_state"]
            and audit["label_use"]["development"] == release["development"],
            "Independent release provenance differs")
    trajectory = audit["stages"]["full"]
    budget, middle, selected = trajectory["continuous"]["stop"], trajectory["prefix"]["stop"], audit["selected_steps"]["full"]
    require(budget >= 512 and 0 < middle < budget and 0 <= selected <= budget,
            "Completed pre-existing longer budget required")
    manifest = json_pin(root, release["development"])
    scene, record = manifest["dataset"], manifest["record"]
    require(scene == "xuzhou" and audit["label_use"]["scene"] == scene
            and digest(record["validation_support"]) == audit["label_use"]["selection_support_hash"]
            and digest(record["train_support"]) == audit["label_use"]["training_support_hash"],
            "Released development scene/role/points differ")
    from loadData.observation_sampling import BandMetadata

    bands = BandMetadata(**manifest["bands"])
    views = {name: [bands.band_ids[i] for i in retained_positions(bands, name)] for name in VIEWS}
    require(len(views["central_gap"]) == len(views["matched_dispersed"]), "Missing-band counts differ")
    registration = json_pin(root, release["registration"])
    require(registration["updates_per_arm"] == budget and registration["source_state"] == release["source_state"],
            "Registered update budget or source differs")
    return dict(policy=POLICY, release=release_pin, audit=release["history_evidence"],
                development=release["development"], contract=release["registration"]["sha256"],
                ancestry=release["source_state"], budget=budget,
                identities=roster(record, scene, 9), statistics=manifest["statistics"]["statistics_hash"],
                full_contract=registration["contracts"]["full"],
                bands=views, views=list(VIEWS), heads=list(HEADS),
                role="devval", graph="learned_from_each_retained_view", rivals={"5": 8, "6": 3},
                stages={"initial": {"step": 0, "state": release["source_weights"]},
                        "prefix": {"step": middle, "state": trajectory["prefix"]["checkpoint"]},
                        "selected": {"step": selected, "state": release["full_weights"]},
                        "last": {"step": budget, "state": trajectory["continuous"]["checkpoint"]}})


def load_stage(root, release, spec, name):
    step, pin = spec["step"], spec["state"]
    path = pinned(root, pin)
    if name in ("initial", "selected"):
        key = "source_metadata" if name == "initial" else "full_metadata"
        pinned(root, release[key])
        model = _load_source(pin, release[key])
        expected = release["source_state" if name == "initial" else "full_state"]
        require(state_hash(model) == expected, "Published model state differs")
    else:
        for field in ("full_weights", "full_metadata"):
            pinned(root, release[field])
        model = _load_source(release["full_weights"], release["full_metadata"])
        packet = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
        require(len(packet["history"]) == step and packet["history"][-1]["step"] == step,
                "Checkpoint does not terminate at the registered step")
        model.load_state_dict(packet["state"], strict=True)
    require(model.backbone.graph_mode == "learned", "Archived static graph cannot be used for missing-band replay")
    return model.eval().requires_grad_(False)


def checked_view(scene, tile, name, expected_bands):
    obs = observe_tile(scene.cube, scene.plan, scene.stats, tile)
    original, _ = sample(obs, scene.bands, {"mode": "none"})
    view, positions, mode = apply_view(original, scene.bands, name)
    require(list(view.bands.band_ids) == expected_bands and mode == "declared"
            and list(view.bands.band_ids) == [scene.bands.band_ids[i] for i in positions]
            and np.array_equal(view.observation.global_ids, original.observation.global_ids),
            "Frozen band order, original IDs or block geometry differs")
    return view


def predict(model, scene, record, band_name, band_ids, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    point_rows, hashes = [], {head: [] for head in HEADS}
    with torch.inference_mode(), count_graphs(model) as graphs:
        for tile, target in validation_blocks(scene, record):
            view = checked_view(scene, tile, band_name, band_ids)
            native = _forward(model, scene, tile, observation_view=view)[:3]
            if band_name == "full":
                original = _forward(model, scene, tile)[:3]
                for left, right in zip(native, original, strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            idx = np.flatnonzero(target)
            values = {}
            for head, p in zip(HEADS, native, strict=True):
                values[head] = probabilities(p.detach().cpu().numpy(), len(target), scene.classes)
                hashes[head].append(digest(values[head].tolist()))
            for index in idx:
                point_rows.append(dict(id=[tile["id"], int(view.observation.global_ids.ravel()[index])],
                                       truth=int(target[index]),
                                       probabilities={head: values[head][index].tolist() for head in HEADS}))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(points=point_rows, output_hash={head: digest(hashes[head]) for head in HEADS},
                graph_calls=graphs["count"], seconds=perf_counter()-started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def verify_original(stage, known):
    for head in HEADS:
        independently = stage["heads"][head]
        require(independently["confusion"] == known[head]["confusion"]
                and independently["support"] == known[head]["support"]
                and independently["count"] == known[head]["count"],
                "Original full-band decisions differ from the independent release")
        np.testing.assert_allclose(independently["nll"], known[head]["nll"], rtol=1e-5, atol=1e-6)


def run(root, plan, device):
    require(plan == stage_plan(root, plan["release"]), "Preregistered stage/seed/checkpoint/input identity differs")
    release = json_pin(root, plan["release"])
    audit = json_pin(root, plan["audit"])
    scene, record, inputs = load_development(plan["development"])
    require(roster(record, scene.dataset, scene.classes) == plan["identities"]
            and scene.stats["statistics_hash"] == plan["statistics"], "Opened development space or scale differs")
    results = {}
    for name, spec in plan["stages"].items():
        model = load_stage(root, release, spec, name).to(device)
        before = state_hash(model)
        by_view = {}
        for view in VIEWS:
            row = predict(model, scene, record, view, plan["bands"][view], device)
            row["heads"] = scores(row["points"], scene.classes)
            require(len(row["points"]) == plan["identities"]["devval_points"], "Incomplete development support")
            if view == "full" and name in audit["endpoints"]["full"]:
                known = audit["endpoints"]["full"][name]
                verify_original(row, known)
                row["original_bitwise_output"] = all(row["output_hash"][head] == known[head]["output_hash"]
                                                     for head in HEADS)
                require(device.type != "cpu" or row["original_bitwise_output"], "CPU output differs from archive")
            if view == "full" and name == "prefix":
                packet = torch.load(pinned(root, spec["state"]), map_location="cpu", weights_only=True)
                known = packet["history"][-1]["validation"]
                verify_original(row, {head: known if head == "fused" else row["heads"][head] for head in HEADS})
            by_view[view] = row
        require(state_hash(model) == before, "Frozen forward altered weights or normalization")
        results[name] = dict(step=spec["step"], checkpoint=spec["state"]["sha256"], state=before, views=by_view)
    for pin in inputs:
        pinned(root, pin)
    return dict(policy=POLICY, status="passed", plan_sha256=digest(plan), identities=plan["identities"],
                device=str(device), cases=results, optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)


def summarize(plan, receipt):
    require(receipt["policy"] == POLICY and receipt["status"] == "passed" and receipt["plan_sha256"] == digest(plan)
            and receipt["identities"] == plan["identities"] and receipt["optimizer_updates"] == 0
            and receipt["query_accessed"] is False and receipt["raw_gt_loaded"] is False
            and set(receipt["cases"]) == set(plan["stages"]), "Incomplete or foreign stage receipt")
    classes = plan["identities"]["classes"]
    rivals = {int(key): value for key, value in plan["rivals"].items()}
    baseline = receipt["cases"]["initial"]["views"]["full"]["points"]
    report = {}
    for name, stage in receipt["cases"].items():
        spec = plan["stages"][name]
        require(stage["step"] == spec["step"] and stage["checkpoint"] == spec["state"]["sha256"]
                and set(stage["views"]) == set(VIEWS), "Stage checkpoint or views differ")
        full = stage["views"]["full"]["points"]
        item = {}
        for view, row in stage["views"].items():
            independent = scores(row["points"], classes)
            require(independent == row["heads"], "Independent released-point reductions differ")
            item[view] = dict(heads=independent, comparison=paired(full, row["points"], classes, rivals))
        report[name] = dict(step=stage["step"], views=item,
                            initial_to_full=paired(baseline, full, classes, rivals))
    return report

def compare_devices(cpu, gpu):
    result = {}
    require(cpu["plan_sha256"] == gpu["plan_sha256"] and cpu["identities"] == gpu["identities"],
            "Device receipts use different experimental plans")
    for stage in cpu["cases"]:
        result[stage] = {}
        for view in VIEWS:
            a, b = (entry["cases"][stage]["views"][view] for entry in (cpu, gpu))
            require(a["graph_calls"] == b["graph_calls"]
                    and [(r["id"], r["truth"]) for r in a["points"]]
                    == [(r["id"], r["truth"]) for r in b["points"]], "Device graph/point roster differs")
            heads = {}
            for head in HEADS:
                p, q = [np.asarray([r["probabilities"][head] for r in item["points"]]) for item in (a, b)]
                heads[head] = dict(max_absolute=float(np.max(np.abs(p-q))),
                                   changed=int(np.count_nonzero(p.argmax(1) != q.argmax(1))))
            result[stage][view] = dict(cpu_seconds=a["seconds"], gpu_seconds=b["seconds"],
                                       peak_cuda_bytes=b["peak_cuda_bytes"], heads=heads)
    return result
