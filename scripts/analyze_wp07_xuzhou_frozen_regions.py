"""Attach existing representation interventions to the accepted WP07 Xuzhou V5 runtime."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace


PRODUCER = "34b9ddbbe75a8f51cc3c86c6694da0c45abe57a8"
SOURCE_HASH = "e0c7955d0387c7a3389dfb9dc92d62c324f99f5b8d057bddf0081b4260d0931d"
SCHEMA = "wp07_real_xuzhou_legacy_v5_region_intervention_v1"
INPUTS = {
    "registration": ("registration.json", "711e90e6b08d9c4554a051bd2e5ae3d133e62a2fc6a897c362b7545ff1984604"),
    "completion": ("batch-complete.json", "07b984a6e49a2c567c93f5713e1f9447db990acda24fab15e1760531da900281"),
    "audit": ("audit.json", "29ed4f67bf08f13070c20fec21ecb77f834a55778c5a090121737dcba0705a32"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("worker", "runs", "out"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    worker = args.worker.resolve(strict=True)
    candidate = Path(__file__).resolve().parents[1]
    # All production model, release, load and forward imports use the frozen checkout.
    sys.path.insert(0, str(worker))
    sys.path.append(str(candidate))
    import numpy as np
    import torch
    from evaluation.classification_metrics import classification_metrics
    from loadData.spatial_protocol import forward_partition
    from loadData.spatial_tiles import observe_tile
    from research_runtime import ROOT, code_snapshot, require_frozen_code, tensor_hash
    from transfer import matched_route_admission as admission, registered_development as dev
    from transfer.g3_training import state_hash
    from transfer.xuzhou_multiscale import ARMS, BUDGET, SEED, load_registration, read_stage
    from transfer.xuzhou_multiscale_audit import HEADS, _packet, _verify_history

    control_path = candidate / "monitoring" / "native_region_analysis.py"
    control_spec = spec_from_file_location("wp07_compat_region_control", control_path)
    control = module_from_spec(control_spec)
    sys.modules[control_spec.name] = control
    control_spec.loader.exec_module(control)
    require(ROOT == worker and Path(control.__file__).resolve() == control_path,
            "Compatibility imports escaped the selected worker or consumer")
    torch.set_num_threads(1)
    code, backend = code_snapshot(candidate), code_snapshot()
    require(not code["dirty"] and code["commit"] == args.expected_commit and
            (backend["commit"], backend["source_hash"]) == (PRODUCER, SOURCE_HASH),
            "Wrong candidate or frozen Xuzhou worker")
    require_frozen_code(backend)
    runs, out = args.runs.resolve(strict=True), args.out.resolve()
    source = runs / "wp07-xuzhou-multiscale-real01"
    require(runs in out.parents and worker not in out.parents and not out.exists(), "Expected isolated analysis output")
    pins = {kind: {"path": str(source / name), "sha256": sha} for kind, (name, sha) in INPUTS.items()}
    root, spec, baseline, scene, record, exposure = load_registration(pins["registration"], backend)
    audit, complete = (admission.read(pins[key]) for key in ("audit", "completion"))
    require(root == source and audit["status"] == "passed" and audit["registration"] == pins["registration"]
            and audit["complete"] == pins["completion"] and audit["independent_forwards"] == 80
            and complete["updates"] == 4096 and complete["registration"] == pins["registration"]
            and audit["query"] == audit["raw_gt"] == audit["independent_updates"] == 0
            and complete["query"] == complete["raw_gt"] == 0,
            "Legacy V5 audit, completion or release scope differs")
    out.mkdir()
    published = dict(zip(record["validation_support"]["indices"], record["validation_support"]["labels"], strict=True))
    width = scene.plan["shape"][1]
    views = []
    seen = []
    for row in record["roles"]:
        if row["role"] != "devval":
            continue
        tile = scene.plan["tiles"][row["id"]]
        actual = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        y0, y1, x0, x1 = row["box"]
        ids = sorted(i for i in published if y0 <= i//width < y1 and x0 <= i%width < x1)
        positions = [(i//width-y0)*(x1-x0)+i%width-x0 for i in ids]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        lattice = SimpleNamespace(factor=1, areas=lambda n=ys.size: np.ones(n, dtype=np.int64),
                                  centers=lambda y=ys, x=xs: np.stack((y.ravel(), x.ravel()), axis=1))
        # The legacy nested protocol places devval inside the original train
        # partition. The signed WP02 role and released V5 roster are authoritative.
        require(actual.partition == tile["partition"] == row["partition"] and ids and np.isfinite(actual.spectral).all(),
                "Expected the original nested V5 block and its unchanged partition")
        views.append((row["id"], actual, lattice, ids, positions))
        seen.extend(ids)
    require(len(views) == 5 and len(seen) == len(set(seen)) == 45 and set(seen) == set(published),
            "Original five-block V5 roster changed")
    ordered = np.asarray([published[i] for i in seen])
    require(np.bincount(ordered, minlength=10)[1:].tolist() == [5]*9, "V5 class support changed")

    def instrument(model, actual, plan=None, branch=None, op=None):
        q_and_a, stage_support = [], []
        graph = model.backbone.learnable_graph
        require("build_graph" not in vars(graph), "Graph builder already instrumented")
        build = graph.build_graph
        ranges = ("narrow", "wide") if isinstance(model.backbone, control.MultiRangeBackbone) else ("single",)

        def collect(*args, **kwargs):
            result = build(*args, **kwargs)
            q_and_a.append((result.Q, result.A0))
            return result

        def stage1(original, values, adj, dataset):
            stage_support.append(tensor_hash(adj))
            return original(values, adj, dataset)

        def stage2(original, values, adj, dataset):
            stage_support.append(tensor_hash(adj))
            result = original(values, adj, dataset)
            if op is not None:
                require(branch in ranges and len(result) % len(ranges) == 0, "Unregistered graph range layout")
                offset = ranges.index(branch)*len(result)//len(ranges)
                result = control.branch_update(result, offset, plan, op, None)
            return result

        graph.build_graph = collect
        binding = model.partition_bands(scene.bands) if hasattr(model, "partition_bands") else nullcontext()
        try:
            with control.stages(model.backbone, stage1, stage2), binding, torch.no_grad():
                prediction = forward_partition(model, scene.dataset, actual, np.asarray(scene.bands.wavelengths, dtype=np.float32), 100)
        finally:
            del graph.build_graph
        require(len(q_and_a) == 1 and len(stage_support) == 2, "Expected original single Q and two HGNN stages")
        q, a = q_and_a[0]
        weights = prediction[3].get("multiscale_graph")
        graph_id = {"q": tensor_hash(q), "a0": tensor_hash(a), "stage_support": stage_support,
                    "ranges": list(ranges), "q_shape": list(q.shape),
                    "range_weight": None if weights is None else weights["weights"].tolist()}
        return prediction[:3], graph_id, q

    models, compatibility = {}, {}
    inception = perf_counter()
    protected = [*spec["protected"].values(), spec["source"], spec["sidecar"], *pins.values()]
    for arm, trial in ARMS.items():
        stage, item = read_stage(root, pins["registration"], arm, "full", None)
        require(item == complete["attempts"][arm]["full"], "Accepted continuous stage changed")
        packet = _packet(stage, spec["contracts"][arm], BUDGET)
        selected = _verify_history(packet, exposure)
        model = deepcopy(baseline) if trial is None else trial.initialize(baseline, SEED).eval()
        model.backbone.adapter.cache_attention = False
        protected.extend((item, stage["checkpoint"]))
        pair = {}
        for name, field, step in (("selected", "selected_state", selected), ("last", "state", BUDGET)):
            model.load_state_dict(packet[field], strict=True)
            before = state_hash(model)
            expected = audit["results"][arm]["endpoints"][name]
            require(expected["step"] == step and before == packet["history"][step-1]["state_hash"],
                    f"{arm}/{name} is not the exact recorded checkpoint")
            replay = []
            for i, (tile, actual, lattice, _, _) in enumerate(views):
                pred, graph_id, q = instrument(model, actual)
                require([tensor_hash(t) for t in pred] == expected["scores"][i],
                        f"{arm}/{name}/{tile} prediction differs from original audit")
                plan, dominant = control.region_options(q, lattice)
                replay.append((pred, graph_id, plan["core"], dominant))
            require(state_hash(model) == before, "Original replay mutated model state")
            pair[name] = (packet[field], step, replay, before)
        compatibility[arm] = {name: {"step": item[1], "state": item[3], "audit_probability_hashes_equal": True}
                              for name, item in pair.items()}
        models[arm] = (model, pair)
    stamp = dev._json_new(out/"compatibility.json", {"schema": SCHEMA, "status": "exact_historical_predictions",
                          "source": pins, "endpoints": compatibility, "original_protocol": spec["policy"],
                          "forward_calls": 80, "optimizer_updates": 0, "seconds": perf_counter()-inception})
    started = perf_counter()
    rows = []
    forwards = 0
    for arm, (model, pair) in models.items():
        for name, (state, step, replay, state_id) in pair.items():
            model.load_state_dict(state, strict=True)
            tiles = []
            for entry, (pred, graph, plan, dominant) in zip(views, replay, strict=True):
                tile, actual, _, coords, positions = entry
                base = np.stack([v[positions].numpy() for v in pred])
                obs = {}
                for operation in ("self", "zero", "replace"):
                    if operation == "replace" and plan.donor is None:
                        continue
                    # Operate on the sole range or the original first (narrow) range.
                    # Graph mixing and the other range keep their original values.
                    changed, updated, _ = instrument(model, actual, plan, graph["ranges"][0], operation)
                    forwards += 1
                    require(graph == updated and torch.equal(changed[2], pred[2]) and
                            (operation != "self" or all(torch.equal(a,b) for a,b in zip(pred, changed, strict=True))),
                            "Graph, CNN or identity intervention changed")
                    obs[operation] = np.stack([v[positions].numpy() for v in changed])
                resting, restored, _ = instrument(model, actual)
                forwards += 1
                require(restored == graph and all(torch.equal(a,b) for a,b in zip(resting, pred, strict=True)),
                        "Restored original outputs changed")
                tiles.append({"tile": tile, "target": plan.target, "donor": plan.donor,
                              "graph": graph, "operated_range": graph["ranges"][0],
                              "points_in_target": int(np.count_nonzero(dominant[positions] == plan.target)),
                              "baseline": base.tolist(), "observed": {key: value.tolist() for key, value in obs.items()},
                              "donor_status": "measured" if "replace" in obs else "no_legal_donor",
                              "labels": [published[i] for i in coords]})
            require(state_hash(model) == state_id, "Region intervention changed source parameters or buffers")
            scores = {}
            for i, head in enumerate(HEADS):
                for op in ("baseline", "self", "zero", "replace"):
                    group = [t for t in tiles if op == "baseline" or op in t["observed"]]
                    if not group:
                        continue
                    array = np.concatenate([np.asarray(t["baseline"] if op == "baseline" else t["observed"][op])[i]
                                            for t in group])
                    before = np.concatenate([np.asarray(t["baseline"])[i] for t in group])
                    truth = np.asarray([v for t in group for v in t["labels"]])
                    pred, prev = array.argmax(1)+1, before.argmax(1)+1
                    m, old = (classification_metrics(truth, p, range(1, 10)) for p in (pred, prev))
                    recall_deltas = [None if a["recall"] is None else a["recall"]-b["recall"]
                                     for a, b in zip(m["per_class"], old["per_class"], strict=True)]
                    scores.setdefault(op, {})[head] = {"metrics": m, "baseline_on_matched_points": old,
                        "nll": float(-np.log(np.maximum(array[np.arange(len(truth)), truth-1], 1e-8)).mean()),
                        "maximum_probability_change": float(np.abs(array-before).max()),
                        "argmax_changes": int(np.count_nonzero(pred != prev)),
                        "affected_by_true_class": np.bincount(truth[pred != prev], minlength=10)[1:].tolist(),
                        "recall_delta": recall_deltas, "protection_comparable": len(truth) == 45,
                        "passes_small_class_guard": (all(x >= spec["recall_floor_delta"] for x in recall_deltas)
                                                     if len(truth) == 45 else None)}
                    if op == "baseline":
                        expected = audit["results"][arm]["endpoints"][name]["metrics"][head]
                        require(m["confusion_matrix"] == expected["confusion_matrix"]
                                and abs(scores[op][head]["nll"]-expected["nll"]) < 1e-6,
                                "Independent V5 metrics differ")
            rows.append({"arm": arm, "endpoint": name, "step": step, "state_sha256": state_id,
                         "original_state_restored": True, "tiles": tiles, "heads": scores})
    admission.check_pins(protected)
    require_frozen_code(backend)
    final = dev._json_new(out/"analysis.json", {"schema": SCHEMA, "status": "measured",
             "scope": "legacy_real_xuzhou_v5_development_only", "backend": {k: backend[k] for k in ("commit", "source_hash")},
             "producer": {k: code[k] for k in ("commit", "source_hash")}, "source": pins, "compatibility": stamp,
             "original_protocol": spec["policy"], "records": rows, "original_forwards": 80,
             "intervention_and_restoration_forwards": forwards, "natural_boundary": "unknown",
             "optimizer_updates": 0, "bn_updates": 0, "calibration_updates": 0, "query_accessed": False,
             "raw_gt_loaded": False, "new_native_protocol": False, "independent_confirmation": False,
             "real_gain": None, "seconds": perf_counter()-started})
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
