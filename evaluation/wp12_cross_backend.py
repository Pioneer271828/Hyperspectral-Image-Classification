"""Frozen multi-backend forecasts and independent WP12 comparison on admitted roles."""

from itertools import product
import math
import re
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, pinned, probabilities, require
from evaluation.wp12_locked_support import transitions
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_qualification import classification_status
from evaluation.wp12_spatial_release import STRATA, classify, read_annotations, score_stratum
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent
from transfer import wp_cross_backend as model_io
from transfer.g3_training import state_hash
from transfer.native_paired_head_campaign import GATES
from transfer.spatial_coverage_protocol import retained_positions

DESIGN = "wp05_wp08_frozen_comparison_v1"
ENDPOINT = "wp05_wp08_frozen_endpoint_v1"
CAPTURE = "wp05_wp08_frozen_capture_v1"
REVIEW = "wp12_cross_backend_review_v1"
SELECTION = "first_min_released_point_fused_nll"
PROTOCOL = {"budget", "data", "exposure", "initialization", "class_axis", "band_axis", "arithmetic", "selection"}
FORMAL_FLOOR = GATES["minimum_real_confirmation_per_class"]


def fingerprint(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_endpoint(body, protocol, seed):
    require(set(body) == {"schema", "protocol", "seed", "phase", "step", "calibration", "checkpoint"}
            and body["schema"] == ENDPOINT and body["protocol"] == protocol and type(body["seed"]) is int
            and body["seed"] == seed, "Checkpoint training context differs from its registered cohort")
    phase, step, budget = body["phase"], body["step"], protocol["budget"]
    require(type(step) is int and phase in ("selected", "terminal") and 0 <= step <= budget,
            "Invalid checkpoint endpoint or training cursor")
    trace = body["calibration"]
    require(isinstance(trace, list) and bool(trace), "Checkpoint requires its frozen calibration trace")
    require(all(set(row) == {"step", "nll"} and type(row["step"]) is int and 0 <= row["step"] <= budget
                and type(row["nll"]) in (float, int) and math.isfinite(row["nll"]) and row["nll"] >= 0
                for row in trace), "Invalid checkpoint calibration history")
    steps = [row["step"] for row in trace]
    require(steps == sorted(set(steps)) and steps[-1] == budget
            and step == (min(trace, key=lambda row: row["nll"])["step"] if phase == "selected" else budget),
            "Checkpoint cursor does not follow the registered first-minimum or terminal rule")
    return steps


def open_case(root, request, *, test_history=None):
    value = json_pin(root, request)
    require(set(value) == {"schema", "intake", "authority", "groups"} and value["schema"] == DESIGN,
            "Exact frozen comparison manifest required")
    bank, intake, roles = independent.open_independent(
        root, value["intake"], value["authority"], test_history=test_history)
    require(any(intake["uses"][use]["eligible"] for use in ("independent_accuracy", "cost")),
            "Bank is ineligible for both classification and cost")
    require(isinstance(value["groups"], dict) and bool(value["groups"]), "Explicit comparison cohorts required")
    groups = {}
    for name, group in value["groups"].items():
        require(re.fullmatch("[a-z0-9_-]+", name) and set(group) == {"seed", "protocol", "arms", "pairs"}
                and type(group["seed"]) is int and 0 <= group["seed"] < 2**32, "Invalid comparison roster")
        protocol, seed = group["protocol"], group["seed"]
        require(set(protocol) == PROTOCOL and type(protocol["budget"]) is int and protocol["budget"] >= 0
                and protocol["selection"] == SELECTION
                and protocol["class_axis"] == list(range(1, bank.release.classes+1))
                and all(type(label) is int for label in protocol["class_axis"])
                and protocol["band_axis"] == bank.release.bands.fingerprint
                and all(fingerprint(protocol[k]) for k in ("data", "exposure", "initialization", "arithmetic")),
                "Budget, original band identity, taxonomy or selection rule differs")
        models = {arm: json_pin(root, pin) for arm, pin in group["arms"].items()}
        require(all(re.fullmatch("[a-z0-9_-]+", arm) for arm in models) and len(models) >= 2,
                "Comparison needs separately pinned model arms")
        schedules = [validate_endpoint(body, protocol, seed) for body in models.values()]
        require(all(schedule == schedules[0] for schedule in schedules)
                and len({model["phase"] for model in models.values()}) == 1,
                "Matched arms use different calibration event schedules or endpoint phases")
        pairs = group["pairs"]
        require(isinstance(pairs, list) and bool(pairs) and all(isinstance(pair, list) and len(pair) == 2
                    and all(isinstance(arm, str) and arm in models for arm in pair) and pair[0] != pair[1]
                    for pair in pairs) and len(set(map(tuple, pairs))) == len(pairs),
                "Invalid explicit model comparisons")
        groups[name] = {**group, "models": models}
    return bank, intake, roles, groups


def emitted_head_scores(output, targets, classes):
    cells = list(targets.cells)
    n = len(cells)
    require(n > 0, "Scored evaluation block has no approved points")
    return {head: probabilities(value.detach().cpu().numpy()[cells].tolist(), n, classes).tolist()
            for head, value in zip(HEADS, output, strict=True)}


def run_case(root, request, out, code, execution, *, test_history=None):
    bank, intake, roles, groups = open_case(root, request, test_history=test_history)
    producer = code["commit"]
    accuracy, cost = (intake["uses"][use]["eligible"] for use in ("independent_accuracy", "cost"))
    annotation_pin = intake.get("spatial_annotations")
    geometry = read_annotations(root, bank, roles, annotation_pin) if annotation_pin else None
    views = intake.get("observation_views", ["full"])
    started, runs = perf_counter(), {}
    for cohort, group in groups.items():
        arms = {}
        for name, body in group["models"].items():
            start = perf_counter()
            spec = body["checkpoint"]
            model = model_io.load_export(root, spec, device=execution.device)
            require(bank.release.scene in model.heads
                    and model.heads[bank.release.scene].out_features == bank.release.classes,
                    "Checkpoint head does not match the approved Bank taxonomy")
            load_seconds = perf_counter()-start
            rows = []
            for band, tile in product(views, roles["evaluation"]):
                row = dict(role="evaluation", tile=tile, view=band, accuracy=None, cost=None)
                if accuracy and bank.role_row("devval", tile)["samples"]:
                    actual, _, targets = independent.materialize_reviewed(
                        bank, intake, roles, "evaluation", tile, producer, band, use="independent_accuracy")
                    output, graph, timing = model_io.run_block(model, actual, producer, "devval", execution)
                    row["accuracy"] = dict(order=digest([key for key, _ in targets.points]),
                        truth=[label for _, label in targets.points], observation=actual.receipt["observation"],
                        bands=actual.receipt["source"]["band_ids"], graph=graph,
                        heads=emitted_head_scores(output, targets, bank.release.classes),
                        spatial=None if geometry is None else classify(bank, roles, geometry, "evaluation", tile, targets.points))
                if cost:
                    timings = []
                    for rep in range(3):
                        execution.sync()
                        if execution.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(execution.device)
                        start = perf_counter()
                        actual, labels, targets = independent.materialize_reviewed(
                            bank, intake, roles, "evaluation", tile, producer, band, use="cost")
                        materialized = perf_counter()
                        require(labels is targets is None, "Cost materialized classification data")
                        output, graph, detail = model_io.run_block(model, actual, producer, "devval", execution)
                        if rep:
                            timings.append(dict(total=perf_counter()-start, materialize=materialized-start,
                                                cuda_peak_bytes=torch.cuda.max_memory_allocated(execution.device)
                                                if execution.device.type == "cuda" else None, **detail))
                    row["cost"] = dict(label_blind=True, samples=timings, graph=graph,
                                       observation=actual.receipt["observation"], bands=actual.receipt["source"]["band_ids"],
                                       outputs=[list(x.shape) for x in output])
                rows.append(row)
            require(state_hash(model) == spec["state"], "Frozen inference modified the recovered checkpoint")
            arms[name] = dict(endpoint=group["arms"][name], state=spec["state"], phase=body["phase"],
                              step=body["step"], semantics=model_io.semantic_identity(model, spec["backend"]),
                              load_seconds=load_seconds, parameters=sum(p.numel() for p in model.parameters()), rows=rows)
        runs[cohort] = dict(arms=arms, pairs=group["pairs"], budget=group["protocol"]["budget"], seed=group["seed"])
    require_frozen_code(code)
    return _write(out, dict(schema=CAPTURE, request=request, producer=producer, numeric=execution.numeric, cohorts=runs,
                            scope=intake["source_mode"], bank=bank.pin, uses=intake["uses"], roles=digest(roles),
                            views=views, optimizer_updates=0, query_accessed=False, raw_gt_loaded=False,
                            wall_seconds=perf_counter()-started))


def raw_pair(truth, before, after, classes, *, floor, delta=-.02):
    require(truth and all(type(y) is int and 1 <= y <= classes for y in truth)
            and len(before) == len(after) == len(truth)
            and all(type(y) is int and 1 <= y <= classes for y in (*before, *after)),
            "The matched roster includes an invalid, missing or zero class")
    first, last = (classification_metrics(truth, pred, range(1, classes+1)) for pred in (before, after))
    changes = transitions(truth, before, after, classes)
    for row in changes.values():
        row["recall_delta"] = (row["gains"]-row["losses"])/row["n"] if row["n"] else None
    enough = all(row["n"] >= floor for row in changes.values())
    return dict(original_oa=first["oa"], candidate_oa=last["oa"], delta_oa=last["oa"]-first["oa"],
                original_aa=first["aa"], candidate_aa=last["aa"], delta_aa=last["aa"]-first["aa"],
                before=first["confusion_matrix"], after=last["confusion_matrix"], transitions=changes,
                min_class_n=min(row["n"] for row in changes.values()), protection_threshold=delta,
                protected=all(row["recall_delta"] >= delta for row in changes.values()) if enough else None)


def checked_blocks(bank, roles, intake, geometry, observations, arm, body, pin, semantics):
    classes = bank.release.classes
    expected = list(product(intake.get("observation_views", ["full"]), roles["evaluation"]))
    seen = [(row["view"], row["tile"]) for row in arm["rows"]]
    require(seen == expected and arm["endpoint"] == pin and arm["state"] == body["checkpoint"]["state"]
            and arm["phase"] == body["phase"] and arm["step"] == body["step"]
            and digest(arm["semantics"]) == digest(semantics),
            "Raw evidence changes the registered endpoint, model arm, or ordered view/role")
    checked = {}
    for row in arm["rows"]:
        band, tile = row["view"], row["tile"]
        class_allowed = intake["uses"]["independent_accuracy"]["eligible"]
        points = bank.role_row("devval", tile)["samples"] if class_allowed else []
        require(set(row) == {"tile", "view", "role", "cost", "accuracy"} and row["role"] == "evaluation"
                and (row["cost"] is not None) == intake["uses"]["cost"]["eligible"]
                and (row["accuracy"] is not None) == bool(points),
                "A capture crosses a role or admitted use")
        bands = [bank.release.bands.band_ids[k] for k in retained_positions(bank.release.bands, band)]
        cost = row["cost"]
        if cost is not None:
            require(cost["label_blind"] is True and len(cost["samples"]) == 2 and cost["bands"] == bands
                    and cost["observation"] == observations[(band,tile)]
                    and cost["outputs"] == [[cost["graph"]["q_shape"][0], classes]]*len(HEADS)
                    and all(math.isfinite(item["total"]) and 0 <= item["materialize"] <= item["total"]
                            and 0 <= item["graph_seconds"] <= item["elapsed"] <= item["total"]
                            for item in cost["samples"]), "Invalid label-blind cost or band identity")
        record = row["accuracy"]
        if record is None:
            checked[(band, tile)] = row
            continue
        points = [tuple(pair) for pair in points]
        truth = [y for _, y in points]
        spatial = None if geometry is None else classify(bank, roles, geometry, "evaluation", tile, points)
        require(record["truth"] == truth and all(type(y) is int and 1 <= y <= classes for y in truth)
                and record["order"] == digest([i for i, _ in points]) and record["bands"] == bands
                and record["observation"] == observations[(band,tile)]
                and record["spatial"] == spatial and set(record["heads"]) == set(HEADS)
                and (cost is None or (record["observation"] == cost["observation"] and record["graph"] == cost["graph"])),
                "The points, observation, band axis or optional trusted geometry differ")
        for head in HEADS:
            probabilities(record["heads"][head], len(truth), classes)
        checked[(band, tile)] = row
    return checked


def matched_view(first, second, band, tiles, scene, classes, *, scope):
    truth, pred_before, pred_after = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    strata = {name: [] for name in STRATA}
    sampled_geometry = False
    costs = []
    for tile in tiles:
        left, right = first[(band,tile)], second[(band,tile)]
        ca, cb = left["cost"], right["cost"]
        if ca is not None:
            require(cb is not None and ca["observation"] == cb["observation"] and ca["bands"] == cb["bands"],
                    "Cost pairs do not observe the same physical pixels or bands")
            costs.append(dict(tile=tile, baseline=ca, candidate=cb))
        a,b = left["accuracy"], right["accuracy"]
        if a is None:
            require(b is None, "Accuracy cannot be borrowed from another model or role")
            continue
        require(b is not None and all(a[k] == b[k] for k in ("truth", "order", "observation", "bands", "spatial")),
                "Accuracy pairs must use identical ordered points, geometry and bands")
        if a["spatial"] is not None:
            sampled_geometry = True
            for name in STRATA:
                strata[name].extend(len(truth)+i for i, use in enumerate(a["spatial"]["strata"][name]) if use)
        truth.extend(a["truth"])
        for h in HEADS:
            pred_before[h].extend((probabilities(a["heads"][h], len(a["truth"]), classes).argmax(1)+1).tolist())
            pred_after[h].extend((probabilities(b["heads"][h], len(b["truth"]), classes).argmax(1)+1).tolist())
    scores, qualifications, spatial = None, None, None
    if truth:
        scores = {head:raw_pair(truth, pred_before[head], pred_after[head], classes, floor=FORMAL_FLOOR)
                  for head in HEADS}
        if scene in GATES["targets"]:
            for head in HEADS:
                score = pair_scores(truth, pred_before[head], pred_after[head], classes, scene,
                                    head_kind=head, guard_population=FORMAL_FLOOR)
                scores[head]["prior_gate"] = score
            qualifications = {head:classification_status(scores[head]["prior_gate"], accuracy_allowed=True,
                              cost_allowed=bool(costs), real=scope=="external_real") for head in HEADS}
            if sampled_geometry:
                spatial = {name: score_stratum(truth, pred_before, pred_after, indices, classes, scene,
                           threshold=FORMAL_FLOOR, real=scope=="external_real") for name, indices in strata.items()}
        elif sampled_geometry:
            spatial = {name: dict(n=len(indices), scores=None if not indices else {
                h:raw_pair([truth[i] for i in indices], [pred_before[h][i] for i in indices],
                           [pred_after[h][i] for i in indices], classes, floor=FORMAL_FLOOR) for h in HEADS})
                for name, indices in strata.items()}
    return dict(accuracy=scores, qualification=qualifications, spatial=spatial, cost=costs)


def compare_devices(reference, capture):
    """Check CPU/CUDA captures of the same frozen exports and ordered Bank observations."""
    reference_numeric, candidate_numeric = reference["numeric"], capture["numeric"]
    require(reference_numeric["device"] == "cpu" and candidate_numeric["device"] == "cuda:0",
            "Device review requires a CPU reference and an admitted CUDA capture")
    require(all(reference[k] == capture[k] for k in
                ("request","producer","bank","roles","uses","views","scope"))
            and reference["cohorts"].keys() == capture["cohorts"].keys(), "Device captures use different releases")
    count = flips = graph_changes = 0
    delta = 0.
    per_backend = {}
    for name, group in capture["cohorts"].items():
        old = reference["cohorts"][name]
        require(set(old["arms"]) == set(group["arms"]), "Device capture changed the registered roster")
        for arm, row in group["arms"].items():
            earlier = old["arms"][arm]
            require(all(earlier[k] == row[k] for k in ("endpoint","state","phase","step","semantics")),
                    "Same-weight device comparison received another model")
            backend = row["semantics"]["backend"]
            local = per_backend.setdefault(backend,dict(point_head=0,prediction_changes=0,max_abs_probability=0.))
            for a,b in zip(earlier["rows"],row["rows"],strict=True):
                require(all(a[k]==b[k] for k in ("role","view","tile")), "Device capture changed roles or row order")
                if a["accuracy"] is None:
                    require(b["accuracy"] is None, "A device acquired a previously absent classification")
                    continue
                y,z = a["accuracy"],b["accuracy"]
                require(z is not None and all(y[k] == z[k] for k in ("truth","order","observation","bands")),
                        "Device capture changed true classes, points or actual bands")
                graph_changes += any(y["graph"][k] != z["graph"][k] for k in ("q_indices","a0_indices"))
                for h in HEADS:
                    p,q = np.asarray(y["heads"][h]),np.asarray(z["heads"][h])
                    require(p.shape == q.shape, "Device head shape differs")
                    maximum = float(np.max(np.abs(p-q)))
                    changed = int((p.argmax(1)!=q.argmax(1)).sum())
                    count += len(p)
                    flips += changed
                    delta = max(delta,maximum)
                    local["point_head"] += len(p)
                    local["prediction_changes"] += changed
                    local["max_abs_probability"] = max(local["max_abs_probability"],maximum)
    require(count > 0, "Same-weight device parity needs approved classification on both captures")
    return dict(point_head=count,prediction_changes=flips,max_abs_probability=delta,
                different_graph_index_rows=graph_changes,backend=per_backend,
                cpu=reference_numeric["protocol"],gpu=candidate_numeric["physical_uuid"])


def review_case(root, capture, out, code, *, test_history=None, reference=None):
    raw = json_pin(root, capture)
    bank, intake, roles, groups = open_case(root, raw["request"], test_history=test_history)
    require(raw["schema"] == CAPTURE and raw["producer"] == code["commit"] and raw["scope"] == intake["source_mode"]
            and raw["bank"] == bank.pin and raw["uses"] == intake["uses"] and raw["roles"] == digest(roles)
            and raw["views"] == intake.get("observation_views", ["full"]) and set(raw["cohorts"]) == set(groups)
            and raw["query_accessed"] is raw["raw_gt_loaded"] is False and raw["optimizer_updates"] == 0,
            "Invalid capture scope, provenance or authority")
    pin = intake.get("spatial_annotations")
    geometry = read_annotations(root, bank, roles, pin) if pin else None
    observations = {}
    use = "cost" if intake["uses"]["cost"]["eligible"] else "independent_accuracy"
    for band, tile in product(raw["views"], roles["evaluation"]):
        actual, _, _ = independent.materialize_reviewed(bank, intake, roles, "evaluation", tile,
                                                         code["commit"], band, use=use)
        observations[band,tile] = actual.receipt["observation"]
    summary = {}
    for key, group in groups.items():
        ready = raw["cohorts"][key]
        require(ready["budget"] == group["protocol"]["budget"] and ready["seed"] == group["seed"]
                and ready["pairs"] == group["pairs"] and set(ready["arms"]) == set(group["arms"]),
                "Capture changes the paired training budget, model seed, or roster")
        checked = {}
        for name, arm in ready["arms"].items():
            body = group["models"][name]
            export = body["checkpoint"]
            model = model_io.load_export(root, export)
            require(arm["parameters"] == sum(p.numel() for p in model.parameters())
                    and bank.release.scene in model.heads
                    and model.heads[bank.release.scene].out_features == bank.release.classes,
                    "Consumption checkpoint has another model or class inventory")
            semantics = model_io.semantic_identity(model, export["backend"])
            checked[name] = checked_blocks(bank, roles, intake, geometry, observations,
                                           arm, body, group["arms"][name], semantics)
        scored = {}
        for base, candidate in group["pairs"]:
            scored[f"{base}/{candidate}"] = {band: matched_view(checked[base], checked[candidate], band,
                roles["evaluation"], bank.release.scene, bank.release.classes, scope=raw["scope"]) for band in raw["views"]}
        summary[key] = dict(seed=ready["seed"], budget=ready["budget"], pairs=scored,
                            arms={name: {k:v for k,v in arm.items() if k!="rows"} for name, arm in ready["arms"].items()})
    require_frozen_code(code)
    parity = None if reference is None else compare_devices(json_pin(root,reference),raw)
    return _write(out, dict(schema=REVIEW, capture=capture, baseline_capture=reference, device_parity=parity,
                            producer=code["commit"], scope=raw["scope"],
                            scene=bank.release.scene, effect_scope="frozen_members_of_explicit_matched_cohort",
                            uses=intake["uses"], roles=intake["role_reports"], cohorts=summary,
                            numeric=raw["numeric"], query_accessed=False, raw_gt_loaded=False, optimizer_updates=0))
