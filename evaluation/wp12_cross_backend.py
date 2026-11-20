"""Frozen multi-backend forecasts and independent WP12 comparison on admitted roles."""

from collections import defaultdict
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
from evaluation.wp12_scientific_claim import COLUMNS
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent
from transfer import wp_cross_backend as model_io
from transfer.g3_training import state_hash
from transfer.native_paired_head_campaign import GATES
from transfer import wp_region_graph as regional, wp_causal_graph_design as causal_design
from transfer import wp_causal_graph as causal_runtime, wp_coverage_release as coverage

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
        factor = protocol.get("factor",1)
        fields = set(protocol)
        columns = "delivered_columns" in intake
        require(PROTOCOL <= fields <= PROTOCOL|{"factor", "observation_protocol", "observation_exposure"}
                and protocol.get("observation_exposure")==intake.get("observation_exposure")
                and (protocol.get("observation_protocol") == COLUMNS if columns
                     else "observation_protocol" not in protocol)
                and type(factor) is int and factor in intake.get("observation_factors",[1])
                and type(protocol["budget"]) is int and protocol["budget"] >= 0
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
        regional_models = [body["checkpoint"]["graph_program"] for body in models.values()
                           if body["checkpoint"]["backend"]==regional.BACKEND]
        coverage_models = [body["checkpoint"]["coverage_graph"] for body in models.values()
                           if body["checkpoint"]["backend"]==coverage.BACKEND]
        assigned_exposure = intake.get("observation_exposure")
        require(bool(coverage_models)==(assigned_exposure is not None),
                "Frozen coverage arms and owner-approved exposure differ")
        if coverage_models:
            scale = 50 if bank.release.scene=="paviaU" else 100
            require(len(coverage_models)==len(models)
                    and len({row["condition"] for row in coverage_models})==len(models)
                    and all(row["anchor"]==coverage_models[0]["anchor"] and row["anchor"]["state"]==protocol["initialization"]
                            and row["scale"]==scale for row in coverage_models),
                    "Coverage requires a common initial retained-observation graph source")
        controlled = None
        if regional_models:
            witness_pins = [r.get("causal_design") for r in regional_models]
            require(all(pin==witness_pins[0] for pin in witness_pins),"Exact witness must be common to all arms")
            if witness_pins[0] is not None:
                require(test_history is not None and intake["source_mode"]=="synthetic",
                        "Controlled paths require independent synthetic custody")
                controlled = causal_design.verify(json_pin(root,witness_pins[0]),bank=bank.pin["sha256"],
                    scene=bank.release.scene,roles=roles,bands=bank.release.bands)
            require(test_history is not None and factor==1 and len(regional_models)==len(models)
                    and len({r["policy"] for r in regional_models})==len(models)
                    and all(row["budget"]==protocol["budget"] and row["seed"]==seed+19000
                            and row["weights"]==regional_models[0]["weights"]
                            and row["state"]==regional_models[0]["state"] for row in regional_models),
                    "Regional endpoints must use one pinned initial source, calibration clock and search seed")
        require(all(schedule == schedules[0] for schedule in schedules)
                and len({model["phase"] for model in models.values()}) == 1,
                "Matched arms use different calibration event schedules or endpoint phases")
        pairs = group["pairs"]
        require(isinstance(pairs, list) and bool(pairs) and all(isinstance(pair, list) and len(pair) == 2
                    and all(isinstance(arm, str) and arm in models for arm in pair) and pair[0] != pair[1]
                    for pair in pairs) and len(set(map(tuple, pairs))) == len(pairs),
                "Invalid explicit model comparisons")
        groups[name] = {**group, "models": models, "factor": factor,"causal_design":controlled}
    return bank, intake, roles, groups


def emitted_head_scores(output, targets, classes):
    cells = list(targets.cells)
    n = len(cells)
    require(n > 0, "Scored evaluation block has no approved points")
    return {head: probabilities(value.detach().cpu().numpy()[cells].tolist(), n, classes).tolist()
            for head, value in zip(HEADS, output, strict=True)}


@torch.no_grad()
def interventions(model, actual, producer, execution, targets):
    rows={}
    for variant in causal_runtime.INTERVENTIONS:
        output,graph,timing=regional.forward_region(model,actual,producer,"devval",execution,
                                                    model.graph_program,intervention=variant)
        p=graph["regional_support"]
        rows[variant]=dict(heads=emitted_head_scores(output[:3],targets,actual.classes),
            q=graph["q_indices"],a0=graph["a0_indices"],a2=p["stage2_indices"],phase=p["causal"]["phase"],
            seconds=timing["forward_seconds"]+timing["regional"]["selection_seconds"])
    return rows


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
        factor = group["factor"]
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
                        bank, intake, roles, "evaluation", tile, producer, band, use="independent_accuracy",factor=factor)
                    output, graph, timing = model_io.run_block(model, actual, producer, "devval", execution)
                    row["accuracy"] = dict(order=digest([key for key, _ in targets.points]),
                        truth=[label for _, label in targets.points], observation=actual.receipt["observation"],
                        bands=actual.receipt["source"]["band_ids"], graph=graph,
                        heads=emitted_head_scores(output, targets, bank.release.classes),
                        spatial=None if geometry is None else classify(bank, roles, geometry, "evaluation", tile, targets.points))
                    if group["causal_design"] is not None and band in ("full","central_gap"):
                        row["accuracy"]["interventions"] = interventions(model,actual,producer,execution,targets)
                    if "regional_support" in graph:
                        origin = bank.role_row("devval",tile)["origin"]
                        row["accuracy"]["signal"] = regional.signal_visibility(origin.get("regional_signal"),
                            graph["regional_support"],row["accuracy"]["bands"],
                            targets.cells.tolist(),row["accuracy"]["truth"])
                if cost:
                    timings, first_run = [],None
                    if hasattr(model,"coverage_program"):
                        model.coverage_program.views.clear()
                    for rep in range(3):
                        execution.sync()
                        if execution.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(execution.device)
                        start = perf_counter()
                        actual, labels, targets = independent.materialize_reviewed(
                            bank, intake, roles, "evaluation", tile, producer, band, use="cost",factor=factor)
                        materialized = perf_counter()
                        require(labels is targets is None and ("observation_exposure" not in intake
                                or "samples" not in actual.receipt["source"]),
                                "Cost materialized classification data")
                        output, graph, detail = model_io.run_block(model, actual, producer, "devval", execution)
                        sample = dict(total=perf_counter()-start, materialize=materialized-start,
                                      cuda_peak_bytes=torch.cuda.max_memory_allocated(execution.device)
                                      if execution.device.type == "cuda" else None, **detail)
                        if rep:
                            timings.append(sample)
                        elif "regional_support" in graph or "coverage" in graph:
                            first_run = sample
                    row["cost"] = dict(label_blind=True, samples=timings, graph=graph,
                                       observation=actual.receipt["observation"], bands=actual.receipt["source"]["band_ids"],
                                       outputs=[list(x.shape) for x in output])
                    if first_run is not None:
                        row["cost"]["first_measurement"] = first_run
                rows.append(row)
            require(state_hash(model) == spec["state"], "Frozen inference modified the recovered checkpoint")
            arms[name] = dict(endpoint=group["arms"][name], state=spec["state"], phase=body["phase"],
                              step=body["step"], semantics=model_io.semantic_identity(model, spec["backend"]),
                              load_seconds=load_seconds, parameters=sum(p.numel() for p in model.parameters()), rows=rows)
        runs[cohort] = dict(arms=arms, pairs=group["pairs"], budget=group["protocol"]["budget"],
                            seed=group["seed"],factor=factor)
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


def checked_blocks(bank, roles, intake, geometry, observations, arm, body, pin, semantics, *, controlled=None):
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
        bands = independent.observation_bands(bank.release.bands, band, intake)
        cost = row["cost"]
        program = semantics.get("regional_support")
        for record in (row["accuracy"],cost):
            if "coverage" in semantics and record is not None:
                check_coverage_trace(record,semantics["coverage"],independent.observation_view(band,intake)[1])
            if program is not None and record is not None:
                graph = record["graph"]
                proof = regional.verify_trace(graph["regional_support"],record["observation"],
                                              require_policy=program["policy"])
                if controlled is not None:
                    causal_runtime.verify_trace(proof,controlled,tile,"devval",program["budget"])
                require(proof["first_indices"]==graph["a0_indices"] and proof["assignment_indices"]==graph["q_indices"]
                        and proof["original_counts"]["nnz"]==graph["a0_nnz"],
                        "Fixed regional support changed the learned first-stage graph")
        if cost is not None:
            if "coverage" in semantics:
                sample = cost["first_measurement"]
                require(0<=sample["coverage_anchor_seconds"]<=sample["elapsed"]<=sample["total"],
                        "Coverage cold cost lacks its retained-graph construction")
            require(cost["label_blind"] is True and len(cost["samples"]) == 2 and cost["bands"] == bands
                    and cost["observation"] == observations["cost"][(band,tile)]
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
                and record["observation"] == observations["independent_accuracy"][(band,tile)]
                and record["spatial"] == spatial and set(record["heads"]) == set(HEADS)
                and (cost is None or "delivered_columns" in intake
                     or "coverage" in semantics
                     or (record["observation"] == cost["observation"] and record["graph"] == cost["graph"])),
                "The points, observation, band axis or optional trusted geometry differ")
        if "coverage" in semantics and cost is not None:
            require(all(record["graph"][key]==cost["graph"][key] for key in ("q","a0")),
                    "Coverage cost and classification observed different retained graphs")
        if program is not None:
            parent = bank.role_row("devval",tile)
            top,_,left,right = parent["box"]
            width = bank.release.shape[1]
            cells = [(pixel//width-top)*(right-left)+pixel%width-left for pixel,_ in points]
            signal = regional.signal_visibility(parent["origin"].get("regional_signal"),
                record["graph"]["regional_support"],bands,cells,truth)
            require(record["signal"]==signal,"Synthetic reachability differs from the emitted Q/A support")
        if controlled is not None:
            check_interventions(record,controlled,tile,band,classes)
        for head in HEADS:
            probabilities(record["heads"][head], len(truth), classes)
        checked[(band, tile)] = row
    return checked


def check_coverage_trace(record, expected, metadata):
    trace = record["graph"]["coverage"]
    axis = "index" if metadata=="ordinal" else "missing" if metadata in ("missing","rejected_axis") else "declared"
    require(trace["observation"]==record["observation"] and trace["policy"]==expected["policy"]
            and trace["anchor"]==expected["anchor"] and trace["mode"]==expected["condition"]
            and trace["axis"]==axis and trace["result"]==("rejected_axis" if metadata=="rejected_axis"
                                                       else "ordinal" if metadata=="ordinal" else None),
            "Coverage graph or metadata scope does not match the approved retained observation")
    fallback = {"index":"index_only_metadata","missing":"missing_metadata"}.get(axis)
    if expected["condition"]=="none" or fallback is not None:
        require(trace["active_anchors"]==0 and trace["fallback"]==(None if expected["condition"]=="none" else fallback),
                "An untrusted metadata axis must not activate physical coverage")


def check_interventions(record, controlled, tile, band, classes):
    expected=band in ("full","central_gap")
    require(("interventions" in record)==expected,"Activation separation changed the registered intervention roster")
    if not expected:
        return
    from research_runtime import tensor_hash

    graph=record["graph"]
    regional_proof=graph["regional_support"]
    body=record["interventions"]
    pairs=causal_design.support(controlled["layout"],controlled["tiles"][str(tile)]["regime"])
    require(set(body)==set(causal_runtime.INTERVENTIONS),"Missing fixed-weight control")
    for label,row in body.items():
        phase="original" if label=="cut" else regional_proof["causal"]["phase"]
        indices=torch.tensor(pairs[phase],dtype=torch.long).T.contiguous()
        require(row["q"]==graph["q_indices"] and row["a0"]==graph["a0_indices"]
                and row["a2"]==tensor_hash(indices) and row["phase"]==phase and row["seconds"]>=0,
                "Fixed-weight control changed the frozen assignment or wrong propagation stage")
        require(set(row["heads"])==set(HEADS),"Fixed-weight control lost a classification head")
        for head in HEADS:
            probabilities(row["heads"][head],len(record["truth"]),classes)


def matched_view(first, second, band, tiles, scene, classes, *, scope):
    truth, pred_before, pred_after = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    strata = {name: [] for name in STRATA}
    sampled_geometry = False
    costs, blocks = [], {}
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
        left_coverage = a["graph"].get("coverage")
        right_coverage = b["graph"].get("coverage")
        if left_coverage is not None or right_coverage is not None:
            require(left_coverage is not None and right_coverage is not None
                    and left_coverage["anchor"]==right_coverage["anchor"]
                    and all(a["graph"][key]==b["graph"][key] for key in ("q","a0")),
                    "E1 pair changed graph weights or support on the same retained observation")
        left_support = a["graph"].get("regional_support")
        right_support = b["graph"].get("regional_support")
        structural = None
        if left_support is not None or right_support is not None:
            require(left_support is not None and right_support is not None
                    and all(left_support[k]==right_support[k] for k in ("original","points","view")),
                    "Regional comparison lacks the same first-stage support, observation or coordinates")
            structural = {role:{k:s[k] for k in ("policy","cursor","counts","positive_nnz","newly_added",
                                                  "same_sender_degree","same_distance_counts","stage2_indices")}
                          for role,s in (("baseline",left_support),("candidate",right_support))}
            structural["same_effective_edges"] = left_support["stage2_indices"]==right_support["stage2_indices"]
            structural["signal"] = dict(baseline=a["signal"],candidate=b["signal"])
        block = {}
        for h in HEADS:
            before, after = ((probabilities(side["heads"][h], len(a["truth"]), classes).argmax(1)+1).tolist()
                             for side in (a, b))
            pred_before[h].extend(before)
            pred_after[h].extend(after)
            block[h] = raw_pair(a["truth"], before, after, classes, floor=FORMAL_FLOOR)
        blocks[str(tile)] = dict(points=len(a["truth"]), heads=block)
        if structural is not None:
            blocks[str(tile)]["regional_support"] = structural
        if left_coverage is not None:
            blocks[str(tile)]["coverage"] = dict(same_graph=True,baseline=left_coverage,candidate=right_coverage)
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
    result = dict(accuracy=scores, qualification=qualifications, spatial=spatial, cost=costs, blocks=blocks)
    sample = first[(band,tiles[0])]["accuracy"]
    if sample is not None and "causal" in sample["graph"].get("regional_support",{}):
        from evaluation.wp12_causal_mechanism import summarize

        result["causal"] = summarize(first,second,band,tiles,scene)
    return result


def band_loss_scores(checked, parent_by_tile, classes):
    """Compare two approved equal-count deletions on one frozen endpoint."""
    parents = defaultdict(list)
    for tile, parent in parent_by_tile.items():
        dispersed = checked["matched_dispersed", tile]["accuracy"]
        central = checked["central_gap", tile]["accuracy"]
        if dispersed is None:
            require(central is None, "Band loss cannot change which points are scored")
            continue
        require(central is not None and all(dispersed[k] == central[k] for k in ("truth", "order", "spatial")),
                "Band loss requires the same ordered points and geometry across views")
        parents[parent].append((dispersed, central))

    whole_truth, whole_before, whole_after = [], {h: [] for h in HEADS}, {h: [] for h in HEADS}
    by_parent = {}
    for parent, blocks in sorted(parents.items()):
        truth = [label for before, _ in blocks for label in before["truth"]]
        whole_truth.extend(truth)
        heads = {}
        for head in HEADS:
            before, after = [], []
            for dispersed, central in blocks:
                before.extend((np.asarray(dispersed["heads"][head]).argmax(1)+1).tolist())
                after.extend((np.asarray(central["heads"][head]).argmax(1)+1).tolist())
            whole_before[head].extend(before)
            whole_after[head].extend(after)
            heads[head] = transitions(truth, before, after, classes)
        by_parent[str(parent)] = dict(points=len(truth), transitions=heads)
    scores = {h: raw_pair(whole_truth, whole_before[h], whole_after[h], classes, floor=FORMAL_FLOOR)
              for h in HEADS}
    return dict(points=len(whole_truth), scores=scores, parents=by_parent)


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
        require(set(old["arms"]) == set(group["arms"]) and old.get("factor",1)==group.get("factor",1),
                "Device capture changed the registered roster or lattice factor")
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
                graph_diff = any(y["graph"][k] != z["graph"][k] for k in ("q_indices","a0_indices"))
                if "regional_support" in y["graph"]:
                    graph_diff |= (y["graph"]["regional_support"]["stage2_indices"]
                                   != z["graph"]["regional_support"]["stage2_indices"])
                graph_changes += graph_diff
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
    uses = [use for use in ("independent_accuracy", "cost") if intake["uses"][use]["eligible"]]
    band_loss = None
    if "independent_accuracy" in uses and {"central_gap", "matched_dispersed"} <= set(raw["views"]):
        retained = {view: independent.observation_bands(bank.release.bands, view, intake)
                    for view in ("matched_dispersed", "central_gap")}
        original_count = sum(bank.release.bands.valid)
        require(len(retained["central_gap"]) == len(retained["matched_dispersed"]) < original_count,
                "Equal-band loss requires two actual deletions of the same size")
        if retained["central_gap"] != retained["matched_dispersed"]:
            band_loss = dict(baseline_view="matched_dispersed", candidate_view="central_gap",
                             retained_columns=retained, removed_count=original_count-len(retained["central_gap"]),
                             support_seed=bank.release.custody.get("support_seed"),
                             support_per_class=intake["role_reports"]["support"]["per_class"],
                             spatial_repeat=bank.release.custody.get("spatial_repeat"),
                             role_assignment_sha256=intake["role_assignment"], space_plan_sha256=bank.release.plan_hash,
                             wavelength_quality=bank.release.bands.wavelength_quality,
                             minimum_protection_per_class=FORMAL_FLOOR)
            parents = {tile: bank.role_row("devval", tile)["parent_tile"] for tile in roles["evaluation"]}
    for factor, band, tile in product(sorted({g["factor"] for g in groups.values()}),raw["views"],roles["evaluation"]):
        proofs = observations.setdefault(factor, {})
        for use in uses:
            actual, _, _ = independent.materialize_reviewed(bank,intake,roles,"evaluation",tile,
                                                             code["commit"],band,use=use,factor=factor)
            proofs.setdefault(use, {})[band,tile] = actual.receipt["observation"]
    summary = {}
    for key, group in groups.items():
        ready = raw["cohorts"][key]
        require(ready["budget"] == group["protocol"]["budget"] and ready["seed"] == group["seed"]
                and ready.get("factor",1) == group["factor"]
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
            checked[name] = checked_blocks(bank, roles, intake, geometry, observations[group["factor"]],
                                           arm, body, group["arms"][name], semantics,
                                           controlled=group["causal_design"])
        scored = {}
        for base, candidate in group["pairs"]:
            scored[f"{base}/{candidate}"] = {band: matched_view(checked[base], checked[candidate], band,
                roles["evaluation"], bank.release.scene, bank.release.classes, scope=raw["scope"]) for band in raw["views"]}
        summary[key] = dict(seed=ready["seed"], budget=ready["budget"], pairs=scored, factor=group["factor"],
                            arms={name: {k:v for k,v in arm.items() if k!="rows"} for name, arm in ready["arms"].items()})
        if band_loss is not None:
            summary[key]["matched_band_loss"] = {
                **band_loss,
                "arms": {name: band_loss_scores(blocks, parents, bank.release.classes)
                         for name, blocks in checked.items()},
            }
        exposure = group["protocol"].get("observation_exposure")
        if exposure is not None:
            summary[key]["observation_exposure"] = exposure
            for views in scored.values():
                for name, result in views.items():
                    result["exposure"] = dict(optimization_seen=name in exposure["support"],
                                              calibration_seen=name in exposure["calibration"],
                                              evaluation_approved=True)
    require_frozen_code(code)
    parity = None if reference is None else compare_devices(json_pin(root,reference),raw)
    return _write(out, dict(schema=REVIEW, capture=capture, baseline_capture=reference, device_parity=parity,
                            producer=code["commit"], scope=raw["scope"],
                            scene=bank.release.scene, effect_scope="frozen_members_of_explicit_matched_cohort",
                            uses=intake["uses"], roles=intake["role_reports"], cohorts=summary,
                            numeric=raw["numeric"], query_accessed=False, raw_gt_loaded=False, optimizer_updates=0))
