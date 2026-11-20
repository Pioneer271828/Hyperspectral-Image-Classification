"""Frozen, same-mask edge contrasts across equally sized retained observations."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from math import sqrt

import numpy as np
import torch

from evaluation import wp12_mechanism_real as real
from evaluation.frozen_training_stages import HEADS, VIEWS, probabilities, require
from evaluation.graph_edge_intervention import delete_support
from evaluation.matched_edge_controls import freeze, verify
from evaluation.wp12_locked_support import transitions
from evaluation.wp12_native_pathways import graph_for
from model.research_config import digest
from model.region_pixel_alignment import pool_regions
from research_runtime import tensor_hash
from transfer.g3_training import state_hash

SCHEMA = "wp06_shared_edge_band_interaction_v1"
CASES = ("selected", "whu-honghu")
ANCHOR = "matched_dispersed"
RANDOM = 7719
CONTROLS = ("low_weight", f"random:{RANDOM}")
MINIMUM_CLASS_N = 20


def registration(producer):
    return dict(schema=SCHEMA, producer=producer, cases=list(CASES), role="devval",
                archive={case: real.LOCKED_PINS[case] for case in CASES},
                blocks="original_first_development_block_per_focus_class",
                views=list(VIEWS), anchor=ANCHOR, per_row=1, seeds=[RANDOM], time_limit=2.0,
                matching="exact_directed_in_out_degree_and_squared_pixel_distance_multiset",
                comparisons={"placement": ["matched_dispersed", "central_gap"],
                             "full_to_central": ["full", "central_gap"]},
                formal_class_floor=MINIMUM_CLASS_N, real_optimizer_updates=0, query_accessed=False)


def validate_observations(views):
    """Check the retained normalized pixels against the ordered full view."""
    require(set(views) == set(VIEWS), "The three previously approved views are required")
    full = views["full"]
    indices = {band: i for i, band in enumerate(full.bands.band_ids)}
    require(len(indices) == len(full.bands.band_ids), "The original column axis contains duplicates")
    columns = {}
    for name, view in views.items():
        ids = view.bands.band_ids
        require(set(ids) <= indices.keys(), "The retained view includes a foreign column")
        positions = [indices[band] for band in ids]
        src, dst = full.observation, view.observation
        require(positions == sorted(set(positions))
                and src.plan_hash == dst.plan_hash and src.partition == dst.partition
                and np.array_equal(src.global_ids, dst.global_ids)
                and np.array_equal(src.coordinates, dst.coordinates)
                and np.array_equal(src.spectral[..., positions], dst.spectral),
                "The spectrum, frozen normalization, or native point order differs")
        require(tuple(full.bands.wavelengths[i] for i in positions) == tuple(view.bands.wavelengths)
                and full.bands.wavelength_quality == view.bands.wavelength_quality,
                "Retained metadata does not match the original registered columns")
        columns[name] = list(ids)
    require(len(columns["central_gap"]) == len(columns[ANCHOR]) < len(columns["full"])
            and columns["central_gap"] != columns[ANCHOR],
            "Band placement requires distinct views with the same deletion count")
    return columns


@torch.inference_mode()
def audited_probe(model, invoke, view, *, graph_action=None):
    owner = graph_for(model)
    require(owner.coverage_modulation is None, "The frozen no-modulation pathway is required")
    trace = {}

    def adapter_input(_module, args, kwargs):
        wavelengths = kwargs.get("wavelengths")
        require(torch.is_tensor(wavelengths) and np.array_equal(
            wavelengths.detach().cpu().numpy(), np.asarray(view.bands.wavelengths, dtype=np.float32)),
            "The adapter used a different wavelength axis")
        trace["adapter"] = args[0]

    def denoiser_input(_module, args):
        trace["denoiser"] = args[0]

    with ExitStack() as stack:
        stack.callback(owner.adapter.register_forward_pre_hook(adapter_input, with_kwargs=True).remove)
        stack.callback(owner.CNN_denoise[0].register_forward_pre_hook(denoiser_input).remove)
        output, features, natural = real.probe(model, invoke, collect=True, graph_action=graph_action, trace=trace)
    obs = view.observation
    require(np.array_equal(trace["adapter"].detach().cpu().numpy(), obs.spectral), "The adapter used different pixels")
    coords = torch.tensor(obs.coordinates, device=trace["denoiser"].device)
    expected = torch.cat((features["vca"], coords), dim=-1).permute(2, 0, 1).unsqueeze(0)
    clean = features["pre_pool"].squeeze(0).permute(1, 2, 0)
    require(torch.equal(trace["denoiser"], expected) and torch.equal(trace["builder_input"], clean)
            and torch.equal(trace["cnn_input"], clean), "CNN and graph did not consume the same retained stem")
    used = natural if graph_action is None else graph_action
    require(tensor_hash(trace["used_q"]) == tensor_hash(natural.Q) == tensor_hash(used.Q)
            and tensor_hash(trace["used_adjacency"]) == tensor_hash(used.A0)
            and torch.equal(trace["pooled"], pool_regions(used.Q, clean.reshape(-1, clean.shape[-1]))),
            "The graph stage used a different assignment, edge matrix, or input")
    proof = {k: tensor_hash(trace[k]) for k in ("adapter", "denoiser", "builder_input", "cnn_input", "pooled")}
    proof.update(q=tensor_hash(used.Q), adjacency=tensor_hash(used.A0),
                 normalized_coordinates=tensor_hash(obs.coordinates))
    return output, natural, proof


def pixel_grid(state):
    """Use integer coordinates with a common denominator for exact pixel distances."""
    gh, gw = state.grid_shape
    h, w = state.image_shape
    ids = torch.arange(gh*gw)
    normalized = torch.stack(((ids//gw + .5)/gh, (ids%gw + .5)/gw), dim=1)
    require(state.diagnostics.get("position_kind") == "grid_seed_centers"
            and torch.equal(state.region_positions.detach().cpu(), normalized.to(state.region_positions.dtype)),
            "Edge masks require the original shared spatial seed centers")
    return integer_grid((h, w), (gh, gw)), gh*gw


def integer_grid(image, shape):
    h, w = image
    gh, gw = shape
    ids = torch.arange(gh*gw)
    return torch.stack(((ids//gw)*(h-1)*gw, (ids%gw)*(w-1)*gh), dim=1)


def squared_distances(indices, integer_grid):
    u, v = indices
    return ((integer_grid[u] - integer_grid[v]).square().sum(dim=1)).tolist()


def confirmed_masks(control):
    if control["status"] in ("different_native_nodes_or_support", "nonpositive_native_edges"):
        return {}
    plan = control["anchor_plan"]
    require(digest({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"]
            and set(plan["controls"]) == set(CONTROLS), "Frozen matching plan identity differs")
    grid = integer_grid(control["image_shape"], control["grid_shape"])
    colocated = torch.tensor(control["native_edges"], dtype=torch.long)
    n = len(grid)
    require(n == plan["node_count"] and tensor_hash(colocated) == control["native_support_sha256"],
            "Frozen grid or edge support identity differs")
    edges = set((colocated[0]*n + colocated[1]).tolist())
    important = plan["baseline"]["removed"]
    masks = {}
    for name, spec in plan["controls"].items():
        if spec["status"] == "optimal":
            require(spec["removed"] != important, "A control duplicated the anchor intervention")
            masks[name] = spec["removed"]
        else:
            require(spec["status"] in ("unresolved", "no_distinct_control") and spec["removed"] is None,
                    "An uncertified control supplied an edge mask")
    if masks:
        masks["important"] = important
    require(control["status"] == ("paired" if masks else "no_matched_arms"),
            "The saved outcome differs from matching feasibility")
    design = {}
    for name, ids in masks.items():
        require(bool(ids) and ids == sorted(set(ids)) and set(ids) <= edges
                and all(edge//n != edge%n for edge in ids), "A mask escapes the native nonself support")
        coords = torch.tensor([(i//n, i%n) for i in ids], dtype=torch.long).T
        design[name] = (torch.bincount(coords[0], minlength=n).tolist(),
                        torch.bincount(coords[1], minlength=n).tolist(),
                        sorted(squared_distances(coords, grid)))
    require(all(values == design["important"] for values in design.values())
            and {name:values[2] for name,values in design.items()} == control["distance_squared"],
            "The saved masks change exact node degree or pixel-distance multiplicities")
    return masks


def shared_masks(states, rule, *, prescribed=None):
    anchor = states[ANCHOR]
    graph = anchor.A0.detach().cpu().coalesce()
    idx = graph.indices()
    if any(state.grid_shape != anchor.grid_shape or state.image_shape != anchor.image_shape
           or not torch.equal(state.region_positions, anchor.region_positions)
           or not torch.equal(state.A0.indices().cpu(), idx) for state in states.values()):
        control = dict(status="different_native_nodes_or_support")
        require(prescribed is None or prescribed == control, "The GPU has a different native support")
        return control, {}
    if any(not bool((state.A0.values() > 0).all()) for state in states.values()):
        control = dict(status="nonpositive_native_edges")
        require(prescribed is None or prescribed == control, "The GPU has a different positive support")
        return control, {}
    grid, divisor = pixel_grid(anchor)
    geometry = dict(grid_shape=list(anchor.grid_shape), image_shape=list(anchor.image_shape),
                    distance_denominator=divisor**2, native_edges=idx.tolist(),
                    native_support_sha256=tensor_hash(idx))
    if prescribed is not None:
        require(all(prescribed[k] == v for k,v in geometry.items()), "The CPU mask belongs to another observed grid")
        return prescribed, confirmed_masks(prescribed)
    keys = sorted(set(squared_distances(idx[:, idx[0] != idx[1]], grid)))
    distances = [sqrt(key)/divisor for key in keys]
    bins = [(a+b)/2 for a,b in zip(distances[:-1], distances[1:], strict=True)] or [1.0]
    positions = grid.double()/divisor
    plan = freeze(graph, positions, per_row=rule["per_row"], distance_bins=bins,
                  seeds=rule["seeds"], time_limit=rule["time_limit"])
    verify(plan, graph, positions, expected_hash=plan["plan_hash"])
    masks = {name: item["removed"] for name, item in plan["controls"].items() if item["status"] == "optimal"}
    if masks:
        masks["important"] = plan["baseline"]["removed"]
    n = graph.shape[0]
    distances_squared = {}
    for name, ids in masks.items():
        edges = torch.tensor([(i//n, i%n) for i in ids], dtype=torch.long).T
        distances_squared[name] = sorted(squared_distances(edges, grid))
    require(len({tuple(v) for v in distances_squared.values()}) <= 1,
            "Matched masks did not preserve exact pixel-distance multiplicities")
    control = dict(status="paired" if masks else "no_matched_arms", anchor_plan=plan,
                   distance_squared=distances_squared, **geometry)
    require(confirmed_masks(control) == masks, "The exact edge accounting did not reproduce")
    return control, masks


def intervention_rows(model, block, band, base, masks, rivals, proof):
    native, natural = base
    out = {"native": real.output_rows(native, block.truth, rivals)}
    if not masks:
        return out, {}
    original = natural.A0.detach().cpu().coalesce()
    encoded = (original.indices()[0]*original.shape[0]+original.indices()[1]).tolist()
    weights = dict(zip(encoded, original.values().tolist(), strict=True))
    rows = {}
    for name, ids in masks.items():
        modified = delete_support(original, ids).to(natural.A0.device)
        target = replace(natural, A0=modified)
        values, rebuilt, traced = audited_probe(model, block.callbacks[band], block.views[band], graph_action=target)
        require(tensor_hash(rebuilt.A0) == tensor_hash(natural.A0)
                and all(traced[k] == v for k,v in proof.items() if k != "adjacency")
                and modified._nnz() == natural.A0._nnz()-len(ids),
                "The fixed deletion changed the native observation, Q, or support count")
        points = real.get_points(values, block.cells)
        require(np.array_equal(points["cnn"], native["cnn"]), "The edge intervention changed the CNN control head")
        out[name] = real.output_rows(points, block.truth, rivals)
        rows[name] = dict(mask_sha256=digest(ids), adjacency=traced["adjacency"], nnz=modified._nnz(),
                          removed_weight_mass=sum(weights[i] for i in ids),
                          cnn_equal=True)
    return out, rows


def run(root, case, producer, device, *, reference=None):
    rule = registration(producer)
    require(case in CASES, "Only the two frozen focal-development releases are admitted")
    endpoint = real.load(root, case, producer, device)
    model = endpoint.model
    old = real.locked_history(root, case, endpoint.checkpoint)
    state_before = state_hash(model)
    require(old is not None, "An externally pinned historical native baseline is required")
    if reference is not None:
        require(device.type == "cuda" and reference["device"] == "cpu" and reference["case"] == case
                and reference["registration"] == digest(rule) and reference["authority"] == endpoint.authority
                and reference["checkpoint"] == endpoint.checkpoint and reference["state"] == state_before,
                "Expected the exact CPU mechanism baseline")
        reference_blocks = {row["ordinal"]: row for row in reference["groups"]}
    else:
        reference_blocks = {}
    groups, prior_focus, offset = [], set(), 0
    focus = real.FOCUS[endpoint.scene]
    rivals = real.RIVALS[endpoint.scene]
    for ordinal, block in enumerate(endpoint.blocks, 1):
        enrolled = (set(block.truth) & set(focus)) - prior_focus
        if enrolled:
            prior_focus.update(enrolled)
            columns = validate_observations(block.views)
            bases, states, proofs = {}, {}, {}
            for band in VIEWS:
                outputs, states[band], proofs[band] = audited_probe(model, block.callbacks[band], block.views[band])
                bases[band] = real.get_points(outputs, block.cells)
                for value in bases[band].values():
                    probabilities(value, len(block.truth), endpoint.classes)
            real.verify_baselines(block, bases, old, offset, device.type == "cpu")
            frozen_block = reference_blocks.get(ordinal)
            require(reference is None or (frozen_block is not None and frozen_block["points_sha256"] == block.point_order
                                          and frozen_block["columns"] == columns),
                    "The CPU mask belongs to another point or retained view")
            plan, masks = shared_masks(states, rule, prescribed=None if frozen_block is None else frozen_block["control"])
            scores, operations = {}, {}
            for band in VIEWS:
                scores[band], operations[band] = intervention_rows(model, block, band, (bases[band], states[band]),
                                                                    masks, rivals, proofs[band])
            groups.append(dict(ordinal=ordinal, tile=block.tile, parent=block.parent, offset=offset,
                               points_sha256=block.point_order, truth=block.truth, first_focus=sorted(enrolled),
                               columns=columns, pathway_proofs=proofs, control=plan, scores=scores,
                               interventions=operations))
        offset += len(block.truth)
    require(prior_focus == set(focus) and state_before == state_hash(model)
            and offset == len(old["result"]["views"]["full"]["truth"]),
            "Focal coverage, model weights, or original development population changed")
    return dict(schema=SCHEMA, registration=digest(rule), producer=producer, case=case,
                scope="historical_released_development_only", scene=endpoint.scene, step=endpoint.step,
                classes=endpoint.classes, checkpoint=endpoint.checkpoint, authority=endpoint.authority,
                state=state_before, release=endpoint.release, focus=focus, rivals=rivals, groups=groups,
                device=str(device), optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)


def checked_receipt(row, producer):
    require(row["schema"] == SCHEMA and row["producer"] == producer and row["case"] in CASES
            and row["registration"] == digest(registration(producer)) and row["device"] in ("cpu", "cuda")
            and row["scope"] == "historical_released_development_only"
            and row["raw_gt_loaded"] is row["query_accessed"] is False and row["optimizer_updates"] == 0
            and row["focus"] == list(real.FOCUS[row["scene"]]), "Foreign roles, scope or frozen mechanism registration")
    reviewed, seen = [], set()
    classes = row["classes"]
    for block in row["groups"]:
        require(block["points_sha256"] not in seen and bool(block["truth"])
                and set(block["scores"]) == set(block["pathway_proofs"]) == set(block["interventions"]) == set(VIEWS),
                "Duplicate points, zero denominator or missing retained cell")
        seen.add(block["points_sha256"])
        masks = confirmed_masks(block["control"])
        for band in VIEWS:
            require(set(block["scores"][band]) == {"native", *masks}
                    and set(block["interventions"][band]) == set(masks), "A view is missing a matched intervention")
            proof = block["pathway_proofs"][band]
            require(proof["builder_input"] == proof["cnn_input"], "CNN and graph stem identity differs")
            native = block["scores"][band]["native"]["cnn"]
            for key, scores in block["scores"][band].items():
                require(set(scores) == set(HEADS)
                        and all(len(points) == len(block["truth"]) and all(
                            type(p["predicted"]) is int and 1 <= p["predicted"] <= classes for p in points)
                                for points in scores.values()), "Head predictions cross the locked point or class order")
                if key != "native":
                    arm = block["interventions"][band][key]
                    require(arm["mask_sha256"] == digest(masks[key]) and arm["cnn_equal"] is True
                            and scores["cnn"] == native
                            and arm["nnz"] == len(block["control"]["native_edges"][0])-len(masks[key]),
                            "A graph arm used different edges or changed the CNN control")
        reviewed.append((block, masks))
    require(bool(reviewed), "No registered focal blocks")
    return reviewed


def decisions(block, view, arm, head):
    return [row["predicted"] for row in block["scores"][view][arm][head]]


def paired_views(blocks, left, right, arm, classes):
    truth = [y for block in blocks for y in block["truth"]]
    return {head: transitions(truth, [p for b in blocks for p in decisions(b, left, arm, head)],
                              [p for b in blocks for p in decisions(b, right, arm, head)], classes) for head in HEADS}


def interaction_counts(predictions, left, right, reference, altered, classes):
    results = {}
    for head in HEADS:
        y, before, before_arm, after, after_arm = map(list, zip(*predictions[head], strict=True))
        small = {}
        for c in range(1, classes+1):
            index = [i for i,t in enumerate(y) if t == c]
            values = [(int(after_arm[i] == c)-int(after[i] == c)) -
                      (int(before_arm[i] == c)-int(before[i] == c)) for i in index]
            small[str(c)] = dict(n=len(index), interaction_correct=sum(values),
                                 interaction_recall=sum(values)/len(values) if values else None,
                                 per_point_delta=dict(sorted(Counter(values).items())),
                                 floor_met=len(index) >= MINIMUM_CLASS_N)
        results[head] = dict(views={left:transitions(y,before,before_arm,classes),
                                   right:transitions(y,after,after_arm,classes)},
                             reference_arm=reference, compared_arm=altered, classes=small)
    return results


def score_interaction(blocks, left, right, reference, altered, classes):
    aligned = {head: [] for head in HEADS}
    for block in blocks:
        for head in HEADS:
            aligned[head].extend(zip(block["truth"], decisions(block, left, reference, head),
                                     decisions(block, left, altered, head), decisions(block, right, reference, head),
                                     decisions(block, right, altered, head), strict=True))
    return interaction_counts(aligned, left, right, reference, altered, classes)


def reduction(receipt, producer):
    audited = checked_receipt(receipt, producer)
    blocks = [block for block, _ in audited]
    classes = receipt["classes"]
    identity = {k:receipt[k] for k in ("scene","case","step","checkpoint","authority","state","release","focus")}
    status = [dict(ordinal=block["ordinal"], parent=block["parent"], points=len(block["truth"]),
                   counts={str(y):block["truth"].count(y) for y in receipt["focus"]},
                   point_identity=block["points_sha256"], columns=block["columns"],
                   status=block["control"]["status"], masks={k:digest(v) for k,v in masks.items()},
                   matching={k:v["status"] for k,v in block["control"].get("anchor_plan", {}).get("controls", {}).items()})
              for block, masks in audited]
    comparisons = {}
    for label,(left,right) in registration(producer)["comparisons"].items():
        comparisons[label] = dict(original=paired_views(blocks,left,right,"native",classes), matched={})
        for reference, altered in (("native","important"), ("low_weight","important"), (CONTROLS[1],"important")):
            eligible = [b for b,masks in audited if altered in masks and (reference == "native" or reference in masks)]
            parents = sorted({block["parent"] for block in eligible})
            entry = dict(eligible_parents=parents, missing_parents=sorted({b["parent"] for b in blocks}-set(parents)),
                         points=sum(len(b["truth"]) for b in eligible), total=None, by_parent={})
            if eligible:
                entry["total"] = score_interaction(eligible,left,right,reference,altered,classes)
                entry["by_parent"] = {str(parent):score_interaction([b for b in eligible if b["parent"]==parent],
                                                                   left,right,reference,altered,classes)
                                     for parent in parents}
            comparisons[label]["matched"][f"{reference}/{altered}"] = entry
    return dict(**identity, blocks=status, total_sampled_points=sum(len(b["truth"]) for b in blocks),
                available_development_points=sum(receipt["release"]["evaluation_per_class"]), contrasts=comparisons)


def device_parity(cpu, cuda, producer):
    before, after = checked_receipt(cpu, producer), checked_receipt(cuda, producer)
    require(cpu["device"] == "cpu" and cuda["device"] == "cuda" and all(cpu[k] == cuda[k] for k in
            ("case","scene","checkpoint","authority","state","release")), "The device replay crossed a frozen release")
    n, flips, margin = 0, 0, 0.0
    for (a, masks_a), (b, masks_b) in zip(before, after, strict=True):
        require(all(a[k] == b[k] for k in ("tile","parent","truth","points_sha256","columns","control"))
                and masks_a == masks_b, "The devices used different geometry, point order, or edge masks")
        for view in VIEWS:
            for arm in ("native", *masks_a):
                for head in HEADS:
                    for x,y in zip(a["scores"][view][arm][head], b["scores"][view][arm][head], strict=True):
                        n += 1
                        flips += x["predicted"] != y["predicted"]
                        margin = max(margin, abs(x["highest_margin"] - y["highest_margin"]))
    return dict(paired_head_points=n, decision_changes=flips, max_highest_margin_difference=margin)
