"""Read bounded HongHu point populations, archived scores and accepted longer states."""

from collections import defaultdict
from dataclasses import replace
import io
from pathlib import Path

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS, hard
from loadData.observation_sampling import replay
from model.research_config import digest
from scripts.run_frozen_band_region import _honghu_context, _read, _state, check, pin
from train_spatial import SpatialConfig
from transfer import frozen_band_region as region, p4_frozen_diagnostics as historical
from transfer import frozen_mechanism_graph as graphs, frozen_mechanism_plan as plan
from transfer import registered_development as transport
from transfer.g3_training import state_hash
from transfer.p4_spatial import initialize as init_honghu


LONG_SCOPE = "a827dec0a6087d1eaf990ca086f1c28dd27e501ba0ae81c3b2c18434672e9ce5"
ARCHIVE_CONTEXT = ("nominal", "full")
STAGES = ("initial", "selected", "last")
LONG_SPATIAL_ARMS = ("none", "continuous")
HONGHU_FOILS = plan.EXPECTED_CONFUSIONS["whu-honghu"]


def accepted_catalog(root, source, old_scope):
    """Match the parent registration and complete independent audit before any scores."""
    docs = {key: _read(link) for key, link in source.items()}
    short = docs["short_scope"]
    evidence, long = docs["long_evidence"], docs["long_scope"]
    check(evidence["new_updates"] == evidence["new_label_reads"] == evidence["new_forwards"] == 0
          and evidence["historical_updates"] == 1200, "Long archive provenance changed")
    check(len(short["cases"]) == 40 and len({x["populations"]["val"]["sha256"] for x in short["cases"]}) == 1,
          "Short archive is not a common population")
    population = short["cases"][0]["populations"]["val"]
    catalog = []
    by_id = {row["id"]: row for row in long["cases"]}
    for name, saved in sorted(evidence["cases"].items()):
        audit_pin = saved["audit_pin"]
        check(Path(audit_pin["path"]).is_relative_to(root), "Long audit outside authorized release")
        audit = _read(audit_pin)
        case = audit["case"]
        check(audit["status"] == "passed" and audit["scope_sha256"] == LONG_SCOPE and audit["optimizer_updates"] == 120
              and audit["audit_optimizer_updates"] == audit["completed_resume_updates"] == 0
              and audit["raw_gt_loaded"] is audit["query_accessed"] is False and audit["validation_steps"] == [30, 60]
              and audit["populations"]["val"] == population and case == saved["case"] == by_id[name],
              "Long case budget, population, arm or status differs")
        folder = Path(audit_pin["path"]).parent / "full"
        state = pin(folder / "state.pt", audit["artifact_pins"][str(folder / "state.pt")])
        result = pin(folder / "result.json", audit["artifact_pins"][str(folder / "result.json")])
        catalog.append(dict(id=name, protocol="honghu_60_cuda_v4", seed=case["model_seed"],
                            support_seed=case["support_seed"], k=case["k"], arm=case["arm"], method=case["method"],
                            budget=60, validation=[30, 60], selected=audit["selected_step"], audit=audit_pin,
                            checkpoint=state, source_result=result, stages=dict(initial=0,
                                selected=audit["selected_step"], last=60),
                            endpoint_hashes=audit["state_sha256"], population=population["sha256"],
                            train_population=audit["populations"]["train"]["sha256"],
                            spatial=case["arm"] in LONG_SPATIAL_ARMS))
    check(len(catalog) == 10 and {x["seed"] for x in catalog} == {512}
          and {x["selected"] for x in catalog} == {30} and {x["k"] for x in catalog} == {1, 5},
          "Long case inventory differs from the declared completed set")
    short_catalog = [dict(id=x["case"]["id"], protocol="honghu_24_cpu_archive", seed=64,
                          support_seed=x["case"]["support_seed"], k=x["case"]["k"], arm=x["case"]["arm"],
                          method=x["case"]["method"], budget=24, validation=[22, 24],
                          selected=x["state"]["selected_step"], endpoint_hashes={
                              stage: x["state"][stage] for stage in STAGES},
                          stages=dict(initial=0, selected=x["state"]["selected_step"], last=24),
                          population=population["sha256"], train_population=x["populations"]["train"]["sha256"],
                          spatial=False)
                     for x in short["cases"]]
    return short_catalog, catalog, dict(
        common_validation=population, unavailable=[
            {"source": "60_step_registered_20_case_scope", "candidate_cases": len(by_id),
             "eligible_cases": 10, "unaccepted_cases": sorted(set(by_id) - {x["id"] for x in catalog}),
             "reason": "No completed independent audit or selected packet in the existing review"},
            {"source": "1500_step_three_arm_real_registration", "candidate_cases": len(docs["long_preflight"]["cases"]),
             "eligible_cases": 0, "reason": "Registered/preflight only; no accepted actual endpoint"},
        ], previous_scope=old_scope)


def _role_points(layout, width, salt, role):
    aliases, points, group_map = [], [], []
    for tile, offsets in layout:
        y0, y1, x0, x1 = tile["box"]
        size = x1 - x0
        grouped = []
        tile_alias = plan.point_alias(salt, "whu-honghu", role + "/tile", tile["id"])
        for position in offsets.tolist():
            y, x = divmod(position, size)
            idx = (y + y0) * width + x + x0
            alias = plan.point_alias(salt, "whu-honghu", role, idx)
            aliases.append(alias)
            points.append(idx)
            grouped.append(alias)
        group_map.append({"block": tile_alias, "aliases": grouped})
    return aliases, points, group_map


def populations(ctx, case, salt):
    _, reg, original_plan, geometry, scene, allowed = ctx
    roles = historical.prior.released_roles(reg, original_plan, geometry, allowed, case)
    width = scene["shape"][1]
    entries = {}
    for role, budget in (("train", case["k"]), ("val", 5)):
        layout, labels, receipt = historical.fixed_population(roles[role], allowed, role, width, budget)
        label_role = "devtrain" if role == "train" else "devval"
        aliases, ids, groups = _role_points(layout, width, salt, label_role)
        entries[role] = dict(layout=layout, labels=labels, aliases=aliases, ids=ids, block_groups=groups,
                             authority=receipt["sha256"], n=len(labels), role=label_role)
    return entries


def serialize_populations(items):
    return {role: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                   for key, value in entry.items() if key not in ("layout", "ids")}
            for role, entry in items.items()}


def short_scores(root, scope, context, registered, salt):
    sources, receipt = scope["short_scope"], scope["short_predictions"]
    frozen = _read(sources)
    summary = _read(receipt)
    case_id = registered["id"]
    item = next(row for row in frozen["cases"] if row["case"]["id"] == case_id)
    authority = populations(context, item["case"], salt)
    check(authority["val"]["authority"] == registered["population"] and
          authority["train"]["authority"] == registered["train_population"], "Short archive label order differs")
    folder = Path(receipt["path"]).parent
    path = pin(folder / f"{case_id}.npz", summary["artifacts"][f"{case_id}.npz"])
    arrays = {}
    with np.load(io.BytesIO(transport._read(path)), allow_pickle=False) as archive:
        for stage in STAGES:
            for role in ("train", "val"):
                for head in HEADS:
                    values = archive[f"fit__{stage}__{role}__{head}"]
                    hard(values, authority[role]["n"], 22)
                    arrays[f"nominal__{stage}__{role}__{head}"] = values
        for head in HEADS:
            values = archive[f"obs__full__{head}"]
            hard(values, authority["val"]["n"], 22)
            arrays[f"full__selected__val__{head}"] = values
    return dict(populations=serialize_populations(authority), forwards=0, source=path["sha256"],
                feature_readout=None, geometry_groups={}, spatial=[], observations=len(arrays),
                arithmetic="pinned_archived_cpu_probabilities"), arrays


def load_long_model(ctx, entry):
    _, reg, *_ = ctx
    audit = _read(entry["audit"])
    terminal = _read(entry["source_result"])
    packet = _state(entry["checkpoint"])
    case = audit["case"]
    check(case["id"] == entry["id"] and terminal["state_sha256"] == entry["checkpoint"]["sha256"]
          and packet["identity"] == terminal["identity"] and packet["cursor"] == terminal["updates"] == 60
          and packet["best"]["step"] == terminal["best_step"] == audit["selected_step"] == entry["selected"]
          and [row["step"] for row in packet["history"]] == [30, 60],
          "Long endpoint packet differs from the independent audit")
    args = (reg["source"]["weight"], reg["source"]["result"]) if case["method"] == "shared_transfer" else ()
    model, initialized = init_honghu(SpatialConfig(reg["config"]), *args,
                                     coverage_arm=case["arm"], model_seed=case["model_seed"])
    check(state_hash(model) == entry["endpoint_hashes"]["initial"] and
          initialized["initial_state_sha256"] == terminal["initialization"]["initial_state_sha256"],
          "Historical seed or initial weights differ")
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model.eval()
    model.backbone.adapter.cache_attention = False
    model.backbone.adapter.clear_attention_cache()
    return model, {"initial": initial, "selected": packet["best"]["model"], "last": packet["model"]}, packet, terminal, case


def _bank_views(bank, authority, observation):
    for tile, offsets in authority["layout"]:
        _, _, full, _ = bank.materialize(bank.role_row(authority["role"], tile["id"]), {"mode": "none"})
        check(tuple(full.observation.spectral.shape[:2]) ==
              (tile["box"][1]-tile["box"][0], tile["box"][3]-tile["box"][2]),
              "Authorized historical block geometry differs from bank")
        view = full if observation is None else replay(full.observation, full.bands, observation)
        yield tile["id"], offsets.tolist(), view


def _population_pass(model, bank, authority, record, identity, *, embed=False):
    scores, features = defaultdict(list), defaultdict(list)
    for _, offsets, view in _bank_views(bank, authority, record):
        result = graphs.execute(model, "whu-honghu", view, offsets, 22, 100,
                                role=authority["role"], model_state=identity, make_embeddings=embed)
        for head in HEADS:
            scores[head].append(result.probabilities[head])
            if embed:
                features[head].append(result.features[head])
    return {key: np.concatenate(vals) for key, vals in scores.items()}, {
        key: np.concatenate(vals) for key, vals in features.items()}


def _separation(model, validation, support, labels):
    heads = dict(fused=model.heads, graph=model.hgnn_heads, cnn=model.cnn_heads)
    output = {}
    for head, module in heads.items():
        train_values, values = support[head].astype(float), validation[head].astype(float)
        classifier = module["whu-honghu"]
        weight = classifier.weight.detach().cpu().numpy().astype(float)
        bias = classifier.bias.detach().cpu().numpy().astype(float)
        head_rows = []
        for cls, foils in HONGHU_FOILS.items():
            subset = values[labels["val"]["labels"] == cls]
            centroid = train_values[labels["train"]["labels"] == cls].mean(0)
            for foil in foils:
                other = train_values[labels["train"]["labels"] == foil].mean(0)
                axis = weight[cls-1] - weight[foil-1]
                within = np.linalg.norm(subset - centroid, axis=1)
                versus = np.linalg.norm(subset - other, axis=1)
                head_rows.append(dict(true_class=cls, foil=foil, validation_count=len(subset),
                    train_true=int((labels["train"]["labels"] == cls).sum()),
                    train_foil=int((labels["train"]["labels"] == foil).sum()),
                    rank_geometry_own_nearer=int((within < versus).sum()),
                    distance_gap_mean=float((versus-within).mean()),
                    bias_gap=float(bias[cls-1]-bias[foil-1]),
                    weight_projection_gap_mean=float((subset @ axis).mean()),
                    train_projection_gap_mean=float((train_values[labels["train"]["labels"] == cls] @ axis).mean())))
        output[head] = head_rows
    return output


def _position_controls(model, bank, authority, identity, salt):
    scores, masks = defaultdict(list), defaultdict(list)
    traces = []
    consumed = 0
    labels = authority["labels"]
    for tile, offsets, view in _bank_views(bank, authority, None):
        local = labels[consumed:consumed + len(offsets)]
        focused = bool(set(local.tolist()).intersection(HONGHU_FOILS))
        if not focused:
            consumed += len(offsets)
            continue
        shape = view.observation.spectral.shape[:2]
        boxes = region.region_boxes(shape, offsets, local.tolist(), plan.FOCUS["whu-honghu"])
        observations = {"clean": view, **{name: replace_spectral(view, box) for name, box in boxes.items()}}
        for name, current in observations.items():
            result = graphs.execute(model, "whu-honghu", current, offsets, 22, 100,
                                    role="devval", model_state=identity)
            for head in HEADS:
                scores[f"spatial__{name}__{head}"].append(result.probabilities[head])
            if name == "clean":
                q = result.graph.state.Q.coalesce()
                memberships = q.indices()[:, q.values() > 0].numpy()
                groups = region.same_region_groups(offsets, boxes["target"], shape, memberships)
                for key, mask in groups.items():
                    masks[key].extend(mask.tolist())
        aliases = authority["aliases"][consumed:consumed+len(offsets)]
        traces.append(dict(tile=plan.point_alias(salt, "whu-honghu", "devval/tile", tile),
                           aliases=aliases, labels=local.tolist(), target_area=(boxes["target"][1]-boxes["target"][0])**2))
        consumed += len(offsets)
    return {key: np.concatenate(parts) for key, parts in scores.items()}, dict(masks), traces


def replace_spectral(view, box):
    return replace(view, observation=region._spatial_view(view.observation, box))


def long_scores(bank, ctx, entry, salt):
    model, states, packet, terminal, case = load_long_model(ctx, entry)
    authority = populations(ctx, case, salt)
    check([authority[r]["authority"] for r in ("train", "val")] ==
          [entry["train_population"], entry["population"]], "Long checkpoint population differs")
    records = terminal["coverage"]["records"]
    arrays, trace, features = {}, {}, {}
    score_forwards = 0
    for stage in STAGES:
        model.load_state_dict(states[stage], strict=True)
        identity = state_hash(model)
        check(identity == entry["endpoint_hashes"][stage], "Long checkpoint model tensor identity drifted")
        for role in ("train", "val"):
            current, embeds = _population_pass(model, bank, authority[role], records[role], identity,
                                               embed=stage == "selected" and entry["spatial"])
            score_forwards += len(authority[role]["layout"])
            if embeds:
                features[role] = embeds
            for head, scores in current.items():
                arrays[f"nominal__{stage}__{role}__{head}"] = scores
            if role == "val" and stage == "selected":
                trace["nominal_cpu_cuda_probability_max_abs"] = {
                    head: float(np.max(np.abs(current[head] - packet["best"]["scores"][head].numpy())))
                    for head in HEADS}
                trace["nominal_cpu_cuda_hard_flips"] = {
                    head: int(np.sum(current[head].argmax(1) != packet["best"]["scores"][head].numpy().argmax(1)))
                    for head in HEADS}
        current, _ = _population_pass(model, bank, authority["val"], None, identity)
        score_forwards += len(authority["val"]["layout"])
        for head, scores in current.items():
            arrays[f"full__{stage}__val__{head}"] = scores
    model.load_state_dict(states["selected"], strict=True)
    identity = state_hash(model)
    spatial, masks, blocks = {}, {}, []
    feature_readout = None
    if entry["spatial"]:
        spatial, masks, blocks = _position_controls(model, bank, authority["val"], identity, salt)
        feature_readout = _separation(model, features["val"], features["train"], authority)
        arrays.update(spatial)
        score_forwards += 3 * len(blocks)
    check(state_hash(model) == identity, "Long replay changed frozen model state")
    return dict(populations=serialize_populations(authority), forwards=score_forwards,
                feature_readout=feature_readout, geometry_groups=masks, spatial=blocks,
                observations=len(arrays), arithmetic="cpu_with_existing_selected_cuda_control", **trace), arrays
