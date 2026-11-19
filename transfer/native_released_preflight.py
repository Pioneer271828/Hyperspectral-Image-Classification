"""Authenticate the available real banks and freeze a read-only sparse-point training proposal."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np
import torch
from model.research_config import digest
from scripts.run_spatial_multiscale import BANK_PRODUCER, NICK, REAL, read
from transfer import development_bank as bank, development_bank_custody as custody
from transfer.development_block_probe import VIEWS
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_released_points import HEADS, LOSS, MODE, bind, scores
from transfer.native_scale_supervision import loss
from transfer.native_released_training import COMPARISONS, PAVIA_COMPARISONS, PAVIA_DESIGN
from transfer.native_scale_observation import FACTORS, Lattice, _materialize, forward
from transfer.native_scale_training import SEEDS
from transfer.spatial_multiscale import initialize


SCHEMA = "wp07_native_released_real_preflight_v1"
SCENES = ("xuzhou", "whu-honghu")
BANK_ROOTS = {"xuzhou": "c877232e6cc38e3f8be37eb1b147d69a7253edbeb665f98e5b8cd6acb8e5cb17",
              "whu-honghu": "0bbe03fc60abe505c2c48d2f4837ba7494fabbcdeda673280d7220d5bc13dfa9"}
EXPECTED = {"xuzhou": (9, 73, 5, 31), "whu-honghu": (22, 261, 10, 66)}


def open_bank(root, scene, *, publication=None):
    if scene == "paviaU" and publication is not None:
        from transfer.paviau_development_bank import open_publication

        return open_publication(root, publication)
    if publication is not None:
        raise ValueError("An explicit completed publication is only expected for PaviaU")
    if scene not in SCENES:
        raise ValueError("DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED or unknown scene")
    prior = {"path": str(root / f"wp02-bank-review-fix01/{NICK[scene]}-replay06.json"), "sha256": REAL[scene]}
    receipt = read(prior, root)
    if (receipt["producer"] != BANK_PRODUCER or receipt["scene"] != scene or receipt["real_updates"] != 0
            or receipt["bank"]["sha256"] != BANK_ROOTS[scene]):
        raise ValueError("Development bank producer or authority differs")
    intake = custody.PROVIDERS[scene](root)
    admitted = bank.Bank(receipt["bank"], root, intake.release)
    release = admitted.release
    if (release.custody["release_seed"], release.custody["validation_seed"], release.custody["per_class"]) != (512, 3031, 5):
        raise ValueError("Original release must be support 512/K5 and validation 3031/V5")
    if state_hash(intake.baseline) != release.source.get("baseline_state", release.source["state"]):
        raise ValueError("Source model differs from the published bank")
    return admitted, prior, intake


def independent_grid(orig, actual, released, row):
    """Scalar membership and integer counts are independent of the training mapper."""
    y0, y1, x0, x1 = row["box"]
    f = actual.lattice.factor
    h, w = actual.lattice.shape
    width = actual.lattice.scene_shape[1]
    counts = np.zeros((h*w, actual.classes), dtype=np.int64)
    for offset, (idx, label) in enumerate(row["samples"]):
        cell = ((idx//width-y0)//f)*w + (idx%width-x0)//f
        counts[cell, label-1] += 1
        if cell != released.cells[offset]:
            raise ValueError("Independent original-pixel mapping differs")
    if not np.array_equal(counts, released.histogram):
        raise ValueError("Independent known class-count grid differs")
    source = orig.observation.spectral
    expected = np.asarray([source[y:min(y+f,y1-y0), x:min(x+f,x1-x0)].mean((0, 1), dtype=np.float64)
                           for y in range(0, y1-y0, f) for x in range(0, x1-x0, f)], dtype=np.float32)
    if not np.array_equal(expected.reshape(actual.view.observation.spectral.shape), actual.view.observation.spectral):
        raise ValueError("Independently reduced retained spectral values differ")
    yy = np.asarray([(y+min(y+f,y1)-1)*.5 for y in range(y0, y1, f)])
    xx = np.asarray([(x+min(x+f,x1)-1)*.5 for x in range(x0, x1, f)])
    sy, sx = actual.lattice.scene_shape
    ys, xs = np.meshgrid(yy/(sy-1), xx/(sx-1), indexing="ij")
    if not np.array_equal(np.stack((ys, xs), -1).astype(np.float32), actual.view.observation.coordinates):
        raise ValueError("Independent native footprint coordinates differ")
    return array_hash(counts)


def draft(scene, admitted, rows, train_tiles, *, comparisons=COMPARISONS):
    classes = admitted.release.classes
    labels = {tile: [label for _, label in admitted.role_row("devtrain", tile)["samples"]] for tile in train_tiles}
    cases = []
    for seed in SEEDS:
        order = [int(tile) for tile in np.random.default_rng(seed).permutation(train_tiles)]
        # One visit per labeled block in each band condition. Same point
        # exposure and events are reused exactly across every factor and arm.
        schedule = [{"tile": tile, "band": band} for band in VIEWS for tile in order]
        total = Counter(label for event in schedule for label in labels[event["tile"]])
        if [total[c] for c in range(1, classes+1)] != [10]*classes:
            raise ValueError("Future matched plan did not expose each released point twice")
        for factor, arm in comparisons:
            cases.append({"id": f"{scene}/{seed}/{factor}/{arm}", "model_seed": seed,
                          "factor": factor, "arm": arm, "planned_updates": len(schedule), "actual_updates": 0,
                          "selected_step": None, "actual_cost": None, "actual_accuracy": None,
                          "train_exposures_per_class": [total[c] for c in range(1, classes+1)],
                          "schedule": schedule, "events": [0, len(order), len(schedule)]})
    params = {arm: count for arm, count in rows.items()}
    return {"status": "NOT_RUN", "authorized_for_training": False,
            "minimum_updates_to_visit_every_released_training_point": len(train_tiles),
            "planned_updates": sum(c["planned_updates"] for c in cases), "cases": cases,
            "checkpoint_selection": "first_min_released_point_fused_nll",
            "fixed_devval": {"tile_ids": [row["id"] for row in admitted.role_rows("devval")],
                             "support_per_class": [5]*classes, "bands": list(VIEWS)},
            "parameter_counts": params,
            "cost_scope": "per-case native forward/backward/Adam, fixed devval and checkpoint phases",
            "estimated_wall_seconds": None, "estimate_reason": "No real training throughput measured under this objective"}


def target_head_baseline(source, scene, seed):
    model = deepcopy(source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for heads in (model.heads, model.hgnn_heads, model.cnn_heads):
            heads[scene].reset_parameters()
    return model


def _add_scores(totals, key, output, released, classes):
    for name, score in scores(output, released, detailed=True).items():
        total = totals.setdefault((*key, name), [0., np.zeros((classes, classes), dtype=np.int64)])
        total[0] += score["nll_sum"]
        total[1] += score["point_confusion"]


def _dev_metrics(totals, classes):
    rows = {}
    for key, (nll, confusion) in totals.items():
        support = confusion.sum(axis=1)
        if support.tolist() != [5]*classes:
            raise ValueError("Zero-update scoring did not cover the entire original point release")
        rows["/".join(map(str, key))] = {"support_per_class": support.tolist(), "nll": nll / int(support.sum()),
            "oa": float(np.trace(confusion)/support.sum()), "recall": (confusion.diagonal()/support).tolist(),
            "confusion": confusion.tolist()}
    return rows


@torch.no_grad()
def scan(root, scene, producer, *, publication=None):
    started = perf_counter()
    admitted, prior, intake = open_bank(root, scene, publication=publication) if publication else open_bank(root, scene)
    pavia = scene == "paviaU"
    classes, ntrain, nval, labeled = (9, 127, 41, None) if pavia else EXPECTED[scene]
    release = admitted.release
    roles = {role: admitted.role_rows(role) for role in ("devtrain", "devval")}
    if (release.classes, *(len(roles[role]) for role in roles)) != (classes, ntrain, nval):
        raise ValueError("Incomplete real published role inventory")
    role_sets = {}
    for role, rows in roles.items():
        pairs = sorted(tuple(point) for row in rows for point in row["samples"])
        if len({point[0] for point in pairs}) != 5*classes:
            raise ValueError("Released original ID duplicates across blocks or missing points")
        role_sets[role] = pairs
    if {p[0] for p in role_sets["devtrain"]} & {p[0] for p in role_sets["devval"]}:
        raise ValueError("Train and devval share released pixels")
    train_tiles = [row["id"] for row in roles["devtrain"] if row["samples"]]
    if labeled is not None and len(train_tiles) != labeled:
        raise ValueError("Released train-block exposure bound changed")

    source = intake.baseline
    base_state = state_hash(source)
    base = target_head_baseline(source, scene, 64) if pavia else source
    design = PAVIA_COMPARISONS if pavia else COMPARISONS
    model_arms = ("off", "fixed", "mean", "content", "metadata", "joint") if pavia else ("off", "mean", "joint")
    models = {arm: initialize(base, arm, 64 + 7321) for arm in model_arms}
    guards = {arm: state_hash(m) for arm, m in models.items()}
    outputs, counts, timings = [], defaultdict(Counter), defaultdict(list)
    zero_update_scores = {}
    roles_and_views = product(roles, VIEWS)
    for role, band in roles_and_views:
        for row in roles[role]:
            _, _, orig, custody_row = admitted.materialize(row, VIEWS[band])
            retained = list(range(len(release.bands.band_ids)))[::1 if band == "full" else 2]
            if custody_row["positions"] != retained or custody_row["band_order"] != orig.bands.fingerprint:
                raise ValueError("Original band positions were changed")
            for axis in ("minimum", "maximum"):
                if custody_row["statistics"][axis] != [release.statistics[axis][i] for i in retained]:
                    raise ValueError("Retained band statistics did not use the original training values")
            for factor in FACTORS:
                lattice = Lattice.make(row["box"], release.shape[:2], factor)
                actual = _materialize(scene, classes, orig, lattice, custody_row, role, producer)
                released = bind(actual, row, producer, role)
                hist = independent_grid(orig, actual, released, row)
                target = released.summary()
                counts[role, band, factor].update(label for _, label in released.points)
                if pavia:
                    arms = tuple(arm for scale, arm in design if scale == factor) if released.points else ("off",)
                else:
                    arms = ("off",) if row["id"] != roles[role][0]["id"] else model_arms
                graphs = {}
                fallback = {}
                losses_checked = 0
                for arm in arms:
                    tick = perf_counter()
                    output, graph = forward(models[arm], actual, producer, role)
                    timings[role, band, factor].append(perf_counter() - tick)
                    if pavia and released.points:
                        _add_scores(zero_update_scores, (role, band, factor, arm), output, released, classes)
                        if arm in ("mean", "content"):
                            fallback[arm] = output[:3]
                        if arm in ("metadata", "joint") and any(not torch.equal(a, b) for a, b in
                            zip(output[:3], fallback["mean" if arm == "metadata" else "content"], strict=True)):
                            raise ValueError("PaviaU missing physical metadata did not take the existing fallback")
                    if arm == "off" and released.points:
                        values, mask = released.weights(MODE)
                        computed = loss(output, values, mask)
                        truth = [label-1 for _, label in released.points]
                        for index, head in enumerate(HEADS):
                            got = float(computed[head + "_supervised"])
                            probs = output[index].detach().cpu().numpy()
                            independent = -np.log(np.clip(probs[released.cells, truth].astype(np.float64), 1e-8, 1.)).mean()
                            if not np.isclose(got, independent, atol=1e-6, rtol=1e-6):
                                raise ValueError("Independent released point loss differs on the real lattice")
                        losses_checked += len(HEADS)
                    if graph["q_shape"][0] != actual.receipt["pixels"]:
                        raise ValueError("Q did not use the actual retained grid")
                    graphs[arm] = {"q": graph["q"], "a0": graph["a0"], "shape": graph["q_shape"]}
                if len({x["q"] for x in graphs.values()}) != 1:
                    raise ValueError("Native arms did not share the same initial Q")
                outputs.append({"role": role, "tile": row["id"], "band": band, "factor": factor,
                                "observation": actual.receipt["observation"], "source": actual.receipt["source"],
                                "spectral": actual.receipt["spectral"], "coords": actual.receipt["coords"],
                                "footprints": actual.receipt["footprints"], "grid": actual.receipt["grid_shape"],
                                "released": target, "scalar_histogram": hist, "graphs": graphs,
                                "initial_head_loss_equivalence_checks": losses_checked})

    expected_keys = set(product(roles, VIEWS, FACTORS))
    if set(counts) != expected_keys or any([row[i] for i in range(1, classes+1)] != [5]*classes for row in counts.values()):
        raise ValueError("Frozen real point populations differ across bands or scales")
    admitted.verify_authority(admitted.pin)
    source_pins = custody.protect(intake)
    if state_hash(source) != base_state or any(state_hash(value) != guards[arm] for arm, value in models.items()):
        raise ValueError("Read-only native preflight changed source state")
    params = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    proposal = draft(scene, admitted, params, train_tiles, comparisons=design)
    medians = {f"{role}/{band}/{factor}": float(np.median(values))
               for (role, band, factor), values in timings.items()}
    # An inference-only proxy records the expected number of forward calls.
    # Backward, Adam, device transfers and artifact writes remain unmeasured.
    proxy = 0.
    for case in proposal["cases"]:
        factor = case["factor"]
        proxy += sum(medians[f"devtrain/{event['band']}/{factor}"] for event in case["schedule"])
        proxy += len(case["events"]) * nval * sum(medians[f"devval/{band}/{factor}"] for band in VIEWS)
    proposal["cpu_forward_only_seconds_proxy"] = proxy
    pavia_evidence = {"design": PAVIA_DESIGN, "initial_model_seed": 64, "graph_scale": 50,
                      "initial_dev_metrics": _dev_metrics(zero_update_scores, classes),
                      "source_initialization": release.source,
                      "missing_metadata_fallback_verified": True} if pavia else {}
    return {"schema": SCHEMA, "status": "preflight_only", "scene": scene, "producer": producer,
            "bank": admitted.pin, "prior_production": prior, "protected_inputs": source_pins,
            "objective": LOSS, "support_train_seed": 512, "validation_seed": 3031,
            "point_set": {role: digest(rows) for role, rows in role_sets.items()},
            "counts": {f"{role}/{band}/{factor}": dict(sorted(row.items())) for (role,band,factor),row in counts.items()},
            "trainable_block_count": len(train_tiles), "role_blocks": {role: len(rows) for role, rows in roles.items()},
            "original_tiles_read": admitted.reads, "views": outputs, "complete_purity": None,
            "elapsed_seconds": perf_counter()-started, "draft": proposal, "forward_median_seconds": medians,
            "optimization_updates": 0, "raw_gt_loaded": False, "query_accessed": False, "labels_added": 0,
            "shared_parent_class21": release.custody.get("class_21_shared_parent"), **pavia_evidence}
