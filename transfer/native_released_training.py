"""Registered sparse-point extension to the original native-grid runner."""
from itertools import product

import numpy as np

from model.research_config import digest
from transfer import native_released_points as points, native_scale_training as training
from transfer.development_bank_training import _initial
from transfer.development_blocks import array_hash
from transfer.native_scale_fixture import BANDS, Fixture, registration
from transfer.native_scale_observation import EDGE_POLICY, FACTORS
from transfer.spatial_multiscale import initialize


FORMAT = "wp07-native-released-continuation-v1"
ARMS = ("off", "mean", "joint")
COMPARISONS = ((1, "off"), (2, "mean"), (2, "joint"), (4, "mean"), (4, "joint"))
STEPS, PREFIX, EVENTS = 6, 2, [0, 3, 6]
scores = points.scores


def materialize(fixture, event, producer):
    actual, labels, _ = fixture.materialize(event["tile"], event["factor"], event["band"],
                                             event["profile"], producer)
    h, w = labels.shape
    scarce = np.argwhere(labels == actual.classes)[0]
    # Select a neighbor even if it has a different label, so collisions
    # involving the singleton class are explicitly retained at coarse factors.
    sy, sx = (int(value) for value in scarce)
    neighbor = sx + 1 if sx % 2 == 0 and sx + 1 < w else max(0, sx - 1)
    candidates = {(sy, sx), (sy, neighbor)}
    for label in range(1, actual.classes + 1):
        y, x = np.argwhere(labels == label)[0]
        candidates.add((int(y), int(x)))
    y0, _, x0, _ = actual.lattice.box
    width = actual.lattice.scene_shape[1]
    pairs = sorted([[(y+y0)*width+x+x0, int(labels[y, x])] for y, x in candidates])
    row = {"id": event["tile"], "role": actual.receipt["role"], "box": actual.lattice.box, "samples": pairs}
    return actual, labels, points.bind(actual, row, producer, row["role"], synthetic_labels=labels)


def _key(role, event):
    return f'{role}/{event["tile"]}/{event["factor"]}/{event["band"]}/{event["profile"]}'


def _evidence(actual, labels, released):
    receipt = actual.receipt
    return {"observation": receipt["observation"], "labels": array_hash(labels),
            "area": int(released.area.sum()), "released": released.summary(),
            "bands": receipt["bands"], "grid": receipt["grid_shape"],
            "statistics": receipt["source"]["statistics"]}


def contract(classes, seed, arm, factor, producer, *, numeric=None):
    if seed not in training.SEEDS or (factor, arm) not in COMPARISONS or not producer:
        raise ValueError("Unknown released point experimental cell")
    train, val = (Fixture(classes, role) for role in ("devtrain", "devval"))
    order = np.random.default_rng(seed).permutation(train.tiles)
    schedule = [dict(step=index, tile=int(tile), factor=factor, band=BANDS[(index-1) % 2],
                     profile="a", condition="matched") for index, tile in enumerate(order, 1)]
    # Every arm shares the held development tiles, bands and point identifiers.
    evaluate = [dict(tile=tile, factor=factor, band=band, profile="b", condition="matched")
                for tile, band in product(val.tiles, BANDS)]
    roster = {}
    for role, fixture, events in (("devtrain", train, schedule), ("devval", val, evaluate)):
        for event in events:
            roster[_key(role, event)] = _evidence(*materialize(fixture, event, producer))
    base = training.baseline(classes, seed)
    model = initialize(base, arm, seed + 7321)
    return dict(format=FORMAT, producer=producer, classes=classes, seed=seed, arm=arm, factor=factor,
                supervision=points.MODE, objective=points.LOSS.copy(), training=schedule, validation=evaluate,
                steps=STEPS, prefix=PREFIX, events=EVENTS.copy(), learning_rate=0.0001, module_seed=seed+7321,
                rng_seed=seed+15000, edge=EDGE_POLICY, selection="first_min_released_point_fused_nll",
                labels_root=digest({k: value["released"]["point_set"] for k, value in roster.items()}),
                inputs=roster, fixture=registration(classes), initialization=_initial(model,base),
                parameters=sum(p.numel() for p in model.parameters()), numeric=numeric,
                raw_gt_loaded=False, query_accessed=False, real_updates=0)


def validate_observation(spec, actual, labels, released, event):
    role = actual.receipt["role"]
    if spec["inputs"][_key(role, event)] != _evidence(actual, labels, released):
        raise ValueError("Released point roster, geometry, statistics or retained band differs")


def validate_history(spec, packet):
    for role, records in (("devtrain", packet["history"]),
                          ("devval", [r for event in packet["validation"] for r in event["rows"]])):
        for row in records:
            intended = spec["inputs"][_key(role, row)]
            if any(intended[key] != row[key] for key in ("observation", "labels", "area", "released")):
                raise ValueError("Recovered released point input differs")
            if role == "devtrain" and row["supervised_area"] != intended["released"]["points"]:
                raise ValueError("Recovered released exposure count differs")
