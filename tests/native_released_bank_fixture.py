"""Synthetic, versioned bank for the production-entry integration test."""
from collections import Counter
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np

from model.research_config import digest
from transfer import development_bank as bank, development_bank_training as original
from transfer import native_released_real as real, registered_development as transport
from transfer.development_block_probe import VIEWS
from transfer.native_released_points import LOSS, bind
from transfer.native_released_training import COMPARISONS
from transfer.native_scale_observation import FACTORS, from_bank
from transfer.native_scale_training import SEEDS


def fixture(root, code, *, design=None):
    receipts, intakes = {}, {}
    for classes, scene in ((9, "xuzhou"), (22, "whu-honghu")):
        release, arrays = original.fixture(classes)
        # Retain only explicit points. Duplicate the class-1 point at a distinct
        # pixel in its first 4x4 cell, yielding a real count-weighted minority.
        rows = [dict(row, samples=[list(p) for p in row["samples"]]) for row in release.rows]
        for row in rows:
            ids = {p[0] for p in row["samples"]}
            low = row["box"][0]*release.shape[1]+row["box"][2]
            for cell in (low+1, low+release.shape[1]):
                if cell not in ids:
                    row["samples"].append([cell, 1])
                    row["samples"].sort()
                    break
        release = replace(release, rows=tuple(rows))
        # Real bank transport performs the same hashing, geometry and decode.
        pin = bank.publish(root / f"bank-{classes}", root, release, lambda row: arrays[row["id"]])
        bank_pin = {key: pin[key] for key in ("root", "sha256", "registration")}
        prior = transport._json_new(root / f"source-{classes}.json",
            {"synthetic": True, "scope": real.ENGINEERING, "bank": bank_pin})
        baseline = original._baseline(release, 81271)
        intakes[scene] = (bank_pin, release, prior, baseline)
        admitted = bank.Bank(bank_pin, root, release)
        train = [row["id"] for row in admitted.role_rows("devtrain") if row["samples"]]
        cases = []
        for seed in SEEDS:
            order = [int(i) for i in np.random.default_rng(seed).permutation(train)]
            schedule = [{"tile": tile, "band": band} for band in VIEWS for tile in order]
            counts = Counter(label for event in schedule
                             for _, label in admitted.role_row("devtrain", event["tile"])["samples"])
            for factor, arm in real.comparisons(design):
                cases.append({"id": f"{scene}/{seed}/{factor}/{arm}", "model_seed": seed,
                              "factor": factor, "arm": arm, "planned_updates": len(schedule),
                              "train_exposures_per_class": [counts[i] for i in range(1, classes+1)],
                              "schedule": schedule, "events": [0, len(order), len(schedule)]})
        views, producer = [], "synthetic-prefrozen-bank-fixture"
        point_set = {role: digest(sorted(p for row in admitted.role_rows(role) for p in row["samples"]))
                     for role in ("devtrain", "devval")}
        for role, band, factor in product(point_set, VIEWS, FACTORS):
            for row in admitted.role_rows(role):
                actual, authority = from_bank(admitted, role, row["id"], VIEWS[band], factor, producer, engineering=True)
                targets = bind(actual, authority, producer, role)
                obs = actual.receipt
                views.append({"role": role, "tile": row["id"], "band": band, "factor": factor,
                              "released": targets.summary(), "source": obs["source"], "observation": obs["observation"],
                              "spectral": obs["spectral"], "coords": obs["coords"], "footprints": obs["footprints"],
                              "grid": obs["grid_shape"]})
        receipts[scene] = transport._json_new(root/f"preflight-{classes}.json", {
            "schema": real.ENGINEERING, "synthetic": True, "status": "preflight_only", "producer": producer,
            "optimization_updates": 0, "query_accessed": False, "raw_gt_loaded": False,
            "scene": scene, "objective": LOSS, "bank": bank_pin, "point_set": point_set, "design": design,
            "prior_production": prior, "views": views,
            "draft": {"cases": cases, "authorized_for_training": False}})
    def open_fixture(authorized, scene):
        if authorized != root:
            raise ValueError("Wrong authorized fixture location")
        pin, release, prior, model = intakes[scene]
        return bank.Bank(pin, root, release), prior, SimpleNamespace(baseline=model)
    return receipts, open_fixture
