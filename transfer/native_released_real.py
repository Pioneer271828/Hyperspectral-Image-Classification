"""A future released-point runner bound to the exact reviewed real draft."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from transfer.development_bank_training import _initial
from transfer.development_block_probe import VIEWS
from transfer.development_blocks import read_pin
from transfer.g3_training import state_hash
from transfer.native_released_points import LOSS, MODE, bind, scores
from transfer.native_released_preflight import BANK_ROOTS, SCHEMA, open_bank
from transfer.native_released_training import COMPARISONS
from transfer.native_scale_observation import EDGE_POLICY, from_bank
from transfer import p4_runtime_device as p4
from transfer.spatial_multiscale import initialize


FORMAT = "wp07-native-real-released-continuation-v1"
RELEASE = "wp07_native_released_execution_release_v1"


class RealReleasedRun:
    """Construct only after a separate approval pins one preflight case and destination."""
    scores = staticmethod(scores)

    def __init__(self, root, preflight, case_id, release, destination, code):
        root = Path(root).resolve(strict=True)
        require_frozen_code(code)
        if release is None:
            raise ValueError("Real sparse optimization requires a separately reviewed execution release")
        destination = Path(destination)
        if destination != destination.resolve() or not destination.is_relative_to(root) or destination == root:
            raise ValueError("Real sparse execution must be a specific external run directory")
        approval = json.loads(read_pin(release, root))
        receipt = json.loads(read_pin(preflight, root))
        scene = receipt.get("scene")
        if (scene not in BANK_ROOTS or receipt.get("bank", {}).get("sha256") != BANK_ROOTS[scene]
                or receipt.get("schema") != SCHEMA or receipt.get("objective") != LOSS
                or receipt.get("status") != "preflight_only" or receipt.get("optimization_updates") != 0
                or receipt.get("query_accessed") is not False or receipt.get("raw_gt_loaded") is not False):
            raise ValueError("Real sparse preflight has unrelated provenance")
        cases = [row for row in receipt["draft"]["cases"] if row["id"] == case_id]
        if len(cases) != 1 or receipt["draft"]["authorized_for_training"] is not False:
            raise ValueError("Expected the one unreleased, preregistered real case")
        case = cases[0]
        expected = {"schema": RELEASE, "decision": "approved", "source_commit": code["commit"],
                    "source_hash": code["source_hash"], "preflight": preflight, "case_sha256": digest(case),
                    "destination": str(destination), "optimizer_updates": case["planned_updates"],
                    "physical_uuid": approval.get("physical_uuid"), "protocol": approval.get("protocol"),
                    "raw_gt_authorized": False, "query_authorized": False}
        uuid = expected["physical_uuid"]
        if (approval != expected or (uuid is not None and (type(uuid) is not str or p4.UUID.fullmatch(uuid) is None))
                or expected["protocol"] != (p4.CPU if uuid is None else p4.CUDA_FP64_WEIGHT)):
            raise ValueError("Real sparse execution approval does not bind this source, case or budget")
        self._bank, prior, self._intake = open_bank(root, scene)
        if self._bank.pin != receipt["bank"] or prior != receipt["prior_production"]:
            raise ValueError("Approved draft differs from the original bank authority")
        self.physical_uuid = uuid
        self._protocol = expected["protocol"]
        self._preflight = deepcopy(preflight)
        self._release = deepcopy(release)
        self._destination = destination
        self._root = root
        self._producer = code["commit"]
        self._source_hash = code["source_hash"]
        self._case = deepcopy(case)
        self._views = {(row["role"], row["tile"], row["band"], row["factor"]): deepcopy(row) for row in receipt["views"]}
        self._point_set = deepcopy(receipt["point_set"])

    def baseline(self, seed):
        # Preserve shared, normalized source tensors. Reset only the existing
        # dataset head to give independent initialization per registered seed.
        source = deepcopy(self._intake.baseline)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            source.heads[self._bank.release.scene].reset_parameters()
        return source

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        code = code_snapshot()
        require_frozen_code(code)
        release = self._bank.release
        expected = (release.classes, self._case["model_seed"], self._case["arm"], self._case["factor"], self._producer)
        self._bank.verify_authority(self._bank.pin)
        read_pin(self._release, self._root)
        read_pin(self._preflight, self._root)
        if ((classes, seed, arm, factor, producer) != expected or (factor, arm) not in COMPARISONS
                or code["source_hash"] != self._source_hash or code["commit"] != producer
                or numeric is None or numeric["physical_uuid"] != self.physical_uuid
                or numeric["protocol"] != self._protocol):
            raise ValueError("Real sparse case or source changed")
        train = [dict(**row, step=i, factor=factor, profile=None, condition="missing")
                 for i, row in enumerate(self._case["schedule"], 1)]
        tile_ids = [row["id"] for row in self._bank.role_rows("devval")]
        val = [dict(tile=tile, band=band, factor=factor, profile=None, condition="missing")
               for tile in tile_ids for band in VIEWS]
        steps = self._case["planned_updates"]
        if len(train) != steps or len({(row["tile"], row["band"]) for row in train}) != steps:
            raise ValueError("Real sparse update schedule duplicates a tile/view or differs from its budget")
        model = self.baseline(seed)
        initial = initialize(model, arm, seed + 7321)
        return {"format": FORMAT, "producer": producer, "classes": classes, "seed": seed, "arm": arm,
                "factor": factor, "supervision": MODE, "objective": LOSS, "training": train, "validation": val,
                "steps": steps, "prefix": self._case["events"][1], "events": self._case["events"],
                "learning_rate": 0.0001, "module_seed": seed+7321, "rng_seed": seed+15000,
                "initialization": _initial(initial, model), "parameters": sum(p.numel() for p in initial.parameters()),
                "edge": EDGE_POLICY, "selection": "first_min_released_point_fused_nll", "numeric": numeric,
                "bank": self._bank.pin, "point_set": self._point_set, "preflight": self._preflight,
                "release": self._release, "destination": str(self._destination), "source_state": state_hash(model)}

    def materialize(self, role, event, producer):
        self._bank.verify_authority(self._bank.pin)
        actual, row = from_bank(self._bank, role, event["tile"], VIEWS[event["band"]], event["factor"], producer)
        points = bind(actual, row, producer, role)
        # Only original released labels are exposed to the generic runner.
        return actual, np.asarray(points.points, dtype=np.int64).reshape((-1, 2)), points

    def validate_observation(self, spec, actual, labels, targets, event):
        receipt = actual.receipt
        reference = self._views[receipt["role"], event["tile"], event["band"], event["factor"]]
        if (reference["released"] != targets.summary() or reference["source"] != receipt["source"]
                or reference["footprints"] != receipt["footprints"] or reference["grid"] != receipt["grid_shape"]
                or reference["spectral"] != receipt["spectral"] or reference["coords"] != receipt["coords"]):
            raise ValueError("Real released native grid, point mapping or band statistics changed")

    def validate_history(self, spec, packet):
        for role, records in (("devtrain", packet["history"]),
                              ("devval", [row for event in packet["validation"] for row in event["rows"]])):
            for row in records:
                reference = self._views[role, row["tile"], row["band"], row["factor"]]["released"]
                if row["released"] != reference or (role == "devtrain" and row["supervised_area"] != reference["points"]):
                    raise ValueError("Real sparse checkpoint exposure changed")

    def check_destination(self, path):
        path = Path(path)
        if path.parent != self._destination or path.resolve() != path:
            raise ValueError("Real sparse checkpoint escaped its approved directory")
