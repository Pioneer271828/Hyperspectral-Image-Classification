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


FORMAT = "wp07-native-real-released-continuation-v2"
RELEASE = "wp07_native_released_execution_release_v2"
ENGINEERING = "wp07_native_released_bank_fixture_v1"
ENGINEERING_RELEASE = "wp07_native_released_fixture_release_v1"


DUTY = "wp07_native_released_duties_v1"
DUTY_COMPARISONS = ((1, "off"), *((factor, arm) for factor in (2, 4) for arm in
                    ("off", "fixed", "mean", "content", "metadata", "joint", "cnn_content", "graph_content", "constant_joint")))


def comparisons(design):
    if design == DUTY:
        return DUTY_COMPARISONS
    if design is None:
        return COMPARISONS
    raise ValueError("Unknown released comparison design")


class RealReleasedRun:
    """Construct only after a separate approval pins one preflight case and destination."""
    scores = staticmethod(scores)

    def __init__(self, root, preflight, case_id, release, destination, code, *, intake=None):
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
        engineering = intake is not None
        schema = ENGINEERING if engineering else SCHEMA
        self.design = receipt.get("design")
        allowed_comparisons = comparisons(self.design)
        if (scene not in BANK_ROOTS or (not engineering and receipt.get("bank", {}).get("sha256") != BANK_ROOTS[scene])
                or receipt.get("schema") != schema or receipt.get("objective") != LOSS
                or receipt.get("synthetic", False) is not engineering
                or receipt.get("status") != "preflight_only" or receipt.get("optimization_updates") != 0
                or receipt.get("query_accessed") is not False or receipt.get("raw_gt_loaded") is not False):
            raise ValueError("Real sparse preflight has unrelated provenance")
        cases = [row for row in receipt["draft"]["cases"] if row["id"] == case_id]
        if len(cases) != 1 or receipt["draft"]["authorized_for_training"] is not False:
            raise ValueError("Expected the one unreleased, preregistered real case")
        case = cases[0]
        if (case["factor"], case["arm"]) not in allowed_comparisons:
            raise ValueError("Released case is outside its registered design")
        expected = {"schema": ENGINEERING_RELEASE if engineering else RELEASE,
                    "decision": "approved", "source_commit": code["commit"],
                    "source_hash": code["source_hash"], "preflight": preflight, "case_sha256": digest(case),
                    "destination": str(destination), "optimizer_updates": case["planned_updates"],
                    "physical_uuid": approval.get("physical_uuid"), "protocol": approval.get("protocol"),
                    "route": approval.get("route"),
                    "raw_gt_authorized": False, "query_authorized": False}
        uuid = expected["physical_uuid"]
        if (approval != expected or expected["route"] not in ("full", "split") or (uuid is not None and (type(uuid) is not str or p4.UUID.fullmatch(uuid) is None))
                or expected["protocol"] != (p4.CPU if uuid is None else p4.CUDA_FP64_WEIGHT)):
            raise ValueError("Real sparse execution approval does not bind this source, case or budget")
        read_bank = intake if engineering else open_bank
        self._bank, prior, self._intake = read_bank(root, scene)
        if (self._bank.pin != receipt["bank"] or prior != receipt["prior_production"]
                or (self._bank.release.profile == "synthetic") != engineering):
            raise ValueError("Approved draft differs from the original bank authority")
        self.engineering = engineering
        self.classes = self._bank.release.classes
        self.physical_uuid = uuid
        self.route = expected["route"]
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
        self._observation_producer = receipt["producer"]
        if len(self._views) != len(receipt["views"]):
            raise ValueError("Preflight contains repeated view identities")
        expected_points = {role: digest(sorted(point for row in self._bank.role_rows(role) for point in row["samples"]))
                           for role in ("devtrain", "devval")}
        if self._point_set != expected_points:
            raise ValueError("Direct bank release identities differ from the draft")

    @property
    def case(self):
        return deepcopy(self._case)

    def baseline(self, seed):
        # Preserve shared, normalized source tensors. Reset only the existing
        # dataset head to give independent initialization per registered seed.
        source = deepcopy(self._intake.baseline)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for heads in (source.heads, source.hgnn_heads, source.cnn_heads):
                heads[self._bank.release.scene].reset_parameters()
        return source

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        code = code_snapshot()
        require_frozen_code(code)
        release = self._bank.release
        expected = (release.classes, self._case["model_seed"], self._case["arm"], self._case["factor"], self._producer)
        self._bank.verify_authority(self._bank.pin)
        read_pin(self._release, self._root)
        read_pin(self._preflight, self._root)
        if ((classes, seed, arm, factor, producer) != expected or (factor, arm) not in comparisons(self.design)
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
        authorized = {(row["id"], band) for row in self._bank.role_rows("devtrain") if row["samples"] for band in VIEWS}
        if len(train) != steps or {(row["tile"], row["band"]) for row in train} != authorized or len(authorized) != steps:
            raise ValueError("Schedule does not cover exactly the authorized labeled tiles and band views")
        if self._case["events"] != [0, steps//2, steps] or steps % 2:
            raise ValueError("Real sparse selection clock differs from the frozen half/end budget")
        model = self.baseline(seed)
        initial = self.model_factory(model, arm, seed + 7321)
        extra = {"registered_graph_scale": 100, "design": self.design,
                 "selector_input": "control_descriptor" if arm == "constant_joint" else "missing"} if self.design else {}
        return {"format": FORMAT, "producer": producer, "classes": classes, "seed": seed, "arm": arm,
                "factor": factor, "supervision": MODE, "objective": LOSS, "training": train, "validation": val,
                "steps": steps, "prefix": self._case["events"][1], "events": self._case["events"],
                "learning_rate": 0.0001, "module_seed": seed+7321, "rng_seed": seed+15000,
                "initialization": _initial(initial, model), "parameters": sum(p.numel() for p in initial.parameters()),
                "edge": EDGE_POLICY, "synthetic": self.engineering, "selection": "first_min_released_point_fused_nll", "numeric": numeric,
                "bank": self._bank.pin, "point_set": self._point_set, "preflight": self._preflight,
                "release": self._release, "route": self.route, "destination": str(self._destination), "source_state": state_hash(model),
                **extra}

    @staticmethod
    def model_factory(baseline, arm, seed):
        return initialize(baseline, "joint" if arm == "constant_joint" else arm, seed)

    def selection_condition(self, actual, spec):
        if spec.get("selector_input") == "control_descriptor":
            from transfer.native_released_capacity import condition_on_bank

            return condition_on_bank(actual, "constant_joint")
        return None

    def materialize(self, role, event, producer):
        self._bank.verify_authority(self._bank.pin)
        actual, row = from_bank(self._bank, role, event["tile"], VIEWS[event["band"]], event["factor"], producer,
                                engineering=self.engineering)
        points = bind(actual, row, producer, role)
        # Only original released labels are exposed to the generic runner.
        return actual, np.asarray(points.points, dtype=np.int64).reshape((-1, 2)), points

    def validate_observation(self, spec, actual, labels, targets, event):
        receipt = actual.receipt
        reference = self._views[receipt["role"], event["tile"], event["band"], event["factor"]]
        if (reference["released"] != targets.summary() or reference["source"] != receipt["source"]
                or reference["footprints"] != receipt["footprints"] or reference["grid"] != receipt["grid_shape"]
                or reference["spectral"] != receipt["spectral"] or reference["coords"] != receipt["coords"]
                or reference["observation"] != self._historical_observation(receipt)):
            raise ValueError("Real released native grid, point mapping or band statistics changed")

    def _historical_observation(self, receipt):
        return digest({**{key: value for key, value in receipt.items() if key != "observation"},
                       "producer": self._observation_producer})

    def validate_history(self, spec, packet):
        for role, records in (("devtrain", packet["history"]),
                              ("devval", [row for event in packet["validation"] for row in event["rows"]])):
            for row in records:
                reference = self._views[role, row["tile"], row["band"], row["factor"]]
                historical = reference["released"]
                if (row["released"] != historical or row["area"] != historical["unknown_pixels"] + historical["points"]
                        or row.get("historical_observation") != reference["observation"] or row["graph"]["q_shape"][0]
                        != int(np.prod(reference["grid"]))
                        or (role == "devtrain" and row["supervised_area"] != historical["points"])):
                    raise ValueError("Real sparse checkpoint exposure changed")

    def stage_evidence(self, actual):
        return {"historical_observation": self._historical_observation(actual.receipt)}

    def check_destination(self, path):
        path = Path(path)
        if path.parent != self._destination or path.resolve() != path:
            raise ValueError("Real sparse checkpoint escaped its approved directory")
