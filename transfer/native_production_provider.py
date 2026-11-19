"""Role-bound provider for the existing native optimizer, selection and recovery."""

from copy import deepcopy
from dataclasses import asdict
from itertools import product

import numpy as np

from evaluation.frozen_training_stages import pinned, require
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from transfer import native_production_fixture as fixture, native_production_model as candidate
from transfer.development_bank_training import _baseline
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_released_points import LOSS, MODE, scores
from transfer.spatial_coverage_protocol import bank_view, model_from_source, retained_positions

FORMAT = "wp05_wp06_production_native_training_v1"


class ProductionProvider:
    scores = staticmethod(scores)

    def __init__(self, root, bank, roles, *, axis, config, source, seed, budget, producer, approval=None,
                 delivered_axis=False):
        bank.verify_authority(bank.pin)
        self.root, self.bank, self.roles = root, bank, deepcopy(roles)
        require(set(roles) == {"support", "calibration", "evaluation"} and roles["support"] and roles["calibration"],
                "Production needs explicit support and calibration spatial roles")
        self.axis = tuple(axis)
        self.config, self.source, self.seed, self.budget, self.producer = config, source, seed, budget, producer
        self.is_synthetic = bank.release.profile == "synthetic"
        self.scene, self.classes = bank.release.scene, bank.release.classes
        self.approval = approval
        require(self.is_synthetic == (approval is None),
                "A real production provider needs separate signed case approval")
        require((self.axis == bank.release.bands.band_ids if delivered_axis else
                 self.axis == tuple(sorted(set(self.axis))) and set(bank.release.bands.band_ids).issubset(self.axis)),
                "Registered axis omits an available original band or changes delivered order")
        self._cache, self._roster = {}, None

    def baseline(self, seed):
        base = _baseline(self.bank.release, seed)
        if self.source is not None:
            path = pinned(self.root, self.source["weights"])
            metadata = pinned(self.root, self.source["metadata"])
            require(metadata == path.with_suffix(".pt.json"), "Source sidecar and weights are not a checkpoint pair")
            original = load_model_checkpoint(path, "cpu")
            require(state_hash(original) == self.source["state"], "Source checkpoint state differs")
            model_from_source(original, base, self.scene, native_source=original.backbone.graph_mode == "learned")
        return base

    def model_factory(self, source, arm, seed):
        require(arm == self.config.arm, "Production model and registration switches differ")
        return candidate.configured_model(source, self.config, len(self.axis), seed)

    @staticmethod
    def key(role, event):
        return f'{role}/{event["tile"]}/{event["factor"]}/{event["band"]}'

    @staticmethod
    def evidence(actual, labels, targets):
        raw = actual.receipt
        return dict(observation=raw["observation"], labels=array_hash(labels), retained_ids=raw["source"]["band_ids"],
                    retained_stats=raw["source"]["retained_statistics"], spectral=raw["spectral"], physical=raw["physical"],
                    released=targets.summary(), grid=raw["grid_shape"])

    def make_observation(self, role, event):
        row = self.bank.role_row(role, event["tile"])
        actual, labels, targets = bank_view(self.bank, row, self.producer, event["band"], factor=event["factor"])
        master = self.bank.release.bands
        expected = [master.band_ids[i] for i in retained_positions(master, event["band"])]
        require(list(actual.view.bands.band_ids) == actual.receipt["source"]["band_ids"] == expected,
                "Materialized spectral stream did not remove the registered original columns")
        return actual, labels, targets

    def materialize(self, role, event, producer):
        purpose = {"devtrain": "support", "devval": "calibration"}[role]
        require(producer == self.producer and event["tile"] in self.roles[purpose]
                and event["factor"] in (1,2) and event["band"] in (*fixture.VIEWS, "missing"),
                "Optimizer or selection attempted to consume an unauthorized role or view")
        key = self.key(role, event)
        if key not in self._cache:
            self._cache[key] = self.make_observation(role, event)
        return self._cache[key]

    def validate_observation(self, spec, actual, labels, targets, event):
        require(self.evidence(actual, labels, targets) == spec["inputs"][self.key(actual.receipt["role"],event)],
                "Production role, retained tensors or registered labels changed")

    def stage_evidence(self, actual):
        value = actual.receipt
        return dict(retained_ids=value["source"]["band_ids"], retained_stats=value["source"]["retained_statistics"],
                    spectral=value["spectral"], grid=value["grid_shape"])

    def registered_forward(self, model, actual, role, execution):
        return candidate.run_forward(model, actual, self.producer, role, self.config, self.axis, execution=execution)

    def validate_history(self, spec, packet):
        for role, rows in (("devtrain", packet["history"]),
                           ("devval", [row for event in packet["validation"] for row in event["rows"]])):
            for row in rows:
                ref = spec["inputs"][self.key(role,row)]
                require(all(row[key] == ref[key] for key in ("observation","labels","retained_ids",
                        "retained_stats","spectral","grid","released")), "Resumed retained exposure differs")
                if role == "devval":
                    require(all(head["count"] == ref["released"]["points"] for head in row["scores"].values()),
                            "Resumed model selection support differs")

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        require((classes,seed,arm,factor,producer) == (self.classes,self.seed,self.config.arm,1,self.producer),
                "Production run parameters differ from case admission")
        tiles = self.roles["support"]
        rng = np.random.default_rng(seed)
        train = []
        slots = ((1,"full"), (1,"central_gap"), (1,"full"), (2,"central_gap"),
                 (1,"matched_dispersed"), (1,"full"))
        while len(train) < self.budget:
            epoch = len(train)//len(tiles)
            for tile in rng.permutation(tiles).tolist():
                scale, band = slots[(epoch+tiles.index(tile))%len(slots)]
                train.append(dict(tile=tile, step=len(train)+1, factor=scale, band=band, condition="registered"))
        train = train[:self.budget]
        val = [dict(tile=tile, factor=1, band="full", condition="registered")
               for tile in self.roles["calibration"]]
        if self._roster is None:
            self._roster = {}
            for role, events in (("devtrain", train), ("devval",val)):
                for event in events:
                    key = self.key(role,event)
                    if key not in self._roster:
                        self._roster[key] = self.evidence(*self.materialize(role,event,producer))
        base = self.baseline(seed)
        model = self.model_factory(base, arm, seed+7321)
        state = state_hash(model)
        events = sorted({0, self.budget//3, 2*self.budget//3, self.budget})
        return dict(format=FORMAT, producer=producer, classes=classes, seed=seed, arm=arm, factor=1,
            supervision=MODE, objective=LOSS, training=train, validation=val, inputs=deepcopy(self._roster),
            steps=self.budget, prefix=self.budget//2, events=events, learning_rate=.002,
            module_seed=seed+7321, rng_seed=seed+15000, switches=asdict(self.config), axis=list(self.axis),
            initialization=dict(state=state, source=None if self.source is None else self.source["state"], fresh={}),
            parameters=sum(p.numel() for p in model.parameters()), bank=self.bank.pin, numeric=numeric,
            roles=digest(self.roles), selection="first_min_released_point_fused_nll",
            scope="synthetic_only" if self.is_synthetic else "externally_approved_real",
            query_accessed=False, raw_gt_loaded=False, real_updates=0 if self.is_synthetic else self.budget,
            frozen_stage_protocol=FORMAT, authority=self.approval)

    def confirmation(self, checkpoint, tile, scale, band):
        require(bool(checkpoint) and tile in self.roles["evaluation"], "Confirmation role is not an optimizer input")
        return self.make_observation("devval", dict(tile=tile,factor=scale,band=band))
