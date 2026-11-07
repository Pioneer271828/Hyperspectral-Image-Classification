"""Bounded LP-FT consumer of a frozen G4 source/target case, without query.

At most twelve updates for engineering comparison. The original losses, input
loader and full-source initializer are reused. Initial ordered BN calibration
is followed by frozen statistics in both phases; no candidate recalibration or
model selection is performed here. A full experiment needs separate registration.
"""
import copy
from pathlib import Path

import torch

from g4_matched_target_execution import case_data, unchanged_producer
from g4_matched_target_study import METHODS, initialize, select_case
from g4_support_context import check_files, read
from g4_support_target import source_pins
from m4_ordered_source import OrderedValidation
from model.research_config import ResearchConfig, digest
from research_runtime import code_snapshot, require_frozen_code
from target_bn_calibration import calibrate_target, POLICY as BN_POLICY, CONTEXT, GRAPH_POLICY
from target_initialization import state_id
from target_training import (TargetRuntime, features, loss_terms, observation, read_rows,
                             selected_outputs)

from .runtime import LPFTRun


REFERENCE_LAUNCH_SHA256 = "0c447087a9679538e0a792aa72cb033197e743bc959cd096ac2e831e46eafdb1"
POLICY = "side_lpft_g4_short_v2"


class G4ShortCase:
    """One explicit target/seed/architecture from the frozen full-source launch.

    Construction reads only the accepted source and sanitized support/validation
    context. It takes no update and writes no files. The caller owns output
    publication and authenticates any snapshot before restore_snapshot().
    """

    def __init__(self, launch_path, launch_sha256, *, expected_commit, target, seed,
                 mode, plan, device="cpu", saved_tensors="device"):
        if saved_tensors not in ("device", "cpu"):
            raise ValueError("saved_tensors must be device or cpu")
        if plan.total_steps > 12:
            raise ValueError("Short LP-FT consumer permits at most twelve updates")
        self.code = code_snapshot()
        if self.code["dirty"] or self.code["commit"] != expected_commit:
            raise ValueError("Clean expected consumer commit required")
        path = str(Path(launch_path).resolve(strict=True))
        if launch_sha256 != REFERENCE_LAUNCH_SHA256:
            raise ValueError("Frozen Issue184 launch required")
        self.pins = {path: launch_sha256}
        check_files(self.pins)
        launch = read(path)
        unchanged_producer(launch["code"], self.code)
        self.reference = select_case(launch, launch["code"], target, seed, METHODS[1], mode,
                                     phase="full_campaign")
        self.items, provenance, self._data_guard = case_data(self.reference)
        self.target, self.item = target, self.items[target]
        # Reuse the exact existing label-budget/field/overlap validator.
        validator = object.__new__(TargetRuntime)
        validator.settings = self.reference["settings"]
        TargetRuntime.validate_inputs(validator, self.items)
        for group in (provenance["protected_files"], source_pins(self.reference["source_spec"])):
            for name, sha in group.items():
                if self.pins.setdefault(name, sha) != sha:
                    raise ValueError("Conflicting LP-FT input identity")
        self.config = ResearchConfig.from_dict(self.reference["settings"]["config"])
        model, initialization = initialize(target, seed, METHODS[1], mode,
                                           source_spec=self.reference["source_spec"])
        if (initialization != self.reference["initialization"] or
                state_id(model) != self.reference["initial_state_sha256"]):
            raise ValueError("Matched source and fresh target initialization changed")
        model = model.to(device).eval()
        with OrderedValidation():
            model, calibration = calibrate_target(model, observation(model, self.item),
                mode=BN_POLICY, context=CONTEXT, graph_policy=GRAPH_POLICY,
                source_state_sha256=state_id(model))
        self.record = {"policy": POLICY, "target": target, "seed": seed, "mode": mode,
                       "saved_tensors": saved_tensors,
                       "reference_case_sha256": digest(self.reference),
                       "launch_sha256": launch_sha256, "plan": plan.contract(),
                       "initialization": initialization, "initial_calibration": calibration,
                       "calibrated_initial_state_sha256": state_id(model),
                       "code": self.code, "source_and_context_sha256": digest(self.pins),
                       "selection": "none_short_endpoint_only",
                       "bn_policy": "initial_ordered_then_frozen_both_phases",
                       "query_labels_accessed": False, "full_experiment": False}
        identity = {"reference": digest(self.reference), "protected_inputs": digest(self.pins),
                    "initial_calibrated_model": state_id(model), "consumer": self.code["source_hash"]}
        self._template = model
        self.run = LPFTRun(model, plan, identity, saved_tensors=saved_tensors)
        self.guard()

    def guard(self):
        self._data_guard()
        check_files(self.pins)
        require_frozen_code(self.code)

    def step(self):
        self.guard()
        terms = {}
        phase = self.run.plan.phase(self.run.completed_steps)
        rates = [g["lr"] for g in self.run.optimizer.param_groups]

        def objective(model):
            outputs = selected_outputs(model, self.target, self.item, "train")
            terms.update(loss_terms(outputs, self.item["train_labels"].to(outputs[0].device), self.config))
            return terms["total"]

        self.run.step(objective)
        self.guard()
        return {"completed_steps": self.run.completed_steps, "update_phase": phase,
                "learning_rates": rates, "loss": {k: float(v.detach()) for k, v in terms.items()}}

    def snapshot(self):
        self.guard()
        return {"case": copy.deepcopy(self.record), "runtime": self.run.snapshot()}

    def restore_snapshot(self, snapshot):
        self.guard()
        if (not isinstance(snapshot, dict) or set(snapshot) != {"case", "runtime"} or
                snapshot["case"] != self.record):
            raise ValueError("LP-FT case snapshot belongs to another run")
        restored = LPFTRun.restore(self._template, self.run.plan, self.run.input_sha256,
                                   snapshot["runtime"], saved_tensors=self.run.saved_tensors)
        self.run = restored
        self.guard()

    def endpoint_metrics(self):
        """Read train/validation endpoints; never select a model or release query."""
        self.guard()
        self.run._validate_live()
        before = state_id(self.run.model)
        metrics = {}
        with OrderedValidation(), torch.no_grad():
            full = features(self.run.model, self.target, self.item)
            for phase in ("train", "val"):
                rows = tuple(v[self.item[phase + "_indices"].to(v.device)] for v in full)
                outputs = selected_outputs(self.run.model, self.target, self.item, phase, rows)
                metrics[phase], _ = read_rows(outputs, self.item[phase + "_labels"], self.item["class_num"])
        if before != state_id(self.run.model):
            raise ValueError("Endpoint metrics changed model state")
        self.guard()
        return {"case": copy.deepcopy(self.record), "completed_steps": self.run.completed_steps,
                "metrics": metrics, "model_state_sha256": before,
                "full_training_completed": False, "accuracy_gain_established": False}
