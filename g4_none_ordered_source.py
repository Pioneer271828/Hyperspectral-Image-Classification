"""Explicit ordered-validation none controls; legacy source defaults stay native."""
import copy
import json
from pathlib import Path

from joint_training import validation_epochs, validation_snapshot
from m4_ordered_source import OrderedValidation, POLICY
from source_pretrain import EXPECTED as NATIVE_STUDY, full_settings as native_full_settings, SourcePretrainRuntime
from source_training import source_settings

EXPECTED = {**copy.deepcopy(NATIVE_STUDY), "protocol": "g4_none_source_pretrain_full_v2",
            "validation_arithmetic": copy.deepcopy(POLICY)}


def load_study(path="configs/experiments/none_source_ordered_full.json"):
    value = json.loads(Path(path).read_text())
    if value != EXPECTED:
        raise ValueError("Modified none source control study")
    return value


def options(held_out, seed=64, *, full=False):
    if type(full) is not bool:
        raise ValueError("Explicit engineering/full source profile required")
    value = native_full_settings(held_out, seed) if full else source_settings(held_out, seed=seed)
    value["protocol"] = EXPECTED["protocol"] if full else "g4_none_source_ordered_engineering_v2"
    value["source_validation_arithmetic"] = copy.deepcopy(POLICY)
    if full:
        value["pretraining_study"] = copy.deepcopy(EXPECTED)
    if value["config"]["model"]["region_residual_mode"] != "none":
        raise ValueError("Only the unchanged none architecture is a source control")
    return value


def select_case(launch, producer, held_out, seed):
    if (launch["code"] != producer or producer["dirty"] or launch["study"] != EXPECTED
            or launch["expected_updates"] != 9018 or launch["test_scoring"] is not False):
        raise ValueError("Frozen none source launch/producer differs")
    expected = {(held, s) for held in EXPECTED["held_targets"] for s in EXPECTED["training_seeds"]}
    cases = launch["cases"]
    if (len(cases) != 9 or {(c["held_out"], c["seed"]) for c in cases} != expected
            or len({c["out"] for c in cases}) != 9 or type(seed) is not int or (held_out, seed) not in expected):
        raise ValueError("Nine distinct full source controls required")
    for case in cases:
        if type(case["seed"]) is not int or case["settings"] != options(case["held_out"], case["seed"], full=True):
            raise ValueError("Frozen none source settings differ")
    return next(c for c in cases if (c["held_out"], c["seed"]) == (held_out, seed))


class NoneOrderedSourceRuntime(SourcePretrainRuntime):
    def validate(self, settings):
        expected = options(settings.get("held_out"), settings.get("model_seed"), full=settings.get("profile") == "full")
        value = {k: v for k, v in settings.items() if k not in ("executed_optimizer_steps", "selected_step")}
        if value != expected:
            raise ValueError("Explicit none source control protocol differs")
        return True

    @staticmethod
    def validation_snapshot(model, items):
        with OrderedValidation():
            return validation_snapshot(model, items)

    def checkpoint_metadata(self, model):
        return {"source_validation_arithmetic": copy.deepcopy(POLICY)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        meta = model.training_config
        settings = meta["joint_runtime"]
        steps = {(e + 1) * settings["steps_per_epoch"] for e in validation_epochs(settings)}
        if (meta.get("source_validation_arithmetic") != POLICY
                or settings.get("executed_optimizer_steps") != settings["optimizer_steps"]
                or settings.get("selected_step") not in steps
                or meta["checkpoint_optimizer_step"] != settings["selected_step"]):
            raise ValueError("None source checkpoint arithmetic/budget/selection differs")
        return model
