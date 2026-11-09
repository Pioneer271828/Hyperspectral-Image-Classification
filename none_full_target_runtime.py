"""Explicit ordered target protocol for the accepted full none source controls."""
import copy
from loadData.dataset_registry import DATASETS
from m4_full_target_runtime import TARGET_ARITHMETIC
from m4_ordered_source import OrderedValidation
from none_full_transfer import SOURCE_BINDING, initialize_none_full_target
from target_initialization import METHODS, state_id
from target_training import TargetRuntime, target_settings


def settings(target, method, seed=64, *, full=False, device="cpu"):
    if method not in METHODS[:2] or type(full) is not bool:
        raise ValueError("Explicit none scratch/full-finetune target profile required")
    value = target_settings(target, method, seed=seed, support_seed=64, per_class=10, device=device)
    value.update(protocol="g4_none_full_source_target_" + ("full" if full else "engineering") + "_v2",
                 full_source_initialization=SOURCE_BINDING,
                 target_validation_arithmetic=copy.deepcopy(TARGET_ARITHMETIC))
    if full:
        value.update(profile="full", epochs=1499, val_interval=30, optimizer_steps=1500)
        value["config"]["training"].update(epochs=1499, val_interval=30)
    return value


class NoneFullSourceTargetRuntime(TargetRuntime):
    def __init__(self, *args, initial_state_sha256=None, **kwargs):
        self.expected_initial_state = initial_state_sha256
        super().__init__(*args, **kwargs)

    def validate(self, options):
        expected = settings(options.get("target"), options.get("method"), options.get("model_seed"),
            full=options.get("profile") == "full", device=options["config"]["device"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Full none target protocol differs")
        return True

    def build_model(self, config, counts):
        target = self.settings["target"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("Full none target class counts differ")
        model, self.initialization = initialize_none_full_target(target, self.settings["method"],
            model_seed=self.settings["model_seed"], head_seed=self.settings["head_seed"],
            source_spec=self.source_spec, device=config.device)
        if self.expected_initial_state is not None and state_id(model) != self.expected_initial_state:
            raise ValueError("Frozen full none target initial state differs")
        model.eval()
        return model

    def validation_candidate(self, model, items, raw_source_path):
        with OrderedValidation():
            return super().validation_candidate(model, items, raw_source_path)

    def validation_snapshot(self, model, items):
        with OrderedValidation():
            return super().validation_snapshot(model, items)

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), "target_validation_arithmetic": copy.deepcopy(TARGET_ARITHMETIC)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        if model.training_config.get("target_validation_arithmetic") != TARGET_ARITHMETIC:
            raise ValueError("Selected full none target arithmetic differs")
        return model
