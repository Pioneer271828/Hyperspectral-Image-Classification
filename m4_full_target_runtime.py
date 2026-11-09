"""Explicit native-validation target runtime for full ordered-source initialization."""
import copy
from contextlib import nullcontext

from loadData.dataset_registry import DATASETS
from m4_full_transfer import initialize_full_target
from m4_transfer import M4TargetRuntime, target_options
from target_initialization import METHODS
from target_initialization import state_id
from m4_ordered_source import OrderedValidation, POLICY as SOURCE_ARITHMETIC

METHODS_ACTIVE = METHODS[:2]
TARGET_ARITHMETIC = {**SOURCE_ARITHMETIC, "scope": "target_calibration_and_validation"}


def settings(target, method, mode, seed=64, *, full=False, device="cpu", ordered=False):
    if method not in METHODS_ACTIVE or mode not in ("capacity_control", "x_residual"):
        raise ValueError("Registered active target modes/methods required")
    value = target_options(target, method, mode, seed=seed, support_seed=64, per_class=10, device=device)
    value["protocol"] = "g4_m4_full_source_target_" + ("full" if full else "engineering") + ("_v2" if ordered else "_v1")
    value["full_source_initialization"] = "g4_m4_full_source_binding_v2"
    if ordered:
        value["target_validation_arithmetic"] = copy.deepcopy(TARGET_ARITHMETIC)
    if full:
        value.update(profile="full", epochs=1499, val_interval=30, optimizer_steps=1500)
        value["config"]["training"].update(epochs=1499, val_interval=30)
    return value


class FullSourceTargetRuntime(M4TargetRuntime):
    def __init__(self, *args, initial_state_sha256=None, **kwargs):
        self.expected_initial_state = initial_state_sha256
        super().__init__(*args, **kwargs)

    def validate(self, options):
        mode = options["config"]["model"]["region_residual_mode"]
        expected = settings(options.get("target"), options.get("method"), mode,
            options.get("model_seed"), full=options.get("profile") == "full", device=options["config"]["device"],
            ordered=options.get("protocol", "").endswith("_v2"))
        actual = {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")}
        if actual != expected:
            raise ValueError("Full-source target protocol differs")
        return True

    def build_model(self, config, counts):
        target = self.settings["target"]
        if counts != {target: DATASETS[target][2]}:
            raise ValueError("Target class counts differ")
        model, self.initialization = initialize_full_target(target, self.settings["method"],
            config.model.region_residual_mode, model_seed=self.settings["model_seed"],
            head_seed=self.settings["head_seed"], source_spec=self.source_spec, device=config.device)
        if self.expected_initial_state is not None and state_id(model) != self.expected_initial_state:
            raise ValueError("Frozen target initialization differs")
        model.eval()
        return model

    def _arithmetic(self):
        return OrderedValidation() if "target_validation_arithmetic" in self.settings else nullcontext()

    def validation_candidate(self, model, items, raw_source_path):
        with self._arithmetic():
            return super().validation_candidate(model, items, raw_source_path)

    def validation_snapshot(self, model, items):
        with self._arithmetic():
            return super().validation_snapshot(model, items)

    def checkpoint_metadata(self, model):
        metadata = super().checkpoint_metadata(model)
        if "target_validation_arithmetic" in self.settings:
            metadata["target_validation_arithmetic"] = copy.deepcopy(TARGET_ARITHMETIC)
        return metadata

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        expected = self.settings.get("target_validation_arithmetic")
        if model.training_config.get("target_validation_arithmetic") != expected:
            raise ValueError("Selected target arithmetic differs")
        return model

