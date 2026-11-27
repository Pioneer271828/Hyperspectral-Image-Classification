"""Exact-capacity residual by message-source control over the bounded G4 runner."""
import copy
from dataclasses import replace
from pathlib import Path

import torch

from g4_branch_local_target import initialize as initialize_message, schema
from g4_interaction_target import PREFIX as MESSAGE, operator_signature
from g4_position_fusion_target import initialize as initialize_residual
from g4_support_context import check_files, read
from g4_support_target import NoneSupportTargetRuntime, prepare as support_prepare, settings as support_settings
from loadData.dataset_registry import DATASETS, file_hash
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.interaction_config import InteractionConfig
from model.research_config import ResearchConfig, digest
from research_runtime import tensor_hash
from target_bn_calibration import CONTEXT, GRAPH_POLICY, verify_target_binding
from target_initialization import METHODS, state_id
from target_training import observation


POLICY = "wp04_residual_message_exact_capacity_v1"
KEY = "wp04_residual_message"
BINDING = "wp04_residual_message_initialization"
RESIDUAL = "backbone.region_residual."
VERSION = MESSAGE + "protocol_version"
TARGET = "xuzhou"
SEEDS = (64, 512, 2048)
ARMS = {"R-cross": ("x_residual", "gated"), "R-local": ("x_residual", "branch_local"),
        "Bcap-cross": ("capacity_control", "gated"), "Bcap-local": ("capacity_control", "branch_local")}
PARAMETERS = {"shared": 61696, "residual": 8256, "message": 58498}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def settings(base, arm):
    mode, message = ARMS[arm]
    config = ResearchConfig.from_dict(base["config"])
    require(base["target"] == TARGET and base["method"] == METHODS[0] and base["profile"] == "engineering"
            and base["optimizer_steps"] == 12 and base["support_seed"] == 512 and base["support_per_class"] == 1
            and config.model.graph_mode == "static" and config.model.region_residual_mode == "none"
            and config.model.coverage_mode == "none" and not config.interaction.active and KEY not in base,
            "Only the registered fixed-Q/A Xuzhou K1/V5 scratch profile is supported")
    gated = replace(InteractionConfig(), interaction_mode="gated", interaction_direction="both", alignment_mode="region")
    config = replace(config, model=replace(config.model, region_residual_mode=mode), interaction=gated)
    return {**copy.deepcopy(base), "protocol": POLICY, "config": config.to_dict(),
            KEY: {"arm": arm, "residual_mode": mode, "message": message, "base_settings_sha256": digest(base),
                  "trainable_parameter_count": sum(PARAMETERS.values()), "steps": 12,
                  "first_min_fused_nll": True, "fixed_q_a": True, "query_access": False}}


def group_fingerprint(state, prefix):
    return digest({k: tensor_hash(v) for k, v in state.items()
                   if (k.startswith(prefix) and k != VERSION if prefix else not k.startswith((MESSAGE, RESIDUAL)))})


def initialize(arm, *, model_seed, head_seed, device="cpu"):
    mode, message = ARMS[arm]
    args = dict(model_seed=model_seed, head_seed=head_seed, device=device)
    residual, _ = initialize_residual(TARGET, METHODS[0], "fixed", mode=mode,
                                      exact_capacity=mode == "capacity_control", **args)
    paired, message_binding = initialize_message(TARGET, METHODS[0], message, **args)
    left, right = residual.state_dict(), paired.state_dict()
    residual_tensors = {k: v for k, v in left.items() if k.startswith(RESIDUAL)}
    shared = {k: v for k, v in left.items() if not k.startswith(RESIDUAL)}
    require(set(shared) == {k for k in right if not k.startswith(MESSAGE)}
            and all(torch.equal(v, right[k]) for k,v in shared.items()),
            "Existing residual and message factories have different common initialization")
    kwargs = {**paired.export_config()["backbone_kwargs"], "region_residual_mode": mode}
    with torch.random.fork_rng(devices=[]):
        model = MultiDatasetVCASAHGCN({TARGET: DATASETS[TARGET][2]}, **kwargs).to(device)
    model.backbone.region_residual = residual.backbone.region_residual
    model.backbone.interaction = paired.backbone.interaction
    model.load_state_dict({**right, **residual_tensors}, strict=True)
    state = model.state_dict()
    trainable = {k: v.numel() for k,v in model.named_parameters() if v.requires_grad}
    counts = {"residual": sum(n for k,n in trainable.items() if k.startswith(RESIDUAL)),
              "message": sum(n for k,n in trainable.items() if k.startswith(MESSAGE))}
    counts["shared"] = sum(trainable.values()) - sum(counts.values())
    require(counts == PARAMETERS and (VERSION in state) == (message == "branch_local"),
            "Exact-capacity modules or local version binding differ")
    binding = {"policy": POLICY, "arm": arm, "model_seed": model_seed, "head_seed": head_seed,
               "initial_state_sha256": state_id(model), "shared_sha256": group_fingerprint(state, ""),
               "residual_sha256": group_fingerprint(state, RESIDUAL), "message_sha256": group_fingerprint(state, MESSAGE),
               "architecture_sha256": digest(model.export_config()), "parameter_schema": trainable,
               "state_schema": schema(model), "operator": operator_signature(model.backbone.interaction),
               "residual_operator": operator_signature(model.backbone.region_residual),
               "local_buffer_bytes": message_binding["additional_state_bytes"], "parameters": counts}
    model.training_config = {"target_initialization": copy.deepcopy(paired.training_config["target_initialization"]),
                             BINDING: copy.deepcopy(binding)}
    return model.eval(), binding


class ResidualMessageRuntime(NoneSupportTargetRuntime):
    def validate(self, options):
        base = support_settings(TARGET, METHODS[0], "none", model_seed=options["model_seed"],
            support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], device=options["config"]["device"])
        expected = settings(base, options[KEY]["arm"])
        require({k: v for k,v in options.items() if k not in ("selected_step", "executed_optimizer_steps")} == expected,
                "Residual-message options, authority or budget differs")
        return True

    def build_model(self, config, counts):
        require(counts == {TARGET: DATASETS[TARGET][2]}, "Unexpected class counts")
        opt = self.settings
        model, self.binding = initialize(opt[KEY]["arm"], model_seed=opt["model_seed"], head_seed=opt["head_seed"],
                                         device=config.device)
        self.initialization = copy.deepcopy(model.training_config["target_initialization"])
        return model

    def verify_operator(self, model):
        require(model.training_config.get(BINDING) == self.binding
                and digest(model.export_config()) == self.binding["architecture_sha256"]
                and schema(model) == self.binding["state_schema"]
                and operator_signature(model.backbone.interaction) == self.binding["operator"]
                and operator_signature(model.backbone.region_residual) == self.binding["residual_operator"],
                "Actual fixed graph or compound operator differs")
        if VERSION in model.state_dict():
            require(int(model.state_dict()[VERSION]) == 1, "Unknown local operator version")

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        self.verify_operator(model)
        parameters = {k:v for k,v in model.named_parameters() if k.startswith((RESIDUAL, MESSAGE))}
        state = {k:v.detach().clone() for k,v in parameters.items()}
        gradient = {}

        def before_step(*_):
            require(all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters.values()),
                    "Missing or nonfinite residual/message gradient before Adam")
            for prefix in (RESIDUAL, MESSAGE):
                gradient[prefix] = sum(float(v.grad.double().square().sum()) for k,v in parameters.items()
                                       if k.startswith(prefix))**.5

        handle = optimizer.register_step_pre_hook(before_step)
        try:
            row = super().train_step_at(cursor, model, optimizer, name, item, config)
        finally:
            handle.remove()
        require(set(gradient) == {RESIDUAL, MESSAGE}, "Missing compound optimizer hook")
        row[KEY] = {k: {"gradient_norm": gradient[k], "changed_elements":
                        sum(int(torch.count_nonzero(v.detach() != state[n])) for n,v in parameters.items() if n.startswith(k))}
                    for k in gradient}
        return row

    def validation_candidate(self, model, items, raw_source_path):
        self.verify_operator(model)
        return super().validation_candidate(model, items, raw_source_path)

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), BINDING: copy.deepcopy(self.binding)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        path = Path(path).resolve(strict=True)
        side = Path(str(path) + ".json")
        pins = {str(p): file_hash(p) for p in (path, side)}
        metadata = read(side)
        meta = metadata["training_config"]
        self.validate(meta["joint_runtime"])
        model = self.build_model(ResearchConfig.from_dict(self.settings["config"]), {TARGET: DATASETS[TARGET][2]})
        require(meta["research_config"] == self.settings["config"] and meta.get(BINDING) == self.binding
                and meta["support_context_binding"] == self.settings["support_context_binding"]
                and meta["target_initialization"] == self.initialization
                and meta.get("target_validation_arithmetic") == self.settings["target_validation_arithmetic"]
                and {k:v for k,v in metadata.items() if k != "training_config"} == model.export_config(),
                "Selected compound checkpoint metadata differs")
        receipt = meta["target_bn_calibration"]
        require(file_hash(raw_source_path) == receipt["raw_source_file_sha256"], "Selected source file changed")
        raw = torch.load(raw_source_path, map_location="cpu", weights_only=True)
        source = digest({k:tensor_hash(v) for k,v in raw.items()})
        model.to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        model.training_config = copy.deepcopy(meta)
        model.eval()
        self.verify_operator(model)
        verify_target_binding(model, observation(model, items[TARGET]), context=CONTEXT,
                              graph_policy=GRAPH_POLICY, source_state_sha256=source)
        check_files(pins)
        return model


def prepare(manifest, manifest_sha256, out, *, arm, model_seed, device="cpu"):
    base, items, provenance, code, _, guard = support_prepare(manifest, manifest_sha256, out, target=TARGET,
        method=METHODS[0], mode="none", model_seed=model_seed, support_seed=512, per_class=1, device=device)
    options = settings(base, arm)
    runtime = ResidualMessageRuntime(options, items, out)
    return options, items, {**provenance, KEY: options[KEY]}, code, runtime, guard
