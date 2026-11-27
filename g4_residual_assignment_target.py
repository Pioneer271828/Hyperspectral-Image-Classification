"""Bounded WP04 residual-only and assignment controls over the original target runtime."""
import copy
from dataclasses import replace

import torch

from g4_support_target import NoneSupportTargetRuntime, prepare as support_prepare, settings as support_settings
from joint_baseline import forward
from loadData.dataset_registry import DATASETS, file_hash
from m4_ordered_source import OrderedValidation
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import load_model_checkpoint
from model.research_config import ResearchConfig, digest
from none_full_transfer import initialize_none_full_target
from research_runtime import tensor_hash
from target_bn_calibration import CONTEXT, GRAPH_POLICY, POLICY as BN_POLICY, calibrate_target, observation_id
from target_initialization import METHODS, state_id
from target_training import observation, save_tensor_mapping


POLICY = "wp04_g4_residual_assignment_v1"
KEY = "wp04_residual_assignment"
CALIBRATION_KEY = "wp04_residual_assignment_calibration"
GRAPH_PREFIX = "backbone.learnable_graph."
RESIDUAL_PREFIX = "backbone.region_residual."
CONDITIONS = {
    "R3": ("residual_only", None), "XRegion": ("x_region", None),
    "Grid0": ("none", "fixed_grid"), "GridR": ("x_residual", "fixed_grid"),
    "Learn0": ("none", "learned"), "LearnR": ("x_residual", "learned"),
}
PARAMETERS = {"R3": 61696, "XRegion": 69952, "Grid0": 70016, "GridR": 78272,
              "Learn0": 70016, "LearnR": 78272}


def settings(base, condition):
    cfg = ResearchConfig.from_dict(base["config"])
    if (base["method"] != METHODS[0] or base["optimizer_steps"] != 12
            or cfg.model.graph_mode != "static" or cfg.model.region_residual_mode != "none"
            or cfg.model.coverage_mode != "none" or cfg.interaction.active or KEY in base):
        raise ValueError("WP04 assignment comparison requires the original bounded scratch options")
    residual, assignment = CONDITIONS[condition]
    overrides = {"region_residual_mode": residual}
    if assignment is not None:
        overrides.update(graph_mode="learned", graph_assignment_mode=assignment,
                         graph_adjacency_mode="fixed_local")
    cfg = replace(cfg, model=replace(cfg.model, **overrides))
    return {**copy.deepcopy(base), "protocol": POLICY, "config": cfg.to_dict(),
            KEY: {"condition": condition, "base_sha256": digest(base), "mode": residual,
                  "assignment": assignment, "dynamic_adjacency": "fixed_local" if assignment else None,
                  "query_access": False, "updates": 12}}


def initialize(target, method, condition, *, model_seed, head_seed, device="cpu"):
    if method != METHODS[0]:
        raise ValueError("Only target scratch may enter the residual assignment campaign")
    original, initial = initialize_none_full_target(target, method, model_seed=model_seed,
                                                     head_seed=head_seed, device=device)
    mode, assignment = CONDITIONS[condition]
    kwargs = dict(original.export_config()["backbone_kwargs"], region_residual_mode=mode)
    if assignment is not None:
        kwargs.update(graph_mode="learned", graph_assignment_mode=assignment, graph_adjacency_mode="fixed_local")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed)
        model = MultiDatasetVCASAHGCN({target: DATASETS[target][2]}, **kwargs).to(device)
    old, state = original.state_dict(), model.state_dict()
    if (not set(old) <= set(state) or any(not name.startswith((GRAPH_PREFIX, RESIDUAL_PREFIX))
                                         for name in set(state) - set(old))):
        raise ValueError("A control changed the original initialization tensor schema")
    model.load_state_dict({**state, **old}, strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[condition]:
        raise ValueError("Unexpected residual assignment capacity")
    binding = {"policy": POLICY, "condition": condition, "architecture_sha256": digest(model.export_config()),
               "initial_state_sha256": state_id(model), "original_state_sha256": state_id(original),
               "trainable": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}}
    model.training_config = {"target_initialization": initial, KEY: binding}
    return model.eval(), binding


def _verify_model(model, binding):
    if (model.training_config.get(KEY) != binding or digest(model.export_config()) != binding["architecture_sha256"]
            or {n: p.numel() for n, p in model.named_parameters() if p.requires_grad} != binding["trainable"]):
        raise ValueError("Residual assignment operator or parameter permissions changed")


def _learned_grid_calibration(model, target, obs):
    """Same sequential population moments, with the model's label-free grid builder."""
    candidate = copy.deepcopy(model)
    before = candidate.state_dict()
    protected = {n: tensor_hash(t) for n, t in before.items()}
    layers = {n: m for n, m in candidate.named_modules()
              if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))}
    if not layers or any(not n.endswith(".norms." + target) or not layer.track_running_stats
                         for n, layer in layers.items()):
        raise ValueError("Calibration requires tracked single-target BN")
    changed, visited, moments = set(), set(), []

    def capture(name):
        def update(module, args):
            x = args[0]
            if name in visited or x.ndim not in (2, 3, 4) or not torch.isfinite(x).all():
                raise ValueError("Invalid or repeated target BN population")
            var, mean = torch.var_mean(x, dim=(0, *range(2, x.ndim)), unbiased=False)
            if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
                raise ValueError("Nonfinite target population moment")
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            visited.add(name)
            changed.update((name + ".running_mean", name + ".running_var"))
            moments.append({"layer": name, "population": x.numel() // x.shape[1]})
        return update

    hooks = [layer.register_forward_pre_hook(capture(name)) for name, layer in layers.items()]
    try:
        with torch.no_grad():
            outputs = forward(candidate, target, obs)
        if visited != set(layers) or not all(torch.isfinite(t).all() for t in outputs[:3]):
            raise ValueError("Incomplete calibration through the declared assignment graph")
    finally:
        for hook in hooks:
            hook.remove()
    if any(tensor_hash(t) != protected[n] for n, t in candidate.state_dict().items() if n not in changed):
        raise ValueError("Population calibration changed a protected tensor")
    return candidate, {"layers": moments, "calibrated_buffers": sorted(changed)}


def calibrate(model, obs, *, target, binding):
    _verify_model(model, binding)
    if any(module.training for module in model.modules()):
        raise ValueError("Read-only calibration requires an evaluation model")
    source = state_id(model)
    with OrderedValidation():
        if CONDITIONS[binding["condition"]][1] is None:
            candidate, details = calibrate_target(model, obs, mode=BN_POLICY, context=CONTEXT,
                graph_policy=GRAPH_POLICY, source_state_sha256=source)
        else:
            candidate, details = _learned_grid_calibration(model, target, obs)
    if state_id(model) != source:
        raise ValueError("Raw target state mutated during calibration")
    receipt = {"policy": POLICY, "binding_sha256": digest(binding), "source_state_sha256": source,
               "state_sha256": state_id(candidate), "observation": observation_id(obs),
               "arithmetic": "fp32_sorted_row_segment_sum_v1", "query_access": False,
               "updates": 0, "layers": details["layers"], "buffers": details["calibrated_buffers"]}
    return candidate, receipt


class ResidualAssignmentRuntime(NoneSupportTargetRuntime):
    calibration_metadata_key = CALIBRATION_KEY

    def validate(self, options):
        base = support_settings(options["target"], options["method"], "none", model_seed=options["model_seed"],
            support_seed=options["support_seed"], per_class=options["support_per_class"],
            binding=options["support_context_binding"], device=options["config"]["device"])
        expected = settings(base, options[KEY]["condition"])
        if {k: v for k, v in options.items() if k not in ("selected_step", "executed_optimizer_steps")} != expected:
            raise ValueError("Residual assignment options differ from the registered scratch contract")
        return True

    def build_model(self, config, counts):
        opt = self.settings
        if counts != {opt["target"]: DATASETS[opt["target"]][2]}:
            raise ValueError("Unexpected target class count")
        model, self.assignment_binding = initialize(opt["target"], opt["method"], opt[KEY]["condition"],
            model_seed=opt["model_seed"], head_seed=opt["head_seed"], device=config.device)
        self.initialization = copy.deepcopy(model.training_config["target_initialization"])
        return model

    def train_step_at(self, cursor, model, optimizer, name, item, config):
        _verify_model(model, self.assignment_binding)
        params = {n: p for n, p in model.named_parameters() if n.startswith((GRAPH_PREFIX, RESIDUAL_PREFIX))}
        before = {n: p.detach().clone() for n, p in params.items()}
        row = super().train_step_at(cursor, model, optimizer, name, item, config)
        if any(not torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None):
            raise ValueError("Nonfinite residual or graph-builder gradient")
        row["assignment_components"] = {n: {"parameters": p.numel(),
            "gradient": float(p.grad.detach().double().norm()) if p.grad is not None else None,
            "changed": int(torch.count_nonzero(p.detach() != before[n]))} for n, p in params.items()}
        return row

    def validation_candidate(self, model, items, raw_source_path):
        save_tensor_mapping({n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, raw_source_path)
        name = self.settings["target"]
        candidate, receipt = calibrate(model, observation(model, items[name]), target=name,
                                       binding=self.assignment_binding)
        receipt["raw_source_file_sha256"] = file_hash(raw_source_path)
        candidate.training_config[CALIBRATION_KEY] = receipt
        return candidate, receipt

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model), KEY: copy.deepcopy(self.assignment_binding)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        model = load_model_checkpoint(path, device=device)
        meta = model.training_config
        self.validate(meta["joint_runtime"])
        if not hasattr(self, "assignment_binding"):
            config = ResearchConfig.from_dict(self.settings["config"])
            self.build_model(config, {name: item["class_num"] for name, item in items.items()})
        _verify_model(model, self.assignment_binding)
        raw = torch.load(raw_source_path, map_location="cpu", weights_only=True)
        receipt = meta[CALIBRATION_KEY]
        name = self.settings["target"]
        if (meta["target_initialization"] != self.initialization or meta["research_config"] != self.settings["config"]
                or meta["support_context_binding"] != self.settings["support_context_binding"]
                or receipt["policy"] != POLICY or receipt["binding_sha256"] != digest(self.assignment_binding)
                or receipt["raw_source_file_sha256"] != file_hash(raw_source_path)
                or receipt["source_state_sha256"] != digest({n: tensor_hash(v) for n, v in raw.items()})
                or receipt["state_sha256"] != state_id(model)
                or receipt["observation"] != observation_id(observation(model, items[name]))):
            raise ValueError("Selected residual assignment checkpoint provenance differs")
        return model


def prepare(manifest, manifest_sha256, out, *, condition, **kwargs):
    base, items, provenance, code, _, guard = support_prepare(manifest, manifest_sha256, out, mode="none", **kwargs)
    options = settings(base, condition)
    runtime = ResidualAssignmentRuntime(options, items, out)
    return options, items, {**provenance, KEY: options[KEY]}, code, runtime, guard
