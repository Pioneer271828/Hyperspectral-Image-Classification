"""Registered spatial updates for an uncoupled WP07 model."""

from dataclasses import asdict
import math
from time import perf_counter

import torch

from inference.light_cnn import validate_source
from model.learnable_superpixel_graph import measure_graph_build
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import evaluate_validation
from transfer.g3_training import _forward, state_hash
from transfer.multiscale_training import MultiscaleTrial


def selector_group(name):
    if name.startswith("backbone.CNN_Branch.0.content_selector."):
        return "cnn_content"
    if name.startswith(("backbone.CNN_Branch.0.condition_encoder.", "backbone.CNN_Branch.0.condition_selector.")):
        return "cnn_metadata"
    if name.startswith("backbone.range_content."):
        return "graph_content"
    if name.startswith(("backbone.range_encoder.", "backbone.range_metadata.")):
        return "graph_metadata"
    return None


def metadata_contract(scene):
    bands = scene.bands
    verified = bands.wavelength_unit == "nm" and bands.wavelength_quality == "verified"
    return dict(bands_sha256=bands.fingerprint, wavelength_quality=bands.wavelength_quality,
                wavelength_unit=bands.wavelength_unit, verified_wavelength_condition=verified,
                fwhm_condition=verified and bands.fwhm is not None, gsd_m=None, coverage_fraction=None,
                adapter_metadata="original_development_forward",
                metadata_fallback=None if verified else "mean", joint_fallback=None if verified else "content")


class MultiscaleRun(CurriculumRun):
    FORMAT = "vca-multiscale-run-v1"
    FAILURE_FORMAT = "vca-multiscale-failure-v1"

    def __init__(self, baseline, scene, record, *, trial, selector_seed, **kwargs):
        validate_source(baseline)
        backbone = baseline.backbone
        if (backbone.graph_assignment_mode != "learned" or backbone.graph_adjacency_mode != "learned"
                or backbone.learnable_graph.graph_offsets.shape[0] != 9):
            raise ValueError("WP07 requires the original learned radius-one source")
        if trial is not None and not isinstance(trial, MultiscaleTrial):
            raise ValueError("Explicit multiscale trial or off required")
        if type(selector_seed) is not int or not 0 <= selector_seed < 2**32:
            raise ValueError("Invalid selector seed")
        if any(kwargs.get(key) is not None for key in ("policy", "expected_policy_hash", "observation_policy",
                                                       "expected_observation_hash")):
            raise ValueError("WP07 disables sample and observation curricula")
        kwargs.setdefault("policy", None)
        kwargs.setdefault("expected_policy_hash", None)
        self.baseline = baseline
        self.baseline_hash = state_hash(baseline)
        self.trial = trial
        self.selector_seed = selector_seed
        self.timings = {"train_forward": [], "train_graph_build": [], "validation": [], "validation_graph_build": []}
        source = baseline
        if trial is not None:
            source = trial.initialize(baseline, selector_seed).eval()
            source.backbone.adapter.cache_attention = False
            source.training_config = dict(getattr(baseline, "training_config", {}))
        shared = baseline.state_dict()
        variants = source.state_dict()
        if any(not torch.equal(value, variants[name]) for name, value in shared.items()):
            raise ValueError("WP07 shared initialization changed")
        self.selector_sizes = {name: p.numel() for name, p in source.named_parameters() if name not in shared}
        if any(selector_group(name) is None for name in self.selector_sizes):
            raise ValueError("Unexpected additional model parameters")
        kwargs.setdefault("expected_source_state", state_hash(source))
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        if type(model) is not ConditionalMultiscaleVCASAHGCN:
            return validate_source(model)
        b = model.backbone
        if (any(m.training for m in model.modules()) or b.graph_mode != "learned" or b.adapter.cache_attention
                or any(getattr(b, k) is not None for k in ("interaction", "region_residual", "coverage_modulation",
                                                          "graph_router", "edge_updater"))
                or any(t.device.type != "cpu" or t.dtype != torch.float32 or not torch.isfinite(t).all()
                       for t in model.state_dict().values())):
            raise ValueError("WP07 requires a sealed uncoupled FP32 evaluation state")

    def _contract_extensions(self):
        return {"multiscale": dict(trial=None if self.trial is None else asdict(self.trial),
                                   selector_seed=self.selector_seed, selector_sizes=self.selector_sizes.copy(),
                                   baseline_state=self.baseline_hash, graph_router="disabled",
                                   metadata=metadata_contract(self.scene))}

    def _check_extensions(self):
        if (state_hash(self.baseline) != self.baseline_hash
                or self.spec["multiscale"] != self._contract_extensions()["multiscale"]):
            raise ValueError("Registered multiscale or metadata contract changed")

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        with measure_graph_build(self.timings["train_graph_build"]):
            result = _forward(self.model, self.scene, tile, multiscale_condition_only=True, **extra)
        self.timings["train_forward"].append(perf_counter() - started)
        return result

    def _evaluate(self, model):
        started = perf_counter()
        with measure_graph_build(self.timings["validation_graph_build"]):
            result = evaluate_validation(model, self.scene, self.record, multiscale_condition_only=True)
        self.timings["validation"].append(perf_counter() - started)
        return result

    def _history_extensions(self, outputs):
        info = outputs[3]
        cnn = info.get("multiscale_cnn")
        graph = info.get("multiscale_graph")
        return {"scales": dict(
            cnn_dilations=[1] if cnn is None else list(cnn["dilations"]),
            cnn_weights=[1.] if cnn is None else cnn["weights"].flatten().tolist(),
            graph_radii=[1] if graph is None else list(graph["radii"]),
            graph_weights=[1.] if graph is None else graph["weights"].tolist(),
            graph_nnz=[info["graph_nnz"]] if graph is None else list(graph["nnz"]),
            assignment_nnz=info["assignment_nnz"], token_count=info["token_count"])}

    def _parameter_history(self, before):
        return {"selectors": {
            name: {"gradient_l1": None if p.grad is None else float(p.grad.abs().sum()),
                   "changed": int((before[name] != p.detach()).sum())}
            for name, p in self.model.named_parameters() if name in self.selector_sizes}}

    def _history_extension_keys(self):
        return {"scales", "selectors"}

    def _validate_history_extensions(self, row):
        scales = row["scales"]
        trial = self.trial or MultiscaleTrial()
        cnn = [trial.cnn_dilation] if trial.cnn_multiscale == "fixed" else [1, 2, 3]
        graph = [trial.graph_radius] if trial.graph_multiscale in ("off", "fixed") else [1, 2]
        if (set(scales) != {"cnn_dilations", "cnn_weights", "graph_radii", "graph_weights", "graph_nnz",
                            "assignment_nnz", "token_count"}
                or scales["cnn_dilations"] != cnn or scales["graph_radii"] != graph
                or type(scales["token_count"]) is not int or scales["token_count"] < 1
                or type(scales["assignment_nnz"]) is not int or scales["assignment_nnz"] < scales["token_count"]
                or len(scales["graph_nnz"]) != len(graph)
                or any(type(n) is not int or not scales["token_count"] <= n <= scales["token_count"]**2
                       for n in scales["graph_nnz"])):
            raise ValueError("Invalid registered multiscale supports")
        for kind, candidates in (("cnn", cnn), ("graph", graph)):
            weights = scales[kind + "_weights"]
            if (len(weights) != len(candidates) or any(type(w) is not float or not 0 <= w <= 1 for w in weights)
                    or not math.isclose(sum(weights), 1., abs_tol=2e-7)):
                raise ValueError("Invalid multiscale weights")
        selectors = row["selectors"]
        if set(selectors) != set(self.selector_sizes):
            raise ValueError("Invalid selector inventory")
        for name, evidence in selectors.items():
            gradient, changed = evidence["gradient_l1"], evidence["changed"]
            if (set(evidence) != {"gradient_l1", "changed"} or type(changed) is not int
                    or not 0 <= changed <= self.selector_sizes[name]
                    or ((gradient is not None) != (name in row["gradient_parameters"]))
                    or (gradient is None and changed)
                    or (gradient is not None and (type(gradient) is not float or not math.isfinite(gradient)
                                                   or gradient < 0))):
                raise ValueError("Invalid selector gradient or update")

    @classmethod
    def load(cls, baseline, scene, record, path, *, plan, trial, selector_seed,
             expected_sha256, expected_contract_hash):
        return super().load(baseline, scene, record, path, plan=plan, policy=None,
                            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(trial=trial, selector_seed=selector_seed))
