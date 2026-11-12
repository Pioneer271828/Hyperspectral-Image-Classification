"""Opt-in native VCA and three-head model with second-stage regional support."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

import torch
from torch import nn

from inference.light_cnn import validate_source
from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCNBackbone
from model.region_pixel_alignment import broadcast_regions, pool_regions
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.g3_training import _forward, state_hash
from transfer.regional_support_evolution import Schedule, development_context
from transfer.regional_support_v2 import RegionalSupport


_active = ContextVar("native_support_forward", default=None)
PROJECTIONS = ("spectrum_score.weight", "graph_score.weight")
SCORE_WIDTH = 16


def _elapsed(timings, name, function, *args, **kwargs):
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings.setdefault(name, []).append(perf_counter() - started)


def _require_backbone_contract(kwargs):
    expected = {
        "graph_mode": "learned", "graph_assignment_mode": "learned", "graph_adjacency_mode": "learned",
        "learned_graph_radius": 1, "interaction_mode": "none", "graph_update_mode": "none",
        "graph_route": None, "region_residual_mode": "none", "coverage_mode": "none",
        "extra_channels": 2, "norm_type": "group_norm",
    }
    required = {"graph_mode", "norm_type"}
    for name, allowed in expected.items():
        value = kwargs.get(name, None if name in required else allowed)
        if value != allowed or type(value) is not type(allowed):
            raise ValueError(f"Native support requires {name}={allowed!r}; received {value!r}")


class NativeSupportBackbone(VCASAHGCNBackbone):
    def __init__(self, **kwargs):
        _require_backbone_contract(kwargs)
        super().__init__(**kwargs)
        self.spectrum_score = nn.Linear(128, SCORE_WIDTH, bias=False)
        self.graph_score = nn.Linear(128, SCORE_WIDTH, bias=False)

    def forward_stages(self, cnn_input, region_input, Q, A, dataset_name=None,
                       collect_stats=False, return_diagnostics=False):
        frame = _active.get()
        if frame is None or frame["backbone"] is not self:
            raise ValueError("Native support requires an admitted block binding")
        timing = frame["timing"]
        pixels = _elapsed(timing, "cnn_stage1", self.cnn_stage1, cnn_input, dataset_name)
        regions = _elapsed(timing, "graph_stage1", self.graph_stage1, region_input, A, dataset_name)
        if frame["capture"]:
            spectra = pool_regions(Q, frame["spectral"])[:, :32]
            frame.update(Q=Q.detach().coalesce(), A0=A.detach().coalesce(), spectra=spectra.detach(),
                         descriptor=regions.detach()[:, :32].contiguous())
            evolution = A
        else:
            selector = frame["selector"]
            if tuple(Q.shape) != (cnn_input.shape[0] * cnn_input.shape[1], selector.n) or tuple(A.shape) != (
                    selector.n, selector.n):
                raise ValueError("Native graph grid differs from the registered block")
            features = _elapsed(timing, "live_projection", lambda: (
                self.spectrum_score(pool_regions(Q, pixels)) + self.graph_score(regions)))
            evolution = _elapsed(timing, "render", selector.render_live, A, features)
            if frame["detail"]:
                frame.update(Q=Q.detach().coalesce(), A0=A.detach().coalesce(),
                             stage2=evolution.detach(), score=features.detach(),
                             stage1=regions.detach())
        cnn = _elapsed(timing, "cnn_stage2", self.cnn_stage2, pixels, cnn_input.shape[:2], dataset_name)
        graph = _elapsed(timing, "graph_stage2", self.graph_stage2, regions, evolution, dataset_name)
        projected = _elapsed(timing, "q_backprojection", broadcast_regions, Q, graph)
        if frame["detail"]:
            frame.update(region_output=graph.detach(), pixel_graph=projected.detach())
        if not frame["capture"]:
            frame["trace"] = dict(q=tensor_hash(Q), a0=tensor_hash(A), support=tensor_hash(evolution),
                                  q_nnz=Q._nnz(), a0_nnz=A._nnz(), stage2_nnz=evolution._nnz(),
                                  snapshot=selector.snapshot())
        return cnn, projected, {}


class NativeSupportModel(MultiDatasetVCASAHGCN):
    def __init__(self, class_counts, *, scorer_seed, **kwargs):
        _require_backbone_contract(kwargs)
        if type(scorer_seed) is not int or not 0 <= scorer_seed < 2**32:
            raise ValueError("Explicit uint32 scorer seed required")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(scorer_seed)
            super().__init__(class_counts, **kwargs)
            self.backbone = NativeSupportBackbone(**self.backbone_kwargs)
        self.scorer_seed = scorer_seed

    @classmethod
    def initialize(cls, baseline, scorer_seed):
        validate_source(baseline)
        if baseline.backbone.learnable_graph.graph_offsets.shape[0] != 9:
            raise ValueError("Native support requires a radius-one source")
        config = baseline.export_config()
        model = cls(config["class_counts"], scorer_seed=scorer_seed, **config["backbone_kwargs"]).eval()
        state = model.state_dict()
        shared = baseline.state_dict()
        if set(state) - set(shared) != {"backbone." + name for name in PROJECTIONS}:
            raise ValueError("Native scorer inventory differs")
        model.load_state_dict({**state, **shared}, strict=True)
        model.training_config = dict(getattr(baseline, "training_config", {}))
        model.backbone.adapter.cache_attention = False
        return model

    def export_config(self):
        return {**super().export_config(), "format_version": 2,
                "regional_support": {"version": 2, "scorer_seed": self.scorer_seed, "device": "cpu_fp32"}}

    def predict_features(self, *args, **kwargs):
        frame = _active.get()
        if frame is None or frame["backbone"] is not self.backbone:
            raise ValueError("Native support requires a bound forward")
        return _elapsed(frame["timing"], "three_heads", super().predict_features, *args, **kwargs)

    def forward(self, dataset_name, x, *args, **kwargs):
        frame = _active.get()
        if (frame is None or frame["backbone"] is not self.backbone or x.device.type != "cpu"
                or x.dtype != torch.float32 or any(kwargs.get(key) is not None for key in ("Q", "A", "graph_state")) or args):
            raise ValueError("Native support requires registered CPU FP32 spatial input")
        if frame["capture"]:
            spectral, _ = self.backbone._split_spectral_extra(x, kwargs.get("wavelengths"))
            frame["spectral"] = spectral.flatten(0, 1)
        return super().forward(dataset_name, x, **kwargs)


@contextmanager
def _measure(module, attribute, timing, name, observer=None):
    if attribute in vars(module):
        raise ValueError("Module timing already bound")
    original = getattr(module, attribute)

    def wrapped(*args, **kwargs):
        result = _elapsed(timing, name, original, *args, **kwargs)
        if observer is not None:
            observer(result)
        return result

    setattr(module, attribute, wrapped)
    try:
        yield
    finally:
        delattr(module, attribute)


def native_forward(model, scene, tile, *, selector=None, capture=False, detail=False, timing=None, observation_view=None):
    """Call the same registered block materializer and original native classifier."""
    if not isinstance(model, NativeSupportModel) or (selector is None) != capture or _active.get() is not None:
        raise ValueError("Explicit source capture or bound selector required")
    if selector is not None and (selector.identity["scene"] != scene.dataset or
            selector.identity["bands"] != scene.bands.fingerprint or
            selector.context["block"] != str(tile["id"])):
        raise ValueError("Selector does not belong to this development block")
    if timing is None:
        timing = {}
    frame = dict(backbone=model.backbone, selector=selector, capture=capture, detail=detail, timing=timing)

    def observe_graph(result):
        frame["grid"] = list(result.grid_shape)

    token = _active.set(frame)
    try:
        with _measure(model.backbone.adapter, "forward", timing, "vca"), _measure(
                model.backbone.learnable_graph, "build_graph", timing, "q_a0_build", observe_graph):
            result = _elapsed(timing, "forward", _forward, model, scene, tile, observation_view=observation_view)
        if selector is not None and frame["grid"] != selector.identity["grid"]:
            raise ValueError("Current graph grid differs from source capture")
        return result, frame
    finally:
        _active.reset(token)


def capture_context(model, scene, role, admission, source_state, timing=None, *, observation_view=None):
    """Capture source descriptions from a previously admitted development tile."""
    tile = scene.plan["tiles"][role["id"]]
    expected_partitions = ("train", "val") if role["role"] == "devval" and observation_view is not None else ("train",)
    if role["role"] not in ("devtrain", "devval") or tile["partition"] not in expected_partitions:
        raise ValueError("Only admitted development blocks can be captured")
    with torch.no_grad():
        _, frame = native_forward(model, scene, tile, capture=True, timing=timing, observation_view=observation_view)
    height, width = frame["grid"]
    # Stable region IDs live in their own namespace inside this block.
    ids = [tile["id"] * 65536 + y * 256 + x for y in range(height) for x in range(width)]
    context = development_context(role=role["role"], block=str(tile["id"]), node_ids=ids, partitions=[tile["id"]] * len(ids))
    identity = dict(scene=scene.dataset, grid=frame["grid"], bands=scene.bands.fingerprint,
                    source=source_state, assignment=tensor_hash(frame["Q"]), admission=admission,
                    block_input=tensor_hash(torch.from_numpy(scene.cube[tile_slice(tile)].copy())))
    return dict(context=context, identity=identity, adjacency=frame["A0"], spectra=frame["spectra"],
                graph_features=frame["descriptor"], mandatory=[])


def make_bank(baseline, scene, record, plan, admission, scorer_seed, timing=None):
    source = NativeSupportModel.initialize(baseline, scorer_seed)
    source_state = state_hash(baseline)
    ids = set(plan.to_dict()["sampler"]["tile_ids"])
    return {str(role["id"]): capture_context(source, scene, role, admission, source_state, timing)
            for role in record["roles"] if role["role"] == "devval" or role["id"] in ids}


def bank_digest(bank):
    return digest({key: {name: tensor_hash(value) if torch.is_tensor(value) else value
                         for name, value in entry.items()} for key, entry in bank.items()})


def make_selector(entry, settings, seed):
    return RegionalSupport(**entry, mode=settings["mode"], seed=seed, schedule=Schedule(**settings["schedule"]))
