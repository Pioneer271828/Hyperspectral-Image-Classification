"""Connect the existing five E1 modes to one retained-observation graph."""

from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import require
from loadData.spatial_protocol import forward_partition
from model.observation_coverage import describe
from research_runtime import tensor_hash
from transfer.band_graph_control import FixedNativeGraph, observation_id
from transfer.e1_ablation import MODES
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.native_support import _measure
from transfer.spectral_spacing_recovery import model_for

BACKEND = "coverage"
GRAPH = "shared_seeded_native_graph_from_retained_columns_v1"


def mode(model):
    layer = model.backbone.coverage_modulation
    return "none" if layer is None else layer.condition


def initialize(base, spec, seed):
    condition = spec.get("condition")
    require(set(spec)=={"backend","condition"} and condition in MODES and mode(base)=="none",
            "Coverage requires one explicit existing E1 condition and an unmodulated native start")
    # Keep the unconditioned graph anchor separate from all trainable case models.
    return model_for(base, condition, seed)


class GraphProgram:
    def __init__(self, anchor, scale):
        require(mode(anchor)=="none" and scale in (50,100)
                and next(anchor.parameters()).device.type=="cpu", "Coverage graph needs one frozen native CPU anchor")
        self.anchor = anchor.eval().requires_grad_(False)
        self.scale = scale
        self.state = state_hash(anchor)
        self.views = {}

    @torch.no_grad()
    def get(self, actual):
        key = observation_id(actual)
        spec = self.views.get(key)
        if spec is None:
            # The graph is a function of this observation's retained values and
            # coordinates. Declared, absent or rejected metadata cannot change it.
            captured = []
            builder = self.anchor.backbone.learnable_graph
            with _measure(builder,"build_graph",{},"seconds",captured.append):
                forward_partition(self.anchor, actual.dataset, actual.view.observation, superpixel_scale=self.scale)
            require(len(captured)==1,"Expected one native graph for the retained observation")
            spec = FixedNativeGraph(key,captured[0])
            self.views[key] = spec
        return spec


def run(model, actual, producer, role, execution, program):
    receipt = actual.verify(producer,role)
    start = perf_counter()
    fixed = program.get(actual)
    timing = {"coverage_anchor_seconds":perf_counter()-start}
    output, graph = forward(model,actual,producer,role,execution=execution,timed=timing,
                            graph_scale=program.scale,graph_override=fixed)
    info = output[3].get("coverage")
    source = receipt["source"]
    graph["coverage"] = dict(policy=GRAPH, anchor=program.state, observation=receipt["observation"],
                             mode=mode(model), axis=source["spectral_metadata_mode"],
                             result=source.get("independent_metadata_policy"),
                             active_anchors=0 if info is None else info["usable_anchors"],
                             fallback=None if info is None else info["fallback_reason"])
    return output,graph,timing


@torch.inference_mode()
def descriptor(model, actual):
    """Use production VCA anchors/mask and the existing E1 description."""
    model.eval()
    bone = model.backbone
    bands = actual.view.bands
    x = torch.from_numpy(np.array(actual.view.observation.spectral,copy=True))
    widths = None if bands.fwhm is None else torch.tensor(bands.fwhm,dtype=x.dtype)
    _, info = bone.adapter(x,wavelengths=torch.tensor(bands.wavelengths,dtype=x.dtype),
        fwhm=widths,wavelength_unit=bands.wavelength_unit,fwhm_unit=bands.fwhm_unit or "auto",
        return_attention=True)
    binary = info["anchor_valid"].bool()
    coverage = describe(bands,info["anchors"],bone.coverage_config)
    return dict(binary=tensor_hash(binary),gate=tensor_hash(binary & coverage.usable),
                continuous=tensor_hash(coverage.features),kept=list(bands.band_ids),
                usable=int((binary & coverage.usable).sum()))
