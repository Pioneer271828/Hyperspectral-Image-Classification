"""Run the existing model with explicitly scoped, whole graph exchanges."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS, hard
from model.coverage_config import coverage_forward_kwargs
from model.graph_state import GraphState
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.development_blocks import array_hash
from transfer.multiscale_region_graph import MultiRangeBackbone, _graph_forward


def geometry_hash(view, role):
    obs = view.observation
    if role not in ("devtrain", "devval") or obs.partition not in ("train", "val"):
        raise ValueError("Only authenticated development geometry is supported")
    return digest(dict(role=role, plan=obs.plan_hash, partition=obs.partition,
                       coordinates=array_hash(obs.coordinates), ids=array_hash(obs.global_ids),
                       shape=list(obs.spectral.shape[:2])))


def view_hash(view):
    return digest(dict(spectral=array_hash(view.observation.spectral), metadata=view.bands.fingerprint))


def _adjacency_identity(state, wider, weights):
    return dict(q=tensor_hash(state.Q), a=tensor_hash(state.A0),
                positions=tensor_hash(state.region_positions), q_nnz=state.Q._nnz(), a_nnz=state.A0._nnz(),
                wide_a=tensor_hash(wider) if wider is not None else None,
                wide_nnz=wider._nnz() if wider is not None else None,
                range_weights=tensor_hash(weights) if weights is not None else None)


@dataclass(frozen=True)
class BoundGraph:
    state: GraphState
    wider: torch.Tensor | None
    weights: torch.Tensor | None
    owner: tuple
    geometry: str
    source_view: str
    source_feature: str
    identity: dict

    @classmethod
    def create(cls, state, wider, weights, owner, geometry, source_view, features):
        state.validate(features)
        if wider is not None:
            replace(state, A0=wider).validate(features)
            if weights is None or weights.shape != (2,) or not torch.equal(weights, weights.new_full((2,), .5)):
                raise ValueError("Only frozen equal range weights admit a complete two-radius exchange")
        return cls(state, wider, weights, owner, geometry, source_view, tensor_hash(features),
                   _adjacency_identity(state, wider, weights))

    def validate(self, owner, geometry, features):
        if (self.owner != owner or self.geometry != geometry
                or self.identity != _adjacency_identity(self.state, self.wider, self.weights)):
            raise ValueError("Foreign model, graph geometry, device, policy or mutated graph")
        self.state.validate(features)
        if self.wider is not None:
            replace(self.state, A0=self.wider).validate(features)
        return self


@dataclass(frozen=True)
class Output:
    probabilities: dict
    features: dict
    graph: BoundGraph
    trace: dict


def _input(model, view, registered_units):
    device = next(model.parameters()).device
    obs, bands = view.observation, view.bands
    value = torch.tensor(np.concatenate((obs.spectral, obs.coordinates), -1), device=device)
    waves = torch.tensor(bands.wavelengths, dtype=torch.float32, device=device)
    kwargs = coverage_forward_kwargs(model, bands)
    if bands.fwhm is not None:
        kwargs.update(fwhm=torch.tensor(bands.fwhm, dtype=torch.float32, device=device), fwhm_unit=bands.fwhm_unit)
    if registered_units:
        kwargs["wavelength_unit"] = bands.wavelength_unit
    return value, waves, kwargs


@torch.no_grad()
def execute(model, dataset, view, offsets, classes, scale, *, role, model_state,
            registered_units=True, frozen=None, make_embeddings=False):
    backbone = model.backbone
    device = next(model.parameters()).device
    if model.training or backbone.graph_mode != "learned" or backbone.adapter.cache_attention:
        raise ValueError("Private eval model with explicit uncached VCA and a learned graph required")
    multirange = isinstance(backbone, MultiRangeBackbone)
    if multirange and getattr(model, "graph_multiscale", None) != "mean":
        raise ValueError("Only the registered fixed-mean multirange branch has legal graph exchange")
    owner = (dataset, model_state, str(device), scale, bool(registered_units))
    geo, observation = geometry_hash(view, role), view_hash(view)
    graph, stages, predictions = [], [], []
    build, stage, predict = backbone.learnable_graph.build_graph, backbone.forward_stages, model.predict_features

    def graph_call(features, *args, **kwargs):
        current = _graph_forward.get() if multirange else None
        if multirange and (current is None or current.owner is not backbone):
            raise ValueError("Two-radius substitution escaped its original live context")
        if frozen is None:
            fresh = build(features, *args, **kwargs)
            chosen = BoundGraph.create(fresh, current.wider_graph if current else None,
                                       current.weights if current else None, owner, geo, observation, features)
        else:
            chosen = frozen.validate(owner, geo, features)
            if current is not None:
                current.wider_graph, current.weights = chosen.wider, chosen.weights
        graph.append((chosen, tensor_hash(features)))
        return chosen.state

    def stage_call(cnn, regions, q, a, *args, **kwargs):
        chosen, digest_features = graph[-1]
        if tensor_hash(q) != chosen.identity["q"] or tensor_hash(a) != chosen.identity["a"]:
            raise ValueError("Actual branch projection Q/A differs from the bound graph")
        if multirange and tensor_hash(_graph_forward.get().wider_graph) != chosen.identity["wide_a"]:
            raise ValueError("Actual second range differs from the bound graph")
        stages.append(dict(cnn=tensor_hash(cnn), region=tensor_hash(regions), clean=digest_features))
        return stage(cnn, regions, q, a, *args, **kwargs)

    def predict_call(name, fused, hgnn, cnn, *args, **kwargs):
        predictions.append(tuple(t[offsets].cpu().numpy().copy() for t in (fused, hgnn, cnn))
                           if make_embeddings else ())
        return predict(name, fused, hgnn, cnn, *args, **kwargs)

    value, waves, kwargs = _input(model, view, registered_units)
    bound = model.partition_bands(view.bands) if hasattr(model, "partition_bands") else nullcontext()
    with (bound, patch.object(backbone.learnable_graph, "build_graph", graph_call),
          patch.object(backbone, "forward_stages", stage_call), patch.object(model, "predict_features", predict_call)):
        result = model(dataset, value, wavelengths=waves, superpixel_scale=scale, return_graph_info=True, **kwargs)
    if len(graph) != 1 or len(stages) != 1 or len(predictions) != 1:
        raise ValueError("Expected exactly one graph build/substitution, projection and prediction")
    chosen, clean = graph[0]
    probabilities = {name: tensor[offsets].cpu().numpy().copy()
                     for name, tensor in zip(HEADS, result[:3], strict=True)}
    for scores in probabilities.values():
        hard(scores, len(offsets), classes)
    return Output(probabilities, dict(zip(HEADS, predictions[0])), chosen,
                  dict(observation=observation, metadata=view.bands.fingerprint,
                       geometry=geo, source_observation=chosen.source_view, graph=chosen.identity,
                       source_graph_features=chosen.source_feature, feature_input=clean, projections=stages[0],
                       probabilities={head: array_hash(scores) for head, scores in probabilities.items()}))


def assert_same_view(native, exchanged):
    if native.trace != exchanged.trace or any(
        not np.array_equal(native.probabilities[head], exchanged.probabilities[head]) for head in HEADS
    ):
        raise ValueError("Self graph substitution changed actual features, projection or probability")
