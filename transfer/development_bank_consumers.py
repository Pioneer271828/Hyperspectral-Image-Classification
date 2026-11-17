"""Shared lazy bank inputs for the original off, WP07, and WP08 classifiers."""

from dataclasses import asdict

import torch

from model.research_config import digest
from research_runtime import tensor_hash
from transfer.development_block_probe import TRIALS, VIEWS, _graph_trace, _inventory
from transfer.g3_training import _forward, state_hash
from transfer.native_support import NativeSupportModel, _measure, capture_context, make_selector, native_forward
from transfer.regional_support_evolution import Schedule


ARMS = ("off", "content", "joint", "native")
MODEL_SEED = 7321
SUPPORT_SEED = 19064


def initialize(baseline, arm, seed):
    if arm == "off":
        from copy import deepcopy
        model = deepcopy(baseline)
    elif arm == "native":
        model = NativeSupportModel.initialize(baseline, seed)
    elif arm in TRIALS:
        model = TRIALS[arm].initialize(baseline, seed)
    else:
        raise ValueError("Unknown bank consumer")
    model.backbone.adapter.cache_attention = False
    return model.eval()


def native_selector(model, materialized, row, baseline_state, steps, seed):
    scene, tile, observed, context = materialized
    entry = capture_context(model, scene, row, digest(context), baseline_state, observation_view=observed)
    schedule = asdict(Schedule(10, 6, 3, 7, 6, steps, pool=5))
    return make_selector(entry, {"mode": "joint", "schedule": schedule}, seed)


def forward(model, arm, materialized, selector=None, *, graph_trace=False):
    scene, tile, observed, _ = materialized
    if arm == "native":
        output, frame = native_forward(model, scene, tile, selector=selector, observation_view=observed)
        return output, frame["trace"]
    kwargs = dict(observation_view=observed, multiscale_condition_only=True)
    if not graph_trace:
        return _forward(model, scene, tile, **kwargs), None
    built = []
    with _measure(model.backbone.learnable_graph, "build_graph", {}, "build",
                  lambda graph: built.append(_graph_trace(graph))):
        output = _forward(model, scene, tile, **kwargs)
    if not built:
        raise ValueError("Consumer did not construct a native graph")
    scales = output[-1].get("multiscale_graph")
    return output, dict(built=built, consumed_graph_nnz=output[-1]["graph_nnz"],
                        scale_nnz=None if scales is None else list(scales["nnz"]))


def heads(output, materialized):
    scene, _, observed, _ = materialized
    pixels = observed.observation.global_ids.size
    result = {}
    for name, prob in zip(("fused", "graph", "cnn"), output[:3], strict=True):
        if (tuple(prob.shape) != (pixels, scene.classes) or prob.dtype != torch.float32
                or not torch.isfinite(prob).all() or (prob < 0).any()
                or not torch.allclose(prob.sum(1), torch.ones_like(prob[:, 0]), atol=1e-6, rtol=1e-6)):
            raise ValueError("Invalid bank three-head probabilities")
        result[name] = dict(shape=list(prob.shape), sha256=tensor_hash(prob))
    return result


def _representatives(bank, historical):
    shapes = set()
    chosen = set(historical)
    for row in bank.rows.values():
        y0, y1, x0, x1 = row["box"]
        signature = (row["role"], y1 - y0, x1 - x0)
        if signature not in shapes:
            chosen.add(row["id"])
            shapes.add(signature)
    return sorted(chosen)


@torch.no_grad()
def scan(bank, baseline, *, prior_tiles=()):
    expected = bank.release.source.get("baseline_state", bank.release.source["state"])
    if state_hash(baseline) != expected:
        raise ValueError("Bank source baseline differs")
    arms = ("off",) if bank.release.profile == "fixed" else ("off", *TRIALS, "native")
    models = {arm: initialize(baseline, arm, MODEL_SEED) for arm in arms}
    common = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    inventory = {arm: _inventory(model, common) for arm, model in models.items()}
    chosen = _representatives(bank, prior_tiles)
    rows, captures = [], 0
    for row in bank.rows.values():
        for name, condition in VIEWS.items():
            if name != "full" and row["id"] not in chosen:
                continue
            materialized = bank.materialize(row, condition)
            active = arms if row["id"] in chosen else ("off",)
            outputs, traces = {}, {}
            for arm in active:
                selector = None
                if arm == "native":
                    selector = native_selector(models[arm], materialized, row, expected, 48, SUPPORT_SEED)
                    captures += 1
                output, trace = forward(models[arm], arm, materialized, selector, graph_trace=True)
                outputs[arm] = heads(output, materialized)
                traces[arm] = trace
            if "native" in traces:
                native = traces["native"]
                original = traces["off"]["built"][0]
                if (native["snapshot"]["step"] != 0 or native["q"] != original["q"] or native["a0"] != original["a0"]
                        or outputs["metadata"] != outputs["mean"] or outputs["joint"] != outputs["content"]):
                    raise ValueError("Bank graph role, step zero or untrusted metadata fallback differs")
            rows.append(dict(view=name, context=materialized[-1], probabilities=outputs, graph=traces))
    if state_hash(baseline) != expected or {arm: _inventory(model, common) for arm, model in models.items()} != inventory:
        raise ValueError("Zero-update bank consumers changed the source or initialized parameters")
    return dict(rows=rows, native_captures=captures, representatives=chosen,
                probability_forwards=sum(len(row["probabilities"]) for row in rows),
                source_unchanged=True, inventory=inventory, materialized_reads=bank.reads,
                materialized_bytes=bank.bytes)
