"""Use admitted blocks in the existing WP07 and WP08 zero-update forward paths."""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter


import torch

from model.research_config import digest
from research_runtime import external_directory, tensor_hash
from transfer import development_block_custody as custody, registered_development as transport
from transfer.development_blocks import Blocks, publish, read_pin, recover, sample_identity, scene_roster, view
from transfer.g3_training import _forward, state_hash
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_support import NativeSupportModel, _measure, capture_context, make_selector, native_forward
from transfer.regional_support_evolution import Schedule


SEED = 7321
SUPPORT_SEED = 19064
VIEWS = {"full": {"mode": "none"}, "even_retained_positions": {"mode": "uniform_stride", "stride": 2, "offset": 0}}
TRIALS = {
    "graph_radius2": MultiscaleTrial(graph_multiscale="fixed", graph_radius=2),
    "mean": MultiscaleTrial(cnn_multiscale="mean", graph_multiscale="mean"),
    "metadata": MultiscaleTrial(cnn_multiscale="metadata", graph_multiscale="metadata"),
    "content": MultiscaleTrial(cnn_multiscale="content", graph_multiscale="content"),
    "joint": MultiscaleTrial(cnn_multiscale="joint", graph_multiscale="joint"),
}


def _inventory(model, shared):
    state = model.state_dict()
    adapter = model.backbone.adapter
    if (any(module.training for module in model.modules()) or adapter.cache_attention or adapter._eval_attention_cache
            or any(name not in state or not torch.equal(value, state[name]) for name, value in shared.items())):
        raise ValueError("Inference mutated shared parameters, evaluation mode or cache")
    return dict(state=state_hash(model), cache_enabled=False, cache_entries=0,
                fresh={name: tensor_hash(value) for name, value in state.items() if name not in shared})


def _probabilities(output, pixels, classes):
    values = {}
    for name, head in zip(("fused", "graph", "cnn"), output[:3], strict=True):
        if (tuple(head.shape) != (pixels, classes) or head.dtype != torch.float32
                or not torch.isfinite(head).all() or (head < 0).any()
                or not torch.allclose(head.sum(1), torch.ones(pixels), atol=1e-6, rtol=1e-6)):
            raise ValueError("Invalid three-head zero-update output")
        values[name] = {"shape": list(head.shape), "sha256": tensor_hash(head)}
    return values


def _graph_trace(state):
    return dict(q=tensor_hash(state.Q), a0=tensor_hash(state.A0), grid=list(state.grid_shape),
                q_nnz=state.Q._nnz(), a0_nnz=state.A0._nnz())


@torch.no_grad()
def forward_blocks(blocks, baseline, pin, prior=None):
    release = blocks.release
    if (baseline.export_config()["class_counts"] != {release.scene: release.classes}
            or state_hash(baseline) != release.source.get("baseline_state", release.source["state"])):
        raise ValueError("Caller source differs from the authorized scene and state")
    initial = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    models = {name: trial.initialize(baseline, SEED).eval() for name, trial in TRIALS.items()}
    native = NativeSupportModel.initialize(baseline, SEED)
    models.update(off=baseline, native_step_zero=native)
    for name, model in models.items():
        if name != "off":
            model.backbone.adapter.cache_attention = False
    before = {name: _inventory(model, initial) for name, model in models.items()}
    rows = []
    for row in release.roles:
        for condition, options in VIEWS.items():
            scene, tile, observed, context = view(blocks, row, options)
            admission = digest(dict(manifest=pin["manifest_sha256"], input=context, source=release.source["shared_state"]))
            probabilities, graph = {}, {}
            for name in (*TRIALS, "off"):
                model = models[name]
                trace = []
                builder = model.backbone.learnable_graph
                with _measure(builder, "build_graph", {}, "build", lambda result: trace.append(_graph_trace(result))):
                    output = _forward(model, scene, tile, observation_view=observed,
                                      multiscale_condition_only=True)
                if not trace or output[-1]["assignment_nnz"] != trace[0]["q_nnz"]:
                    raise ValueError("Consumer did not build its native block assignment")
                probabilities[name] = _probabilities(output, observed.observation.global_ids.size, release.classes)
                scales = output[-1].get("multiscale_graph")
                graph[name] = dict(built=trace, consumed_graph_nnz=output[-1]["graph_nnz"],
                                   scale_nnz=None if scales is None else list(scales["nnz"]))
            entry = capture_context(native, scene, row, admission, state_hash(baseline), observation_view=observed)
            selector = make_selector(entry, dict(mode="joint", schedule=asdict(Schedule(10, 6, 3, 7, 6, 48, pool=5))),
                                     SUPPORT_SEED)
            output, frame = native_forward(native, scene, tile, selector=selector, observation_view=observed)
            if (frame["trace"]["snapshot"]["step"] != 0 or frame["trace"]["q"] != entry["identity"]["assignment"]
                    or frame["trace"]["a0"] != tensor_hash(entry["adjacency"])
                    or frame["trace"]["q"] != graph["off"]["built"][0]["q"]
                    or frame["trace"]["a0"] != graph["off"]["built"][0]["a0"]):
                raise ValueError("Step-zero native graph differs from the captured/off block")
            probabilities["native_step_zero"] = _probabilities(output, observed.observation.global_ids.size, release.classes)
            graph["native_step_zero"] = dict(frame["trace"], grid=entry["identity"]["grid"],
                                             capture=entry["identity"])
            if (probabilities["metadata"] != probabilities["mean"]
                    or probabilities["joint"] != probabilities["content"]):
                raise ValueError("Unverified physical metadata did not follow the original fallback")
            rows.append(dict(condition=condition, context=context, admission=admission, probabilities=probabilities,
                             graph=graph, released_samples=sample_identity(row, release.shape, release.classes),
                             native_capture_forwards=1, actual_forwards=8))
    if {name: _inventory(model, initial) for name, model in models.items()} != before:
        raise ValueError("Zero-update consumer state changed")
    regression = _legacy(prior, release, rows) if prior is not None else None
    return dict(status="PASSED", profile=release.profile, block_package=pin, source=release.source,
                actual_forwards=sum(row["actual_forwards"] for row in rows), physical=release.physical,
                roles=[row["role"] for row in release.roles], models=before, seeds={"modules": SEED, "support": SUPPORT_SEED},
                rows=rows, fallback={"metadata": "mean", "joint": "content"},
                source_unchanged=True, legacy_regression=regression)


def _legacy(prior, release, rows):
    if [row["origin"] for row in release.roles] != prior["role_arrays"]:
        raise ValueError("Historical Xuzhou block pins changed")
    for row, old in zip((row for row in rows if row["condition"] == "full"), prior["rows"], strict=True):
        context = row["context"]
        if ((context["tile"], context["role"], context["box"]) != (old["tile"], old["role"], old["box"])
                or {name: [heads[key]["sha256"] for key in ("fused", "graph", "cnn")]
                    for name, heads in row["probabilities"].items()} != old["probability_hashes"]
                or row["graph"]["graph_radius2"]["consumed_graph_nnz"] != old["radius_two_nnz"]):
            raise ValueError("Historical Xuzhou full-band outputs changed")
        current, original = row["graph"]["native_step_zero"], old["stage_two"]
        if any(current[key] != original[key] for key in ("q", "a0", "support", "q_nnz", "a0_nnz", "stage2_nnz")):
            raise ValueError("Historical Xuzhou native graph changed")
        if current["snapshot"]["selected"] != original["snapshot"]["selected"]:
            raise ValueError("Historical Xuzhou step-zero support edge set changed")
    return dict(preserved=True, original_arrays=prior["role_arrays"], probability_hashes=42, graph_contexts=2,
                new_admission_identity=True)


def describe(results):
    """Freeze observation requests with exact source, context and physical claim scope."""
    by_consumer = {}
    for consumer in ("WP05", "WP07", "WP08"):
        rows = {}
        for scene, result in results.items():
            if result["status"] != "PASSED":
                rows[scene] = {key: result[key] for key in ("status", "reason_code", "missing")}
                continue
            rows[scene] = dict(status="OBSERVATIONS_READY", profile=result["profile"],
                package=result["block_package"], source=result["source"], physical=result["physical"],
                views=[dict(condition=item["condition"], admission=item["admission"], **item["context"])
                       for item in result["rows"]],
                optimizer_authorized=False, scope="zero_update_development_observation")
        by_consumer[consumer] = rows
    return by_consumer


def probe(root, out, code, names=("paviaU", "xuzhou", "whu-honghu"), *, resume=False):
    started = perf_counter()
    names = scene_roster(names)
    root = external_directory(Path(root).resolve(strict=True))
    if root.name != "runs":
        raise ValueError("Expected the existing authorized runs root")
    out = external_directory(out)
    if not out.is_relative_to(root):
        raise ValueError("Publish development artifacts within the authorized runs root")
    out.mkdir(parents=False, exist_ok=resume)
    previous = out / "admission.json"
    if resume and previous.exists():
        raise FileExistsError("Completed publication is immutable; use the pinned receipt to replay")
    results = {}
    providers = {"xuzhou": custody.xuzhou, "whu-honghu": custody.honghu}
    for name in names:
        try:
            if name == "paviaU":
                results[name] = custody.pavia_status(root)
                continue
            admitted = providers[name](root)
        except FileNotFoundError as error:
            results[name] = dict(status="BLOCKED", reason_code="PINNED_INPUT_FILE_MISSING",
                                 missing=error.filename or str(error), actual_forwards=0)
            continue
        package = out / f"{name}.zip"
        if resume and package.exists():
            pin = recover(package, out, admitted.release, admitted.reader)
        else:
            pin = publish(package, admitted.release, admitted.reader)
        blocks = Blocks(pin, out, admitted.release)
        result = forward_blocks(blocks, admitted.baseline, pin, admitted.regression)
        custody._protect(root, admitted.protected)
        result["protected_inputs"] = list(admitted.protected)
        results[name] = result
    request_file = out / "frozen-requests.json"
    requested = dict(version=1, code=code, consumers=describe(results), updates=0, query=0, raw_gt=0, labels_added=0)
    if resume and request_file.exists():
        requests = transport._pin(request_file)
        if json.loads(read_pin(requests, out)) != requested:
            raise ValueError("Prior experiment request differs from the recovered publication")
    else:
        requests = transport._json_new(request_file, requested)
    return transport._json_new(out / "admission.json", dict(version=1, code=code,
        updates=0, query=0, raw_gt=0, labels_added=0, scope="authorized_spatial_development_zero_update",
        scenes=results, requests=requests, elapsed_seconds=perf_counter() - started))
