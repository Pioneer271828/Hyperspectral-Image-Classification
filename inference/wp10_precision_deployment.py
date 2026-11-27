"""Deploy archived WP10 precision choices with their original release guards."""

from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import torch

from compression.int8_torchao_onednn import OneDNNExecutionError, OneDNNUnavailable, load_backend
from compression.int8_torchao_reference import TorchAOAffine
from compression.low_rank_recovery_audit import HEADS
from compression.precision_cost import ResidentPrecisionTile, _tile_identity
from compression.precision_fallback import LAYER_ORDER
from compression.precision_recovery_audit import load_package
from inference.native_deployment import Request, RouteFailure, Runtime, clock_ms
from inference.wp11_frozen_deployment import deployment_arithmetic, place_requests, sha
from research_runtime import external_directory, tensor_hash, write_json
from transfer.p4_runtime_device import snapshot
from transfer.registered_development import _read


SOURCE_COMMIT = "e6a652e9466ca7ca1cb5455e7e19a258b80b5fc3"
QAT_PLAN = "de7e38acc032e82902bd008ca7d48aea7257e787c7d90c9a9361fb6d09c3167e"
RANK_PLAN = "1b8913423805192d801098abfa46eaec9045b128baf10d4fb3290fd4a08174a5"
QAT_ARMS = ("fp32", "qat-first", "qat-both", "ternary")
SEEDS = (64, 512, 2048)
RANK_ARMS = tuple(f"s{seed}-{arm}" for seed in SEEDS for arm in ("dense", "svd-r16"))


def _passes(guards):
    if set(guards) != set(HEADS):
        raise ValueError("Original fused, graph and CNN guards required")
    return all(row["passes_metric_guards"] is True for row in guards.values())


def read_source(path, expected):
    if expected not in (QAT_PLAN, RANK_PLAN):
        raise ValueError("Approved frozen WP10 release digest required")
    raw = _read({"path":str(path), "sha256":expected})
    plan = json.loads(raw)
    if (plan["commit"] != SOURCE_COMMIT or plan["dataset"] != "xuzhou"
            or plan["class_ids"] != list(range(1, 10)) or len(plan["tiles"]) != 5
            or sum(row["pixels"] for row in plan["tiles"]) != 3920):
        raise ValueError("Frozen producer, role, observation or class axis differs")
    if expected == QAT_PLAN:
        if (plan["kind"] != "wp10_frozen_onednn_pair_v1"
                or not _passes(plan["guards"]) or _passes(plan["rejected"])):
            raise ValueError("Original WP10 single/double layer protection differs")
    elif (plan["policy"] != "wp10_label_free_warm_plan_v1" or plan["seeds"] != list(SEEDS)
          or plan["observation_role"] != "devval_image_only" or set(plan["cases"]) != set(RANK_ARMS)
          or any(_passes(plan["proofs"][f"s{s}-svd-r16"]["guards"]) for s in SEEDS)):
        raise ValueError("Original rank safeguards and dense controls required")
    return raw, plan


def _approved_packages(plan):
    if plan["kind"] == "wp10_frozen_onednn_pair_v1":
        return plan["packages"]
    return {f"s{s}-dense": plan["cases"][f"s{s}-dense"]["package"] for s in SEEDS}


def _assets(plan):
    pins = {"observations.pt":plan["resident_inputs"]}
    if plan["kind"] == "wp10_frozen_onednn_pair_v1":
        pins["backend.json"] = plan["backend"]
    for name, package in _approved_packages(plan).items():
        pins[name+".pt"] = package["weights"]
        if package["sidecar"] is not None:
            pins[name+".pt.json"] = package["sidecar"]
    return pins


def _new_output(out):
    out = external_directory(out)
    if out.exists():
        raise FileExistsError(out)
    return out


def export_capsule(path, expected, out):
    """Preserve the original plan bytes and only the eligible inference assets."""
    out = _new_output(out)
    raw, plan = read_source(path, expected)
    with TemporaryDirectory(dir=out.parent) as temporary:
        staging = Path(temporary)/"capsule"
        staging.mkdir()
        (staging/"source-plan.json").write_bytes(raw)
        for name, pin in _assets(plan).items():
            (staging/name).write_bytes(_read(pin))
        if out.exists():
            raise FileExistsError(out)
        staging.rename(out)


def _local(root, name, pin):
    path = root/name
    if path.is_symlink():
        raise ValueError("Use a regular file inside the authenticated WP10 capsule")
    return {"path":str(path), "sha256":pin["sha256"]}


def read_requests(root, plan):
    pins = _assets(plan)
    raw = _read(_local(root, "observations.pt", pins["observations.pt"]))
    rows = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if any(set(row) != {"tile_id", "image", "wavelengths"} for row in rows):
        raise ValueError("Original image-only observations required")
    tiles = [ResidentPrecisionTile(**row) for row in rows]
    if _tile_identity(tiles) != plan["tiles"]:
        raise ValueError("Frozen WP10 observations or order changed")
    return [Request(str(tile.tile_id), "xuzhou", obs["image_sha256"], pins["observations.pt"]["sha256"],
                    "original_full", tile.image, tile.wavelengths, None, None, 100, 1., True)
            for tile, obs in zip(tiles, plan["tiles"], strict=True)]


def _choice(plan, requested, partition, gpu, provider):
    if plan["kind"] != "wp10_frozen_onednn_pair_v1":
        if requested not in RANK_ARMS:
            raise ValueError("Specify an archived rank case and its recovery seed")
        if requested.endswith("-dense"):
            return requested, requested, "requested_fp32"
        return requested, requested.split("-")[0]+"-dense", "original_rank_three_head_guard"
    requested = requested or "fp32"
    if requested not in QAT_ARMS:
        raise ValueError("Use an original QAT or ternary precision request")
    if requested == "fp32":
        return requested, requested, "requested_fp32"
    if requested in ("qat-both", "ternary"):
        return requested, "fp32", "original_precision_three_head_guard"
    if gpu is not None:
        return requested, "fp32", "cpu_integer_backend_only"
    if partition != "native":
        return requested, "fp32", "integer_partition_not_qualified"
    if provider is None:
        return requested, "fp32", "backend_source_unconfigured"
    if not (torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled
            and hasattr(torch.ops.onednn, "qconv_prepack") and hasattr(torch.ops.onednn, "qconv2d_pointwise")):
        return requested, "fp32", "onednn_unavailable"
    return requested, requested, "original_three_head_qualified"


def _load(root, plan, arm, provider):
    package = deepcopy(_approved_packages(plan)[arm])
    package["weights"] = _local(root, arm+".pt", package["weights"])
    if package["sidecar"] is not None:
        package["sidecar"] = _local(root, arm+".pt.json", package["sidecar"])
    model = load_package(package)
    model.backbone.adapter.cache_attention = False
    if arm == "qat-first":
        model = load_backend(_local(root, "backend.json", plan["backend"]), TorchAOAffine(provider),
                             enabled=True, native=model)
    return model.requires_grad_(False).eval(), package


def _expected(plan, arm):
    if plan["kind"] == "wp10_frozen_onednn_pair_v1":
        return plan["hashes"][arm]
    return plan["proofs"][arm]["output_hashes"]


def execute(root, expected, out, code, *, requested=None, partition="native", gpu=None, provider=None):
    root = root.resolve(strict=True)
    out = _new_output(out)
    if out.is_relative_to(root):
        raise ValueError("Outputs must be outside the original capsule")
    if (root/"source-plan.json").is_symlink():
        raise ValueError("Source plan must be a regular capsule file")
    started = perf_counter_ns()
    _, plan = read_source(root/"source-plan.json", expected)
    requests = read_requests(root, plan)
    requested, arm, reason = _choice(plan, requested, partition, gpu, provider)
    prep_ms = clock_ms(started)
    before = snapshot(gpu) if gpu is not None else None
    kernels = Counter()
    recovered = None
    load_ms = []
    with deployment_arithmetic(gpu, kernel_calls=kernels) as numeric:
        device = numeric["device"]
        if gpu is not None:
            torch.cuda.reset_peak_memory_stats()
            requests = place_requests(requests, device)

        def run(which):
            load_start = perf_counter_ns()
            model, package = _load(root, plan, which, provider)
            runtime = Runtime(device=device, full=model, partition=partition)
            load_ms.append(clock_ms(load_start))
            return runtime.run(requests, "full"), package

        try:
            result, package = run(arm)
        except (OneDNNUnavailable, RouteFailure) as error:
            operator_failure = isinstance(error, RouteFailure) and isinstance(error.__cause__, OneDNNExecutionError)
            if arm != "qat-first" or (not isinstance(error, OneDNNUnavailable) and not operator_failure):
                raise
            recovered = error.stats if operator_failure else {"graph_builds":0, "full_calls":0}
            arm, reason = "fp32", "onednn_execution_failed" if operator_failure else "onednn_prepack_unavailable"
            result, package = run(arm)
        predictions = [tuple(value.detach().cpu() for value in row) for row in result.pop("predictions")]
        peaks = (dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
                 if gpu is not None else None)
    hashes = [dict(zip(HEADS, map(tensor_hash, row), strict=True)) for row in predictions]
    archived_exact = hashes == _expected(plan, arm) if gpu is None and partition == "native" else None
    if archived_exact is False:
        raise ValueError("Native CPU output differs from the archived selected package")
    after = snapshot(gpu) if gpu is not None else None
    out.mkdir()
    path = out/"predictions.pt"
    with path.open("xb") as stream:
        torch.save(predictions, stream)
    write_json(out/"receipt.json", dict(result, tested_commit=code["commit"], source_plan_sha256=expected,
               requested=requested, executed=arm, reason=reason, partition=partition,
               selected_weight_sha256=package["weights"]["sha256"], selected_state=package["state"],
               bytes=package["bytes"], quantized_paths=[LAYER_ORDER[0]] if arm=="qat-first" else [],
               precision="onednn_u8s8" if arm=="qat-first" else "fp32", calibration_updates=0,
               optimizer_updates=0, query_reads=0, label_reads=0, recovered_attempt=recovered,
               arithmetic=numeric, graph_kernels=dict(kernels), device_before=before, device_after=after,
               cpu_archived_exact=archived_exact, predictions_sha256=sha(path), wall_ms=clock_ms(started),
               prep_ms=prep_ms, load_ms=load_ms, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               cuda_peak_bytes=peaks))
