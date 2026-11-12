"""Measure the three admitted WP11 static paths with common resident observations."""

from contextlib import nullcontext
from itertools import permutations
import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter_ns

import torch

from evaluation.wp11_confirmation import build_requests, confirmation_environment
from inference.route_controls import request_hash
from inference.trained_light import load_trained_light
from model.research_config import digest
from research_runtime import ROOT, tensor_hash, write_json
from transfer.g3_training import state_hash
from transfer.registered_development import _load_source, _read, load_development
from transfer.registered_wp11_learning import RELEASE
from transfer.wp11_matched_learning import HEADS, count_graphs


PATHS = ("light", "full_source", "full_matched")
WARMUPS = 2
ORDERS = tuple(permutations(PATHS)) * 2
STARTS = 3


def elapsed_ms(start):
    return (perf_counter_ns() - start) / 1e6


def environment():
    return dict(confirmation_environment(), scope="static_cpu_resident_requests",
                fs_cache="uncontrolled", attention_cache=False)


def read_release(pin):
    release = json.loads(_read(pin))
    if release["policy"] != RELEASE:
        raise ValueError("Audited matched release required")
    return release


def prepare(release):
    scene, record, _ = load_development(release["development"])
    blocks = [dict(id=str(r["id"]), tile_id=r["id"]) for r in record["roles"] if r["role"] == "devval"]
    if not blocks:
        raise ValueError("Existing permitted development observations required")
    requests = build_requests(scene, blocks, 50 if scene.dataset == "paviaU" else 100)
    return requests


def load_static(release, path):
    if path == "light":
        source = _load_source(release["source_weights"], release["source_metadata"])
        if state_hash(source) != release["source_state"]:
            raise ValueError("Original full source differs")
        return load_trained_light(source, release["light"]["path"], expected_sha256=release["light"]["sha256"],
                                  expected_training_sha256=release["light_training_sha256"],
                                  expected_contract_hash=release["light_contract_hash"]).model
    if path == "full_source":
        model = _load_source(release["source_weights"], release["source_metadata"])
        expected = release["source_state"]
    elif path == "full_matched":
        model = _load_source(release["full_weights"], release["full_metadata"])
        expected = release["full_state"]
    else:
        raise ValueError("Explicit static path required")
    if state_hash(model) != expected:
        raise ValueError("Frozen full state differs")
    return model.requires_grad_(False).eval()


def scan(model, path, requests):
    light = path == "light"
    with torch.inference_mode():
        result = []
        for row in requests:
            options = dict(wavelengths=row["wavelengths"])
            if light:
                outputs = (model(row["dataset"], row["x"], **options),)
            else:
                outputs = model(row["dataset"], row["x"], superpixel_scale=row["scale"], **options)[:3]
            if any(not torch.isfinite(p).all() or len(p) != row["x"].shape[0] * row["x"].shape[1]
                   for p in outputs):
                raise ValueError("Finite probabilities at every pixel required")
            result.append(outputs)
    return result


def probability_hash(tensors):
    """Preserve block order and shapes, including different-sized edge blocks."""
    return digest([tensor_hash(value) for value in tensors])


def output_hashes(path, outputs):
    names = ("light",) if path == "light" else HEADS
    return {name: probability_hash([p[i] for p in outputs]) for i, name in enumerate(names)}


def measure(model, path, requests):
    light = path == "light"
    with (nullcontext({"count": 0}) if light else count_graphs(model)) as graphs:
        started = perf_counter_ns()
        result = scan(model, path, requests)
        wall_ms = elapsed_ms(started)
    expected = 0 if light else len(requests)
    if graphs["count"] != expected or (light and hasattr(model, "backbone")):
        raise ValueError("Static path did not execute its registered graph count")
    return dict(wall_ms=wall_ms, graph_builds=expected, graph_skips=len(requests)-expected,
                output_hashes=output_hashes(path, result))


def startup(pin, path):
    env = environment()
    begin = perf_counter_ns()
    release = read_release(pin)
    req_begin = perf_counter_ns()
    requests = prepare(release)
    prepare_ms = elapsed_ms(req_begin)
    load_begin = perf_counter_ns()
    model = load_static(release, path)
    load_ms = elapsed_ms(load_begin)
    first = measure(model, path, requests)
    e2e = elapsed_ms(begin)
    return dict(path=path, release=pin["sha256"], environment=env, input_hash=request_hash(requests),
                prepare_ms=prepare_ms, load_ms=load_ms, first=first, ready_to_first_output_ms=e2e,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                parameters=sum(p.numel() for p in model.parameters()), optimizer_updates=0, query=0)


def summary(values):
    return dict(p50=statistics.median(values),
                p95=statistics.quantiles(values, n=100, method="inclusive")[94])


def run(pin, commit, out):
    if out.exists():
        raise ValueError("New external cost output required")
    env = environment()
    release = read_release(pin)
    requests = prepare(release)
    identity = request_hash(requests)
    models = {path: load_static(release, path) for path in PATHS}
    frozen = {path: state_hash(model) for path, model in models.items()}
    hashes = {path: measure(model, path, requests)["output_hashes"] for path, model in models.items()}
    for _ in range(WARMUPS):
        for path, model in models.items():
            measure(model, path, requests)
    samples = []
    for order in ORDERS:
        row = {path: measure(models[path], path, requests) for path in order}
        if any(sample["output_hashes"] != hashes[path] for path, sample in row.items()):
            raise ValueError("Resident output differs")
        samples.append(dict(order=list(order), paths=row))
    if (any(state_hash(model) != frozen[path] for path, model in models.items())
            or request_hash(requests) != identity or env != environment()):
        raise ValueError("Resident models, requests or environment changed")
    starts = {path: [] for path in PATHS}
    for offset in range(STARTS):
        for path in PATHS[offset:] + PATHS[:offset]:
            argv = [sys.executable, str(ROOT / "scripts/run_wp11_static_cost.py"), "--expected-commit", commit,
                    "--release", pin["path"], "--release-sha256", pin["sha256"], "--worker", path]
            work = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
            if work.returncode:
                raise RuntimeError(work.stderr)
            item = json.loads(work.stdout)
            if (item["input_hash"] != identity or item["environment"] != env
                    or item["first"]["output_hashes"] != hashes[path] or item["release"] != pin["sha256"]):
                raise ValueError("Fresh process does not reproduce the resident identity")
            starts[path].append(dict(item, command=argv))
    model_pins = {"light": ("light",), "full_source": ("source_weights", "source_metadata"),
                  "full_matched": ("full_weights", "full_metadata")}
    result = {}
    for path in PATHS:
        files = [release[k] for k in model_pins[path]]
        for file in files:
            _read(file)
        result[path] = dict(weights=frozen[path], package_bytes=sum(Path(f["path"]).stat().st_size for f in files),
                            parameters=sum(p.numel() for p in models[path].parameters()), output_hashes=hashes[path],
                            resident_ms=summary([row["paths"][path]["wall_ms"] for row in samples]),
                            prepare_ms=summary([row["prepare_ms"] for row in starts[path]]),
                            load_ms=summary([row["load_ms"] for row in starts[path]]),
                            first_ms=summary([row["first"]["wall_ms"] for row in starts[path]]),
                            ready_to_first_output_ms=summary([row["ready_to_first_output_ms"] for row in starts[path]]),
                            peak_rss_kib=summary([row["peak_rss_kib"] for row in starts[path]]))
    out.mkdir()
    receipt = dict(status="passed", commit=commit, release=pin, scope="existing_devval_image_only",
                   input_hash=identity, blocks=len(requests), pixels=sum(int(r["x"].shape[0] * r["x"].shape[1]) for r in requests),
                   environment=env, resident_warmups=WARMUPS, resident=samples, startup=starts, summary=result,
                   light_load_includes_source_authentication=True, startup_excludes_process_creation_import_and_code_check=True,
                   original_gt_read=False, query=0, optimizer_updates=0)
    write_json(out / "static-cost.json", receipt)
    return receipt
