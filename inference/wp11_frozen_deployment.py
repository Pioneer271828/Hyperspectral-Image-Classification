"""Open the pinned Xuzhou WP11 development capsule without reading labels."""

import hashlib
import io
import json
from time import perf_counter

import torch

from inference.light_cnn import load_light
from inference.native_deployment import Request, Runtime
from inference.route_controls import request_hash
from model.research_config import digest
from transfer.registered_development import _load_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(base, expected):
    manifest = base/"inputs.json"
    if sha(manifest) != expected:
        raise ValueError("Frozen input capsule differs")
    inputs = json.loads(manifest.read_text())
    if (inputs["release"] != "9a4493d2c25c6cff64d6b8ee8958e467546985865b72df41540327262d24aacb"
            or inputs["original_manifest"] != "583e3949b6ed1aad09025b06984123d05654c88424d2598bc874f93f4df096fb"):
        raise ValueError("Original WP11 release differs")
    raw = (base/"requests.pt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != inputs["files"]["requests.pt"]:
        raise ValueError("Frozen request bytes differ")
    rows = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if len(rows) != 5 or request_hash(rows) != inputs["input_hash"]:
        raise ValueError("Frozen request tensors differ")
    requests = [Request(row["id"], "xuzhou", digest(row["id"]), inputs["input_hash"],
                        "original_full", row["x"], row["wavelengths"], None, None,
                        row["scale"], 1., True) for row in rows]
    if len({x.key for x in requests}) != len(requests) or any(row["dataset"] != "xuzhou" for row in rows):
        raise ValueError("Expected exact Xuzhou development requests")
    return inputs, requests


def open_runtime(base, inputs, mode, *, partition="native", full_loads=None):
    files = inputs["files"]
    states = dict(inputs["states"])

    def load_full():
        start = perf_counter()
        full = _load_source({"path":str(base/"full.pt"), "sha256":files["full.pt"]},
                            {"path":str(base/"full.json"), "sha256":files["full.json"]})
        if full_loads is not None:
            full_loads.append((perf_counter()-start)*1000)
        return full.requires_grad_(False).eval()

    if mode == "full":
        del states["light"]
        light = None
    else:
        light = load_light(base/"light.pt", files["light.pt"])
    return Runtime(device="cpu", light=light, full_loader=load_full, identity=states, partition=partition)
