"""Model load and first-sweep measurements for an isolated Linux worker."""

import os
from pathlib import Path
import time

import torch

from compression.int8_checkpoint import load_int8
from compression.int8_model import validate_plan
from compression.precision_cost import _state_identity, _tile_identity
from compression.ternary_checkpoint import load_ternary
from compression.ternary_model import validate_model
from model.checkpoint_io import load_model_checkpoint
from monitoring.dependency_groups import HEADS
from research_runtime import hash_file, tensor_hash


def _memory_kib():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        name, _, contents = line.partition(':')
        if name in ('VmRSS', 'VmHWM'):
            value, unit = contents.split()
            if unit != 'kB':
                raise RuntimeError('Expected Linux process memory in KiB')
            fields[name] = int(value)
    if set(fields) != {'VmRSS', 'VmHWM'} or min(fields.values()) <= 0:
        raise RuntimeError('Linux resident and peak process memory unavailable')
    return {'rss_kib': fields['VmRSS'], 'peak_rss_kib': fields['VmHWM']}


def measure_precision_startup(arm, checkpoint, tiles, dataset, *, expected_sha256, expected_sidecar_sha256=None,
                              expected_recovered_state=None):
    """Load one pinned package, then time its first full resident-input sweep.

    Run one arm per process. Peak RSS includes imports and input preparation.
    File authentication precedes the load timer. The OS cache is unspecified.
    """
    checkpoint = Path(checkpoint)
    if (arm not in ('fp32', 'ptq', 'ternary') or torch.get_num_threads() != 1 or
            (arm == 'fp32') != (expected_sidecar_sha256 is not None) or not tiles):
        raise ValueError('One CPU thread, resident tiles and an explicit package format are required')
    pins = {checkpoint: expected_sha256}
    if arm == 'fp32':
        pins[Path(str(checkpoint) + '.json')] = expected_sidecar_sha256
    if any(hash_file(path) != sha for path, sha in pins.items()):
        raise ValueError('Precision startup package hash mismatch')
    inputs = _tile_identity(tiles)
    before = _memory_kib()
    start = time.perf_counter_ns()
    if arm == 'fp32':
        model = load_model_checkpoint(checkpoint)
    else:
        loader = load_int8 if arm == 'ptq' else load_ternary
        model = loader(checkpoint, expected_sha256)
    load_ns = time.perf_counter_ns() - start
    loaded = _memory_kib()
    model.backbone.adapter.cache_attention = False
    if expected_recovered_state is not None and _state_identity(model) != expected_recovered_state:
        raise ValueError('Startup package differs from the audited recovery state')
    if model.backbone.graph_mode != 'learned' or model.backbone.coverage_modulation is not None:
        raise ValueError('Expected the learned-graph model without a coverage override')
    if arm == 'ptq':
        plan = validate_plan(model)
        if plan['phase'] != 'integer' or plan['contract']['qat'] is not (expected_recovered_state is not None):
            raise ValueError('Integer package does not match the registered PTQ/QAT mode')
    elif arm == 'ternary' and validate_model(model)['phase'] != 'packed':
        raise ValueError('Expected packed ternary weights')
    scale = 50 if dataset == 'paviaU' else 100

    def forward(tile):
        return model(dataset, tile.image, wavelengths=tile.wavelengths, superpixel_scale=scale,
                     return_graph_info=True)[:3]

    with torch.inference_mode():
        start = time.perf_counter_ns()
        for tile in tiles:
            forward(tile)
        first_ns = time.perf_counter_ns() - start
        first = _memory_kib()
        outputs = []
        for tile in tiles:
            values = forward(tile)
            if len(values) != len(HEADS) or not all(torch.isfinite(value).all() for value in values):
                raise ValueError('Expected finite three-head output on validation replay')
            outputs.append({head: tensor_hash(value) for head, value in zip(HEADS, values, strict=True)})
    if _tile_identity(tiles) != inputs or any(hash_file(path) != sha for path, sha in pins.items()):
        raise RuntimeError('Startup measurement inputs changed')
    return {'arm': arm, 'load_ms': load_ns / 1e6, 'first_sweep_ms': first_ns / 1e6,
            'before_model_load': before, 'after_model_load': loaded, 'after_first_sweep': first,
            'package_bytes': sum(path.stat().st_size for path in pins), 'pid': os.getpid(),
            'affinity': sorted(os.sched_getaffinity(0)), 'tiles': inputs, 'output_hashes': outputs}
