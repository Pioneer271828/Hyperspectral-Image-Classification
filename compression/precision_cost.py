"""Paired CPU cost measurement for already selected precision assignments."""

from dataclasses import dataclass
from itertools import permutations
import os
import statistics
import time

import numpy as np
import torch

from compression.g4_cpu_cost import _p95
from compression.int8_model import _dense, validate_plan
from compression.ternary_model import validate_model
from loadData.observation_sampling import sample
from loadData.spatial_tiles import observe_tile, tile_slice, validate_tile_plan
from model.research_config import digest
from monitoring.dependency_groups import HEADS
from research_runtime import tensor_hash
from transfer.g3_training import state_hash


@dataclass(frozen=True)
class ResidentPrecisionTile:
    tile_id: int
    image: torch.Tensor
    wavelengths: torch.Tensor


def prepare_resident_precision_tiles(scene, registered_tiles):
    """Prepare only the original registered input tiles, without labels."""
    validate_tile_plan(scene.plan)
    if not registered_tiles or len({row['id'] for row in registered_tiles}) != len(registered_tiles):
        raise ValueError('Distinct registered cost tiles required')
    if list(scene.cube.shape[:2]) != scene.plan['shape']:
        raise ValueError('Scene and registered tile plan differ')
    tiles = []
    for row in registered_tiles:
        tile_id = row['id']
        if type(tile_id) is not int or not 0 <= tile_id < len(scene.plan['tiles']):
            raise ValueError('Invalid registered cost tile')
        tile = scene.plan['tiles'][tile_id]
        if tensor_hash(scene.cube[tile_slice(tile)]) != row['sha256']:
            raise ValueError('Registered cost tile bytes changed')
        observation = observe_tile(scene.cube, scene.plan, scene.stats, tile)
        view, _ = sample(observation, scene.bands, {'mode': 'none'})
        image = np.concatenate((view.observation.spectral, view.observation.coordinates), axis=-1)
        tiles.append(ResidentPrecisionTile(tile_id, torch.from_numpy(image),
                                           torch.tensor(view.bands.wavelengths, dtype=torch.float32)))
    return tiles


def _quantized_identity(value):
    if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        parameters = [value.q_scale(), value.q_zero_point()]
    else:
        parameters = [value.q_per_channel_axis(), tensor_hash(value.q_per_channel_scales()),
                      tensor_hash(value.q_per_channel_zero_points())]
    return {'scheme': str(value.qscheme()), 'dtype': str(value.dtype),
            'integers': tensor_hash(value.int_repr()), 'parameters': parameters}


def _state_identity(model):
    def describe(value):
        if torch.is_tensor(value):
            return _quantized_identity(value) if value.is_quantized else {'tensor': tensor_hash(value)}
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(item) for item in value]
        return str(value) if isinstance(value, torch.dtype) else value

    return digest(describe(model.state_dict()))


def _arms(models, dataset):
    if 'fp32' not in models or not 2 <= len(models) <= 3 or not set(models) <= {'fp32', 'ptq', 'ternary'}:
        raise ValueError('Expected FP32 and at least one selected precision variant')
    reference = models['fp32']
    _dense(reference)
    config = reference.export_config()
    if dataset not in config['class_counts']:
        raise ValueError('Missing registered dataset head')
    initial = state_hash(reference)
    initial_state = reference.state_dict()
    details = {}
    for arm, model in models.items():
        if (model.export_config() != config or any(module.training for module in model.modules()) or
                any(value.device.type != 'cpu' for value in (*model.parameters(), *model.buffers())) or
                model.backbone.graph_mode != 'learned' or model.backbone.coverage_modulation is not None or
                model.backbone.adapter.cache_attention):
            raise ValueError('Expected matched eval CPU models with learned graph and VCA cache disabled')
        paths = []
        if arm != 'fp32':
            plan = validate_plan(model) if arm == 'ptq' else validate_model(model)
            phase, spec = ('integer', 'contract') if arm == 'ptq' else ('packed', 'spec')
            if (plan['phase'] != phase or plan[spec]['source_state'] != initial or
                    (arm == 'ptq' and plan[spec]['qat'] is not False)):
                raise ValueError('Precision variant must derive from the exact frozen FP32 model')
            paths = list(plan[spec]['paths'])
            prefixes = tuple(path + '.' for path in paths)
            unchanged = {name: value for name, value in model.state_dict().items() if not name.startswith(prefixes)}
            expected = {name: value for name, value in initial_state.items() if not name.startswith(prefixes)}
            if set(unchanged) != set(expected) or any(not torch.equal(value, expected[name])
                                                     for name, value in unchanged.items()):
                raise ValueError('Precision variant changed a layer outside its declared assignment')
        details[arm] = {'state_sha256': _state_identity(model), 'precision_paths': paths,
                        'parameter_count': sum(value.numel() for value in model.parameters())}
    return config['class_counts'][dataset], details


def _tile_identity(tiles):
    return [{'tile_id': tile.tile_id, 'pixels': int(tile.image.shape[0] * tile.image.shape[1]),
             'image_sha256': tensor_hash(tile.image), 'wavelengths_sha256': tensor_hash(tile.wavelengths)}
            for tile in tiles]


def benchmark_precision_cost(models, tiles, dataset, *, warmups=3, repeats=12):
    """Measure full resident-input model forwards in balanced order on one CPU thread."""
    models = {arm: models[arm] for arm in sorted(models)}
    class_count, details = _arms(models, dataset)
    orders = tuple(permutations(models))
    if (not tiles or len({tile.tile_id for tile in tiles}) != len(tiles) or
            type(warmups) is not int or warmups < 1 or type(repeats) is not int or
            repeats < 2 * len(orders) or repeats % len(orders) or torch.get_num_threads() != 1):
        raise ValueError('Matched CPU cost requires unique tiles, warmups and balanced single-thread repeats')
    if any(tile.image.device.type != 'cpu' or tile.wavelengths.device.type != 'cpu' or
           tile.image.dtype != torch.float32 or tile.wavelengths.dtype != torch.float32 or
           tile.image.ndim != 3 or min(tile.image.shape) < 1 or tile.wavelengths.ndim != 1 or
           tile.image.shape[-1] != len(tile.wavelengths) + models['fp32'].backbone.extra_channels or
           not torch.isfinite(tile.image).all() or not torch.isfinite(tile.wavelengths).all() for tile in tiles):
        raise ValueError('Expected finite, matched resident FP32 input tensors')
    inputs = _tile_identity(tiles)
    pixels = sum(row['pixels'] for row in inputs)
    scale = 50 if dataset == 'paviaU' else 100

    def forward(model, tile):
        return model(dataset, tile.image, wavelengths=tile.wavelengths, superpixel_scale=scale,
                     return_graph_info=True)[:3]

    def validate_outputs(model):
        result = []
        for tile, entry in zip(tiles, inputs, strict=True):
            values = forward(model, tile)
            if len(values) != len(HEADS) or any(tuple(value.shape) != (entry['pixels'], class_count)
                                                or not torch.isfinite(value).all() for value in values):
                raise ValueError('Expected finite full-tile three-head probabilities')
            result.append({head: tensor_hash(value) for head, value in zip(HEADS, values, strict=True)})
        return result

    samples = []
    with torch.inference_mode():
        before = {arm: validate_outputs(model) for arm, model in models.items()}
        for _ in range(warmups):
            for model in models.values():
                for tile in tiles:
                    forward(model, tile)
        for index in range(repeats):
            order = orders[index % len(orders)]
            sweep = {}
            for arm in order:
                start_cpu = time.process_time_ns()
                start_wall = time.perf_counter_ns()
                for tile in tiles:
                    forward(models[arm], tile)
                wall = time.perf_counter_ns() - start_wall
                cpu = time.process_time_ns() - start_cpu
                if wall <= 0 or cpu <= 0:
                    raise RuntimeError('Invalid measured CPU or wall interval')
                sweep[arm] = {'wall_ms': wall / 1e6, 'process_cpu_ms': cpu / 1e6}
            samples.append({'order': list(order), 'timings': sweep})
        after = {arm: validate_outputs(model) for arm, model in models.items()}
    if before != after or _tile_identity(tiles) != inputs or any(
        _state_identity(model) != details[arm]['state_sha256'] for arm, model in models.items()
    ):
        raise RuntimeError('Frozen precision cost states, inputs or outputs changed')
    summary = {}
    for arm in models:
        wall = [row['timings'][arm]['wall_ms'] for row in samples]
        median = statistics.median(wall)
        summary[arm] = {'wall_p50_ms': median, 'wall_p95_ms': _p95(wall),
                        'process_cpu_p50_ms': statistics.median(row['timings'][arm]['process_cpu_ms'] for row in samples),
                        'pixels_per_second_at_p50': 1000 * pixels / median,
                        'median_paired_fp32_over_arm': statistics.median(
                            row['timings']['fp32']['wall_ms'] / row['timings'][arm]['wall_ms'] for row in samples)}
    return {'status': 'passed', 'scope': 'cpu_resident_full_model_forward', 'dataset': dataset,
            'tiles': inputs, 'pixels_per_sweep': pixels, 'models': details, 'output_hashes': before,
            'warmups_per_arm': warmups, 'repeats': repeats, 'threads': torch.get_num_threads(),
            'affinity': sorted(os.sched_getaffinity(0)), 'torch_version': str(torch.__version__),
            'quantized_engine': torch.backends.quantized.engine, 'samples': samples, 'summary': summary,
            'optimizer_updates': 0, 'query_labels_accessed': False}
