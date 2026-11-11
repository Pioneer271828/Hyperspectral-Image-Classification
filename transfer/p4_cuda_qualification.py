"""Qualify the registered CUDA FP32 graph operators on HongHu tile geometry."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer import p4_coverage, p4_coverage_batch as batch, p4_runtime_device as device
from transfer.g2_contract import read_pinned


SCHEMA = 'p4_cuda_fp32_operator_qualification_v1'
WEIGHT64_SCHEMA = 'p4_cuda_fp64_sparse_weight_operator_qualification_v1'
FIXTURE_SEED = 20261111
RTOL, ATOL = 1e-5, 1e-6
GRADIENTS = {'pool': ('input',), 'sparse': ('weights', 'features')}


def geometry_contract(plan, allowed, config):
    size = config['tile_size']
    edge_height, edge_width = (side % size or size for side in plan['shape'])
    shapes = {(size, size), (edge_height, size), (size, edge_width), (edge_height, edge_width)}
    counts = {}
    for role in ('train', 'val'):
        role_shapes = Counter((y1-y0, x1-x0) for y0, y1, x0, x1 in (tile['box'] for tile in allowed[role]))
        counts[role] = [{'shape': list(shape), 'tiles': n} for shape, n in sorted(role_shapes.items())]
        shapes.update(role_shapes)
    config = config['model']
    return {'geometry_sha256': plan['plan_hash'], 'shapes': [list(shape) for shape in sorted(shapes)],
            'released_role_shapes': counts, 'token_stride': 10, 'assignment_radius': config['learned_assignment_radius'],
            'adjacency_radius': config['learned_graph_radius'], 'pool_channels': 64, 'graph_channels': [64, 128]}


def fixture_specs(contract):
    names = []
    for height, width in contract['shapes']:
        prefix = f'{height}x{width}'
        names.extend((f'{prefix}/adaptive_pool', f'{prefix}/assignment', f'{prefix}/assignment_transposed',
                      f'{prefix}/adjacency64', f'{prefix}/adjacency128'))
    return [*names, 'coo_duplicate_empty_row', 'coo_duplicate_empty_row_transposed']


def neighbor_indices(height, width, grid_height, grid_width, radius):
    y, x = np.mgrid[:height, :width]
    rows = np.arange(height*width)
    center_y = y.ravel() * grid_height // height
    center_x = x.ravel() * grid_width // width
    links = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            iy, ix = center_y + dy, center_x + dx
            valid = (0 <= iy) & (iy < grid_height) & (0 <= ix) & (ix < grid_width)
            links.append(np.stack((rows[valid], (iy*grid_width + ix)[valid])))
    return np.concatenate(links, axis=1)


def pool_reference(value, size, sensitivity):
    height, width = value.shape[-2:]
    output = np.empty(sensitivity.shape, dtype=np.float64)
    gradient = np.zeros(value.shape, dtype=np.float64)
    for row in range(size[0]):
        y0, y1 = row * height // size[0], math.ceil((row+1) * height / size[0])
        for col in range(size[1]):
            x0, x1 = col * width // size[1], math.ceil((col+1) * width / size[1])
            output[..., row, col] = value[..., y0:y1, x0:x1].mean(axis=(-2, -1), dtype=np.float64)
            gradient[..., y0:y1, x0:x1] += sensitivity[..., row, col, None, None].astype(np.float64) / (
                (y1-y0) * (x1-x0))
    return output, gradient


def sparse_reference(indices, weights, features, sensitivity):
    rows, cols = indices
    weights, features, sensitivity = (value.astype(np.float64) for value in (weights, features, sensitivity))
    output, feature_grad = np.zeros(sensitivity.shape), np.zeros(features.shape)
    np.add.at(output, rows, weights[:, None] * features[cols])
    np.add.at(feature_grad, cols, weights[:, None] * sensitivity[rows])
    weight_grad = (features[cols] * sensitivity[rows]).sum(axis=1)
    return output, weight_grad, feature_grad


def compare(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    error = np.abs(actual-expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {'compatible': False, 'max_absolute_error': None, 'max_relative_error': None,
                'max_tolerance_ratio': None}
    nonzero = expected != 0
    bound = ATOL + RTOL * np.abs(expected)
    ratio = error / bound
    summary = {'compatible': bool((error <= bound).all()), 'max_absolute_error': float(error.max()),
            'max_relative_error': float((error[nonzero] / np.abs(expected[nonzero])).max(initial=0)),
            'max_tolerance_ratio': float(ratio.max())}
    if not summary['compatible']:
        index = np.unravel_index(ratio.argmax(), ratio.shape)
        summary['out_of_bound_elements'] = int((ratio > 1).sum())
        summary['worst_tolerance_element'] = {'index': [int(i) for i in index], 'actual': float(actual[index]),
            'reference': float(expected[index]), 'absolute_error': float(error[index]), 'bound': float(bound[index])}
    return summary


def outputs(result, parameters, sensitivity):
    gradient = torch.autograd.grad(result, parameters, sensitivity)
    return tuple(tensor.detach().cpu().numpy() for tensor in (result, *gradient))


def evaluate_pool(rng, height, width, channels, size, target):
    # CHW permutation matches the real pooling input's non-contiguous layout.
    values = rng.uniform(-1, 1, (height, width, channels)).astype(np.float32)
    weights = rng.uniform(.1, .9, (1, channels, *size)).astype(np.float32)
    reference = pool_reference(values.transpose(2, 0, 1)[None], size, weights)
    results = []
    for where, use_registered in (('cpu', False), (target, True)):
        image = torch.tensor(values, device=where).permute(2, 0, 1).unsqueeze(0).requires_grad_()
        result = (device.adaptive_bin_mean(image, size) if use_registered and where == 'cpu'
                  else F.adaptive_avg_pool2d(image, size))
        results.append(outputs(result, (image,), torch.tensor(weights, device=where)))
    return comparisons('pool', reference, *results)


def evaluate_sparse(rng, indices, shape, channels, target, protocol=device.CUDA_AUDITED):
    raw = rng.uniform(.2, 1, indices.shape[1]).astype(np.float32)
    counts = np.bincount(indices[0], weights=raw, minlength=shape[0])
    weights = (raw / counts[indices[0]]).astype(np.float32)
    raw_features = rng.uniform(-1, 1, (channels, shape[1])).astype(np.float32)
    sensitivity = rng.uniform(.1, .9, (shape[0], channels)).astype(np.float32)
    reference = sparse_reference(indices, weights, raw_features.T, sensitivity)
    results = []
    for where, use_registered in (('cpu', False), (target, True)):
        values = torch.tensor(weights, device=where, requires_grad=True)
        features = torch.tensor(raw_features, device=where).T.requires_grad_()
        graph = torch.sparse_coo_tensor(torch.tensor(indices, device=where), values, shape)
        replacement = (device.sparse_row_product_weight64 if protocol == device.CUDA_FP64_WEIGHT
                       else device.sparse_row_product_gemm)
        result = (replacement(graph, features) if use_registered and where == 'cpu'
                  else torch.sparse.mm(graph, features))
        results.append(outputs(result, (values, features), torch.tensor(sensitivity, device=where)))
    return comparisons('sparse', reference, *results)


def comparisons(operator, reference, native, replacement):
    names = ('forward', *GRADIENTS[operator])
    return {name: {'replacement_vs_math64': compare(new, ref), 'native_cpu32_vs_math64': compare(old, ref),
                   'replacement_vs_native_cpu32': compare(new, old)}
            for name, ref, old, new in zip(names, reference, native, replacement, strict=True)}


def sparse_pattern(indices, shape, channels):
    return {'shape': list(shape), 'stored_entries': indices.shape[1], 'channels': channels,
            'indices_sha256': digest(indices.tolist())}


def sparse_fixtures(contract, height, width):
    stride = contract['token_stride']
    grid = (math.ceil(height / stride), math.ceil(width / stride))
    pixels, tokens = height*width, math.prod(grid)
    q = neighbor_indices(height, width, *grid, contract['assignment_radius'])
    a = neighbor_indices(*grid, *grid, contract['adjacency_radius'])
    return grid, (
        ('assignment', q, (pixels, tokens), 64), ('assignment_transposed', q[::-1].copy(), (tokens, pixels), 64),
        ('adjacency64', a, (tokens, tokens), 64), ('adjacency128', a, (tokens, tokens), 128))


def adversarial_fixtures():
    indices = np.array([[0, 0, 0, 2, 2, 3], [0, 1, 1, 3, 3, 4]])
    return (('coo_duplicate_empty_row', indices, (5, 6), 64),
            ('coo_duplicate_empty_row_transposed', indices[::-1].copy(), (6, 5), 64))


def expected_patterns(contract):
    patterns = {}
    for height, width in contract['shapes']:
        _, fixtures = sparse_fixtures(contract, height, width)
        for label, indices, shape, channels in fixtures:
            patterns[f'{height}x{width}/{label}'] = sparse_pattern(indices, shape, channels)
    for key, indices, shape, channels in adversarial_fixtures():
        patterns[key] = sparse_pattern(indices, shape, channels)
    return patterns


def run_fixtures(contract, target, protocol=device.CUDA_AUDITED):
    rng = np.random.default_rng(FIXTURE_SEED)
    results, matrices = {}, {}
    for height, width in contract['shapes']:
        name = f'{height}x{width}'
        grid, fixtures = sparse_fixtures(contract, height, width)
        results[f'{name}/adaptive_pool'] = evaluate_pool(rng, height, width, contract['pool_channels'], grid, target)
        for label, indices, shape, channels in fixtures:
            key = f'{name}/{label}'
            matrices[key] = sparse_pattern(indices, shape, channels)
            results[key] = evaluate_sparse(rng, indices, shape, channels, target, protocol)
    for key, indices, shape, channels in adversarial_fixtures():
        matrices[key] = sparse_pattern(indices, shape, channels)
        results[key] = evaluate_sparse(rng, indices, shape, channels, target, protocol)
    return results, matrices


def native_cuda_diagnostic():
    """Try the original kernel under strict mode in the same isolated process."""
    prior = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        value = torch.ones((1, 64, 32, 32), device='cuda:0', requires_grad=True)
        result = F.adaptive_avg_pool2d(value, (4, 4))
        try:
            torch.autograd.grad(result.sum(), value)
        except RuntimeError as error:
            if 'deterministic' not in str(error):
                raise
            return {'adaptive_avg_pool2d_backward': 'unsupported_under_strict_mode', 'detail': str(error)}
        return {'adaptive_avg_pool2d_backward': 'available_under_strict_mode'}
    finally:
        torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])


def check_receipt(row, protocol, code, contract):
    stable = protocol == device.CUDA_FP64_WEIGHT
    device.require(row.get('schema') == (WEIGHT64_SCHEMA if stable else SCHEMA) and row.get('status') == 'passed'
        and row['commit'] == code['commit'] and row['source_hash'] == code['source_hash']
        and row['parent_sha256'] == batch.PARENT_SHA and row['geometry'] == contract
        and row['tolerances'] == {'rtol': RTOL, 'atol': ATOL}
        and protocol in device.CUDA_QUALIFIED and row['numeric']['protocol'] == protocol
        and row['numeric']['device'] == 'cuda:0' and row['numeric']['deterministic_algorithms']
        and not row['numeric']['deterministic_warn_only'] and row['optimizer_updates'] == 0
        and row['query_accessed'] is row['raw_gt_loaded'] is False and row['scope_restored']
        and list(row['results']) == fixture_specs(contract)
        and row['interceptions'] == {'adaptive_avg_pool2d': len(contract['shapes']),
                                    'sparse_mm': 4*len(contract['shapes'])+2}
        and all(check_comparisons(name, results) for name, results in row['results'].items()),
        'Missing or failed CUDA FP32 forward/gradient qualification')
    if stable:
        numeric = row['numeric']
        device.require(numeric.get('sparse_weight_contract') == device.FP64_WEIGHT_CONTRACT
            and numeric.get('tf32') is False
            and numeric.get('float32_matmul_precision') == 'highest'
            and numeric.get('graph_kernels') == 'adaptive_bin_mean_and_sparse_row_product_weight64_v4'
            and device.UUID.fullmatch(numeric['physical_uuid'])
            and row.get('fixture_seed') == FIXTURE_SEED and row['coo_patterns'] == expected_patterns(contract),
            'FP64 sparse weight qualification dtype, device or fixture identity differs')
    return row['numeric']['physical_uuid']


def check_comparisons(name, results):
    operator = 'pool' if name.endswith('/adaptive_pool') else 'sparse'
    comparisons = {'replacement_vs_math64', 'native_cpu32_vs_math64', 'replacement_vs_native_cpu32'}
    return set(results) == {'forward', *GRADIENTS[operator]} and all(
        set(output) == comparisons and all(proof['compatible'] is True
            and 0 <= proof['max_tolerance_ratio'] <= 1
            and all(math.isfinite(proof[k]) and proof[k] >= 0 for k in ('max_absolute_error', 'max_relative_error'))
            for proof in output.values()) for output in results.values())


def runtime_settings():
    return (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.get_num_threads(), torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)


def sparse_operator_cost(contract):
    """Same-device forward/backward cost on fixed shapes, with zero optimizer updates."""
    rng = np.random.default_rng(20261112)
    _, cases = sparse_fixtures(contract, 28, 23)
    rows = {}
    kernels = {'v3_fp32_gemm': device.sparse_row_product_gemm, 'v4_edge_fp64': device.sparse_row_product_weight64}
    for name, indices, shape, channels in (cases[0], cases[-1]):
        coordinates = torch.tensor(indices, device='cuda:0')
        weights = torch.tensor(rng.uniform(.2, 1, len(indices[0])).astype(np.float32),
                               device='cuda:0', requires_grad=True)
        x = torch.tensor(rng.standard_normal((channels,shape[1])).astype(np.float32),
                         device='cuda:0').T.requires_grad_()
        dy = torch.tensor(rng.standard_normal((shape[0],channels)).astype(np.float32), device='cuda:0')
        rows[name] = {}
        for label, operator in kernels.items():
            def call():
                graph = torch.sparse_coo_tensor(coordinates, weights, shape)
                return torch.autograd.grad(operator(graph, x), (weights, x), dy)
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            initial = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(12):
                call()
            end.record()
            end.synchronize()
            rows[name][label] = {'gpu_ms_per_call': start.elapsed_time(end)/12,
                'extra_peak_allocated_bytes': torch.cuda.max_memory_allocated()-initial}
    return rows


def qualify(parent, gpu, out, code, protocol=device.CUDA_AUDITED):
    from transfer.p4_full_budget_registration import role_inputs

    started = perf_counter()
    require_frozen_code(code)
    out = external_directory(out)
    out.mkdir(exist_ok=False)
    parent = Path(parent).resolve()
    reg = p4_coverage.prepare(parent, batch.PARENT_SHA, commit=code['commit'],
                              observation=batch.OBSERVATION, seed=3031, metadata='nominal')
    plan, allowed, _ = role_inputs(reg)
    contract = geometry_contract(plan, allowed, reg['config'])
    calls = Counter()
    stable = protocol == device.CUDA_FP64_WEIGHT
    settings = runtime_settings if stable else lambda: (torch.are_deterministic_algorithms_enabled(),torch.get_num_threads())
    before = settings()
    cost = None
    with device.arithmetic(protocol, gpu, kernel_calls=calls) as numeric:
        # A CPU pooling call and ordinary CUDA product must not affect interceptions.
        host = F.adaptive_avg_pool2d(torch.ones(1, 1, 2, 2), (1, 1))
        unrelated = torch.eye(2, device='cuda:0') @ torch.ones((2, 2), device='cuda:0')
        device.require(host.item() == 1 and bool((unrelated == 1).all()) and not calls,
                       'Unrelated CPU/CUDA operations changed')
        results, matrices = run_fixtures(contract, 'cuda:0', protocol)
        if stable and all(check_comparisons(name, rows) for name, rows in results.items()):
            try:
                cost = sparse_operator_cost(contract)
            except RuntimeError as error:
                cost = {'status': 'measurement_failed', 'reason': str(error)}
    native = native_cuda_diagnostic()
    after = settings()
    restored = before == after and calls == {'adaptive_avg_pool2d': len(contract['shapes']),
                                            'sparse_mm': 4*len(contract['shapes'])+2}
    passed = restored and all(proof['compatible'] for result in results.values()
                              for output in result.values() for proof in output.values())
    path = out / 'qualification.json'
    row = {'schema': WEIGHT64_SCHEMA if stable else SCHEMA, 'status': 'passed' if passed else 'failed', 'commit': code['commit'],
           'source_hash': code['source_hash'], 'parent_sha256': hash_file(parent), 'geometry': contract,
           'numeric': numeric, 'tolerances': {'rtol': RTOL, 'atol': ATOL}, 'interceptions': dict(calls),
           'scope_restored': restored, 'results': results, 'coo_patterns': matrices, 'native_cuda_diagnostic': native,
           'optimizer_updates': 0, 'raw_gt_loaded': False, 'query_accessed': False,
           'wall_seconds': perf_counter() - started}
    if stable:
        row['fixture_seed'] = FIXTURE_SEED
        row['same_device_sparse_operator_cost'] = cost
    require_frozen_code(code)
    write_json(path, row)
    check_receipt(row, protocol, code, contract)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--protocol', choices=device.CUDA_QUALIFIED, default=device.CUDA_AUDITED)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    device.require(code['commit'] == args.expected_commit, 'Unexpected CUDA qualification source commit')
    path = qualify(args.parent, args.gpu, args.out, code, args.protocol)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}))


if __name__ == '__main__':
    main()
