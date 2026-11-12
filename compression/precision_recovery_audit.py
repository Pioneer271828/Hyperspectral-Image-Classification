"""Independently account for matched updates and replay every packaged endpoint."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compression.int8_checkpoint import load_int8, save_int8
from compression.precision_cost import _state_identity
from compression.precision_fallback import LAYER_ORDER, LIMITS, SMALL_CLASS_MAX_SUPPORT
from compression.precision_recovery import ARMS, HEADS, PrecisionRecoveryRun, deployment, frozen_buffers
from compression.registered_precision_recovery import ARM_NAMES, initial_model, load_registration, stages
from compression.ternary_checkpoint import load_ternary, save_ternary
from evaluation.classification_metrics import metrics_from_confusion
from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import observe_tile
from model.checkpoint_io import save_model_checkpoint
from monitoring.dependency_groups import _protection
from research_runtime import require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


@torch.no_grad()
def independent_predictions(model, scene, record):
    """Read only devval, locate published coordinates directly and accumulate each cell."""
    model.eval()
    before = _state_identity(model)
    classes = record['classes']
    k = len(classes)
    w = scene.plan['shape'][1]
    released = dict(zip(record['validation_support']['indices'], record['validation_support']['labels'], strict=True))
    seen, targets, scores = set(), [], []
    matrices = {head: np.zeros((k, k), np.int64) for head in HEADS}
    predictions = {head: [] for head in HEADS}
    all_probabilities = {head: [] for head in HEADS}
    nll = dict.fromkeys(HEADS, 0.)
    for row in record['roles']:
        if row['role'] != 'devval':
            continue
        y0, y1, x0, x1 = row['box']
        indices = sorted(i for i in released if y0 <= i // w < y1 and x0 <= i % w < x1)
        if not indices:
            continue
        if seen.intersection(indices):
            raise ValueError('Duplicate released development sample')
        seen.update(indices)
        local = [(i // w - y0) * (x1 - x0) + (i % w - x0) for i in indices]
        target = [released[i] - 1 for i in indices]
        targets.extend(released[i] for i in indices)
        observation = observe_tile(scene.cube, scene.plan, scene.stats, scene.plan['tiles'][row['id']])
        outputs = forward_partition(model, scene.dataset, observation, np.asarray(scene.bands.wavelengths, np.float32),
                                    50 if scene.dataset == 'paviaU' else 100)
        scores.append([tensor_hash(output) for output in outputs[:3]])
        for head, output in zip(HEADS, outputs[:3], strict=True):
            full = output.cpu().numpy()
            if not np.isfinite(full).all():
                raise ValueError('Independent nonfinite deployment probabilities')
            all_probabilities[head].append(full.copy())
            p = full[local]
            hard = p.argmax(1)
            for i, j in zip(target, hard, strict=True):
                matrices[head][i, j] += 1
            predictions[head].extend(classes[j] for j in hard)
            nll[head] -= float(np.log(np.maximum(p[np.arange(len(target)), target], 1e-8)).sum())
    if seen != released.keys() or _state_identity(model) != before:
        raise ValueError('Independent validation changed model or omitted development samples')
    metrics = {}
    for head, matrix in matrices.items():
        value = metrics_from_confusion(matrix, classes)
        value['nll'] = nll[head] / len(targets)
        value['zero_recall'] = [row['class_id'] for row in value['per_class'] if row['recall'] == 0.]
        metrics[head] = value
    return dict(metrics=metrics, scores=scores, predictions=predictions, targets=targets,
                probabilities=all_probabilities, forwards=len(scores))


def save_package(model, arm, path):
    mode = ARMS[arm][0]
    start = perf_counter()
    if mode == 'fp32':
        save_model_checkpoint(model, path)
        weight, sidecar = dev._pin(path), dev._pin(str(path) + '.json')
    else:
        save = save_int8 if mode == 'qat' else save_ternary
        weight, sidecar = dict(path=str(path), sha256=save(model, path)), None
    return dict(weights=weight, sidecar=sidecar, state=_state_identity(model),
                bytes=path.stat().st_size + (Path(sidecar['path']).stat().st_size if sidecar else 0),
                save_seconds=perf_counter()-start, format={'fp32': 'fp32', 'qat': 'qat_integer', 'ternary': 'ternary_packed'}[mode])


def load_package(package):
    fmt, weights = package['format'], package['weights']
    if fmt == 'fp32':
        model = dev._load_source(weights, package['sidecar'])
    else:
        loader = {'qat_integer': load_int8, 'ternary_packed': load_ternary}[fmt]
        model = loader(weights['path'], weights['sha256'])
    if _state_identity(model) != package['state']:
        raise ValueError('Deployed reloaded package state differs')
    return model


def _guard(reference, candidate, classes, small):
    if candidate['targets'] != reference['targets']:
        raise ValueError('Unpaired development sample identities')
    labels = np.asarray(reference['targets'])
    return {head: _protection(reference['metrics'][head], candidate['metrics'][head], labels,
                              np.asarray(reference['predictions'][head]), np.asarray(candidate['predictions'][head]),
                              classes, small, LIMITS) for head in HEADS}


def select_families(guards):
    results = {}
    for mode in ('qat', 'ternary'):
        passed, rows = {}, []
        best = 'fp32'
        for suffix in ('first', 'second', 'both'):
            key = f'{mode}-{suffix}'
            missing = [f'{mode}-{name}' for name in ('first', 'second')
                       if suffix == 'both' and not passed[f'{mode}-{name}']]
            ok = guards[key]['fused']['passes_metric_guards'] and not missing
            passed[key] = ok
            if ok and len(ARMS[key][1]) > len(ARMS[best][1]):
                best = key
            rows.append(dict(arm=key, passed=ok, failed_single_layers=missing, heads=guards[key]))
        selected = ARMS[best][1]
        results[mode] = dict(arm=best, low_precision_paths=list(selected),
                             fp32_paths=[path for path in LAYER_ORDER if path not in selected], trials=rows)
    return results


def _comparison(ref, observed):
    return {head: max(float(np.max(np.abs(a - b))) for a, b in
                      zip(ref['probabilities'][head], observed['probabilities'][head], strict=True)) for head in HEADS}


def _endpoint(seed, state, arm, scene, record, path, step, row):
    model = deepcopy(seed).eval()
    model.load_state_dict(state, strict=True)
    if state_hash(model) != row['state_hash'] or frozen_buffers(model, arm) != frozen_buffers(seed, arm):
        raise ValueError('Independent state hash or frozen observer evidence differs')
    train = independent_predictions(model, scene, record)
    started = perf_counter()
    converted = deployment(model, arm)
    seconds = perf_counter() - started
    converted.training_config = dict(model.training_config, precision_recovery_arm=arm, precision_recovery_step=step,
                                     precision_recovery_state=row['state_hash'])
    observed = independent_predictions(converted, scene, record)
    for head, metric in observed['metrics'].items():
        registered = row['deployment']['heads'][head]
        if (metric['confusion_matrix'] != registered['confusion'] or metric['nll'] != registered['nll']
                or metric['oa'] != registered['oa']):
            raise ValueError('Independent endpoint differs from the recorded deployment validation event')
    package = save_package(converted, arm, path)
    started = perf_counter()
    reloaded = load_package(package)
    load_seconds = perf_counter() - started
    replay = independent_predictions(reloaded, scene, record)
    if observed['scores'] != replay['scores']:
        raise ValueError('Reloaded packaged endpoint probabilities differ')
    info = dict(step=step, state=row['state_hash'], metrics=observed['metrics'], probabilities=observed['scores'],
                predictions=observed['predictions'], train_representation_metrics=train['metrics'],
                train_representation_probabilities=train['scores'], conversion_max_abs=_comparison(train, observed),
                conversion_seconds=seconds, independent_reload_seconds=load_seconds, package=package,
                forwards=train['forwards'] + observed['forwards'] + replay['forwards'])
    return info, observed


def audit(pin, complete, code):
    start = perf_counter()
    root, value, source, scene, record, plan = load_registration(pin, code)
    batch = admission.read(complete)
    if (complete['path'] != str(root / 'batch-complete.json') or batch['registration'] != pin
            or batch['updates'] != value['maximum_updates'] or batch['gt'] != 0 or batch['query'] != 0):
        raise ValueError('Wrong completed WP10 batch')
    intervals = stages(value['budget'], value['prefix'])
    out = root / 'deployment'
    out.mkdir(exist_ok=False)
    reference = independent_predictions(deepcopy(source), scene, record)
    historical = value['source'].get('historical_baseline')
    if historical is not None and (reference['metrics'] != historical['metrics']
                                   or reference['scores'] != historical['scores']):
        raise ValueError('Cross-commit dense source replay differs from original WP08 three-head baseline')
    reports, endpoints, forward_count, updates = {}, {}, reference['forwards'], 0
    for arm in ARMS:
        receipts, packets, status_rows = {}, {}, {}
        for name, (_, stop, previous) in intervals.items():
            status, receipts[name] = read_stage(root, pin, arm, name, receipts.get(previous),
                                                arms=ARM_NAMES, stages=intervals)
            if receipts[name] != batch['attempts'][arm][name]:
                raise ValueError('Stage receipt differs from completed batch')
            packets[name] = _packet(status, value['contracts'][arm], stop, packet_format=PrecisionRecoveryRun.FORMAT)
            status_rows[name] = status
            updates += status['new_updates']
        full = packets['full']
        if (any(not _exact(full[key], packets[name][key]) for name in ('resume', 'completed') for key in SCIENCE)
                or not _exact(full['history'][:value['prefix']], packets['prefix']['history'])):
            raise ValueError('Continuous/recovered precision model, optimizer, observers, RNG or events differ')
        chosen = _verify_history(full, plan)
        seed = initial_model(root, value, source, scene, record, arm)
        reports[arm], endpoints[arm] = {}, {}
        arm_out = out / arm
        arm_out.mkdir()
        initial_live = independent_predictions(deepcopy(seed), scene, record)
        began = perf_counter()
        initial_deployment = deployment(seed, arm)
        conversion_seconds = perf_counter() - began
        initial_outputs = independent_predictions(initial_deployment, scene, record)
        reports[arm]['initial'] = dict(step=0, state=state_hash(seed), metrics=initial_outputs['metrics'],
            probabilities=initial_outputs['scores'], predictions=initial_outputs['predictions'],
            train_representation_metrics=initial_live['metrics'], train_representation_probabilities=initial_live['scores'],
            conversion_max_abs=_comparison(initial_live, initial_outputs), conversion_seconds=conversion_seconds,
            forwards=initial_live['forwards'] + initial_outputs['forwards'])
        endpoints[arm]['initial'] = initial_outputs
        forward_count += reports[arm]['initial']['forwards']
        for name, step, state in (('selected', chosen, full['selected_state']),
                                  ('last', value['budget'], full['state'])):
            info, outputs = _endpoint(seed, state, arm, scene, record, arm_out / (name + '.pt'),
                                      step, full['history'][step - 1])
            reports[arm][name] = info
            endpoints[arm][name] = outputs
            forward_count += info['forwards']
        reports[arm]['updates'] = sum(row['new_updates'] for row in status_rows.values())
        reports[arm]['stage_costs'] = {name: {k: v for k, v in status.items()
            if k.endswith('_seconds') or k in ('timing', 'new_updates')} for name, status in status_rows.items()}
        reports[arm]['parameter_activity'] = {name: dict(grad_steps=sum(r['precision_parameters'][name]['nonzero_gradient'] > 0
            for r in full['history']), changed_steps=sum(r['precision_parameters'][name]['changed'] > 0 for r in full['history']))
            for name in full['history'][0]['precision_parameters']}
        reports[arm]['deploy_kernel_calls'] = {layer: sum(row['deployment']['kernel_calls'][layer] for row in full['history'])
                                                for layer in ARMS[arm][1]}
        reports[arm]['history'] = full['history']
        reports[arm]['identical_fields'] = list(SCIENCE)
        print(f'{arm}: deployment endpoints and all intervals verified', flush=True)
    if updates != value['maximum_updates']:
        raise ValueError('Independent optimizer budget accounting differs')
    small = [int(k) for k, n in value['validation']['train_support'].items() if n <= SMALL_CLASS_MAX_SUPPORT]
    guards = {name: {arm: _guard(endpoints['fp32'][name], result[name], record['classes'], small)
                     for arm, result in endpoints.items() if arm != 'fp32'} for name in ('initial', 'selected', 'last')}
    original = {arm: {name: _guard(reference, data, record['classes'], small) for name, data in result.items()}
                for arm, result in endpoints.items()}
    decisions = {name: select_families(guards[name]) for name in ('selected', 'last')}
    admission.check_pins([pin, complete, *value['protected'].values(), *[row['checkpoint'] for row in value['initial'].values()]])
    require_frozen_code(code)
    return dev._json_new(root / 'audit.json', dict(status='passed_precision_recovery_independent',
        registration=pin, completion=complete, scope=value['scope'], source=dict(metrics=reference['metrics'],
        probabilities=reference['scores'], predictions=reference['predictions'], targets=reference['targets']),
        report=reports, small_classes=small, decisions=decisions, initial_guards=guards['initial'], versus_original_fp32=original,
        actual_updates=updates, independent_forwards=forward_count, independent_updates=0, gt=0, query=0,
        elapsed_seconds=perf_counter()-start))
