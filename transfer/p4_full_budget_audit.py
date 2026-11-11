"""Accept a completed P4 case from its committed worker ledger and frozen probabilities."""
from pathlib import Path
from time import perf_counter

import math
import numpy as np
import torch

from joint_training import _rng_state, _tree_fingerprint
from model.research_config import digest
from research_runtime import hash_file
from train_spatial import SpatialConfig
from transfer import p4_coverage_batch as batch, p4_coverage_review as prior, p4_exposure
from transfer import p4_frozen_diagnostic_metrics as independent, p4_frozen_diagnostics as diag
from transfer import p4_full_budget_registration as registration, p4_runtime_device as device
from transfer.g2_contract import read_pinned
from transfer.g3_training import _optimizer, state_hash


def ledger(folder, case, sha):
    """A checkpoint without a matching committed update receipt requires diagnosis."""
    folder = Path(folder)
    cursor, receipts = 0, []
    for path in sorted((folder / 'calls').glob('*.json')):
        row = read_pinned(path, hash_file(path))
        until = row['until']
        device.require(row['case'] == case['id'] and row['scope_sha256'] == sha and row['start'] == cursor
            and type(until) is int and until > cursor and row['new_optimizer_updates'] == until - cursor
            and path.name == f'{cursor:06d}-{until:06d}.json',
            'Untrusted or overlapping worker update ledger')
        cursor = until
        receipts.append(row)
    checkpoint = folder / 'state.pt'
    device.require(checkpoint.exists() == bool(receipts), 'Uncommitted worker checkpoint; preserve for diagnosis')
    if receipts:
        device.require(hash_file(checkpoint) == receipts[-1]['state_sha256'], 'Worker checkpoint differs from its ledger')
    return cursor, receipts


def train_state(folder, scope, reg, case, sha, numeric):
    start, receipts = ledger(folder, case, sha)
    steps = scope['budget']['steps']
    device.require(start == steps and all(r['numeric'] == numeric for r in receipts),
                   'Full budget or same-device numerical history is incomplete')
    checkpoint, result_path = (folder / name for name in ('state.pt', 'result.json'))
    result = read_pinned(result_path, hash_file(result_path))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    runtime = {'scope_sha256': sha, 'case': case, 'numeric': numeric}
    identity = digest({'registration': reg, 'case': case['index'], 'code': scope['commit'],
                       'init': case['initialization'], 'runtime': runtime})
    fingerprints = {key: _tree_fingerprint(state[key]) for key in batch.FIELDS}
    device.require(state['runtime'] == result['runtime'] == runtime and
                   state['identity'] == result['identity'] == identity and
                   all(r['identity'] == identity for r in receipts) and
                   state['fingerprints'] == fingerprints and result['state_sha256'] == hash_file(checkpoint) and
                   state['cursor'] == result['updates'] == steps and result['initialization'] == case['initialization'] and
                   result['coverage'] == {**reg['coverage'], 'arm': case['arm'], 'pair_sha256': case['pair_sha256']} and
                   result['raw_gt_loaded'] is result['query_accessed'] is False and
                   (state['rng']['cuda'] is None) == (numeric['device'] == 'cpu'),
                   'Runtime, state fingerprint, coverage, seed or device identity differs')
    proof_path = folder / 'completed-resume.json'
    proof = read_pinned(proof_path, hash_file(proof_path))
    device.require(proof == {'scope_sha256': sha, 'identity': identity, 'case': case['id'],
            'new_optimizer_updates': 0, 'cursor': steps, 'numeric': numeric,
            'state_sha256': hash_file(checkpoint), 'result_sha256': hash_file(result_path)},
            'Completed zero-update resume proof differs')
    return state, result, receipts


def check_optimizer(model, config, state):
    """Reconstruct every scheduler update from validation, without optimizing."""
    optimizer = _optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=20)
    validation = {e['step']: e['metrics']['fused']['nll'] for e in state['history']}
    for trace in state['trace']:
        device.require(trace['lr'] == [g['lr'] for g in optimizer.param_groups]
            and all(math.isfinite(v) for v in trace['loss'].values())
            and math.isfinite(trace['graph_gradient_l1']) and trace['graph_gradient_l1'] > 0
            and trace['graph_changed_elements'] > 0, 'LR, finite loss or graph gradient/update evidence differs')
        if trace['step'] in validation:
            scheduler.step(validation[trace['step']])
    expected = optimizer.state_dict()
    actual = state['optimizer']
    device.require(state['scheduler'] == scheduler.state_dict() and actual['param_groups'] == expected['param_groups'],
                   'Adam or ReduceLROnPlateau state differs from the frozen schedule')
    parameters = {key: parameter for group, spec in zip(optimizer.param_groups, expected['param_groups'])
                  for key, parameter in zip(spec['params'], group['params'])}
    device.require(bool(actual['state']) and set(actual['state']) <= parameters.keys(), 'Unregistered Adam moments')
    for key, entry in actual['state'].items():
        parameter = parameters[key]
        device.require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}
            and 1 <= float(entry['step']) <= state['cursor'] and float(entry['step']).is_integer()
            and all(entry[name].shape == parameter.shape and entry[name].dtype == parameter.dtype
                    and bool(torch.isfinite(entry[name]).all()) for name in ('exp_avg', 'exp_avg_sq')),
            'Invalid Adam moment shape, dtype or update count')


def readout(model, scene, views, labels, arm):
    scores, graphs = diag.forward_rows(model, scene, views, arm)
    metrics = {}
    for head, values in scores.items():
        check = independent.recompute(values.numpy(), labels)
        device.require(independent.equivalent(check, diag.score(values, labels)),
                       'Independent fixed-point metrics differ')
        metrics[head] = check
    return scores, {'heads': metrics, 'graphs': graphs}


def case_audit(folder, scope, reg, case, sha, numeric):
    tick = perf_counter()
    folder = Path(folder)
    known = [folder / name for name in ('state.pt', 'result.json', 'completed-resume.json')]
    known.extend(sorted((folder / 'calls').glob('*.json')))
    case_pins = {str(path): hash_file(path) for path in known}
    state, result, receipts = train_state(folder, scope, reg, case, sha, numeric)
    plan = read_pinned(Path(reg['plan']), reg['pins'][reg['plan']])
    exposure_path = reg['exposure']['plan']
    exposure = read_pinned(Path(exposure_path), reg['pins'][exposure_path])
    geometry, allowed, scene, frozen = prior.inputs(reg, plan, exposure)
    release = prior.released_roles(reg, plan, geometry, allowed, case)
    release_path = reg['cases'][case['index']]['release']
    raw = read_pinned(Path(release_path), reg['pins'][release_path])
    cohort = f's{case["support_seed"]}-k{case["k"]}'
    expected = scope['exposure'][cohort]
    _, exposure_row = p4_exposure.bind(reg, reg['cases'][case['index']], release['train'])
    counts = prior.check_exposure(state, result, release, exposure_row, exposure)
    sequence = [row['tile'] for row in state['trace']]
    actual = registration.per_label_exposure(raw, release, scene.plan['shape'][1], sequence, exposure['validation_steps'])
    device.require(all(expected[k] == v for k, v in actual.items()) and
                   expected['sequence_sha256'] == digest(sequence), 'Per-label exposure differs from the new scope')
    chosen = prior.chosen_event(state, result, scope['budget']['validation_steps'])
    populations = {role: diag.fixed_population(release[role], allowed, role, scene.plan['shape'][1], budget)
                   for role, budget in (('train', case['k']), ('val', 5))}
    device.require({role: value[2] for role, value in populations.items()} == expected['populations'],
                   'Fixed train/devval population identity differs')
    conditions = diag.freeze_conditions(scene.bands, scene.plan['plan_hash'], frozen['records']['val'])
    device.require(conditions == scope['observation_diagnostics'], 'Seven frozen observation contracts differ')
    reference = {role: diag.fixed_views(scene, allowed, populations[role][0],
                 {'sampling': frozen['records'][role], 'metadata': 'nominal'}, role) for role in populations}
    target_device = torch.device(numeric['device'])
    rng = _tree_fingerprint(_rng_state(target_device))
    with torch.random.fork_rng(devices=[0] if target_device.type == 'cuda' else []):
        model, initial = registration.initialize_case(reg, case)
    device.require(initial == case['initialization'], 'Audit initialization identity differs')
    model.to(target_device).eval()
    check_optimizer(model, SpatialConfig(reg['config']), state)
    snapshots = {'initial': initial['initial_state_sha256']}
    readouts, cache, selected, forwards = {}, {}, None, 0
    for stage in ('initial', 'selected', 'last'):
        if stage != 'initial':
            weights = state['best']['model'] if stage == 'selected' else state['model']
            model.load_state_dict(weights, strict=True)
        digest_before = state_hash(model)
        snapshots[stage] = digest_before
        if digest_before not in cache:
            cache[digest_before] = {role: readout(model, scene, view, populations[role][1], case['arm'])
                                    for role, view in reference.items()}
            forwards += sum(len(view) for view in reference.values())
        outputs = cache[digest_before]
        if stage == 'selected':
            selected = outputs['val']
            device.require(all(torch.equal(selected[0][head], state['best']['scores'][head]) for head in prior.HEADS),
                           'Selected probabilities differ after independent same-device reload')
            prior.metrics(selected[0], populations['val'][1], chosen['metrics'])
        readouts[stage] = {role: outcome[1] for role, outcome in outputs.items()}
        device.require(state_hash(model) == digest_before, 'Frozen fitting diagnostic changed model or DomainBN')
    model.load_state_dict(state['best']['model'], strict=True)
    observation_cache = {diag.REFERENCE: selected}
    observation_rows = {}
    for name, condition in conditions.items():
        if condition['status'] != 'ready':
            observation_rows[name] = condition
            continue
        alias = condition['same_bands_as']
        if alias is None and name != diag.REFERENCE:
            views = diag.fixed_views(scene, allowed, populations['val'][0], condition, 'val')
            observation_cache[name] = readout(model, scene, views, populations['val'][1], case['arm'])
            forwards += len(views)
        scores, evidence = observation_cache[alias or name]
        observation_rows[name] = {**evidence, 'versus_reference': {
            head: independent.difference(scores[head].numpy(), selected[0][head].numpy(),
                evidence['heads'][head], selected[1]['heads'][head]) for head in prior.HEADS}}
    device.require(state_hash(model) == snapshots['selected'] and _tree_fingerprint(_rng_state(target_device)) == rng,
                   'Frozen observation review changed model, DomainBN or RNG')
    batch.check_pins({**scope['protected_files'], **case_pins})
    return {'status': 'passed', 'scope_sha256': sha, 'case': case, 'numeric': numeric,
        'optimizer_updates': sum(row['new_optimizer_updates'] for row in receipts),
        'completed_resume_updates': 0, 'validation_steps': scope['budget']['validation_steps'],
        'nonselected_event_probability_replays': 0, 'selected_step': chosen['step'],
        'initialization': initial, 'state_sha256': snapshots, 'artifact_pins': case_pins,
        'first_all_classes_step': counts['first_all_classes_step'], 'exposure': actual,
        'populations': expected['populations'], 'fit': readouts, 'observations': observation_rows,
        'actual_forward_calls': forwards, 'coverage_participation': prior.participation(
            state['trace'], initial, reg['coverage']['records']['train']),
        'phases': [{'start': row['start'], 'until': row['until'], 'wall_seconds': row['wall_seconds']}
                   for row in receipts], 'audit_wall_seconds': perf_counter() - tick,
        'audit_optimizer_updates': 0, 'raw_gt_loaded': False, 'query_accessed': False}
