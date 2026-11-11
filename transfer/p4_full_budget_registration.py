"""Freeze a new 120-case P4 budget without changing accepted short-run inputs."""
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch

from loadData.data_read import read_band_metadata
from model.research_config import digest
from research_runtime import hash_file, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer import p4_coverage as coverage, p4_coverage_batch as batch, p4_exposure as exposure
from transfer import p4_frozen_diagnostics as diag, p4_guarded as guarded, p4_runtime_device as device
from transfer import p4_update_evidence as updates
from transfer.g2_contract import read_pinned
from transfer.g3_training import _optimizer
from transfer.p4_spatial import initialize, load_release


POLICY = 'wp05_full_target_budget_v1'
LEGACY = 'full_budget'
AUDITED = 'audited_full_budget'
ADMISSION = 'real_admission'
POLICIES = {LEGACY: POLICY, AUDITED: 'wp05_full_target_budget_v2', ADMISSION: 'wp05_honghu_cuda_admission_v1'}
ADMISSION_STEPS = 60
ADMISSION_PATHS = {'full': [60], 'split': [2, 30, 60]}
MODEL_SEEDS = (64, 512, 2048)
SAMPLER_SEED = 64
STEPS = 1500
INTERVAL = 30
SOURCE_PRODUCER = '52f32978419cbeda216a90bfb0ed76c393ef4f5d'
SOURCE_WEIGHT_SHA = '92f732e8ab9b675cb2379d30e12cd6521415926c4e59d3f2fdb02948d09b09c9'


def events(steps, interval):
    device.require(type(steps) is type(interval) is int and 3 <= interval <= steps and steps % interval == 0,
                   'Require ordered equal intervals including the terminal validation')
    return list(range(interval, steps + 1, interval))


def matrix(reg, protocol, gpus, purpose=LEGACY):
    slots = device.slots(protocol, gpus)
    keys = [(r['support_seed'], r['k'], r['method']) for r in reg['cases']]
    expected = {(seed, k, method) for seed, k in batch.COHORTS for method in batch.METHODS}
    device.require(len(keys) == len(expected) and set(keys) == expected, 'Exactly eight paired source/target roles required')
    cases = []
    for index, (support_seed, k, method) in enumerate(keys):
        if purpose == ADMISSION and support_seed != 2048:
            continue
        seeds = (512,) if purpose == ADMISSION else MODEL_SEEDS
        for ordinal, model_seed in enumerate(seeds):
            # All five arms of a comparison run on the same physical card.
            pair = (k == 5) + (method == 'shared_transfer') if purpose == ADMISSION else index * len(seeds) + ordinal
            gpu = slots[pair % len(slots)]
            for arm in coverage.ARMS:
                cases.append({'id': f's{support_seed}-k{k}-{method}-m{model_seed}-{arm}', 'index': index,
                    'support_seed': support_seed, 'k': k, 'method': method, 'model_seed': model_seed,
                    'sampler_seed': SAMPLER_SEED, 'arm': arm, 'physical_uuid': gpu, 'protocol': protocol,
                    'observation_sha256': reg['coverage']['records']['train']['record_hash']})
    return cases


def check_budget(purpose, protocol, gpus, synthetic, steps, interval):
    device.require(purpose in POLICIES, 'Unknown P4 registration purpose')
    if purpose == ADMISSION:
        device.require((steps, interval) == (ADMISSION_STEPS, INTERVAL)
                       and (synthetic or (protocol == device.CUDA_AUDITED and len(gpus) == 2)),
                       'Admission requires two registered audited CUDA GPUs and the 60-step paired budget')
    elif not synthetic:
        device.require((steps, interval) == (STEPS, INTERVAL), 'Full research budget differs')
    device.require(protocol != device.CUDA_AUDITED if purpose == LEGACY else protocol != device.CUDA,
                   'Numerical protocol does not match the evidence schema')


def qualification_bindings(paths, gpus, protocol, code, shape_contract):
    from transfer.p4_cuda_qualification import check_receipt

    receipts, pins = {}, {}
    for path in paths:
        path = Path(path).resolve()
        checksum = hash_file(path)
        row = read_pinned(path, checksum)
        gpu = check_receipt(row, protocol, code, shape_contract)
        device.require(gpu in gpus and gpu not in receipts, 'Duplicate or unregistered GPU qualification')
        receipts[gpu] = str(path)
        pins[str(path)] = checksum
    device.require(set(receipts) == set(gpus), 'Every registered CUDA device needs its own qualification')
    return receipts, pins


def role_inputs(reg):
    pins = reg['pins']
    plan, geometry = [read_pinned(Path(reg[k]), pins[reg[k]]) for k in ('plan', 'development')]
    plan_guarded, allowed = guarded.bind_inputs(reg, plan, geometry)
    roles = {}
    for case in reg['cases']:
        path = case['release']
        if path not in roles:
            raw = read_pinned(Path(path), pins[path])
            released = load_release(raw, plan, case['support_seed'], case['k'], geometry, guarded_tiles=allowed)
            roles[path] = (raw, released)
    return plan_guarded, allowed, roles


def per_label_exposure(raw, released, width, sequence, validation_steps):
    """Counts indexed by the ordered released support list, with no coordinates in the receipt."""
    ids = raw['support']['indices']
    id_slot = {index: slot for slot, index in enumerate(ids)}
    membership = np.full(len(ids), -1, dtype=np.int64)
    for tile, local in released['train']:
        y0, _, x0, _ = tile['box']
        y, x = np.nonzero(local)
        membership[[id_slot[i] for i in ((y + y0) * width + x + x0).tolist()]] = tile['id']
    device.require(bool((membership >= 0).all()), 'Support point has no authorized training tile')
    count = np.cumsum(np.asarray(sequence)[:, None] == membership, axis=0)
    first = np.flatnonzero((count > 0).all(axis=1))
    labels = np.asarray(raw['support']['labels'])
    return {'support_order_sha256': digest(raw['support']), 'support_label_count': len(ids),
        'first_all_support_step': int(first[0]) + 1 if len(first) else None,
        'unique_support_at_end': int((count[-1] > 0).sum()),
        'per_label_counts_at_events': {str(step): count[step - 1].tolist() for step in validation_steps},
        'class21_counts_at_events': {str(step): int(count[step - 1][labels == 21].sum())
                                     for step in validation_steps}}


def common_init(initialization):
    return {key: value for key, value in initialization.items() if key not in ('coverage', 'initial_state_sha256')}


def pair_identity(reg, case, initialization):
    fields = ('index', 'support_seed', 'k', 'method', 'model_seed', 'sampler_seed',
              'physical_uuid', 'protocol', 'observation_sha256')
    return digest({'registration_sha256': digest(reg), 'case': {k: case[k] for k in fields},
                   'common_initialization': common_init(initialization)})


def initialize_case(reg, case):
    source = reg['source'] if case['method'] == 'shared_transfer' else None
    with torch.random.fork_rng(devices=[]):
        return initialize(SpatialConfig(reg['config']), source['weight'] if source else None,
                          source['result'] if source else None, coverage_arm=case['arm'],
                          model_seed=case['model_seed'])


def prepare(parent, parent_sha, prior, prior_sha, out, code, *, protocol, gpus, synthetic=False,
            steps=None, interval=INTERVAL, purpose=LEGACY, qualifications=()):
    tick = perf_counter()
    require_frozen_code(code)
    steps = (ADMISSION_STEPS if purpose == ADMISSION else STEPS) if steps is None else steps
    check_budget(purpose, protocol, gpus, synthetic, steps, interval)
    parent, prior = Path(parent).resolve(), Path(prior).resolve()
    if not synthetic:
        device.require((parent_sha, prior_sha) == (batch.PARENT_SHA, batch.PRIOR_SHA),
                       'Unaccepted HongHu source')
    accepted = read_pinned(prior, prior_sha)
    device.require(accepted['status'] == 'passed' and accepted['query_accessed'] is False
                   and accepted['protected_files'].get(str(parent)) == parent_sha, 'Parent release was not accepted')
    reg = coverage.prepare(parent, parent_sha, commit=code['commit'], observation=batch.OBSERVATION,
                           seed=3031, metadata='nominal')
    original_plan = reg['exposure']['plan']
    if not synthetic:
        device.require(reg['pins'][original_plan] == batch.PLAN_SHA
                       and reg['pins'][reg['source']['weight']] == SOURCE_WEIGHT_SHA,
                       'Original exposure or fixed seed-64 source differs')
    batch.check_pins(reg['pins'])
    guarded_plan, allowed, releases = role_inputs(reg)
    requires_qualification = not synthetic and protocol == device.CUDA_AUDITED
    qualification, qualification_pins, shape_contract = {}, {}, None
    if requires_qualification:
        from transfer.p4_cuda_qualification import geometry_contract

        shape_contract = geometry_contract(guarded_plan, allowed, reg['config'])
        qualification, qualification_pins = qualification_bindings(qualifications, gpus, protocol, code, shape_contract)
    else:
        device.require(not qualifications, 'Qualification receipts apply only to real audited CUDA scopes')
    floor_path = reg['exposure']['floor']
    floor = read_pinned(Path(floor_path), reg['pins'][floor_path])
    validation_steps = events(steps, interval)
    rows, evidence, seen = [], {}, set()
    for case in reg['cases']:
        if purpose == ADMISSION and case['support_seed'] != 2048:
            continue
        k, path = case['k'], case['release']
        cohort_id = f's{case["support_seed"]}-k{k}'
        if cohort_id in seen:
            continue
        seen.add(cohort_id)
        raw, release = releases[path]
        row = exposure.cohort(floor, reg, case, release['train'], steps=steps,
                              events=validation_steps, seed=SAMPLER_SEED)
        rows.append(row)
        populations = {role: diag.fixed_population(release[role], allowed, role, guarded_plan['shape'][1], budget)[2]
                       for role, budget in (('train', k), ('val', 5))}
        counts = per_label_exposure(raw, release, guarded_plan['shape'][1], row['tile_sequence'], validation_steps)
        evidence[cohort_id] = {'release_sha256': row['release_sha256'], 'populations': populations,
            'sequence_sha256': digest(row['tile_sequence']), 'first_all_classes_step': row['first_all_classes_step'],
            **counts}
    device.require(len({r['populations']['val']['sha256'] for r in evidence.values()}) == 1,
                   'All cases must use the same fixed development validation')
    plan = {'policy': exposure.POLICY, 'geometry_sha256': reg['pins'][reg['development']],
            'floor_sha256': reg['pins'][floor_path], 'steps': steps, 'validation_steps': validation_steps,
            'sampler_seed': SAMPLER_SEED, 'cohorts': rows}
    plan['plan_hash'] = digest(plan)
    out = batch.sibling(out, parent)
    device.require(not out.exists(), 'Full budget registration output already exists')
    out.mkdir(exist_ok=False)
    full_plan_path = out / 'exposure.json'
    write_json(full_plan_path, plan)
    reg['exposure'] = {'plan': str(full_plan_path), 'floor': floor_path}
    reg['pins'][str(full_plan_path)] = hash_file(full_plan_path)
    reg['full_budget_policy'] = POLICIES[purpose]
    if purpose != LEGACY:
        reg['parameter_audit'] = updates.SCHEMA
    stats = read_pinned(Path(reg['statistics']), reg['pins'][reg['statistics']])
    bands = read_band_metadata(guarded.SCENE, len(stats['minimum']))
    reg['coverage'] = coverage.freeze(reg, plan, bands, observation=batch.OBSERVATION, seed=3031, metadata='nominal')
    conditions = diag.freeze_conditions(bands, guarded_plan['plan_hash'], reg['coverage']['records']['val'])
    cases = matrix(reg, protocol, gpus, purpose)
    common = {}
    for case in cases:
        model, init = initialize_case(reg, case)
        pair = pair_identity(reg, case, init)
        key = case['index'], case['model_seed']
        device.require(common.setdefault(key, pair) == pair, 'Coverage arms have different common initializations')
        case.update(initialization=init, pair_sha256=pair)
        if purpose != LEGACY:
            optimizer = _optimizer(model, SpatialConfig(reg['config']))
            case['parameter_layout_sha256'] = digest(updates.layout(model, optimizer))
    if purpose != ADMISSION:
        for index in range(len(reg['cases'])):
            fresh = [r['initialization']['base_initial_state_sha256'] for r in cases if r['index'] == index and r['arm'] == 'none']
            device.require(len(set(fresh)) == len(MODEL_SEEDS), 'Model seeds did not yield distinct initial states')
    reg_path = out / 'registration.json'
    write_json(reg_path, reg)
    pins = {**reg['pins'], str(reg_path): hash_file(reg_path), str(parent): parent_sha, str(prior): prior_sha,
            **qualification_pins}
    batch.check_pins(pins)
    require_frozen_code(code)
    paths = ADMISSION_PATHS if purpose == ADMISSION else None
    scope = {'policy': POLICIES[purpose], 'commit': code['commit'], 'source_hash': code['source_hash'],
        'synthetic': synthetic, 'registration': str(reg_path), 'parent_registration': str(parent),
        'prior_receipt': str(prior), 'protected_files': pins, 'protocol': protocol, 'gpus': list(gpus),
        'source': {'fixed_seed': 64, 'producer_commit': None if synthetic else SOURCE_PRODUCER,
                   'weight_sha256': reg['pins'][reg['source']['weight']]},
        'budget': {'steps': steps, 'validation_steps': validation_steps, 'selection': coverage.SELECTION,
            'planned_optimizer_updates': steps * len(cases) * (len(paths) if paths else 1), 'checkpoint_interval': 1},
        'cases': cases, 'exposure': evidence, 'observation_diagnostics': conditions,
        'preflight': {'optimizer_updates': 0, 'forward_calls': 0, 'query_accessed': False, 'raw_gt_loaded': False,
                      'initializations': len(cases), 'wall_seconds': perf_counter() - tick}}
    if purpose != LEGACY:
        scope.update(purpose=purpose, parameter_audit=updates.SCHEMA, paths=paths, qualifications=qualification,
                     operator_geometry=shape_contract)
    scope['scope_hash'] = digest(scope)
    path = out / 'scope.json'
    write_json(path, scope)
    return path


def load(path, sha, code, *, check_files=True):
    require_frozen_code(code)
    path = Path(path).resolve()
    frozen = read_pinned(path, sha)
    purpose = frozen.get('purpose', LEGACY)
    device.require(purpose in POLICIES and frozen['policy'] == POLICIES[purpose] and frozen['commit'] == code['commit']
                   and frozen['source_hash'] == code['source_hash']
                   and frozen['scope_hash'] == digest({k: v for k, v in frozen.items() if k != 'scope_hash'}),
                   'Full budget checkout or scope identity differs')
    pins = frozen['protected_files']
    reg_path = frozen['registration']
    reg = read_pinned(Path(reg_path), pins[reg_path])
    parent, prior = (frozen[k] for k in ('parent_registration', 'prior_receipt'))
    original = read_pinned(Path(parent), pins[parent])
    accepted = read_pinned(Path(prior), pins[prior])
    synthetic = frozen['synthetic']
    budget = frozen['budget']
    expected_keys = matrix(reg, frozen['protocol'], frozen['gpus'], purpose)
    validation = budget['validation_steps']
    check_budget(purpose, frozen['protocol'], frozen['gpus'], synthetic, budget['steps'], validation[0])
    paths = ADMISSION_PATHS if purpose == ADMISSION else None
    schema = None if purpose == LEGACY else updates.SCHEMA
    quals = frozen.get('qualifications', {})
    if not synthetic and frozen['protocol'] == device.CUDA_AUDITED:
        from transfer.p4_cuda_qualification import check_receipt

        device.require(set(quals) == set(frozen['gpus']) and all(
            check_receipt(read_pinned(Path(path), pins[path]), frozen['protocol'], code, frozen['operator_geometry']) == gpu
            for gpu, path in quals.items()), 'Registered GPU operator qualification differs')
    else:
        device.require(not quals, 'Unregistered GPU operator qualification')
    device.require(type(synthetic) is bool and accepted['status'] == 'passed'
        and accepted['query_accessed'] is False and accepted['protected_files'].get(parent) == pins[parent]
        and set(pins) == set(reg['pins']) | {reg_path, parent, prior} | set(quals.values())
        and all(pins[k] == v for k, v in reg['pins'].items())
        and all(reg[k] == original[k] for k in ('image', 'plan', 'development', 'statistics', 'config', 'source', 'cases', 'guarded'))
        and all(reg['pins'].get(k) == v for k, v in original['pins'].items())
        and reg['full_budget_policy'] == frozen['policy'] and reg.get('parameter_audit') == frozen.get('parameter_audit') == schema
        and frozen.get('paths') == paths and validation == events(budget['steps'], validation[0])
        and reg['coverage']['steps'] == budget['steps'] and reg['coverage']['validation_steps'] == budget['validation_steps']
        and reg['coverage']['selection'] == budget['selection'] == coverage.SELECTION
        and reg['coverage']['observation'] == batch.OBSERVATION and reg['coverage']['metadata'] == 'nominal'
        and reg['coverage']['observation_seed'] == 3031 and budget['checkpoint_interval'] == 1
        and budget['planned_optimizer_updates'] == len(expected_keys) * budget['steps'] * (len(paths) if paths else 1)
        and (synthetic or (pins[parent], pins[prior]) == (batch.PARENT_SHA, batch.PRIOR_SHA)),
        'Full budget input, observation or validation contract differs')
    got = [{k: v for k, v in case.items() if k not in ('initialization', 'pair_sha256', 'parameter_layout_sha256')}
           for case in frozen['cases']]
    device.require(got == expected_keys and all(case['pair_sha256'] == pair_identity(reg, case, case['initialization'])
           and (case.get('parameter_layout_sha256') is None if schema is None else
                bool(re.fullmatch('[a-f0-9]{64}', case.get('parameter_layout_sha256', ''))))
           for case in frozen['cases']), 'Full case identity matrix differs')
    if check_files:
        batch.check_pins(pins)
    return frozen, reg
