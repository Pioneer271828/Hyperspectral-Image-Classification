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
from transfer.g2_contract import read_pinned
from transfer.p4_spatial import initialize, load_release


POLICY = 'wp05_full_target_budget_v1'
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


def matrix(reg, protocol, gpus):
    slots = device.slots(protocol, gpus)
    keys = [(r['support_seed'], r['k'], r['method']) for r in reg['cases']]
    expected = {(seed, k, method) for seed, k in batch.COHORTS for method in batch.METHODS}
    device.require(len(keys) == len(expected) and set(keys) == expected, 'Exactly eight paired source/target roles required')
    cases = []
    for index, (support_seed, k, method) in enumerate(keys):
        for ordinal, model_seed in enumerate(MODEL_SEEDS):
            # All five arms of a comparison run on the same physical card.
            gpu = slots[(index * len(MODEL_SEEDS) + ordinal) % len(slots)]
            for arm in coverage.ARMS:
                cases.append({'id': f's{support_seed}-k{k}-{method}-m{model_seed}-{arm}', 'index': index,
                    'support_seed': support_seed, 'k': k, 'method': method, 'model_seed': model_seed,
                    'sampler_seed': SAMPLER_SEED, 'arm': arm, 'physical_uuid': gpu, 'protocol': protocol,
                    'observation_sha256': reg['coverage']['records']['train']['record_hash']})
    return cases


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
            steps=STEPS, interval=INTERVAL):
    tick = perf_counter()
    require_frozen_code(code)
    parent, prior = Path(parent).resolve(), Path(prior).resolve()
    if not synthetic:
        device.require((parent_sha, prior_sha, steps, interval) == (batch.PARENT_SHA, batch.PRIOR_SHA, STEPS, INTERVAL),
                       'Unaccepted HongHu source or full research budget differs')
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
    floor_path = reg['exposure']['floor']
    floor = read_pinned(Path(floor_path), reg['pins'][floor_path])
    validation_steps = events(steps, interval)
    rows, evidence, seen = [], {}, set()
    for case in reg['cases']:
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
    reg['full_budget_policy'] = POLICY
    stats = read_pinned(Path(reg['statistics']), reg['pins'][reg['statistics']])
    bands = read_band_metadata(guarded.SCENE, len(stats['minimum']))
    reg['coverage'] = coverage.freeze(reg, plan, bands, observation=batch.OBSERVATION, seed=3031, metadata='nominal')
    conditions = diag.freeze_conditions(bands, guarded_plan['plan_hash'], reg['coverage']['records']['val'])
    cases = matrix(reg, protocol, gpus)
    common = {}
    for case in cases:
        _, init = initialize_case(reg, case)
        pair = pair_identity(reg, case, init)
        key = case['index'], case['model_seed']
        device.require(common.setdefault(key, pair) == pair, 'Coverage arms have different common initializations')
        case.update(initialization=init, pair_sha256=pair)
    for index in range(len(reg['cases'])):
        fresh = [r['initialization']['base_initial_state_sha256'] for r in cases if r['index'] == index and r['arm'] == 'none']
        device.require(len(set(fresh)) == len(MODEL_SEEDS), 'Model seeds did not yield distinct initial states')
    reg_path = out / 'registration.json'
    write_json(reg_path, reg)
    pins = {**reg['pins'], str(reg_path): hash_file(reg_path), str(parent): parent_sha, str(prior): prior_sha}
    batch.check_pins(pins)
    require_frozen_code(code)
    scope = {'policy': POLICY, 'commit': code['commit'], 'source_hash': code['source_hash'],
        'synthetic': synthetic, 'registration': str(reg_path), 'parent_registration': str(parent),
        'prior_receipt': str(prior), 'protected_files': pins, 'protocol': protocol, 'gpus': list(gpus),
        'source': {'fixed_seed': 64, 'producer_commit': None if synthetic else SOURCE_PRODUCER,
                   'weight_sha256': reg['pins'][reg['source']['weight']]},
        'budget': {'steps': steps, 'validation_steps': validation_steps, 'selection': coverage.SELECTION,
                   'planned_optimizer_updates': steps * len(cases), 'checkpoint_interval': 1},
        'cases': cases, 'exposure': evidence, 'observation_diagnostics': conditions,
        'preflight': {'optimizer_updates': 0, 'forward_calls': 0, 'query_accessed': False, 'raw_gt_loaded': False,
                      'initializations': len(cases), 'wall_seconds': perf_counter() - tick}}
    scope['scope_hash'] = digest(scope)
    path = out / 'scope.json'
    write_json(path, scope)
    return path


def load(path, sha, code, *, check_files=True):
    require_frozen_code(code)
    path = Path(path).resolve()
    frozen = read_pinned(path, sha)
    device.require(frozen['policy'] == POLICY and frozen['commit'] == code['commit']
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
    expected_keys = matrix(reg, frozen['protocol'], frozen['gpus'])
    device.require(type(synthetic) is bool and accepted['status'] == 'passed'
        and accepted['query_accessed'] is False and accepted['protected_files'].get(parent) == pins[parent]
        and set(pins) == set(reg['pins']) | {reg_path, parent, prior}
        and all(pins[k] == v for k, v in reg['pins'].items())
        and all(reg[k] == original[k] for k in ('image', 'plan', 'development', 'statistics', 'config', 'source', 'cases', 'guarded'))
        and all(reg['pins'].get(k) == v for k, v in original['pins'].items())
        and reg['full_budget_policy'] == POLICY
        and reg['coverage']['steps'] == budget['steps'] and reg['coverage']['validation_steps'] == budget['validation_steps']
        and reg['coverage']['selection'] == budget['selection'] == coverage.SELECTION
        and reg['coverage']['observation'] == batch.OBSERVATION and reg['coverage']['metadata'] == 'nominal'
        and reg['coverage']['observation_seed'] == 3031 and budget['checkpoint_interval'] == 1
        and budget['planned_optimizer_updates'] == 120 * budget['steps']
        and (synthetic or (pins[parent], pins[prior], budget['steps'], budget['validation_steps'])
             == (batch.PARENT_SHA, batch.PRIOR_SHA, STEPS, events(STEPS, INTERVAL))),
        'Full budget input, observation or validation contract differs')
    got = [{k: v for k, v in case.items() if k not in ('initialization', 'pair_sha256')} for case in frozen['cases']]
    device.require(got == expected_keys and all(case['pair_sha256'] == pair_identity(reg, case, case['initialization'])
                                               for case in frozen['cases']), 'Full case identity matrix differs')
    if check_files:
        batch.check_pins(pins)
    return frozen, reg
