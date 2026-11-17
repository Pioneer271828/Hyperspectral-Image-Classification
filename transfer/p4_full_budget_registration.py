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
RECOVERY = 'synthetic_same_device_recovery'
POLICIES = {LEGACY: POLICY, AUDITED: 'wp05_full_target_budget_v2', ADMISSION: 'wp05_honghu_cuda_admission_v1',
            RECOVERY: 'wp05_cuda_fp64_weight_same_device_recovery_v1'}
ADMISSION_STEPS = 60
ADMISSION_PATHS = {'full': [60], 'split': [2, 30, 60]}
PAIRED = (ADMISSION, RECOVERY)
RECOVERY_SCHEMA = 'p4_cuda_fp64_weight_recovery_acceptance_v1'
REAL_GEOMETRY = 'ca67834056aebdf44e47315bfa4de971283746c7679f63ce0489e99617fd41aa'
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
    slots = device.slots(protocol, gpus, recovery=purpose == RECOVERY)
    keys = [(r['support_seed'], r['k'], r['method']) for r in reg['cases']]
    expected = {(seed, k, method) for seed, k in batch.COHORTS for method in batch.METHODS}
    device.require(len(keys) == len(expected) and set(keys) == expected, 'Exactly eight paired source/target roles required')
    cases = []
    for index, (support_seed, k, method) in enumerate(keys):
        if purpose in PAIRED and support_seed != 2048:
            continue
        if purpose == RECOVERY and (k, method) != (5, 'scratch'):
            continue
        seeds = (512,) if purpose in PAIRED else MODEL_SEEDS
        for ordinal, model_seed in enumerate(seeds):
            # All five arms of a comparison run on the same physical card.
            pair = (k == 5) + (method == 'shared_transfer') if purpose == ADMISSION else index * len(seeds) + ordinal
            gpu = slots[pair % len(slots)]
            for arm in (('none',) if purpose == RECOVERY else coverage.ARMS):
                cases.append({'id': f's{support_seed}-k{k}-{method}-m{model_seed}-{arm}', 'index': index,
                    'support_seed': support_seed, 'k': k, 'method': method, 'model_seed': model_seed,
                    'sampler_seed': SAMPLER_SEED, 'arm': arm, 'physical_uuid': gpu, 'protocol': protocol,
                    'observation_sha256': reg['coverage']['records']['train']['record_hash']})
    return cases


def check_budget(purpose, protocol, gpus, synthetic, steps, interval):
    device.require(purpose in POLICIES, 'Unknown P4 registration purpose')
    device.require(protocol != device.CUDA_FP64_WEIGHT or purpose in PAIRED,
                   'FP64 sparse weight arithmetic requires an explicit paired scope')
    device.require(not synthetic or protocol != device.CUDA_FP64_WEIGHT or purpose == RECOVERY,
                   'Synthetic FP64-weight tests must use the single-device recovery scope')
    if purpose == ADMISSION:
        device.require((steps, interval) == (ADMISSION_STEPS, INTERVAL)
                       and (synthetic or (protocol in device.CUDA_QUALIFIED and len(gpus) == 2)),
                       'Admission requires two registered audited CUDA GPUs and the 60-step paired budget')
    elif purpose == RECOVERY:
        device.require(synthetic is True and (steps, interval) == (ADMISSION_STEPS, INTERVAL)
                       and (protocol, len(gpus)) in ((device.CPU, 0), (device.CUDA_FP64_WEIGHT, 1)),
                       'Recovery requires a bounded synthetic scope and a single physical CUDA UUID')
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


def recovery_geometry(qualifications):
    device.require(len(qualifications) == 1, 'One physical qualification is required for recovery')
    path = Path(qualifications[0]).resolve(strict=True)
    geometry = read_pinned(path, hash_file(path))['geometry']
    device.require(geometry['geometry_sha256'] == REAL_GEOMETRY, 'Recovery qualification needs the original real geometry')
    return geometry


def bind_recovery(path, purpose, protocol, synthetic, code):
    required = purpose == ADMISSION and protocol == device.CUDA_FP64_WEIGHT and not synthetic
    device.require((path is not None) == required, 'Real FP64-weight admission requires the accepted physical recovery receipt')
    if not required:
        return {}
    path = Path(path).resolve(strict=True)
    checksum = hash_file(path)
    proof = read_pinned(path, checksum)
    gpu = proof['physical_uuid']
    device.require(proof['schema'] == RECOVERY_SCHEMA and proof['status'] == 'passed'
        and (proof['commit'], proof['source_hash']) == (code['commit'], code['source_hash'])
        and proof['protocol'] == protocol and device.UUID.fullmatch(gpu)
        and proof['optimizer_updates'] == 240 and proof['completed_resume_updates'] == 0
        and [row['classes'] for row in proof['cases']] == [9, 22],
        'Physical FP64-weight recovery source, device or budget differs')
    pins = {str(path): checksum}
    for row in proof['cases']:
        scope_path, acceptance_path = (str(Path(row[key]).resolve(strict=True)) for key in ('scope', 'acceptance'))
        scope = read_pinned(Path(scope_path), row['scope_sha256'])
        receipt = read_pinned(Path(acceptance_path), row['acceptance_sha256'])
        device.require(scope['synthetic'] is True and scope['purpose'] == RECOVERY
            and scope['commit'] == code['commit'] and scope['source_hash'] == code['source_hash']
            and scope['recovery_classes'] == row['classes'] and scope['gpus'] == [gpu]
            and scope['paths'] == ADMISSION_PATHS and scope['protocol'] == protocol
            and receipt['status'] == 'passed' and receipt['scope_sha256'] == row['scope_sha256']
            and receipt['purpose'] == RECOVERY and receipt['producer_commit'] == code['commit']
            and receipt['synthetic'] is True and receipt['cases'] == 1 and receipt['optimizer_updates'] == 120
            and receipt['audit_optimizer_updates'] == 0
            and receipt['query_accessed'] is receipt['raw_gt_loaded'] is False,
            'Paired physical recovery acceptance differs')
        audit_path, audit_sha = next(iter(receipt['case_audits'].items()))
        audit = read_pinned(Path(audit_path), audit_sha)
        device.require(audit['numeric']['physical_uuid'] == gpu and audit['numeric']['protocol'] == protocol
            and audit['class_count'] == row['classes'] and audit['optimizer_updates'] == 120
            and audit['completed_resume_updates'] == 0 and audit['status'] == 'passed'
            and audit['exact_path_fingerprints']['full'] == audit['exact_path_fingerprints']['split']
            and audit['parameter_audit']['schema'] == updates.SCHEMA,
            'Physical paired recovery did not verify class or scientific state identity')
        for name, sha in {scope_path: row['scope_sha256'], acceptance_path: row['acceptance_sha256'],
                          **receipt['protected_files']}.items():
            device.require(pins.setdefault(name, sha) == sha, 'Conflicting physical recovery input fingerprint')
    return pins


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


def fixed_population(tiles, allowed, role, width, budget, recovery_classes=None):
    if recovery_classes != 9:
        return diag.fixed_population(tiles, allowed, role, width, budget)
    permitted = {tile['id']: tile for tile in allowed[role]}
    layout, labels, coordinates = [], [], []
    for tile, local in tiles:
        y0,y1,x0,x1 = tile['box']
        device.require(permitted.get(tile['id']) == tile and tile['partition'] == role
            and local.shape == (y1-y0,x1-x0) and np.all((local>=0)&(local<=9)),
            'Nine-class recovery labels escape the authorized role')
        rows = np.flatnonzero(local.ravel())
        iy,ix = np.divmod(rows,x1-x0)
        labels.extend(local.ravel()[rows].tolist())
        coordinates.extend(((iy+y0)*width+ix+x0).tolist())
        layout.append((tile,rows))
    device.require(len(set(coordinates)) == 9*budget
        and np.bincount(labels,minlength=10).tolist() == [0]+[budget]*9,
        'Nine-class fixed support or coordinate uniqueness differs')
    return layout, np.asarray(labels), {'sha256':digest({'coordinates':coordinates,'labels':labels}),
        'labels':len(labels),'per_class':budget,'tiles':len(layout)}


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
            steps=None, interval=INTERVAL, purpose=LEGACY, qualifications=(), recovery_receipt=None,
            recovery_classes=None):
    tick = perf_counter()
    require_frozen_code(code)
    steps = (ADMISSION_STEPS if purpose in PAIRED else STEPS) if steps is None else steps
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
    requires_qualification = ((not synthetic and protocol in device.CUDA_QUALIFIED)
                              or (purpose == RECOVERY and protocol == device.CUDA_FP64_WEIGHT))
    qualification, qualification_pins, shape_contract = {}, {}, None
    if requires_qualification:
        from transfer.p4_cuda_qualification import geometry_contract

        shape_contract = (recovery_geometry(qualifications) if purpose == RECOVERY else
                          geometry_contract(guarded_plan, allowed, reg['config']))
        qualification, qualification_pins = qualification_bindings(qualifications, gpus, protocol, code, shape_contract)
    else:
        device.require(not qualifications, 'Qualification receipts apply only to real audited CUDA scopes')
    if purpose == RECOVERY:
        device.require(recovery_classes in (9, 22) and all(
            len(raw['validation']['labels']) == 5*recovery_classes for raw, _ in releases.values()),
            'Synthetic recovery requires the declared 9- or 22-class release')
    else:
        device.require(recovery_classes is None, 'Synthetic recovery class count cannot enter other scopes')
    recovery_pins = bind_recovery(recovery_receipt, purpose, protocol, synthetic, code)
    floor_path = reg['exposure']['floor']
    floor = read_pinned(Path(floor_path), reg['pins'][floor_path])
    validation_steps = events(steps, interval)
    rows, evidence, seen = [], {}, set()
    for case in reg['cases']:
        if purpose in PAIRED and case['support_seed'] != 2048:
            continue
        if purpose == RECOVERY and case['k'] != 5:
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
        populations = {role: fixed_population(release[role], allowed, role, guarded_plan['shape'][1], budget, recovery_classes)[2]
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
    if purpose not in PAIRED:
        for index in range(len(reg['cases'])):
            fresh = [r['initialization']['base_initial_state_sha256'] for r in cases if r['index'] == index and r['arm'] == 'none']
            device.require(len(set(fresh)) == len(MODEL_SEEDS), 'Model seeds did not yield distinct initial states')
    reg_path = out / 'registration.json'
    write_json(reg_path, reg)
    pins = {**reg['pins'], str(reg_path): hash_file(reg_path), str(parent): parent_sha, str(prior): prior_sha,
            **qualification_pins, **recovery_pins}
    batch.check_pins(pins)
    require_frozen_code(code)
    paths = ADMISSION_PATHS if purpose in PAIRED else None
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
    if recovery_pins:
        scope['recovery_receipt'] = next(iter(recovery_pins))
    if purpose == RECOVERY:
        scope['recovery_classes'] = recovery_classes
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
    paths = ADMISSION_PATHS if purpose in PAIRED else None
    schema = None if purpose == LEGACY else updates.SCHEMA
    quals = frozen.get('qualifications', {})
    protocol = frozen['protocol']
    if (not synthetic and protocol in device.CUDA_QUALIFIED) or (purpose == RECOVERY and protocol == device.CUDA_FP64_WEIGHT):
        from transfer.p4_cuda_qualification import check_receipt

        device.require(set(quals) == set(frozen['gpus']) and all(
            check_receipt(read_pinned(Path(path), pins[path]), frozen['protocol'], code, frozen['operator_geometry']) == gpu
            for gpu, path in quals.items()), 'Registered GPU operator qualification differs')
    else:
        device.require(not quals, 'Unregistered GPU operator qualification')
    recovery = frozen.get('recovery_receipt')
    recovery_pins = bind_recovery(recovery, purpose, protocol, synthetic, code)
    if purpose == RECOVERY:
        device.require(frozen.get('recovery_classes') in (9, 22)
            and (not quals or frozen['operator_geometry']['geometry_sha256'] == REAL_GEOMETRY),
            'Synthetic recovery class or qualification geometry differs')
    device.require(type(synthetic) is bool and accepted['status'] == 'passed'
        and accepted['query_accessed'] is False and accepted['protected_files'].get(parent) == pins[parent]
        and set(pins) == set(reg['pins']) | {reg_path, parent, prior} | set(quals.values()) | set(recovery_pins)
        and all(pins[k] == v for k,v in recovery_pins.items())
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
