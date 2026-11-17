"""Register matched precision recovery and execute each claimed interval once."""
import os
from pathlib import Path
import subprocess
from time import perf_counter

import torch

from compression.int8_checkpoint import load_int8, save_int8
from compression.int8_model import calibration_contract
from compression.precision_fallback import LIMITS, SMALL_CLASS_MAX_SUPPORT, _validation_inputs
from compression.precision_recovery import ARMS, TERNARY_RATIO, PrecisionRecoveryRun, initialization, new_run, transform
from compression.ternary_checkpoint import load_ternary, save_ternary
from inference.light_cnn import validate_source
from model.research_config import canonical_json
from research_runtime import ROOT, external_directory, require_frozen_code, tensor_hash
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import state_hash
from transfer.region_message_run import RegionMessageRun
from transfer.xuzhou_multiscale import LR, SEED, _accepted_inputs, execute_stage, read_stage
from transfer.xuzhou_multiscale_audit import _packet


POLICY = 'wp10_precision_recovery_v1'
HISTORICAL = '4f86ff180b666fa494cd9732840fcf5ef7714b4e'
HISTORICAL_PINS = {
    'registration': 'c19dfa3fe958b07935a17ad9ce45644a003cb396fbce927fa767c00c1b5cfd37',
    'completion': '5fb1eff61844f8f8c2b2688b261e797d49caf30dea4735ad3413ab4c0781ae08',
    'audit': '8df3b5c1d9c2e95a7ca04a713185a9af67187b3998e559b51762e242e3354f2e',
}
COMPATIBILITY_PATHS = (
    'model', 'loadData', 'configs', 'transfer/g3_training.py', 'transfer/development_training.py',
    'transfer/observation_policy.py', 'transfer/curriculum_exposure.py', 'transfer/curriculum_training.py',
    'transfer/sample_curriculum.py', 'transfer/tile_sampling.py', 'transfer/region_message_run.py',
    'transfer/xuzhou_region_message.py', 'transfer/xuzhou_multiscale.py', 'transfer/xuzhou_multiscale_audit.py',
    'transfer/registered_development.py', 'transfer/wp02_route_development.py',
)
REAL_INTERVAL = (155, 7)
ARM_NAMES = {key: key for key in ARMS}


def runtime():
    engine = torch.backends.quantized.engine
    threads = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    if (engine not in ('x86', 'fbgemm') or torch.get_num_threads() != 1
            or set(threads.values()) != {'1'} or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Frozen one-thread CPU environment and validated INT8 engine required')
    return dict(admission.runtime_identity(), quantized_engine=engine, thread_environment=threads,
                torch_interop_threads=torch.get_num_interop_threads(), mkldnn=torch.backends.mkldnn.enabled)


def stages(budget, prefix):
    return dict(full=(0, budget, None), prefix=(0, prefix, None),
                resume=(prefix, budget, 'prefix'), completed=(budget, budget, 'resume'))


def _historical(pin):
    origin = admission.read(pin)
    if (origin['kind'] != 'wp08_off_selected_precision_origin_v1' or origin['producer_commit'] != HISTORICAL
            or any(origin[name]['sha256'] != sha for name, sha in HISTORICAL_PINS.items())
            or any(origin[name] != 0 for name in ('gt', 'query', 'optimizer_updates'))):
        raise ValueError('Expected accepted WP08 off selected-state export')
    differing = subprocess.check_output(
        ['git', '-C', str(ROOT), 'diff', '--name-only', HISTORICAL, 'HEAD', '--', *COMPATIBILITY_PATHS], text=True)
    if differing.strip():
        raise ValueError('WP08 source migration has incompatible model, normalization or training consumers')
    old, completed, audited = [admission.read(origin[name]) for name in ('registration', 'completion', 'audit')]
    if (old['commit'] != HISTORICAL or old['source_hash'] != origin['producer_source_hash']
            or completed['registration'] != origin['registration'] or audited['registration'] != origin['registration']
            or audited['complete'] != origin['completion'] or audited['status'] != 'passed'
            or completed['attempts']['off']['full'] != origin['off_full']):
        raise ValueError('WP08 completion and source export are unrelated')
    original_off = admission.read(origin['off_full'])
    chosen = audited['results']['off']['endpoints']['selected']
    if (origin['off_checkpoint'] != original_off['checkpoint'] or origin['baseline'] != chosen
            or origin['selected_step'] != chosen['step'] or origin['selected_state'] != original_off['finish']['selected_state_hash']):
        raise ValueError('Wrong historical off checkpoint or selected step')
    old_packet = _packet(original_off, old['contracts']['off'], old['budget'], packet_format=RegionMessageRun.FORMAT)
    scene, record, prior, _, protected = _accepted_inputs(old['prior'], old['donor_audit'])
    coverage = admission.read(origin['coverage'])
    expected_rows = {str(n): prior.progress(n) for n in (64, 154, 155)}
    original = prior.to_dict()
    first_visits = []
    for index, class_id in zip(record['train_support']['indices'], record['train_support']['labels'], strict=True):
        first = next(e for e in original['events'] if index in e['support_ids'])
        first_visits.append(dict(index=index, class_id=class_id, tile=first['tile_id'], first_step=first['step']))
    if (coverage['producer_commit'] != HISTORICAL or coverage['original_registration'] != origin['registration']
            or coverage['original_exposure'] != prior.plan_hash or coverage['seed'] != SEED
            or coverage['prefixes'] != expected_rows or coverage['first_visits'] != first_visits
            or coverage['minimum_eligible_steps'] != 155
            or expected_rows['154']['exposure']['meets_minimum'] or not expected_rows['155']['exposure']['meets_minimum']):
        raise ValueError('Historical 64-step and shortest 155-step exposure evidence differs')
    source = dev._load_source(origin['source'], origin['sidecar'])
    old_config = admission.read(old['sidecar'])
    if (old_packet['selected_step'] != chosen['step'] or state_hash(source) != origin['selected_state']
            or source.state_dict().keys() != old_packet['selected_state'].keys()
            or any(tensor_hash(tensor) != tensor_hash(old_packet['selected_state'][name])
                   for name, tensor in source.state_dict().items())
            or source.export_config() != {k: v for k, v in old_config.items() if k != 'training_config'}
            or source.training_config != old_config['training_config']):
        raise ValueError('Source bytes differ from the accepted dense selected tensors or sidecar')
    expected_pins = {item['path']: item for item in (
        *protected.values(), *(origin[name] for name in (*HISTORICAL_PINS, 'off_full', 'off_checkpoint')),
        old['source'], old['sidecar'])}
    if origin['protected'] != expected_pins:
        raise ValueError('Original protection inventory differs from the producer export')
    all_pins = {**expected_pins, **{item['path']: item for item in (pin, origin['coverage'], origin['source'], origin['sidecar'])}}
    source_info = dict(weights=origin['source'], sidecar=origin['sidecar'], origin=pin, coverage=origin['coverage'],
                       historical_selected_step=chosen['step'], historical_baseline=chosen,
                       compatibility_paths=list(COMPATIBILITY_PATHS))
    return source, scene, record, source_info, all_pins, prior


def _inputs(request):
    v = admission.read(request)
    scope = v.get('scope')
    if scope == 'wp08_xuzhou_off':
        if set(v) != {'scope', 'origin', 'budget', 'prefix'} or (v['budget'], v['prefix']) != REAL_INTERVAL:
            raise ValueError('Real WP10 requires the approved shortest 155-step prefix and 7+148 resume')
        source, scene, record, source_info, pins, prior = _historical(v['origin'])
    elif scope == 'synthetic':
        if set(v) != {'scope', 'development', 'weights', 'sidecar', 'budget', 'prefix'}:
            raise ValueError('Explicit synthetic development and dense source required')
        source = dev._load_source(v['weights'], v['sidecar'])
        scene, record, files = dev.load_development(v['development'])
        source_info = dict(weights=v['weights'], sidecar=v['sidecar'], development=v['development'])
        pins = {item['path']: item for item in (*files, v['weights'], v['sidecar'])}
        prior = None
    else:
        raise ValueError('Unsupported WP10 source scope')
    if (type(v['budget']) is not int or type(v['prefix']) is not int or not 1 <= v['prefix'] < v['budget']
            or source_info['sidecar']['path'] != source_info['weights']['path'] + '.json'):
        raise ValueError('Explicit recovery budget, prefix and adjacent source sidecar required')
    validate_source(source)
    if source.backbone.norm_type != 'group_norm':
        raise ValueError('Matched recovery requires the original GroupNorm source')
    plan = ExposurePlan.freeze(scene, record, updates=v['budget'], seed=SEED,
                               minimum_class_blocks={str(c): 1 for c in record['classes']}, minimum_support_visits=1)
    plan.require_eligible()
    if prior is not None and (plan.to_dict()['events'] != prior.to_dict()['events'][:plan.updates]
                              or plan.progress(plan.updates)['prefix_hash'] != prior.progress(plan.updates)['prefix_hash']):
        raise ValueError('New recovery changed the accepted class-first exposure sequence')
    calibration_tiles = list(dict.fromkeys(row['tile_id'] for row in plan.to_dict()['events']))[:2]
    if not calibration_tiles or any(record['roles'][i]['role'] != 'devtrain' for i in calibration_tiles):
        raise ValueError('Registered calibration must use development training only')
    return v, source, scene, record, source_info, {**pins, request['path']: request}, plan, calibration_tiles


def _rules(value):
    return dict(budget=value['budget'], prefix=value['prefix'], maximum_updates=2 * len(ARMS) * value['budget'],
        arm_order=list(ARMS), assignments={key: dict(mode=mode, paths=list(paths)) for key, (mode, paths) in ARMS.items()},
        lr=LR, optimizer='adam_original', scheduler=None, seed=SEED, ternary_ratio=TERNARY_RATIO,
        validation_events=list(range(1, value['budget'] + 1)), selection='first_min_deployment_fused_nll',
        validation_fit_reads=0, completed_reload_updates=0, channel_pruning=False,
        guards=dict(max_fused_drop={key: float(limit) for key, limit in LIMITS.items()},
                    small_support_max=SMALL_CLASS_MAX_SUPPORT, small_new_errors_max=0, joint_requires_singles=True))


def register(request, out, code):
    require_frozen_code(code)
    environment = runtime()
    value, source, scene, record, source_info, protected, plan, tiles = _inputs(request)
    root = external_directory(out)
    if any(root == Path(path) or root in Path(path).parents for path in protected):
        raise ValueError('Registered output must not contain existing inputs')
    root.mkdir(parents=True, exist_ok=False)
    contracts, seeds = {}, {}
    for arm in ARMS:
        started = perf_counter()
        candidate = transform(source, scene, record, arm, tiles)
        evidence = initialization(source, candidate, arm)
        mode, paths = ARMS[arm]
        calibrated = {path: int(candidate.get_submodule(path).batches_seen) for path in paths} if mode == 'qat' else {}
        if any(count != len(tiles) for count in calibrated.values()):
            raise ValueError('QAT calibration did not observe exactly the fixed number of blocks')
        prepared = perf_counter()
        if mode == 'fp32':
            packet = source_info['weights']
        else:
            path = root / 'initial' / (arm + '.pt')
            save = save_int8 if mode == 'qat' else save_ternary
            packet = dict(path=str(path), sha256=save(candidate, path))
        run = new_run(candidate, scene, record, plan, arm)
        contracts[arm] = run.contract_hash
        seeds[arm] = dict(checkpoint=packet, evidence=evidence, calibration_batches=calibrated,
                          preparation_seconds=prepared-started, publication_seconds=perf_counter()-prepared)
    validation, _ = _validation_inputs(scene, record)
    admission.check_pins(protected.values())
    require_frozen_code(code)
    return dev._json_new(root / 'registration.json', dict(
        policy=POLICY, commit=code['commit'], source_hash=code['source_hash'], root=str(root), request=request,
        runtime=environment, scope=value['scope'], source=source_info, protected=protected,
        exposure=plan.to_dict(), exposure_prefix=plan.progress(plan.updates)['prefix_hash'],
        validation=validation, calibration_tiles=tiles, contracts=contracts, initial=seeds,
        registration_optimizer_updates=0, **_rules(value)))


def load_registration(pin, code):
    require_frozen_code(code)
    reg = admission.read(pin)
    root = external_directory(reg['root'])
    if (reg['policy'] != POLICY or pin['path'] != str(root / 'registration.json')
            or reg['commit'] != code['commit'] or reg['source_hash'] != code['source_hash']
            or reg['runtime'] != runtime()):
        raise ValueError('Wrong registered code, runtime, output root or precision engine')
    value, source, scene, record, source_info, protected, plan, tiles = _inputs(reg['request'])
    if (any(reg[key] != item for key, item in _rules(value).items()) or reg['scope'] != value['scope']
            or reg['source'] != source_info or reg['protected'] != protected or reg['calibration_tiles'] != tiles
            or reg['exposure'] != plan.to_dict() or reg['exposure_prefix'] != plan.progress(plan.updates)['prefix_hash']
            or reg['validation'] != _validation_inputs(scene, record)[0]
            or set(reg['initial']) != set(ARMS) or set(reg['contracts']) != set(ARMS)):
        raise ValueError('Source, data, calibration, validation, precision assignment or update budget changed')
    return root, reg, source, scene, record, plan


def initial_model(root, reg, source, scene, record, arm):
    if arm not in ARMS:
        raise ValueError('Unknown registered precision arm')
    mode, paths = ARMS[arm]
    entry = reg['initial'][arm]
    pin = entry['checkpoint']
    if mode == 'fp32':
        if pin != reg['source']['weights']:
            raise ValueError('FP32 control must use the same dense source')
        candidate = source
    else:
        if pin['path'] != str(root / 'initial' / (arm + '.pt')):
            raise ValueError('Wrong frozen calibration or initialization checkpoint')
        load = load_int8 if mode == 'qat' else load_ternary
        candidate = load(pin['path'], pin['sha256'])
    if canonical_json(initialization(source, candidate, arm)) != canonical_json(entry['evidence']):
        raise ValueError('Transformed initial state differs from the registered dense masters or observer state')
    if mode == 'qat':
        registered = candidate._int8_plan['contract']
        if (registered != calibration_contract(source, scene, record, paths, reg['calibration_tiles'], qat=True)
                or any(int(candidate.get_submodule(path).batches_seen) != len(reg['calibration_tiles']) for path in paths)):
            raise ValueError('Frozen QAT initialization is unrelated to the registered calibration blocks')
    if mode == 'ternary' and candidate._tq_plan['spec']['source_state'] != state_hash(source):
        raise ValueError('Ternary masters are unrelated to the registered dense source')
    return candidate


def execute(pin, code, notice, arm, stage, prior):
    root, value, source, scene, record, plan = load_registration(pin, code)
    disclosed = admission.read(notice)
    if (notice['path'] != str(root / 'notice.json') or disclosed['registration'] != pin
            or disclosed['maximum_updates'] != value['maximum_updates'] or disclosed['cpu_workers'] != 1):
        raise ValueError('Announce the exact registered CPU budget before execution')
    seed = initial_model(root, value, source, scene, record, arm)
    return execute_stage(root, pin, arm, stage, prior, value=value, baseline=seed, scene=scene, record=record,
                         plan=plan, arms=ARM_NAMES, stages=stages(value['budget'], value['prefix']),
                         run_factory=new_run, run_type=PrecisionRecoveryRun)


def complete(pin, code):
    root, value, _, _, _, _ = load_registration(pin, code)
    intervals = stages(value['budget'], value['prefix'])
    attempts, total = {}, 0
    for arm in ARMS:
        receipts = {}
        for stage, (_, _, previous) in intervals.items():
            status, receipts[stage] = read_stage(root, pin, arm, stage, receipts.get(previous),
                                                arms=ARM_NAMES, stages=intervals)
            total += status['new_updates']
        attempts[arm] = receipts
    if total != value['maximum_updates']:
        raise ValueError('Actual stage counts exceed or omit registered optimization')
    admission.check_pins([*value['protected'].values(), *[row['checkpoint'] for row in value['initial'].values()], pin])
    require_frozen_code(code)
    return dev._json_new(root / 'batch-complete.json', dict(registration=pin, attempts=attempts, updates=total, gt=0, query=0))
