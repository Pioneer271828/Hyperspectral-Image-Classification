"""Prepare and independently audit the bounded Xuzhou route data preflight."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash
from train_spatial import SpatialConfig
from transfer import matched_route_admission as admission, registered_development as transport
from transfer import wp02_route_development as wp02
from transfer.container_runtime import training_container_kind
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import _forward
from transfer.graph_route_context import _context, _model, route_state_hash


BUDGET, MODEL_SEED, EXPOSURE_SEED, LR = 24, 64, 64, 0.0001
ROUTE = {'budget':4, 'mode':'weight', 'seed':64}


def _source(config):
    kwargs = SpatialConfig(config).research.model_kwargs()
    if kwargs['norm_type'] != 'domain_bn':
        raise ValueError('Expected the accepted WP02 DomainBN architecture record')
    kwargs.update(norm_type='group_norm', graph_route=ROUTE.copy())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = MultiDatasetVCASAHGCN({wp02.SCENE:len(wp02.CLASSES)}, **kwargs).eval()
    model.backbone.adapter.cache_attention = False
    model.training_config = {'initialization':'fresh_group_norm', 'model_seed':MODEL_SEED,
                             'shared_pretrained_tensors':0, 'architecture_ancestor':wp02.REGISTRATION}
    return model


def _geometry(scene, record):
    rows = [_context(scene, record, tile['id']) for tile in record['roles']
            if tile['role'] in ('devtrain','devval')]
    largest = [min((row for row in rows if row['role']==role),
                   key=lambda row: (-row['shape'][0]*row['shape'][1], row['tile_id']))
               for role in ('devtrain','devval')]
    return rows, largest


@torch.no_grad()
def _probe(model, scene, context, step):
    router = _model(model, scene)
    before = router.get_extra_state()
    router.set_step(step)
    try:
        started = time.perf_counter()
        outputs = _forward(model, scene, scene.plan['tiles'][context['tile_id']])
        elapsed = time.perf_counter() - started
    finally:
        router.set_extra_state(before)
    probabilities = outputs[:3]
    if any(not torch.isfinite(x).all() for x in probabilities):
        raise ValueError('Nonfinite route probe probabilities')
    trace = outputs[3]['interaction']
    return {'tile_id':context['tile_id'], 'role':context['role'], 'seconds':elapsed,
            'route_step':step, 'nnz':trace['route_output_nnz'],
            'head_probability_sha256':[tensor_hash(x) for x in probabilities]}


def prepare(registration, acceptance, out, code):
    out = external_directory(out)
    out.mkdir(parents=True, exist_ok=False)
    development, config = wp02.bridge(registration, acceptance, out/'development')
    scene, record, input_files = transport.load_development(development)
    plan = ExposurePlan.freeze(scene, record, updates=BUDGET, seed=EXPOSURE_SEED,
                               minimum_class_blocks={str(c):1 for c in wp02.CLASSES}, minimum_support_visits=1)
    plan_pin = transport._json_new(out/'exposure.json', plan.to_dict())
    exposure = plan.exposure()
    support_tiles = len(plan.to_dict()['sampler']['tile_ids'])
    model = _source(config)
    _model(model, scene)
    source_path = out/'source.pt'
    save_model_checkpoint(model, source_path)
    source, sidecar = transport._pin(source_path), transport._pin(str(source_path)+'.json')
    state = route_state_hash(model)
    rng = tensor_hash(torch.get_rng_state())
    rows, probes = _geometry(scene, record)
    pin = transport._json_new(out/'preflight.json', {
        'policy':'wp02_xuzhou_route_preflight_v1', 'commit':code['commit'], 'source_hash':code['source_hash'],
        'runtime':admission.runtime_identity(), 'development':development, 'source':source, 'source_sidecar':sidecar,
        'source_state':state, 'architecture':model.export_config(), 'initialization':model.training_config,
        'scene':wp02.SCENE, 'upstream':record['wp02'], 'development_stats_sha256':scene.stats['statistics_hash'],
        'bands_sha256':scene.bands.fingerprint, 'exposure':plan_pin, 'exposure_hash':plan.plan_hash,
        'context_population_sha256':digest(rows), 'contexts':{'train':73,'validation':5},
        'geometry_counts':dict(Counter('x'.join(map(str, row['shape'])) for row in rows)),
        'registered_probes':probes, 'probe_forwards_maximum':2, 'optimizer_updates':0,
        'donor_contract':{'lr':LR,'model_seed':MODEL_SEED,'exposure_seed':EXPOSURE_SEED,'updates':BUDGET,
                          'train_route_step':'one_based_optimizer_step','validation_route_step':0,
                          'loss':'original_three_probability_losses_class_balanced_default_coefficients',
                          'optimizer':'Adam, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False',
                          'selection':'first_min_full_devval_fused_NLL_each_update'},
        'exposure_gate':{'passed':exposure['meets_minimum'], 'support_bearing_tiles':support_tiles,
                         'scheduled_unique_tiles':sum(value > 0 for value in exposure['tile_counts'].values()),
                         'missing_support_points':sum(value > 0 for value in exposure['support_deficits'].values()),
                         'missing_classes':sum(value > 0 for value in exposure['class_deficits'].values()),
                         'seed_independent_minimum_updates':support_tiles},
        'conditional_limits':{'donor_updates':48,'retraining_updates':288,'completed_updates':0,
                              'planning_forwards':317,'review_forwards_maximum':35,
                              'solver_calls_maximum':312,'solver_timeout_per_call':30,
                              'solver_seconds_maximum':9360},
        'raw_gt_accessed':False, 'query_accessed':False})
    timing = [_probe(model, scene, context, step) for context, step in zip(probes, (1,0), strict=True)]
    if (route_state_hash(model) != state or tensor_hash(torch.get_rng_state()) != rng):
        raise ValueError('Timing probes changed source state or CPU RNG')
    if exposure['meets_minimum']:
        raise ValueError('This preflight records the frozen 24-step exposure barrier only')
    try:
        transport.register_route(source_path, development, plan, out/'donor-registration.json',
                                 lr=LR, torch_seed=MODEL_SEED, expected_commit=code['commit'])
    except ValueError as error:
        if 'Frozen budget misses preregistered exposure' not in str(error):
            raise
        rejection = str(error)
    else:
        raise ValueError('GraphRouteRun unexpectedly accepted the insufficient exposure')
    if (out/'donor-registration.json').exists():
        raise ValueError('Rejected registration produced an output')
    admission.check_pins([registration, acceptance, source, sidecar, plan_pin, *input_files])
    require_frozen_code(code)
    return transport._json_new(out/'blocked.json', {
        'status':'blocked_exposure', 'preflight':pin, 'registration_rejection':rejection,
        'probe_forwards':2, 'timing':timing, 'source_and_rng_unchanged':True,
        'donor_registration_created':False, 'optimizer_updates':0, 'solver_calls':0,
        'raw_gt_accessed':False, 'query_accessed':False})


def audit(blocked_pin, out, code):
    blocked = wp02._read(blocked_pin)
    registration = wp02._read(blocked['preflight'])
    if (registration['commit'], registration['source_hash']) != (code['commit'], code['source_hash']):
        raise ValueError('Preflight producer and audit source differ')
    if (blocked['status'] != 'blocked_exposure' or blocked['probe_forwards'] != 2
            or blocked['donor_registration_created'] or blocked['solver_calls'] != 0
            or blocked['query_accessed'] or blocked['raw_gt_accessed']):
        raise ValueError('Preflight action or data access differs')
    _, pins, plan, geometry, release, original_stats = wp02.accepted_inputs(
        registration['upstream']['registration'], registration['upstream']['acceptance'])
    native = wp02.read_observations(pins['image'])
    scene, record, files = transport.load_development(registration['development'])
    wp02.verify_scene(scene, original_stats)
    population = np.zeros(plan['shape'], dtype=bool)
    pixel_counts, block_counts = Counter(), Counter()
    train_native_min, train_native_max = [], []
    for row in geometry['roles']:
        if row['role'] not in ('devtrain','devval'):
            continue
        area = tile_slice(row)
        if population[area].any() or not np.array_equal(scene.cube[area], native[area].astype(np.float32)):
            raise ValueError('Development pixels or role geometry differ from the accepted source')
        population[area] = True
        pixel_counts[row['role']] += int(population[area].size)
        block_counts[row['role']] += 1
        if row['role'] == 'devtrain':
            train_native_min.append(native[area].min(axis=(0,1)))
            train_native_max.append(native[area].max(axis=(0,1)))
    if np.any(scene.cube[~population]) or not np.array_equal(np.min(train_native_min,axis=0), original_stats['minimum']) \
            or not np.array_equal(np.max(train_native_max,axis=0), original_stats['maximum']):
        raise ValueError('Excluded pixels or native training statistics changed')
    width = plan['shape'][1]
    label_counts, support_tiles = {}, set()
    scheduled = {row['tile_id'] for row in wp02._read(registration['exposure'])['events']}
    visited = 0
    for key, old, role in (('train_support','support','devtrain'), ('validation_support','validation','devval')):
        published, transported = release[old], record[key]
        if any(published[k]!=transported[k] for k in ('indices','labels','seed')):
            raise ValueError('Released label coordinates, classes or seed changed')
        label_counts[role] = dict(Counter(str(c) for c in published['labels']))
        for index in published['indices']:
            y, x = divmod(index, width)
            allowed = [t['id'] for t in geometry['roles'] if t['role']==role and
                       t['box'][0]<=y<t['box'][1] and t['box'][2]<=x<t['box'][3]]
            if len(allowed)!=1:
                raise ValueError('Published label has no unique allowed coordinate')
            if role=='devtrain':
                support_tiles.add(allowed[0])
                visited += int(allowed[0] in scheduled)
    role_geometry = [dict(t,role=r['role']) for t,r in zip(plan['tiles'], geometry['roles'], strict=True)]
    if record['roles'] != geometry['roles'] or geometry['roles'] != role_geometry:
        raise ValueError('Transport changed the frozen role geometry')
    if len(support_tiles) <= BUDGET or visited >= 45 or blocked['optimizer_updates']!=0:
        raise ValueError('The seed-independent exposure barrier did not reproduce')
    admission.check_pins([blocked_pin, blocked['preflight'], registration['source'], registration['source_sidecar'],
                          registration['exposure'], *pins.values(), *files])
    require_frozen_code(code)
    return transport._json_new(external_directory(out), {
        'status':'passed_bridge_and_exposure_block', 'commit':code['commit'], 'blocked':blocked_pin,
        'geometry_exact':True,
        'label_counts':label_counts, 'labels_compared':90,
        'blocks_compared':dict(block_counts), 'pixels_compared':dict(pixel_counts),
        'spectral_values_compared':sum(pixel_counts.values())*len(scene.bands.band_ids),
        'bands_sha256':scene.bands.fingerprint, 'native_statistics_sha256':original_stats['statistics_hash'],
        'fp32_statistics_sha256':scene.stats['statistics_hash'],
        'excluded_pixels_zero':int((~population).sum()), 'support_bearing_tiles':len(support_tiles),
        'scheduled_tiles':len(scheduled), 'support_visited':visited, 'support_missed':45-visited,
        'budget':BUDGET, 'min_updates_any_seed':len(support_tiles), 'optimizer_updates':0,
        'raw_gt_accessed':False, 'query_accessed':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep, check = commands.add_parser('prepare'), commands.add_parser('audit')
    for command in (prep, check):
        command.add_argument('--expected-commit', required=True)
        command.add_argument('--out', required=True, type=Path)
    for name in ('registration','acceptance'):
        prep.add_argument('--'+name, required=True)
        prep.add_argument('--'+name+'-sha256', required=True)
    check.add_argument('--blocked', required=True)
    check.add_argument('--blocked-sha256', required=True)
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error('Authorized Linux container with assertions required')
    torch.set_num_threads(1)
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        parser.error('Expected exact producer commit')
    require_frozen_code(code)
    def pin(name):
        return {'path':str(Path(getattr(args, name)).resolve(strict=True)), 'sha256':getattr(args, name+'_sha256')}
    result = (prepare(pin('registration'), pin('acceptance'), args.out, code) if args.action=='prepare'
              else audit(pin('blocked'), args.out, code))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
