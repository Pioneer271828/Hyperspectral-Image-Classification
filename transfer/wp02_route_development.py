"""Transport the independently accepted WP02 Xuzhou development release to E3."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from loadData.data_read import read_band_metadata
from loadData.spatial_tiles import tile_slice
from model.research_config import canonical_json, digest
from research_runtime import external_directory, hash_file
from transfer import registered_development as transport
from transfer.development_training import WP02_SPATIAL_RECORD, prepare_scene, validate
from transfer.p4_spatial import development_tiles, fit_development_statistics, load_release


SCENE = 'xuzhou'
CLASSES = list(range(1, 10))
TRAIN_SEED, VAL_SEED, K = 512, 3031, 5
SHAPE = (500, 260, 436)
REGISTRATION = 'b96a7304ed4499a5d98f4d88697d8e02c95038031727aa2c0032f7cf4e948f05'
ACCEPTANCE = '0bcc860206106b9cdcb155416f56022e3bd9712fc9d95a44f69bd392841e8ccb'


def _read(pin):
    return json.loads(transport._read(pin))


def _only(rows, predicate):
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError('Expected one accepted WP02 release entry')
    return matches[0]


def _member(registration, path):
    return {'path':path, 'sha256':registration['pins'][path]}


def accepted_inputs(registration_pin, acceptance_pin):
    if (registration_pin['sha256'], acceptance_pin['sha256']) != (REGISTRATION, ACCEPTANCE):
        raise ValueError('Expected the independently accepted Xuzhou P4 registration and acceptance')
    registration, acceptance = _read(registration_pin), _read(acceptance_pin)
    if (acceptance['status'] != 'passed' or acceptance['commit'] != registration['commit']
            or acceptance['raw_gt_loaded'] is not False or acceptance['query_accessed'] is not False
            or registration['config']['dataset'] != SCENE or registration['config']['training']['seed'] != 64):
        raise ValueError('WP02 acceptance or source case differs')
    case = _only(registration['cases'], lambda row: (row['support_seed'], row['k'], row['method']) == (TRAIN_SEED, K, 'scratch'))
    _only(acceptance['rows'], lambda row: row['case'] == case)
    roles_path = Path(registration['development'])
    pins = {'registration':registration_pin, 'acceptance':acceptance_pin,
            'image':_member(registration, registration['image']),
            'plan':_member(registration, registration['plan']),
            'roles':_member(registration, str(roles_path)),
            'release':_member(registration, case['release']),
            'statistics':_member(registration, registration['statistics']),
            'curator':_member(registration, str(roles_path.parent/'receipt.json')),
            'curator_audit':_member(registration, str(roles_path.parent/'independent01.json'))}
    curator, audit = _read(pins['curator']), _read(pins['curator_audit'])
    item = _only(curator['rows'], lambda row: row['scene'] == SCENE)
    approved = _only(audit['checks'], lambda row: row['scene'] == SCENE)
    release_row = _only(item['releases'], lambda row: (row['seed'], row['k']) == (TRAIN_SEED, K))
    if (audit['status'] != 'passed' or audit['curator_receipt_sha256'] != pins['curator']['sha256']
            or audit['raw_gt_loaded'] is not False or audit['query_accessed'] is not False
            or approved['status'] != 'passed' or approved['original_test_geometry_retained'] is not True
            or item['status'] != 'prepared'
            or (item['geometry'], item['geometry_sha256']) != (roles_path.name, pins['roles']['sha256'])
            or release_row['file'] != Path(case['release']).name or release_row['sha256'] != pins['release']['sha256']):
        raise ValueError('Original geometry or support release is outside accepted WP02 custody')
    plan, roles, release, statistics = [_read(pins[key]) for key in ('plan', 'roles', 'release', 'statistics')]
    allowed = development_tiles(roles, plan)
    if (plan['shape'] != list(SHAPE[:2]) or [len(allowed[r]) for r in ('train','val')] != [73,5]
            or item['devval_blocks'] != roles['heldout_tile_ids']
            or statistics['plan_hash'] != plan['plan_hash'] or statistics['fit_role'] != 'devtrain'
            or statistics['development_roles_hash'] != roles['development_roles_hash']
            or statistics['statistics_hash'] != digest({k:v for k,v in statistics.items() if k!='statistics_hash'})):
        raise ValueError('Expected the original Xuzhou geometry and devtrain statistics')
    return registration['config'], pins, plan, roles, release, statistics


def released_record(plan, geometry, release, pins):
    """Map the original per-point release. Its order and coordinates stay intact."""
    load_release(release, plan, TRAIN_SEED, K, geometry)
    train, val = ({key:deepcopy(release[name][key]) for key in ('seed','indices','labels')}
                  for name in ('support','validation'))
    record = dict(schema_version=1, kind=WP02_SPATIAL_RECORD, parent_plan_hash=plan['plan_hash'],
                  shape=plan['shape'], classes=CLASSES.copy(), roles=deepcopy(geometry['roles']),
                  heldout_tile_ids=geometry['heldout_tile_ids'], training_candidates=geometry['training_candidates'],
                  validation_candidates=geometry['validation_candidates'], train_support=train, validation_support=val,
                  training_labels=len(train['indices']), validation_labels=len(val['indices']),
                  total_training_selection_labels=len(train['indices'])+len(val['indices']),
                  selection='minimum_validation_nll_then_earliest_step', test_unchanged=True,
                  rule='exact accepted WP02 role and support release', oracle_disclosure=geometry['oracle_disclosure'],
                  normalization_contract='fit devtrain tiles only; no devval/test context in adaptation',
                  runner_compatible=False, wp02=deepcopy(pins))
    record['validation_hash'] = digest(record)
    validate(record, plan)
    return record


def verify_record(record, plan, dataset):
    pins = record['wp02']
    _, original, expected_plan, roles, release, statistics = accepted_inputs(pins['registration'], pins['acceptance'])
    if (dataset != SCENE or pins != original or plan != expected_plan
            or record != released_record(plan, roles, release, original)):
        raise ValueError('Development record differs from the original Xuzhou WP02 release')
    return statistics


def read_observations(image):
    path = external_directory(Path(image['path']).resolve(strict=True))
    if hash_file(path) != image['sha256']:
        raise ValueError('Frozen Xuzhou source image identity differs')
    cube = loadmat(path, variable_names=[SCENE])[SCENE]
    if cube.shape != SHAPE or cube.dtype != np.float64 or hash_file(path) != image['sha256']:
        raise ValueError('Original Xuzhou image shape, dtype or identity differs')
    return cube


def development_observations(cube, record):
    """Copy exactly the released role boxes into a zero-filled FP32 view."""
    result = np.zeros(cube.shape, dtype=np.float32)
    for tile in record['roles']:
        if tile['role'] in ('devtrain','devval'):
            box = tile_slice(tile)
            result[box] = cube[box]
            if not np.isfinite(result[box]).all():
                raise ValueError('Nonfinite authorized development observation')
    return result


def verify_scene(scene, original_statistics):
    expected_bands = read_band_metadata(SCENE, SHAPE[-1])
    if canonical_json(asdict(scene.bands)) != canonical_json(asdict(expected_bands)):
        raise ValueError('Xuzhou physical band metadata differs')
    for key in ('minimum','maximum'):
        if scene.stats[key] != np.asarray(original_statistics[key], dtype=np.float32).tolist():
            raise ValueError('Development statistics differ from the accepted devtrain extrema after FP32 conversion')


def bridge(registration_pin, acceptance_pin, out):
    config, pins, plan, roles, release, statistics = accepted_inputs(registration_pin, acceptance_pin)
    record = released_record(plan, roles, release, pins)
    cube = read_observations(pins['image'])
    if fit_development_statistics(cube, plan, roles) != statistics:
        raise ValueError('Native development statistics differ from the original acceptance')
    scene = prepare_scene(development_observations(cube, record), read_band_metadata(SCENE, SHAPE[-1]), plan, record)
    verify_scene(scene, statistics)
    return transport.export_development(scene, record, out), config
