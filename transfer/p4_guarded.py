"""Bind guarded HongHu geometry, release files and observations to the P4 runner."""
import json
from pathlib import Path

import numpy as np

from loadData.observation_sampling import sample
from loadData.spatial_protocol import PartitionObservation
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from transfer.guarded_spatial_roles import guarded_roles, SUPPORT_SEEDS, MINIMUM


SCENE = 'whu-honghu'
KIND = 'guarded_honghu_development_v1'
EVIDENCE = ('scan_registration', 'cut_audit', 'release_receipt')


def validate_geometry(plan, geometry, scan, audit):
    """Verify the entire role list, including the untouched original held-out tiles."""
    identity = {name: value for name, value in geometry.items() if name != 'development_roles_hash'}
    if (geometry['schema_version'] != 1 or geometry['kind'] != KIND or geometry['scene'] != SCENE
            or geometry['parent_plan_hash'] != plan['plan_hash'] or geometry['shape'] != plan['shape']
            or geometry['development_roles_hash'] != digest(identity) or scan['scene'] != SCENE
            or audit['status'] != 'passed'
            or geometry['roles'] != guarded_roles(plan, scan['blocker'], audit['chosen_geometry_for_followup'])):
        raise ValueError('Guarded spatial geometry differs from the audited cut')
    # The preselected observation API checks tile IDs against the indexed scene table.
    # The superseded parent keeps its index but is absent from both allowed roles.
    tiles = list(plan['tiles'])
    allowed = {'train': [], 'val': []}
    for row in geometry['roles']:
        if row['role'] in ('devtrain', 'devval'):
            partition = {'devtrain': 'train', 'devval': 'val'}[row['role']]
            tile = {'id': row['id'], 'partition': partition, 'box': row['box']}
            allowed[partition].append(tile)
            if row['id'] == len(tiles):
                tiles.append(tile)
            else:
                tiles[row['id']] = tile
    return {'shape': plan['shape'], 'tiles': tiles,
            'plan_hash': geometry['development_roles_hash']}, allowed


def bind_inputs(registration, plan, geometry):
    paths, pins = registration['guarded'], registration['pins']
    if (set(paths) != set(EVIDENCE) or set(paths.values()) - pins.keys()
            or len(set(paths.values())) != len(EVIDENCE)):
        raise ValueError('All guarded evidence requires distinct pinned paths')
    scan, audit, receipt = [json.loads(Path(paths[name]).read_text(encoding='utf-8')) for name in EVIDENCE]
    if (scan['plan_sha256'] != pins[registration['plan']]
            or audit['protected_files'].get(paths['scan_registration']) != pins[paths['scan_registration']]
            or geometry['audit_sha256'] != pins[paths['cut_audit']]
            or receipt['status'] != 'prepared'
            or receipt['roles_sha256'] != pins[registration['development']]):
        raise ValueError('Guarded geometry and release evidence identities differ')
    manifest = {(row['seed'], row['k']): row for row in receipt['releases']}
    budget = {(seed, k) for seed in SUPPORT_SEEDS for k in (1, MINIMUM)}
    expected = {(seed, k, method) for seed, k in budget for method in ('scratch', 'shared_transfer')}
    cases = registration['cases']
    if (len(receipt['releases']) != len(budget) or manifest.keys() != budget or len(cases) != len(expected)
            or {(c['support_seed'], c['k'], c['method']) for c in cases} != expected):
        raise ValueError('Guarded study requires the four support cohorts and paired methods')
    for case in cases:
        entry = manifest[case['support_seed'], case['k']]
        if (Path(case['release']) != Path(paths['release_receipt']).parent / entry['file']
                or pins.get(case['release']) != entry['sha256']):
            raise ValueError('Guarded label file differs from the released support cohort')
    return validate_geometry(plan, geometry, scan, audit)


def observation_provider(scene, allowed):
    """Expose only development roles as separate model inputs."""
    registered = {tile['id']: tile for tiles in allowed.values() for tile in tiles}
    h, w = scene.plan['shape']
    low = np.asarray(scene.stats['minimum'], dtype=np.float32)
    high = np.asarray(scene.stats['maximum'], dtype=np.float32)
    scale = np.where(high > low, high - low, 1)
    views = {}

    def view(tile):
        if registered.get(tile['id']) != tile:
            raise ValueError('Guarded observation escapes the development roles')
        if tile['id'] not in views:
            spectral = np.array(scene.cube[tile_slice(tile)], dtype=np.float32, copy=True)
            if not np.isfinite(spectral).all():
                raise ValueError('Nonfinite guarded observation')
            y0, y1, x0, x1 = tile['box']
            yy, xx = np.mgrid[y0:y1, x0:x1]
            coordinates = np.stack((yy / max(h - 1, 1), xx / max(w - 1, 1)), axis=-1).astype(np.float32)
            observation = PartitionObservation(tile['partition'], (spectral - low) / scale,
                                               coordinates, yy * w + xx, scene.plan['plan_hash'])
            views[tile['id']], _ = sample(observation, scene.bands, {'mode': 'none'})
        return views[tile['id']]
    return view
