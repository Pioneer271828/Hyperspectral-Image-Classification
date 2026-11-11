"""Bind guarded HongHu geometry, release files and observations to the P4 runner."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from loadData.observation_sampling import ObservationView, sample, subset_statistics
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


def retained_view(scene, tile, condition):
    """Construct a band-deleted view after the caller checks tile permission."""
    record = condition['sampling']
    if record['partition'] != tile['partition'] or record['plan_hash'] != scene.plan['plan_hash']:
        raise ValueError('Spectral sampling record must bind the observation role')
    statistics = subset_statistics(scene.stats, scene.bands, record)
    positions = record['selected_positions']
    metadata = scene.bands.subset(positions)
    if (metadata.fingerprint != record['selected_metadata_hash']
            or list(metadata.band_ids) != record['selected_band_ids']):
        raise ValueError('Selected band metadata differs')
    y0, y1, x0, x1 = tile['box']
    tile_values = scene.cube[y0:y1, x0:x1]
    retained = tile_values if len(positions) == tile_values.shape[-1] else tile_values[:, :, positions]
    values = np.array(retained, dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite retained observation')
    low, high = (np.asarray(statistics[key], dtype=np.float32) for key in ('minimum', 'maximum'))
    spectral = (values - low) / np.where(high > low, high - low, 1)
    h, w = scene.plan['shape']
    yy, xx = np.mgrid[y0:y1, x0:x1]
    coordinates = np.stack((yy / max(h - 1, 1), xx / max(w - 1, 1)), axis=-1).astype(np.float32)
    observation = PartitionObservation(tile['partition'], spectral, coordinates, yy * w + xx,
                                       scene.plan['plan_hash'])
    return ObservationView(observation, model_metadata(metadata, condition['metadata']))


def model_metadata(metadata, mode):
    if mode == 'shift10nm':
        if metadata.wavelength_unit != 'nm':
            raise ValueError('Wavelength shift requires physical coordinates')
        return replace(metadata, wavelengths=tuple(value + 10 for value in metadata.wavelengths))
    if mode not in ('nominal', 'missing'):
        raise ValueError('Unknown metadata perturbation')
    # Missing preserves the retained band identity internally. The forward
    # call supplies neither wavelengths, FWHM nor an E1 descriptor.
    return metadata
