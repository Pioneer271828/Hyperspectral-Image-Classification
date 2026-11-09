"""Materialize only registered, label-free source observations for G2."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO

import numpy as np

from loadData.spatial_tiles import fit_tile_statistics, observe_tile, tile_slice, validate_tile_plan
from transfer.g2_contract import sha, validate_contract


def _tiles(plan, partition):
    return {f"tile_{tile['id']}": tile for tile in plan['tiles'] if tile['partition'] == partition}


def _metadata(scene, artifact_sha256, plan, partition):
    return dict(scene=scene, artifact_sha256=artifact_sha256, plan_hash=plan['plan_hash'], partition=partition)


def _wavelengths(value):
    if (value.dtype != np.float32 or value.ndim != 1 or not value.size
            or not np.isfinite(value).all() or (value <= 0).any()
            or np.unique(value).size != value.size):
        raise ValueError('Source wavelengths require distinct positive float32 nanometres')
    return value


def pack_source_observations(scene, artifact_sha256, plan, partition, cube, wavelengths_nm):
    """Offline curator entry. Read and package exactly one source role's blocks."""
    if partition not in ('train', 'val'):
        raise ValueError('Source observation release requires a train/val partition')
    return _pack_observations(scene, artifact_sha256, plan, partition, cube, wavelengths_nm)


def _pack_observations(scene, artifact_sha256, plan, partition, cube, wavelengths_nm):
    validate_tile_plan(plan)
    sha(artifact_sha256)
    if not isinstance(scene, str) or not scene:
        raise ValueError('Observation release requires a scene')
    wave = _wavelengths(np.asarray(wavelengths_nm, dtype=np.float32))
    if cube.shape != (*plan['shape'], wave.size):
        raise ValueError('Source image shape and wavelength count do not match the plan')
    arrays = {key: np.asarray(cube[tile_slice(tile)], dtype=np.float32)
              for key, tile in _tiles(plan, partition).items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('Nonfinite source observation')
    stream = BytesIO()
    np.savez(stream, **_metadata(scene, artifact_sha256, plan, partition), wavelengths_nm=wave, **arrays)
    return stream.getvalue()


def _read(payload, registered, scene, artifact_sha256, plan, partition):
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != registered:
        raise ValueError('Source observation release hash mismatch')
    tiles = _tiles(plan, partition)
    metadata = _metadata(scene, artifact_sha256, plan, partition)
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            expected = tiles.keys() | metadata.keys() | {'wavelengths_nm'}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise ValueError('Source observation archive has incorrect role coverage')
            for key, value in metadata.items():
                field = archive[key]
                if field.ndim != 0 or field.dtype.kind != 'U' or field.item() != value:
                    raise ValueError('Source observation scene, artifact, role or geometry mismatch')
            wave = _wavelengths(archive['wavelengths_nm'])
            cube = np.zeros((*plan['shape'], wave.size), dtype=np.float32)
            for key, tile in tiles.items():
                values = archive[key]
                block = cube[tile_slice(tile)]
                if values.shape != block.shape or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError('Invalid source observation block')
                block[:] = values
    except TypeError as exc:
        raise ValueError('Expected a source observation NPZ archive') from exc
    wave.setflags(write=False)
    cube.setflags(write=False)
    return wave, cube


@dataclass(frozen=True)
class SourceObservations:
    _plan: dict
    _partition: str
    _cube: np.ndarray
    _statistics: dict
    wavelengths_nm: np.ndarray

    @property
    def statistics(self):
        return deepcopy(self._statistics)

    def tile(self, tile_id):
        tiles = self._plan['tiles']
        if (type(tile_id) is not int or not 0 <= tile_id < len(tiles)
                or tiles[tile_id]['partition'] != self._partition):
            raise ValueError('Source observation tile is outside the released role')
        return observe_tile(self._cube, self._plan, self._statistics, tiles[tile_id])


def source_observations(catalog, contract, scene, partition, plan, train_payload, *, val_payload=None):
    """Build one role view, fitting min/max exclusively from the registered train release."""
    validate_contract(catalog, contract)
    validate_tile_plan(plan)
    if scene not in contract['sources'] or partition not in ('train', 'val'):
        raise ValueError('Observations require a registered source train/val role')
    if (partition == 'val') != (val_payload is not None):
        raise ValueError('Validation bytes are required only for a validation view')
    artifact = catalog['scenes'][scene]['artifact_sha256']
    releases = contract['source_releases'][scene]
    wave, cube = _read(train_payload, releases['train']['observation_sha256'], scene, artifact, plan, 'train')
    statistics = fit_tile_statistics(cube, plan)
    if partition == 'val':
        val_wave, cube = _read(val_payload, releases['val']['observation_sha256'], scene, artifact, plan, 'val')
        if not np.array_equal(wave, val_wave):
            raise ValueError('Source wavelength order differs between train and validation')
    return SourceObservations(deepcopy(plan), partition, cube, statistics, wave)
