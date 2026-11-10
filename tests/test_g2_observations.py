import hashlib
from io import BytesIO
import unittest

import numpy as np

from loadData.spatial_tiles import fit_tile_statistics, make_tile_plan, observe_tile, tile_slice
from test_g2_contract import fixture
from transfer.g2_observations import pack_source_observations, source_observations


class SourceObservationTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.contract, _, _ = fixture()
        self.scene = 'paviaU'
        self.artifact = self.catalog['scenes'][self.scene]['artifact_sha256']
        self.plan = make_tile_plan((25, 34), tile_size=8, gap=1)
        self.wave = np.array([400, 500, 600], dtype=np.float32)
        self.cube = np.random.default_rng(7).random((25, 34, 3), dtype=np.float32)
        self.releases = {role: self.pack(role) for role in ('train', 'val')}

    def pin(self, role, payload):
        self.contract['source_releases'][self.scene][role]['observation_sha256'] = hashlib.sha256(payload).hexdigest()
        return payload

    def pack(self, role, cube=None, wave=None):
        payload = pack_source_observations(self.scene, self.artifact, self.plan, role,
                                           self.cube if cube is None else cube, self.wave if wave is None else wave)
        return self.pin(role, payload)

    def view(self, role, **kwargs):
        return source_observations(self.catalog, self.contract, self.scene, role, self.plan,
                                   self.releases['train'], **kwargs)

    def tile(self, role):
        return next(tile for tile in self.plan['tiles'] if tile['partition'] == role)

    def alter(self, role, **changes):
        with np.load(BytesIO(self.releases[role]), allow_pickle=False) as archive:
            fields = dict(archive)
        stream = BytesIO()
        np.savez(stream, **(fields | changes))
        return self.pin(role, stream.getvalue())

    def test_existing_spatial_statistics_and_observations_match_exactly(self):
        stats = fit_tile_statistics(self.cube, self.plan)
        for role in ('train', 'val'):
            options = {'val_payload': self.releases['val']} if role == 'val' else {}
            view = self.view(role, **options)
            self.assertEqual(view.statistics, stats)
            self.assertTrue(np.array_equal(view.wavelengths_nm, self.wave))
            for tile in self.plan['tiles']:
                if tile['partition'] != role:
                    continue
                actual = view.tile(tile['id'])
                expected = observe_tile(self.cube, self.plan, stats, tile)
                for name in ('spectral', 'global_ids', 'coordinates'):
                    self.assertTrue(np.array_equal(getattr(actual, name), getattr(expected, name)))
                self.assertFalse(actual.spectral.flags.writeable)

    def test_poisoned_test_and_validation_pixels_do_not_affect_train(self):
        poison = self.cube.copy()
        for tile in self.plan['tiles']:
            if tile['partition'] != 'train':
                poison[tile_slice(tile)] = np.nan
        self.assertEqual(self.pack('train', poison), self.releases['train'])
        for tile in self.plan['tiles']:
            if tile['partition'] == 'val':
                poison[tile_slice(tile)] = 200
        val = self.pack('val', poison)
        self.assertNotEqual(val, self.releases['val'])
        view = self.view('val', val_payload=val)
        self.assertEqual(view.statistics, self.view('train').statistics)
        self.assertGreater(view.tile(self.tile('val')['id']).spectral.min(), 1)

    def test_role_api_refuses_wrong_inputs_and_tiles(self):
        with self.assertRaisesRegex(ValueError, 'required only'):
            self.view('train', val_payload=self.releases['val'])
        with self.assertRaisesRegex(ValueError, 'required only'):
            self.view('val')
        train = self.view('train')
        for role in ('val', 'test'):
            with self.assertRaisesRegex(ValueError, 'outside'):
                train.tile(self.tile(role)['id'])
        with self.assertRaisesRegex(ValueError, 'registered source'):
            self.view('test')
        with self.assertRaisesRegex(ValueError, 'registered source'):
            source_observations(self.catalog, self.contract, self.contract['target'], 'train',
                                self.plan, b'invalid')

    def test_changed_hash_rejected_before_npz_decode(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.view('val', val_payload=b'invalid')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.view('val', val_payload=self.releases['train'])

    def test_rehashed_identity_and_heldout_block_injection_rejected(self):
        for fields in ({'scene': 'whu-honghu'}, {'artifact_sha256': 'f'*64},
                       {'partition': 'test'}, {'plan_hash': 'e'*64}):
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                self.view('val', val_payload=self.alter('val', **fields))
        key = f"tile_{self.tile('test')['id']}"
        with self.assertRaisesRegex(ValueError, 'role coverage'):
            self.view('val', val_payload=self.alter('val', **{key: np.zeros((6, 6, 3), np.float32)}))

    def test_rehashed_invalid_values_and_changed_wavelength_order_rejected(self):
        key = f"tile_{self.tile('val')['id']}"
        with self.assertRaisesRegex(ValueError, 'Invalid source observation block'):
            self.view('val', val_payload=self.alter('val', **{key: np.full((6, 6, 3), np.nan, np.float32)}))
        with self.assertRaisesRegex(ValueError, 'wavelength order'):
            self.view('val', val_payload=self.pack('val', wave=self.wave[::-1]))
