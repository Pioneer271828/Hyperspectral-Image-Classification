"""Verify exact WP02 label transport and observation isolation."""

from copy import deepcopy
import unittest

import numpy as np

from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import make_tile_plan, tile_slice
from model.research_config import digest
from transfer.development_training import development_counts, prepare_scene
from transfer.wp02_route_development import development_observations, released_record


class WP02RouteDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_tile_plan((64,64),8,2,2026)
        plan = self.plan
        held = next(tile['id'] for tile in plan['tiles'] if tile['partition']=='train')
        roles = [dict(tile, role=('devval' if tile['id']==held else 'devtrain')
                      if tile['partition']=='train' else 'unused' if tile['partition']=='val' else 'test')
                 for tile in plan['tiles']]
        self.roles = roles
        self.geometry = dict(scene='xuzhou', parent_plan_hash=plan['plan_hash'], shape=plan['shape'],
            roles=roles, heldout_tile_ids=[held], training_candidates=[50]*9, validation_candidates=[5]*9,
            oracle_disclosure='synthetic fixture with explicit labels')
        self.geometry['development_roles_hash'] = digest(self.geometry)
        self.release = dict(scene='xuzhou', development_roles_hash=self.geometry['development_roles_hash'])
        for key,role,seed in (('support','devtrain',512),('validation','devval',3031)):
            tile = next(t for t in roles if t['role']==role)
            y0,y1,x0,x1 = tile['box']
            indices = [y*64+x for y in range(y0,y1) for x in range(x0,x1)][:45]
            self.release[key] = dict(seed=seed, per_class=5, indices=indices, labels=list(range(1,10))*5)
        self.release['release_hash'] = digest(self.release)

    def record(self):
        return released_record(self.plan, self.geometry, self.release, {})

    def test_exact_support_and_validation_order_and_original_roles(self):
        record = self.record()
        self.assertEqual(development_counts(record), (45,45))
        self.assertEqual(record['roles'], self.roles)
        self.assertEqual(record['classes'], list(range(1,10)))
        for old,new in (('support','train_support'),('validation','validation_support')):
            self.assertEqual(record[new], {k:self.release[old][k] for k in ('seed','indices','labels')})
        with self.assertRaisesRegex(ValueError,'fixed per-class'):
            development_counts({**record,'kind':'oracle_spatial_development_v1'})

    def test_heldout_and_buffer_poison_cannot_change_transport_or_statistics(self):
        record = self.record()
        cube = np.random.default_rng(12).uniform(size=(64,64,3))
        bands = BandMetadata('synthetic', (0,1,2), (410.,500.,600.), 'nm', 'approximate', (True,)*3)
        plain = development_observations(cube, record)
        authorized = np.zeros(cube.shape[:2], dtype=bool)
        for row in self.roles:
            if row['role'] in ('devtrain','devval'):
                authorized[tile_slice(row)] = True
        poisoned = cube.copy()
        poisoned[~authorized] = np.nan
        np.testing.assert_array_equal(development_observations(poisoned, record), plain)
        for row in self.roles:
            if row['role']=='devval':
                poisoned[tile_slice(row)] = 12345.
        altered = development_observations(poisoned, record)
        self.assertFalse(np.array_equal(altered, plain))
        self.assertEqual(prepare_scene(altered,bands,self.plan,record).stats,
                         prepare_scene(plain,bands,self.plan,record).stats)
        next_val = next(t for t in self.roles if t['role']=='devval')
        poisoned[tile_slice(next_val)] = np.inf
        with self.assertRaisesRegex(ValueError,'Nonfinite authorized'):
            development_observations(poisoned, record)

    def test_rehashed_cross_role_outside_tile_and_wrong_seed_are_rejected(self):
        tile_ids = {r:next(t for t in self.roles if t['role']==r) for r in ('devval','unused','test')}
        targets = [tile['box'][0]*64+tile['box'][2] for tile in tile_ids.values()]
        occupied = {y*64+x for t in self.roles for a,b,c,d in (t['box'],) for y in range(a,b) for x in range(c,d)}
        targets.extend((next(i for i in range(64*64) if i not in occupied),64*64))
        for index in targets:
            with self.subTest(index=index):
                altered = deepcopy(self.release)
                support = altered['support']
                pairs = sorted([(index,support['labels'][0]), *zip(support['indices'][1:],support['labels'][1:])])
                support['indices'], support['labels'] = [list(values) for values in zip(*pairs)]
                altered['release_hash'] = digest({k:v for k,v in altered.items() if k!='release_hash'})
                with self.assertRaises(ValueError):
                    released_record(self.plan,self.geometry,altered,{})
        for altered in ({**self.release,'scene':'paviaU'}, deepcopy(self.release)):
            if altered['scene']=='xuzhou':
                altered['support']['seed'] = 2048
            altered['release_hash'] = digest({k:v for k,v in altered.items() if k!='release_hash'})
            with self.assertRaises(ValueError):
                released_record(self.plan,self.geometry,altered,{})
