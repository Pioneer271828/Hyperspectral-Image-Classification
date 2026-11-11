import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from model.research_config import digest
from research_runtime import hash_file, write_json
from transfer.p4_exposure import POLICY, bind, cohort, schedule


class ExposureScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tiles = [({'id': 8}, np.array([[1, 2], [0, 2]])),
                      ({'id': 3}, np.array([[3, 0], [0, 0]])),
                      ({'id': 5}, np.array([[1, 0], [0, 0]]))]

    def test_permutation_and_tail_use_only_released_tiles(self):
        options = {'steps': 6, 'events': [2, 6], 'seed': 64, 'classes': 3}
        plan = schedule(self.tiles, [8, 3], **options)
        self.assertEqual(schedule(self.tiles[::-1], [3, 8], **options), plan)
        self.assertEqual(set(plan['tile_sequence'][:2]), {3, 8})
        self.assertTrue(set(plan['tile_sequence']) <= {3, 5, 8})
        self.assertEqual(plan['first_all_classes_step'], 2)
        self.assertEqual(plan['label_exposure_at_events']['2'], [1, 2, 1])

    def test_reject_unreleased_tile_or_missed_coverage_before_selection(self):
        options = {'steps': 5, 'events': [2, 5], 'seed': 64, 'classes': 3}
        for witness in ([8, 99], [8, 8]):
            with self.subTest(witness=witness), self.assertRaisesRegex(ValueError, 'released training'):
                schedule(self.tiles, witness, **options)
        with self.assertRaisesRegex(ValueError, 'before first validation'):
            schedule(self.tiles, [8, 5], **options)
        with self.assertRaisesRegex(ValueError, 'before first validation'):
            schedule(self.tiles, [8, 3], **dict(options, events=[1, 5]))

    def test_rehashed_plan_with_changed_sequence_is_rejected(self):
        case = {'support_seed': 512, 'k': 1, 'release': '/released'}
        tiles = [({'id': 8}, np.arange(1, 23).reshape(1, 22))]
        with TemporaryDirectory() as folder:
            plan, floor = (Path(folder) / name for name in ('plan.json', 'floor.json'))
            evidence = {'status': 'passed', 'protected_files': {'/released': 'a' * 64, '/roles': 'b' * 64},
                        'rows': [{'seed': 512, 'k': 1, 'witness_tiles': [8],
                                  'minimum_tiles_to_see_all_classes': 1}]}
            write_json(floor, evidence)
            registration = {'exposure': {'plan': str(plan), 'floor': str(floor)}, 'development': '/roles',
                            'pins': {**evidence['protected_files'], str(floor): hash_file(floor)}}
            row = cohort(evidence, registration, case, tiles, steps=2, events=[1, 2], seed=64)
            record = {'policy': POLICY, 'geometry_sha256': 'b' * 64, 'floor_sha256': hash_file(floor),
                      'steps': 2, 'validation_steps': [1, 2], 'sampler_seed': 64, 'cohorts': [row]}
            write_json(plan, dict(record, plan_hash=digest(record)))
            registration['pins'][str(plan)] = hash_file(plan)
            self.assertEqual(bind(registration, case, tiles)[1], row)
            row['tile_sequence'][0] = 99
            write_json(plan, dict(record, plan_hash=digest(record)))
            registration['pins'][str(plan)] = hash_file(plan)
            with self.assertRaisesRegex(ValueError, 'sequence or counts differ'):
                bind(registration, case, tiles)
