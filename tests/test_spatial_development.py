import unittest
import numpy as np
from loadData.spatial_tiles import make_tile_plan,tile_slice
from loadData.spatial_development import make_development


class DevelopmentTests(unittest.TestCase):
    def test_isolation_coverage_and_determinism(self):
        plan=make_tile_plan((96,96),32,2,2026)
        labels=np.resize(np.arange(1,10),(96,96))
        record=make_development(labels,plan,list(range(1,10)))
        poisoned=labels.copy()
        for tile in plan['tiles']:
            if tile['partition']!='train':poisoned[tile_slice(tile)]=-999
        self.assertEqual(record,make_development(poisoned,plan,list(range(1,10))))
        self.assertEqual(record['training_labels'],90)
        self.assertEqual(record['validation_labels'],45)
        self.assertFalse(set(record['train_support']['indices'])&set(record['validation_support']['indices']))
        for c in range(1,10):
            self.assertEqual(record['train_support']['labels'].count(c),10)
            self.assertEqual(record['validation_support']['labels'].count(c),5)
        for tile in record['roles']:
            if tile['partition']=='test':self.assertEqual(tile['role'],'test')

    def test_impossible_class_is_rejected(self):
        with self.assertRaises(ValueError):
            make_development(np.ones((96,96)),make_tile_plan((96,96),32,2,2026),[1,2])
