import unittest
import numpy as np
from scripts.audit_target_coverage import audit
from loadData.spatial_tiles import make_tile_plan


class CoverageTests(unittest.TestCase):
    def test_counts_conserved_and_holdout_independent_of_labels(self):
        plan=make_tile_plan((96,96),32,2,2026)
        labels=np.resize(np.arange(1,4),(96,96))
        result=audit(labels,plan,3,{'indices':[]})
        other=audit(np.zeros_like(labels),plan,3,{'indices':[]})
        self.assertEqual(result['heldout_tile_ids'],other['heldout_tile_ids'])
        self.assertEqual(sum(r['full'] for r in result['rows']),96*96)
        self.assertTrue(result['feasible_10_train_5_val_each'])
        self.assertFalse(other['feasible_10_train_5_val_each'])

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError):
            audit(np.full((96,96),4),make_tile_plan((96,96),32,2,2026),3,{'indices':[]})
