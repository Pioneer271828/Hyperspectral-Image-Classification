import unittest
import numpy as np
from loadData.spatial_tiles import make_tile_plan, tile_slice
from loadData.transfer_protocol import make_support
from loadData.target_validation import make_validation


class ValidationTests(unittest.TestCase):
    def test_only_val_labels_affect_selection_and_support_unchanged(self):
        plan=make_tile_plan((96,96),32,2,2026)
        labels=np.resize(np.arange(1,10),(96,96))
        support=make_support(labels,plan,list(range(1,10)),10,2027)
        expected=make_validation(labels,plan,support)
        poisoned=labels.copy()
        for tile in plan['tiles']:
            if tile['partition']!='val':poisoned[tile_slice(tile)]=-999
        self.assertEqual(expected,make_validation(poisoned,plan,support))
        self.assertEqual(expected['total_training_selection_labels'],135)
        self.assertFalse(set(expected['indices'])&set(support['indices']))
        for cls in range(1,10):self.assertEqual(expected['labels'].count(cls),5)
        self.assertEqual(support,make_support(labels,plan,list(range(1,10)),10,2027))

    def test_insufficiency_and_invalid_values_fail_without_redraw(self):
        plan=make_tile_plan((96,96),32,2,2026)
        labels=np.resize(np.arange(1,10),(96,96))
        support=make_support(labels,plan,list(range(1,10)),10,2027)
        for tile in plan['tiles']:
            if tile['partition']=='val':labels[tile_slice(tile)]=0
        with self.assertRaises(ValueError):make_validation(labels,plan,support)
        with self.assertRaises(ValueError):make_validation(labels,plan,support,seed=-1)
