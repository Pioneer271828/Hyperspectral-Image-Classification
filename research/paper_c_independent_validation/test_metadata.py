"""Check source/role consistency and known counterexamples without real data access."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import build_tables as b


class MetadataEvidence(unittest.TestCase):
    def test_original_roster_and_family_relations_remain_complete(self):
        _, _, rows, _, _ = b.tables()
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(r['C9_relation'] == 'same_published_family' for r in rows), 6)
        self.assertEqual(sum(r['C9_relation'] == 'different_published_families' for r in rows), 14)
        self.assertEqual(sum(r['C9_relation'] == 'local_identity_unbound' for r in rows), 10)
        self.assertEqual(sum(r['C9_C7_common2_development_inputs_admitted'] for r in rows), 2)

    def test_CDL_intersections_are_raw_ids_not_native_head_ordinals(self):
        *_, intersections = b.tables()
        self.assertEqual([len(intersections[k]) for k in ('EnMAP-DESIS', 'EnMAP-EO1')], [9, 11])
        self.assertNotIn(0, intersections['EnMAP-DESIS'])

    def test_current_spatial_counts_do_not_certify_independent_parcels(self):
        _, _, _, spatial, _ = b.tables()
        self.assertEqual(sum(r['released_points'] for r in spatial), 40)
        self.assertEqual(sum(r['selected_parent_count_reported'] for r in spatial), 17)
        self.assertEqual(sum(r['untouched_confirmation_units_released'] for r in spatial), 0)
        self.assertTrue(all(r['validated_independent_parcels'] is None for r in spatial))

    def test_false_candidate_admission_is_rejected(self):
        original = b.read
        def modified(path):
            value = original(path)
            if Path(path).name == 'comparisons.json':
                value['comparisons'][0]['second_device_pair_ready'] = True
            return value
        with patch('build_tables.read', side_effect=modified):
            with self.assertRaises(ValueError):b.tables()

    def test_relabeling_whu_scene_as_different_family_is_rejected(self):
        original = b.read
        def modified(path):
            value = original(path)
            if Path(path).name == 'comparisons.json':
                next(r for r in value['comparisons'] if r['id'] == 'WHU-HH-HC-control')['relation'] = 'different_published_families'
            return value
        with patch('build_tables.read', side_effect=modified):
            with self.assertRaisesRegex(ValueError, 'Different-family'):b.tables()

    def test_exports_preserve_legacy_cells_and_unknown_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'tables'
            receipt = b.build(out)
            self.assertEqual(receipt['newly_ready_second_HSI_pairs'], 0)
            self.assertEqual(len(list(out.iterdir())), 6)
            with (out/'original30-current-qualification.csv').open(encoding='utf-8') as stream:
                projected = list(__import__('csv').DictReader(stream))
            with (b.HERE/'reuse/D-candidate-pairs30.csv').open(encoding='utf-8-sig') as stream:
                original = list(__import__('csv').DictReader(stream))
            for a, old in zip(projected, original):
                self.assertEqual({k:a[k] for k in old}, old)

    def test_unobserved_HyRANK_classes_and_hidden_validation_stay_unreleased(self):
        data, rows, *_ = b.tables()
        self.assertTrue(all(d['project_common_fit_points'] == 0 for d in data if d['id'].startswith('hyrank')))
        control = next(r for r in rows if r['id'] == 'HyRANK-scene-control')
        self.assertIn('COVERAGE_UNKNOWN', control['semantic_status'])
        self.assertFalse(any(d['id'] in ('erato', 'nefeli', 'kirki') for d in data))


if __name__ == '__main__':
    unittest.main()
