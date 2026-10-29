"""Hand-calculated E0a contracts. Execution pending in the remote container."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.classification_metrics import classification_metrics, confusion_matrix, metrics_from_confusion
from evaluation.spatial_metrics import boundary_pixels, boundary_bands, spatial_metrics
from evaluation.branch_diagnostics import branch_diagnostics
from evaluation.region_diagnostics import graph_diagnostics
from export_research_diagnostics import evaluate_predictions, export_frozen_predictions


class ClassificationTests(unittest.TestCase):
    def test_hand_confusion_metrics_and_fixed_order(self):
        # Rows GT, columns predicted: class 1 has P=2/3,R=1/2,F1=4/7;
        # class 2 has P=2/3,R=4/5,F1=8/11. OA=6/9, expected=14/27,
        # so kappa=(18/27-14/27)/(1-14/27)=4/13.
        matrix = np.array([[2, 2], [1, 4]])
        result = metrics_from_confusion(matrix, [1, 2])
        self.assertAlmostEqual(result['oa'], 6 / 9)
        self.assertAlmostEqual(result['aa'], (.5 + .8) / 2)
        self.assertAlmostEqual(result['macro_f1'], (4 / 7 + 8 / 11) / 2)
        self.assertAlmostEqual(result['kappa'], 4 / 13)
        self.assertEqual(result['macro_class_ids'], [1, 2])
        actual = confusion_matrix(np.array([3, 1, 3, 1]), np.array([1, 1, 3, 3]), [3, 1])
        np.testing.assert_array_equal(actual, [[1, 1], [1, 1]])

    def test_absent_class_no_predictions_and_empty_selection(self):
        result = metrics_from_confusion([[2, 0, 0], [1, 0, 0], [0, 0, 0]], [1, 2, 3])
        self.assertEqual(result['macro_class_ids'], [1, 2])
        self.assertEqual(result['missing_class_ids'], [3])
        self.assertIsNone(result['aa_full'])
        self.assertIsNone(result['per_class'][1]['precision'])
        self.assertEqual(result['per_class'][1]['f1'], 0.)
        self.assertIsNone(result['per_class'][2]['f1'])
        empty = classification_metrics(np.array([0, 0]), np.array([999, -1]), [1, 2])
        self.assertEqual(empty['count'], 0)
        for key in ('oa', 'aa', 'macro_f1', 'kappa'): self.assertIsNone(empty[key])
        json.dumps(empty, allow_nan=False)

    def test_mask_excludes_invalid_predictions_and_unlabeled_pixels(self):
        labels = np.array([[1, 2], [0, 2]])
        prediction = np.array([[1., np.nan], [-999., 1.]])
        mask = np.array([[True, False], [True, True]])
        result = classification_metrics(labels, prediction, [1, 2], mask)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['oa'], .5)
        with self.assertRaises(ValueError): classification_metrics(labels, prediction, [1, 2])
        with self.assertRaises(ValueError): classification_metrics(labels, labels, [1, 2], mask.astype(int))
        with self.assertRaises(ValueError): classification_metrics(labels, labels, [1])
        with self.assertRaises(ValueError): metrics_from_confusion([[-1]], [1])

    def test_kappa_single_class_is_undefined(self):
        result = metrics_from_confusion([[4]], [7])
        self.assertEqual(result['oa'], 1.)
        self.assertIsNone(result['kappa'])


class SpatialTests(unittest.TestCase):
    def test_boundary_incident_pixels_and_widths(self):
        labels = np.array([[1, 1, 1, 2, 2, 2]])
        bands = boundary_bands(labels)
        np.testing.assert_array_equal(bands[1], [[False, False, True, True, False, False]])
        np.testing.assert_array_equal(bands[2], [[False, True, True, True, True, False]])
        np.testing.assert_array_equal(bands[3], np.ones((1, 6), dtype=bool))
        result = spatial_metrics(labels, labels, [1, 2])
        for width, expected in (('1', 2), ('2', 4), ('3', 6)):
            self.assertEqual(result['widths'][width]['boundary']['count'], expected)
        self.assertIsNone(result['widths']['3']['interior']['oa'])

    def test_no_unlabeled_or_diagonal_boundary_and_no_hole_crossing(self):
        labels = np.array([[1, 0], [0, 2]])
        self.assertFalse(boundary_pixels(labels).any())
        labels = np.array([[1, 2, 0, 2, 2]])
        np.testing.assert_array_equal(boundary_bands(labels, widths=(3,))[3], [[True, True, False, False, False]])
        excluded = np.array([[True, False, False, True, True]])
        self.assertFalse(boundary_pixels(labels, excluded).any())
        with self.assertRaises(ValueError): boundary_bands(labels, widths=(0,))


class BranchTests(unittest.TestCase):
    def test_four_strata_correction_and_harm_denominators(self):
        truth = np.ones(4, dtype=int)
        cnn = np.array([1, 1, 2, 2]); graph = np.array([1, 2, 1, 2]); fused = np.array([2, 1, 2, 1])
        result = branch_diagnostics(truth, cnn, graph, fused, [1, 2])
        self.assertEqual([v['count'] for v in result['strata'].values()], [1, 1, 1, 1])
        self.assertEqual(result['correction'], {'numerator': 1, 'denominator': 2, 'rate': .5,
                                                'population': 'exactly_one_branch_correct'})
        self.assertEqual(result['harm']['rate'], 1.)
        self.assertEqual(result['branch_oracle_accuracy'], .75)
        self.assertEqual(result['strata']['both_wrong']['fused_correct'], 1)
        self.assertEqual(sum(s['fused_correct'] for s in result['strata'].values()), 2)

    def test_empty_strata_rates_are_na(self):
        values = np.array([1, 1])
        result = branch_diagnostics(values, values, values, values, [1, 2])
        self.assertIsNone(result['correction']['rate'])
        self.assertIsNone(result['strata']['both_wrong']['fused_accuracy'])
        empty = branch_diagnostics(values, values, values, values, [1, 2], np.zeros(2, dtype=bool))
        self.assertIsNone(empty['fused_accuracy'])


class SparseGraphTests(unittest.TestCase):
    def test_components_degrees_and_aligned_weight_change(self):
        indices = np.array([[0, 0, 1, 2, 3], [0, 1, 0, 2, 3]])
        before = np.array([.5, .5, 1., 1., 1.])
        after = np.array([1., 0., 1., 1., 1.])
        result = graph_diagnostics(5, indices, before, after)
        self.assertEqual(result['weak_component_count'], 4)  # {0,1}, {2}, {3}, {4}
        self.assertEqual(result['largest_weak_component'], 2)
        self.assertEqual(result['isolated_without_nonself_neighbors'], 3)
        self.assertEqual(result['positive_edge_count'], 4)
        self.assertAlmostEqual(result['weight_change_mean'], .2)
        self.assertEqual(result['support_change_count'], 1)
        self.assertEqual(result['max_row_mass_error'], 1.)
        self.assertIsNone(graph_diagnostics(5, indices, before)['weight_change_mean'])

    def test_empty_graph_and_invalid_sparse_inputs(self):
        empty = graph_diagnostics(0, np.empty((2, 0), dtype=int), np.empty(0))
        self.assertEqual(empty['weak_component_count'], 0)
        self.assertIsNone(empty['max_out_degree'])
        with self.assertRaises(ValueError): graph_diagnostics(2, [[0, 0], [1, 1]], [1., 1.])
        with self.assertRaises(ValueError): graph_diagnostics(2, [[0], [2]], [1.])
        with self.assertRaises(ValueError): graph_diagnostics(2, [[0], [1]], [-1.])


class FrozenExportTests(unittest.TestCase):
    def test_programmatic_report_predeclared_small_classes(self):
        labels = np.array([[1, 1], [2, 0]])
        arrays = {key: labels.copy() for key in ('prediction', 'cnn_prediction', 'graph_prediction')}
        report = evaluate_predictions(labels, np.ones_like(labels, dtype=bool), arrays, [1, 2, 3], [2])
        self.assertEqual(report['classification']['prediction']['missing_class_ids'], [3])
        self.assertEqual(report['small_classes']['metrics']['prediction']['count'], 1)
        self.assertIsNone(report['graph'])
        json.dumps(report, allow_nan=False)
        with self.assertRaises(ValueError): evaluate_predictions(labels, labels, arrays, [1, 2])
        with self.assertRaises(ValueError): evaluate_predictions(labels, labels > 0, arrays, [1, 2], [3])

    def test_export_inputs_immutable_hashes_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([[1, 2], [0, 1]])
            np.save(root / 'labels.npy', labels)
            np.save(root / 'mask.npy', np.ones_like(labels, dtype=bool))
            np.savez_compressed(root / 'arrays.npz', **{key: labels for key in ('prediction', 'cnn_prediction', 'graph_prediction')})
            (root / 'checkpoint.pt').write_bytes(b'frozen fixture; never deserialized')
            args = (root / 'arrays.npz', root / 'labels.npy', root / 'mask.npy', root / 'checkpoint.pt', [1, 2], root / 'report')
            kwargs = dict(prediction_commit='a' * 40, exporter_commit='b' * 40,
                          dataset='fixture', protocol_id='fixed', partition='test')
            report = export_frozen_predictions(*args, **kwargs)
            self.assertEqual(report['selected_labeled_pixels'], 3)
            self.assertEqual(report['provenance']['inputs']['checkpoint']['sha256'],
                             hashlib.sha256(b'frozen fixture; never deserialized').hexdigest())
            np.testing.assert_array_equal(np.load(root / 'labels.npy', allow_pickle=False), labels)
            self.assertTrue((root / 'report' / 'research_diagnostics.json').is_file())
            with self.assertRaises(FileExistsError): export_frozen_predictions(*args, **kwargs)
            with self.assertRaises(ValueError): export_frozen_predictions(*args[:-1], root / 'other', **{**kwargs, 'partition': 'train'})

    def test_changed_frozen_input_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([[1, 2]])
            np.save(root / 'labels.npy', labels)
            np.save(root / 'mask.npy', np.ones_like(labels, dtype=bool))
            np.savez_compressed(root / 'arrays.npz', **{key: labels for key in ('prediction', 'cnn_prediction', 'graph_prediction')})
            (root / 'checkpoint.pt').write_bytes(b'frozen')
            def mutate(*args, **kwargs):
                result = evaluate_predictions(*args, **kwargs)
                np.save(root / 'labels.npy', np.array([[2, 2]]))
                return result
            with patch('export_research_diagnostics.evaluate_predictions', side_effect=mutate):
                with self.assertRaises(RuntimeError):
                    export_frozen_predictions(root / 'arrays.npz', root / 'labels.npy', root / 'mask.npy',
                        root / 'checkpoint.pt', [1, 2], root / 'report', prediction_commit='a' * 40,
                        exporter_commit='b' * 40, dataset='fixture', protocol_id='fixed', partition='test')
            self.assertFalse((root / 'report').exists())


if __name__ == '__main__': unittest.main()
