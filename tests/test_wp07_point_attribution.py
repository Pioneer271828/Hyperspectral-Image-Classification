import unittest
from transfer.wp07_point_attribution import decompose, concentration, identity


class PointBookkeepingTests(unittest.TestCase):
    def test_exact_decomposition_and_error_transition(self):
        value = decompose([.1, .8, .1], [.2, .7, .1], [.5, .4, .1], 2)
        self.assertAlmostEqual(value['total_delta'], value['coefficient_delta']+value['training_residual_delta'])
        self.assertEqual(value['transition'], 'wrong->correct')
        self.assertLess(value['coefficient_delta'], 0)
        self.assertEqual(value['pred_content_weights_mean_coeff'], 2)
        self.assertGreater(value['true_class_margin_content'], 0)
        self.assertLess(value['true_class_margin_trained_mean'], 0)

    def test_wrong_class_change_is_not_correctness_change(self):
        value = decompose([.1,.3,.6], [.1,.7,.2], [.1,.7,.2], 1)
        self.assertNotEqual(value['pred_content'], value['pred_content_weights_mean_coeff'])
        self.assertEqual(value['coefficient_transition'], 'wrong->wrong')
        self.assertEqual(value['coefficient_delta'], 0)

    def test_concentration_is_not_positive_support_count(self):
        self.assertEqual(concentration([.5, .5]), 2)
        self.assertLess(concentration([.99, .01]), 1.1)
        self.assertEqual(len([x for x in [.99, .01] if x > 0]), 2)

    def test_invalid_vectors_and_stable_anonymous_pairing(self):
        with self.assertRaisesRegex(ValueError, 'mass'):
            decompose([.1, .1], [.5, .5], [.5, .5], 1)
        self.assertEqual(identity('scene', 'bank', 2, 1), identity('scene', 'bank', 2, 1))
        self.assertNotEqual(identity('scene', 'bank', 2, 1), identity('scene', 'other', 2, 1))


if __name__ == '__main__':
    unittest.main()
