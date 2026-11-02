import copy
import unittest
from scripts.diagnose_accuracy_variance import curve_summary, exposure_summary, seed_arm_decomposition


class AccuracyDiagnosticsTest(unittest.TestCase):
    def result(self):
        return {'steps': 3, 'trace': [{'step': i, 'tile_id': 1, 'anchor_class': 1,
                                     'loss': {'total': 1/i}} for i in range(1, 4)],
                'validation': [{'step': 1, 'nll': 1., 'oa': .8},
                               {'step': 2, 'nll': .5, 'oa': .7},
                               {'step': 3, 'nll': .5, 'oa': .9}], 'selected_step': 2}

    def test_earliest_nll_and_accuracy_tradeoff(self):
        d = curve_summary(self.result())
        self.assertEqual((d['selected_step'], d['selected_oa'], d['best_oa_in_history']), (2, .7, .9))

    def test_reject_incomplete_or_reselected_or_nonfinite(self):
        for field in ['trace', 'selected_step', 'validation']:
            r = self.result()
            if field == 'trace': r[field].pop()
            elif field == 'selected_step': r[field] = 3
            else: r[field][0]['nll'] = float('nan')
            with self.assertRaises(ValueError): curve_summary(r)

    def test_prefix_is_saved_trajectory(self):
        r = self.result(); r['steps'] = 2000
        r['trace'] = [{'step': i, 'loss': {'total': 1.}} for i in range(1, 2001)]
        r['validation'] = [{'step': 1000, 'nll': 1., 'oa': .5}, {'step': 2000, 'nll': .8, 'oa': .4}]
        r['selected_step'] = 2000
        self.assertEqual([x['oa'] for x in curve_summary(r)['prefixes']], [.5, .4])

    def test_participation_is_not_label_visits_or_loss_share(self):
        r = self.result(); r['sampler'] = {'classes': [1, 2], 'tile_ids': [1, 2],
            'tile_class_counts': [[3, 1], [0, 1]], 'tile_probabilities': [.75, .25]}
        r['trace'][-1]['tile_id'] = 2; r['trace'][-1]['anchor_class'] = 2
        d = exposure_summary(r)
        self.assertEqual(d['classes'][0]['updates_present'], 2)
        self.assertEqual(d['classes'][0]['label_visits'], 6)
        self.assertEqual(d['total_label_visits'], 9)
        self.assertAlmostEqual(d['classes'][0]['expected_balanced_loss_share'], .375)
        self.assertAlmostEqual(d['classes'][1]['expected_balanced_loss_share'], .625)

    def test_unknown_training_tile_rejected(self):
        r = self.result(); r['sampler'] = {'classes': [1], 'tile_ids': [2],
            'tile_class_counts': [[1]], 'tile_probabilities': [1.]}
        with self.assertRaises(ValueError): exposure_summary(r)

    def test_descriptive_seed_effect_not_arm_effect(self):
        rows = [{'seed': s, 'mode': m, 'oa': v} for s, v in [(1, .5), (2, .9)] for m in ['a', 'b']]
        d = seed_arm_decomposition(rows)
        self.assertAlmostEqual(d['seed_ss_fraction'], 1.)
        self.assertAlmostEqual(d['arm_ss_fraction'], 0.)
        with self.assertRaises(ValueError): seed_arm_decomposition(rows[:-1])


if __name__ == '__main__': unittest.main()
