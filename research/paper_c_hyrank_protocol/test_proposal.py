"""Policy and budget checks only, without scientific data or model execution."""
import copy
import tempfile
from pathlib import Path
import unittest

import check_proposal as c


class FocusedProposal(unittest.TestCase):
    def proposal(self):return c.read(c.HERE/'hypotheses.json')
    def test_unique_source_fit_budget_not_nine_cell_retraining(self):
        r = c.review();self.assertEqual(r['conditional_counts']['producer_updates'],576);self.assertEqual(r['conditional_counts']['prediction_presentations'],360);self.assertEqual(r['current_optimizer_updates'],0)
    def test_C7_bare_ground_mapping_cannot_be_reused(self):
        p = self.proposal();p['classes'][0]['xuzhou_original_ids'] = [1,7]
        with self.assertRaisesRegex(ValueError,'water/woody'):c.review(p)
    def test_illustrated_ordinal_is_not_a_delivered_array_value(self):
        p = self.proposal();p['classes'][0]['HyRANK_delivered_values'] = [13]
        with self.assertRaisesRegex(ValueError,'ordinal guess'):c.review(p)
    def test_same_Hyperion_control_cannot_be_promoted_to_new_family(self):
        p = self.proposal();p['scenes'][2]['family'] = 'invented_new_sensor'
        with self.assertRaisesRegex(ValueError,'same-family'):c.review(p)
    def test_weights_target_updates_and_legacy_target_statistics_rejected(self):
        for key, value in [('external_pretrained_weights_allowed',True),('per_transfer_arm_target_optimizer_updates',1),('representation','legacy_scene_distribution')]:
            p = self.proposal();p['fitting'][key] = value
            with self.assertRaisesRegex(ValueError,'source fits'):c.review(p)
    def test_fake_update_budget_or_hidden_source_scene_rejected(self):
        p = self.proposal();p['budget_if_separately_authorized']['producer_updates'] = 1728
        with self.assertRaisesRegex(ValueError,'unique source'):c.review(p)
        p = self.proposal();p['fitting']['source_scenes'].append('Erato')
        with self.assertRaisesRegex(ValueError,'hidden validation'):c.review(p)
    def test_English_and_request_delivery_preserves_all_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'for-D';r=c.export(out)
            self.assertEqual(len(r['files']),10)
            for name in c.DOCUMENTS:self.assertEqual(c.digest(c.HERE/name),c.digest(out/name))
            self.assertEqual(r['outbound_requests_sent'],0);self.assertEqual(len(list(out.iterdir())),11)


if __name__ == '__main__':unittest.main()
