"""Verify small-cluster denominators and fail-closed stage access, using synthetic records."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from metrics import measures,cluster_weights,sensitivity,paired
from phase_data import PhaseData
from c8_bridge import require_learning_enabled


class FixedProtocol(unittest.TestCase):
    def test_paused_real_entries_reject_before_argument_or_input_access(self):
        import run_all
        import train_stage
        for entry in (require_learning_enabled,run_all.main,train_stage.main):
            with self.assertRaisesRegex(RuntimeError,'training is paused'):entry()
    def record(self,correct):
        y=[0]*5+[1]*5;pred=y if correct else [0]*10
        return {'id':str(correct),'scene':'synthetic','point_ids':list(range(10)),'parent_ids':[1]*5+[2]*5,'truth':y,'predictions':pred,'logits':[0.]*10,'probabilities':[.5]*10,'metrics':measures(y,[0.]*10,[.5]*10,pred)}
    def test_balanced_ground_constant_reference_has_expected_denominators(self):
        r=self.record(False)['metrics'];self.assertEqual(r['confusion'],[[5,0],[5,0]]);self.assertEqual(r['class_denominators'],[5,5]);self.assertEqual(r['OA'],.5);self.assertEqual(r['AA'],.5);self.assertAlmostEqual(r['macroF1'],1/3);self.assertAlmostEqual(r['NLL'],np.log(2));self.assertEqual(r['binary_Brier'],.25)
    def test_two_parent_bootstrap_keeps_missing_class_metrics_undefined(self):
        r=self.record(True);s=sensitivity(r);self.assertEqual(len(cluster_weights(r['parent_ids'])),4);self.assertEqual(s['AA']['valid_draws'],2);self.assertEqual(s['recall_ground']['valid_draws'],3);self.assertEqual(s['OA']['range_2_5_97_5'],[1.,1.])
    def test_pairing_counts_five_true_corrections_without_new_points(self):
        r=paired(self.record(True),self.record(False));self.assertEqual(r['points'],10);self.assertEqual(r['A_correct_B_wrong'],5);self.assertEqual(r['A_wrong_B_correct'],0);self.assertEqual(r['OA_delta'],.5)
    def test_different_point_sets_are_not_paired_by_position(self):
        a=self.record(True);b=self.record(False);b['point_ids'][0]=99
        with self.assertRaisesRegex(ValueError,'Point pairing'):paired(a,b)
    def test_fractional_labels_and_nonprobabilities_rejected(self):
        with self.assertRaises(ValueError):measures([.5],[0.],[.5],[0])
        with self.assertRaises(ValueError):measures([0],[0.],[1.1],[0])
    def test_phase_is_rejected_before_adapter_or_payload_access(self):
        obj=PhaseData.__new__(PhaseData);obj.allowed={'source_fit'}
        with self.assertRaisesRegex(ValueError,'Phase not available'):obj.load('target_development',None)
    def test_hook_forbids_target_NPY_and_historical_MAT_or_weights(self):
        with patch('phase_data.PairInputs') as adapter,patch('phase_data.sys.addaudithook') as install:
            adapter.return_value.tile_ids.return_value=(1,)
            view=PhaseData(expected_commit=None,c7_root=None,inputs='/tmp/c8-synthetic',plan=None,direction=('xuzhou','whu-honghu'),allowed={'source_fit'});hook=install.call_args.args[0]
        hook('open',('/tmp/c8-synthetic/xuzhou/tile-1.npy','rb',0));self.assertEqual(len(view.opens),1)
        for path in ('/tmp/c8-synthetic/whu-honghu/tile-1.npy','/tmp/old.mat','/tmp/old.pt'):
            with self.assertRaises(PermissionError):hook('open',(path,'rb',0))


if __name__=='__main__':unittest.main()
