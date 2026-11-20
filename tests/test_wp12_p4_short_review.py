import unittest

import numpy as np

from evaluation.wp12_p4_short_review import (
    CLASSES, HEADS, descriptive_cross_cohorts, focus_readout, nearest_other_labels, roster_pair,
    threshold_comparison,
)
from transfer import p4_frozen_diagnostic_metrics as metrics


class P4ShortReviewTests(unittest.TestCase):
    def test_same_point_wrong_to_wrong_differs_from_marginal_recall(self):
        truth = np.repeat(CLASSES, 5)
        baseline = np.eye(22)[truth-1]
        candidate = baseline.copy()
        where = np.where(truth==15)[0][:2]
        baseline[where] = np.eye(22)[[0,3]]
        candidate[where] = np.eye(22)[[3,0]]
        first, last = (metrics.recompute(value,truth) for value in (baseline,candidate))
        evidence = threshold_comparison(truth,baseline,candidate,first,last)
        self.assertEqual(evidence["recall_loss_classes_gt_2pp"], [])
        self.assertEqual(evidence["delta"]["prediction_changes"], 2)
        self.assertEqual(evidence["focus"]["15"]["destinations"], [[1,4,1],[4,1,1],[15,15,3]])
        self.assertEqual(focus_readout(truth,candidate,15)["misclassified_into"], {1:1,4:1})
        candidate[where[0]+2] = np.eye(22)[3]
        lost = threshold_comparison(truth,baseline,candidate,first,metrics.recompute(candidate,truth))
        self.assertEqual(lost["recall_loss_classes_gt_2pp"], [15])
        self.assertFalse(lost["oa_aa_limits_passed"])

    def test_spatial_neighbours_stay_inside_the_released_block_and_record_ties(self):
        layout = [({"box":[0,3,0,4]},np.array([1,4,6])),
                  ({"box":[3,5,0,4]},np.array([2]))]
        truth = np.array([15,3,4,21])
        actual = nearest_other_labels(layout,truth)
        self.assertEqual(actual[0],([3,4],2))
        self.assertEqual(actual[3],([],None))
        scores = np.eye(22)[[2,2,3,20]]
        checked = focus_readout(truth,scores,15,actual)["nearest_released_other_class"]
        self.assertEqual(checked["wrong_predicted_one_of_nearest"],1)
        self.assertEqual(checked["ties"],1)

    def test_validation_signature_and_support_cannot_be_exchanged(self):
        sample = dict(pair_sha256="paired",validation={"sha256":"val"},support={"sha256":"train"},
                      observation="view",physical_uuid="gpu",scope="scope",case="case")
        lines = {arm:dict(sample) for arm in ("none","anchor_only","binary","continuous","shuffled")}
        lines["continuous"]["validation"] = {"sha256":"other_equal_size_val"}
        with self.assertRaisesRegex(ValueError,"do not share"):
            roster_pair("group",lines,{})

    def test_cross_group_tables_link_only_one_scientific_factor_and_show_device(self):
        arms = ("none","anchor_only","binary","continuous","shuffled")
        one = dict(validation={"sha256":"validation"},observation="training_view",physical_uuid="a",
                   selected_step=30,heads={h:dict(oa=.1,aa=.1,kappa=.01,macro_f1=.02,nll=3.) for h in HEADS})
        other = {**one,"physical_uuid":"b"}
        a,b,c,d = [(2,k,method,512,64) for k in (1,5) for method in ("scratch","shared_transfer")]
        rows = {k:dict.fromkeys(arms,row) for k,row in zip((a,b,c,d),(one,other,other,one),strict=True)}
        self.assertEqual(descriptive_cross_cohorts({b:rows[b],c:rows[c]}),[])
        compared = descriptive_cross_cohorts(rows)
        self.assertEqual(len(compared),4)
        self.assertTrue(all("physical_device" in x["changed_factors"] for x in compared))
