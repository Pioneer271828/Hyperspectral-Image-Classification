"""Verify the historical count contract and approved E1 observation roles."""

import unittest

from evaluation.wp12_m1_e1_history import scores
from transfer.native_independent_release import PHYSICAL_COVERAGE_VIEWS, observation_view, validate_exposure


class CoverageReleaseTests(unittest.TestCase):
    def setUp(self):
        self.views = list(PHYSICAL_COVERAGE_VIEWS)
        self.exposure = dict(support=["full","central_gap","matched_dispersed"],
                             calibration=["full"],evaluation=self.views)

    def test_held_observation_is_not_an_optimizer_or_selector_view(self):
        self.assertEqual(validate_exposure(self.exposure,self.views,columns=False),self.exposure)
        for role in ("support","calibration"):
            altered = {**self.exposure,role:[*self.exposure[role],"held_range"]}
            with self.assertRaises(ValueError):
                validate_exposure(altered,self.views,columns=False)
        with self.assertRaises(ValueError):
            validate_exposure(self.exposure,self.views,columns=True)

    def test_rejected_table_falls_back_on_same_retained_columns(self):
        intake = {"observation_views":self.views,"observation_exposure":self.exposure}
        self.assertEqual(observation_view("central_gap",intake),("central_gap",None))
        self.assertEqual(observation_view("misaligned_gap",intake),("central_gap","rejected_axis"))
        self.assertEqual(observation_view("metadata_missing",intake),("full","missing"))
        self.assertEqual(observation_view("central_gap",{}),("central_gap",None))
        with self.assertRaises(ValueError):
            observation_view("unapproved",intake)


    def test_exposure_protocol_cost_consumes_only_observation_rows(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from model.research_config import digest
        from transfer.native_independent_release import materialize_reviewed, scope_identity

        bank = SimpleNamespace(pin={"sha256":"bank","registration":"release"},
                               release=SimpleNamespace(plan_hash="geometry"),
                               observation_row=Mock(return_value={"id":8}), role_row=Mock())
        roles={"support":[1],"calibration":[2],"evaluation":[8]}
        report=dict(bank=scope_identity(bank),role_assignment=digest(roles),observation_views=self.views,
                    observation_exposure=self.exposure,uses={"cost":{"eligible":True}})
        with patch("transfer.native_independent_release.bank_observation",return_value="unlabelled") as read:
            self.assertEqual(materialize_reviewed(bank,report,roles,"evaluation",8,"sha",
                                                 "misaligned_gap",use="cost"),("unlabelled",None,None))
        bank.role_row.assert_not_called()
        self.assertEqual(read.call_args.kwargs,{"factor":1,"metadata":"rejected_axis","observation_only":True})

    def test_historical_counts_never_infer_integer_counts_from_rounded_scores(self):
        matrix = [[1 if i==j else 0 for j in range(9)] for i in range(9)]
        matrix[4][3] = 1
        entry = scores({"oa":.9,"aa":8.5/9,"confusion_matrix":matrix},"xuzhou")
        self.assertEqual(entry["focus"]["5"],{"n":2,"correct":1})
        self.assertEqual(scores({"oa":.9,"aa":.9},"xuzhou")["focus"]["5"],{"n":None,"correct":None})
        with self.assertRaises(ValueError):
            scores({"oa":.8,"aa":.9,"confusion":matrix},"xuzhou")
