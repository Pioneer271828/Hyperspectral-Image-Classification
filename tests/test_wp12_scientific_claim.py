"""Prospective and legacy publication decisions use held-pixel counts."""

import unittest

from evaluation import wp12_scientific_claim as science
from evaluation.wp12_cross_backend import raw_pair

VIEWS = ("full","central_gap","matched_dispersed")
PAIR = dict(baseline="off",candidate="scale",comparison="architecture_with_capacity_change")


def scores(scene="paviaU", *, gains=(), losses=(), count=21):
    classes = 22 if scene=="whu-honghu" else 9
    truth = [y for y in range(1,classes+1) for _ in range(count)]
    baseline,candidate = list(truth),list(truth)
    for y in gains:
        baseline[(y-1)*count] = y%classes+1
    for y in losses:
        candidate[(y-1)*count+1] = y%classes+1
    row = raw_pair(truth,baseline,candidate,classes,floor=20)
    return {head:dict(row,protected=True,accepted=True,target_gain=True,delta_aa=1.)
            for head in ("fused","graph","cnn")}


def seed_case(seed, *, scene="paviaU", policy=science.PAVIA, gains=(), losses=(), adverse=None, missing=None):
    profiles = {view:dict(accuracy=scores(scene,gains=gains if view=="central_gap" else (),
                                        losses=losses if view=="central_gap" else ())) for view in VIEWS}
    if adverse:
        profiles[adverse] = dict(accuracy=scores(scene,losses=(1,)))
    if missing in VIEWS:
        profiles.pop(missing)
    cohort = {"pairs":{} if missing=="arm" else {"off/scale":profiles}}
    _,proof = science.paired_result(PAIR,cohort,scene,science.PHYSICAL,policy,VIEWS)
    return dict(seed=seed,**proof)


class ScientificClaimTest(unittest.TestCase):
    def test_legacy_does_not_adopt_three_identical_or_improved_results(self):
        for possible in ((),(1,)):
            rows = [seed_case(seed,policy=None,gains=possible) for seed in science.SEEDS]
            result = science.conclusion(rows,VIEWS,authentic_real=True)
            self.assertTrue(result["comparable"] and result["protection_passed"])
            self.assertIsNone(result["directed_effect"])
            self.assertFalse(result["adopted"])

    def test_same_output_and_only_one_winner_cannot_support_prospective_gain(self):
        for winners in (0,1):
            rows = [seed_case(seed,gains=(1,) if i<winners else ())
                    for i,seed in enumerate(science.SEEDS)]
            result = science.conclusion(rows,VIEWS,authentic_real=True)
            self.assertTrue(result["protection_passed"])
            self.assertFalse(result["directed_effect"] or result["adopted"])

    def test_primary_only_win_or_raised_aa_with_rare_loss_fails(self):
        rows = [seed_case(seed,gains=(1,3,4),losses=(2,)) for seed in science.SEEDS]
        result = science.conclusion(rows,VIEWS,authentic_real=True)
        self.assertTrue(result["directed_effect"])
        self.assertFalse(result["protection_passed"] or result["adopted"])
        rows = [seed_case(seed,gains=(1,),adverse="full") for seed in science.SEEDS]
        self.assertFalse(science.conclusion(rows,VIEWS,authentic_real=True)["directed_effect"])

    def test_incomplete_seed_arm_and_view_and_unrepresented_class(self):
        full = [seed_case(seed,gains=(1,)) for seed in science.SEEDS]
        for rows in (full[:-1], [*full[:-1],seed_case(2048,missing="arm")],
                               [*full[:-1],seed_case(2048,missing="matched_dispersed")]):
            result = science.conclusion(rows,VIEWS,authentic_real=True)
            self.assertFalse(result["complete"] or result["adopted"])
        self.assertFalse(science.view_result(None,"paviaU",science.PAVIA)["comparable"])
        self.assertFalse(science.view_result(scores(count=9),"paviaU",science.PAVIA)["directional_evaluable"])

    def test_positive_scope_and_exact_guard_counts(self):
        rows = [seed_case(seed,gains=(1,) if seed!=2048 else ()) for seed in science.SEEDS]
        eligible = science.conclusion(rows,VIEWS,authentic_real=True)
        self.assertTrue(eligible["adopted"])
        engineering = science.conclusion(rows,VIEWS,authentic_real=False)
        self.assertTrue(engineering["adoption_requirements_met"])
        self.assertFalse(engineering["adopted"])
        hh = science.view_result(scores("whu-honghu",gains=(15,16,17),losses=(21,)),"whu-honghu",science.FOCUS)
        self.assertFalse(hh["directed_effect"] or hh["protection_passed"])
        xz = science.view_result(scores("xuzhou",gains=(5,)),"xuzhou",science.FOCUS)
        self.assertTrue(xz["directed_effect"] and xz["protection_passed"])


if __name__ == "__main__":
    unittest.main()
