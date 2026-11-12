"""Check independent supervision reduction and role-specific report boundaries."""
from copy import deepcopy
import unittest

from evaluation.wp12_curriculum_evidence import exposure_totals
from evaluation.wp12_production_evidence import provenance


class CurriculumEvidenceTests(unittest.TestCase):
    def test_weighted_exposure_oracles_for_nine_and_twenty_two_classes(self):
        for classes in (9, 22):
            with self.subTest(classes=classes):
                support = [dict(index=i, label=i//2+1, tile=7) for i in range(2*classes)]
                history = [dict(support_ids=list(range(2*classes)), applied_weights=[.25, 1.75]*classes),
                           dict(support_ids=list(range(2*classes)), applied_weights=[1.25, .75]*classes)]
                e = dict(by_support={str(i): dict(raw=2, effective=1.5 if i%2 == 0 else 2.5) for i in range(2*classes)},
                         by_class={str(i+1): dict(raw=4, effective=4.) for i in range(classes)}, total_label_visits=4*classes)
                summary = exposure_totals(e, history, support)
                self.assertEqual(summary, dict(raw_support_visits=4*classes, effective_support_mass=4.*classes, minimum_applied_weight=.25))
                altered = deepcopy(e)
                altered["by_support"]["0"]["effective"] += .25
                with self.assertRaisesRegex(ValueError, "support mass"):
                    exposure_totals(altered, history, support)
                bad = deepcopy(history)
                bad[0]["applied_weights"][0] = 0.
                with self.assertRaisesRegex(ValueError, "applied curriculum"):
                    exposure_totals(e, bad, support)
                with self.assertRaisesRegex(ValueError, "producer SHA"):
                    provenance("b"*40, "a"*40, source_evidence={"path":"/tmp/witness", "sha256":"1"*64},
                               source_steps=256)
