"""Oracle for supervision through the selected checkpoint, with a live suffix."""
from copy import deepcopy
import unittest

from model.research_config import digest
from transfer.curriculum_support_review import _stratify, supervision_prefix


class Plan:
    def __init__(self, events):
        self.events = events
        self.updates = len(events)

    def event(self, step):
        return self.events[step-1]


class EndpointExposureTests(unittest.TestCase):
    def test_nine_and_twenty_two_class_suffix_cannot_change_selected(self):
        for k in (9,22):
            with self.subTest(classes=k):
                ids = list(range(2*k))
                events = [dict(step=i, tile_id=0, anchor=1, support_ids=ids) for i in range(1,5)]
                plan = Plan(events)
                records = [dict(event, applied_weights=weights*k) for event,weights in
                           zip(events,([1.,1.],[.5,1.5],[.25,1.75],[1.,1.]), strict=True)]
                altered = deepcopy(records)
                altered[2]["applied_weights"] = [1.75,.25]*k
                teacher = {i:dict(label=i//2+1,tile=0,nll=float(i),prediction=i//2+1) for i in ids}
                for selected in (0,1,2):
                    a = supervision_prefix(records,plan,ids,selected)
                    b = supervision_prefix(altered,plan,ids,selected)
                    self.assertEqual(a,b)
                    self.assertEqual(a["history_sha256"],digest(records[:selected]))
                    self.assertEqual(_stratify(teacher,teacher,a["exposure"]),_stratify(teacher,teacher,b["exposure"]))
                    expected = {0:[0.,0.],1:[1.,1.],2:[1.5,2.5]}[selected]
                    self.assertEqual(a["exposure"],{str(i):dict(raw=selected,effective=expected[i%2]) for i in ids})
                    if selected in (0,1):
                        self.assertIsNone(_stratify(teacher,teacher,a["exposure"])["1"]["association"]["teacher_nll_vs_effective"])
                a = supervision_prefix(records,plan,ids,4)
                b = supervision_prefix(altered,plan,ids,4)
                self.assertNotEqual(a["history_sha256"],b["history_sha256"])
                self.assertEqual([a["exposure"][str(i)]["effective"] for i in (0,1)], [2.75,5.25])
                self.assertEqual([b["exposure"][str(i)]["effective"] for i in (0,1)], [4.25,3.75])
                self.assertNotEqual(_stratify(teacher,teacher,a["exposure"]),_stratify(teacher,teacher,b["exposure"]))
                for step in (-1,5,True,None,2.5):
                    with self.assertRaisesRegex(ValueError,"frozen curriculum budget"):
                        supervision_prefix(records,plan,ids,step)
                with self.assertRaisesRegex(ValueError,"complete history"):
                    supervision_prefix(records[:-1],plan,ids,1)
                bad = deepcopy(records)
                del bad[0]["applied_weights"]
                with self.assertRaises(KeyError):
                    supervision_prefix(bad,plan,ids,1)
