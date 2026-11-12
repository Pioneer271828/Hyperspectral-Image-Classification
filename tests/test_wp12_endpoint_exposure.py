"""Endpoint-prefix fixtures independent of the production review reducer."""
from copy import deepcopy
import unittest

from evaluation.wp12_curriculum_evidence import endpoint_supervision
from model.research_config import digest


class TableEndpointTests(unittest.TestCase):
    def test_endpoint_exposure_schema_and_tamper_on_nine_and_twenty_two_classes(self):
        for classes in (9,22):
            with self.subTest(classes=classes):
                ids = list(range(classes*2))
                support = [dict(index=i,label=i//2+1,tile=0) for i in ids]
                history = [dict(support_ids=ids,applied_weights=weights*classes)
                           for weights in ([1.,1.],[.25,1.75],[.5,1.5])]
                masses = {0:[0.,0.],1:[1.,1.],2:[1.25,2.75],3:[1.75,4.25]}
                def summary(step):
                    return dict(step=step,history_sha256=digest(history[:step]),
                        exposure={str(i):dict(raw=step,effective=masses[step][i%2]) for i in ids})
                def endpoint(step):
                    value = summary(step)
                    value["train_support"] = {str(c+1):dict(samples=[
                        dict(index=i,class_id=c+1,**value["exposure"][str(i)]) for i in (2*c,2*c+1)])
                        for c in range(classes)}
                    return value
                for selected in (1,2,3):
                    item = dict(history=history,initial={"step":0},selected={"step":selected},last={"step":3},
                                exposure={"by_support":summary(3)["exposure"]})
                    proof = dict(full_run=summary(3),endpoints=dict(initial=endpoint(0),selected=endpoint(selected),last=endpoint(3)))
                    accepted = endpoint_supervision(proof,item,support,3)
                    self.assertEqual(accepted["selected"],summary(selected))
                    self.assertEqual(accepted["initial"]["exposure"]["0"],dict(raw=0,effective=0.))
                    for field,value in (("history_sha256","a"*64),("step",True),("exposure",{})):
                        bad = deepcopy(proof)
                        bad["endpoints"]["selected"][field] = value
                        with self.assertRaises(ValueError):
                            endpoint_supervision(bad,item,support,3)
                    bad = deepcopy(proof)
                    del bad["endpoints"]["initial"]
                    with self.assertRaisesRegex(ValueError,"Versioned"):
                        endpoint_supervision(bad,item,support,3)
                    bad = deepcopy(proof)
                    bad["endpoints"]["selected"]["train_support"]["1"]["samples"][0]["effective"] += 1.
                    with self.assertRaisesRegex(ValueError,"strata"):
                        endpoint_supervision(bad,item,support,3)
                    if selected < 3:
                        bad = deepcopy(proof)
                        bad["endpoints"]["selected"] = endpoint(3)
                        with self.assertRaisesRegex(ValueError,"step"):
                            endpoint_supervision(bad,item,support,3)
