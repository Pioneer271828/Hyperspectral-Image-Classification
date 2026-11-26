"""Actual-cohort random matching with unequal shapes, without any neural call."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from audit_routes import plans
from route_transfer import previous_policy


@unittest.skipUnless(os.environ.get('PAPER_C_NATIVE'),'explicit pinned dependency required')
class ActualShapeMatching(unittest.TestCase):
    def test_random_matches_shape_not_just_band_and_coverage(self):
        sys.path.insert(0,os.environ['PAPER_C_NATIVE'])
        from inference.native_deployment import matched_random
        policy=previous_policy(Path(__file__).resolve().parents[2]);roster=[];light={};requests=[]
        for i,(band,shape,cov,margin) in enumerate((('b',[2,2],.5,.05),('b',[2,2],.5,.1),
                ('b',[3,3],.5,.8),('b',[3,3],.5,.9),('v',[2,2],1.,.01),('v',[2,2],1.,.05))):
            key=str(i);roster.append(dict(key=key,parent=str(i//2),band=band,shape=shape,coverage=cov))
            light['p'+key]=np.tile(np.asarray([(1+margin)/2,(1-margin)/2],dtype=np.float32),(4,1))
            requests.append(SimpleNamespace(key=key,known=True,coverage=cov,cohort=(band,tuple(shape),cov,True),
                parent=str(i//2),band=band,x=SimpleNamespace(shape=shape)))
        c={'seed':64,'full_fraction':.5};expected,margins,_,_=plans(roster,light,c)
        actual,_=policy.plan(requests,margins,'content_coverage',seed=64)
        self.assertEqual(actual,expected['content_coverage'])
        routes={'routes':[{'key':r.key,'route':'full' if r.key in actual else 'light'} for r in requests]}
        actual_full=sorted(set(r.key for r in requests)-set(matched_random(requests,routes,64,0.)))
        self.assertEqual(actual_full,expected['matched_random'])
        self.assertTrue({'0','1'}<=set(actual_full));self.assertFalse({'2','3'}&set(actual_full))


if __name__=='__main__':unittest.main()
