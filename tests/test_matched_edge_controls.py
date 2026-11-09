import copy
import itertools
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evaluation.matched_edge_controls import freeze,verify
from evaluation.graph_edge_intervention import delete_support
from model.research_config import digest
from research_runtime import tensor_hash


def fixture():
    a=torch.ones(4,4,dtype=torch.float64)
    for i in range(4): a[i,(i+1)%4]=8
    a/=a.sum(1,keepdim=True)
    return a.to_sparse().coalesce(),torch.tensor([[0.,0.],[0.,1.],[1.,1.],[1.,0.]],dtype=torch.float64)


class MatchedControlsTests(unittest.TestCase):
    def test_exhaustive_optima_degree_distance_and_actual_deletion(self):
        a,xy=fixture(); before=tensor_hash(a); rng=torch.get_rng_state().clone()
        plan=freeze(a,xy,per_row=1,distance_bins=[1.1],seeds=[64,512,2048])
        ids=[i*4+j for i in range(4) for j in range(4) if i!=j]; candidates=[]
        for edges in itertools.combinations(ids,4):
            rows=[e//4 for e in edges]; cols=[e%4 for e in edges]
            if sorted(rows)!=list(range(4)) or sorted(cols)!=list(range(4)):continue
            if any(float(torch.linalg.vector_norm(xy[e//4]-xy[e%4]))>=1.1 for e in edges):continue
            if list(edges)!=plan['baseline']['removed']:candidates.append(edges)
        self.assertTrue(candidates)
        for name,control in plan['controls'].items():
            self.assertEqual(control['status'],'optimal'); self.assertIn(tuple(control['removed']),candidates)
            cost=a.to_dense().numpy().ravel()[ids] if name=='low_weight' else np.random.default_rng(int(name.split(':')[1])).random(12)
            expected=min(sum(cost[ids.index(e)] for e in c) for c in candidates)
            self.assertAlmostEqual(control['objective'],expected,places=12)
            pruned=delete_support(a,control['removed']);self.assertEqual(pruned._nnz(),12)
            self.assertEqual(torch.bincount(pruned.indices()[0]).tolist(),[3]*4)
            self.assertEqual(torch.bincount(pruned.indices()[1]).tolist(),[3]*4)
        self.assertEqual(before,tensor_hash(a));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        restored=json.loads(json.dumps(plan));self.assertTrue(verify(restored,a,xy,expected_hash=plan['plan_hash'])['discrete_constraints_verified'])

    def test_no_alternative_zero_and_all_masks_never_relax(self):
        a,xy=fixture()
        for k in (0,3):
            plan=freeze(a,xy,per_row=k,distance_bins=[.5],seeds=[64])
            self.assertTrue(all(x['status']=='no_distinct_control' and x['removed'] is None for x in plan['controls'].values()))
            verify(plan,a,xy,expected_hash=plan['plan_hash'])
        a=torch.tensor([[.5,.5],[.5,.5]],dtype=torch.float64).to_sparse().coalesce()
        plan=freeze(a,xy[:2],per_row=1,distance_bins=[.5],seeds=[64])
        self.assertEqual(plan['controls']['low_weight']['status'],'no_distinct_control')

    def test_time_limit_result_does_not_accept_feasible_incumbent(self):
        from types import SimpleNamespace
        a,xy=fixture()
        with patch('evaluation.matched_edge_controls.milp',return_value=SimpleNamespace(status=1,success=False,message='time limit',x=np.ones(12))):
            plan=freeze(a,xy,per_row=1,distance_bins=[1.1],seeds=[64])
        self.assertTrue(all(x['status']=='unresolved' and x['removed'] is None for x in plan['controls'].values()))

    def test_identity_constraints_and_invalid_geometry(self):
        a,xy=fixture();plan=freeze(a,xy,per_row=1,distance_bins=[1.1],seeds=[64])
        with self.assertRaises(ValueError):verify(plan,a,xy+1,expected_hash=plan['plan_hash'])
        bad=copy.deepcopy(plan);bad['controls']['low_weight']['removed']=[0]
        with self.assertRaises(ValueError):verify(bad,a,xy,expected_hash=plan['plan_hash'])
        bad.pop('plan_hash');bad['plan_hash']=digest(bad)
        with self.assertRaises(ValueError):verify(bad,a,xy,expected_hash=bad['plan_hash'])
        for bins in ([],[0.],[2.,1.],[float('inf')]):
            with self.assertRaises(ValueError):freeze(a,xy,per_row=1,distance_bins=bins,seeds=[64])
        with self.assertRaises(ValueError):freeze(a,xy*float('nan'),per_row=1,distance_bins=[1.],seeds=[64])


if __name__=='__main__':unittest.main()
