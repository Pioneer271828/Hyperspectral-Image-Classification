"""Independent directed support and numerical gradient checks."""

from copy import deepcopy
import unittest

import torch

from evaluation.controlled_signal_learning import Trajectory, adjacency, geometry, model, predict, support, topology
from model.regional_support_evolution import Schedule, SupportEvolution, development_context
from model.research_config import digest
from transfer.xuzhou_multiscale_audit import _exact


def small(mode="joint", **changes):
    a=torch.eye(6)
    a[2,1]=.5
    a[2,3]=.7
    a[2,0]=.4
    feats=torch.tensor([[1.,1.],[2.,1.],[1.,2.],[3.,1.],[1.,3.],[2.,2.]])
    options=dict(context=development_context(role="synthetic",block="unit",node_ids=[91,82,73,64,55,46],partitions=[0,0,0,0,0,1]),
                 mandatory=[[2,1]],schedule=Schedule(4,2,2,2,2,12,pool=2),mode=mode,seed=64)
    options.update(changes)
    return SupportEvolution(a,feats,feats,**options)


class RegionalSupportTests(unittest.TestCase):
    def test_bounded_modes_actual_addition_reduction_and_gradient(self):
        for mode in ("local","random","spectral","graph","joint"):
            r=small(mode)
            for step in range(1,8):
                r.advance(step)
            x=r.features.clone().requires_grad_()
            coo=r.render(scores=x)
            pairs=[tuple(v) for v in coo.indices().t().tolist()]
            self.assertTrue(coo.is_coalesced())
            self.assertTrue(all((i==5)==(j==5) for i,j in pairs))
            self.assertIn((2,1),pairs)
            self.assertLessEqual(max(coo.to_dense().count_nonzero(1)).item(),2)
            self.assertTrue((coo.values()>0).all())
            self.assertTrue(torch.allclose(coo.to_dense().sum(1),torch.ones(6)))
            if mode=="local":
                self.assertTrue(set(pairs)<=r.local)
                self.assertLess(coo._nnz(),len(r.local))
            else:
                self.assertTrue(set(pairs)-r.local)
            check=torch.zeros((6,6))
            for i,j in pairs:
                v=torch.dot(x[i],x[j])/(x[i].norm()*x[j].norm())
                check[i,j]=.5*r.base.to_dense()[i,j]+.5*torch.exp(v/2)
            check=check/check.sum(1,keepdim=True)
            self.assertTrue(torch.allclose(coo.to_dense(),check,atol=2e-7))
            grad=torch.autograd.grad(coo.to_dense().square().sum(),x)[0]
            expected=torch.autograd.grad(check.square().sum(),x)[0]
            self.assertTrue(torch.allclose(grad,expected,atol=1e-6))
            self.assertGreater(float(grad.abs().sum()),0)

    def test_no_refresh_on_render_and_exact_resume_rejects_rehashed_mutations(self):
        source=small()
        for s in range(1,6):
            source.advance(s)
        value=source.receipt()
        source.advance(5)
        source.render()
        self.assertEqual(source.receipt(),value)
        dest=small()
        dest.restore(value)
        for s in range(6,10):
            source.advance(s);dest.advance(s)
        self.assertEqual(source.receipt(),dest.receipt())
        for key,wrong in (("pool",[]),("budget",16),("rng",{}),("refresh_step",1),("selected",[[0,5]])):
            modified=dict(value,**{key:wrong})
            modified["state_sha256"]=digest({k:v for k,v in modified.items() if k!="state_sha256"})
            with self.assertRaisesRegex(ValueError,"history"):
                small().restore(modified)
        for changed in (dict(seed=512),dict(schedule=Schedule(4,2,2,2,3,12))):
            with self.assertRaisesRegex(ValueError,"contract"):
                small(**changed).restore(value)
        fresh=small()
        with self.assertRaisesRegex(ValueError,"partition"):
            fresh.render(support=[[0,0],[0,5],[1,1],[2,1],[2,2],[3,3],[4,4],[5,5]])
        with self.assertRaisesRegex(ValueError,"cursor"):
            fresh.advance(2)

    def test_invalid_context_mandatory_out_of_range_nan_and_devval(self):
        with self.assertRaisesRegex(ValueError,"Mandatory"):
            small(mandatory=[[2,0],[2,1]])
        for role in ("query","test"):
            with self.assertRaisesRegex(ValueError,"development"):
                development_context(role=role,block="no",node_ids=[0],partitions=[0])
        r=small(context=development_context(role="devval",block="held",node_ids=list(range(6)),partitions=[0,0,0,0,0,1]))
        with self.assertRaisesRegex(ValueError,"cannot advance"):
            r.advance(1)
        with self.assertRaisesRegex(ValueError,"Finite"):
            r.render(scores=torch.full((6,2),float("nan")))
        with self.assertRaisesRegex(ValueError,"node range"):
            r.render(support=[[0,6]])

    def test_stable_id_permutation_for_ties(self):
        x=torch.ones(7,2);a=torch.eye(7)
        ids=[11,80,34,67,49,23,55]
        order=[4,0,5,6,3,1,2]
        def run(a,x,ids):
            r=SupportEvolution(a,x,x,context=development_context(role="synthetic",block="perm",node_ids=ids,partitions=[0]*7),
                               mandatory=[],schedule=Schedule(3,2,0,2,2,6,pool=3),mode="joint",seed=64)
            for i in range(1,6):
                r.advance(i)
            return {(ids[i],ids[j]) for i,j in r.selected}
        self.assertEqual(run(a,x,ids),run(a[order][:,order],x[order],[ids[i] for i in order]))

    def test_two_hops_direction_leakage_and_trainable_eval_batchnorm(self):
        for graph_name in ("chain","bridge","bottleneck"):
            top=topology(graph_name)
            signals=torch.tensor([-.7,.7])
            m,_=model(64)
            for mode in ("local","cut","reachable"):
                edges=support(top,mode,64)
                d=geometry(top,edges)["distance"]
                adj=adjacency(edges,top["nodes"])
                result=predict(m,adj,top,signals)
                if mode=="reachable":
                    self.assertEqual(d,1)
                    self.assertNotEqual(float(result[0]),float(result[1]))
                else:
                    self.assertTrue(d is None or d>2)
                    self.assertEqual(float(result[0]),float(result[1]))
                self.assertTrue(torch.allclose(result,predict(m,adj,top,signals,permutation=list(reversed(range(12))))))
            self.assertFalse(m.batch_normalzation2.norm.training)
            self.assertTrue(m.batch_normalzation2.norm.weight.requires_grad)

    def test_real_refresh_boundary_optimizer_and_validation_state(self):
        case=dict(key="test",kind="evolution",topology="bottleneck",mode="regenerate",seed=64,budget=120)
        torch.manual_seed(64)
        run=Trajectory(case)
        for _ in range(23):
            run.step()
        cp=deepcopy(run.packet())
        fresh=Trajectory(case)
        fresh.restore(cp)
        for _ in range(4):
            run.step();fresh.step()
        self.assertTrue(_exact(run.packet(),fresh.packet()))
        self.assertEqual(run.router.refresh_step,24)
        self.assertGreater(sum(x["changes"]["score.weight"] for x in run.history),0)
        wrong=deepcopy(cp)
        wrong["history"][0]["sample_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"sample"):
            Trajectory(case).restore(wrong)
