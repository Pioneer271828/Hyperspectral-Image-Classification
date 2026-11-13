"""Verify the pinned external oracle against the real G4 checkpoint and graph entry."""

from io import BytesIO
import os
from pathlib import Path
import subprocess
from unittest import TestCase

import torch

from compression.g4_cnn_stage1 import ANCHOR
from research_runtime import tensor_hash
from test_g4_compact_cnn_stage1 import fixture
from transfer.oss_a_dependency import compare_stage1


PIN = "e80127d7c6935c319a7dd6719b8a72e3c1fcbff9"
DEVICE = os.environ.get("OSS_A_DEVICE","cpu")


def library():
    import torch_pruning

    root = Path(torch_pruning.__file__).resolve().parents[1]
    head = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if head != PIN or subprocess.check_output(["git","-C",str(root),"status","--porcelain"]):
        raise ValueError("Clean fixed Torch-Pruning source required")
    return torch_pruning


def snapshot(model):
    return {name:tensor_hash(value) for name,value in model.state_dict().items()}


def step(model, optimizer, value, q, a):
    optimizer.zero_grad(set_to_none=True)
    output = model("scene",value,q,a)[:3]
    loss = sum((head[:,0]**2).mean() for head in output)
    loss.backward()
    optimizer.step()
    return tuple(head.detach() for head in output)


class DependencyOracleTests(TestCase):
    def test_axes_and_stock_forward_gradient(self):
        for norm in ("batch_norm","domain_bn"):
            with self.subTest(norm=norm):
                model,values,_ = fixture(norm)
                model.to(DEVICE)
                v,q,a = [values[name].to(DEVICE) for name in ("x","q","a")]
                before = snapshot(model)
                compact,proof = compare_stage1(model,torch.rand(1,64,5,6,device=DEVICE),[2,95,126],
                                               depgraph=library())
                self.assertEqual(len(proof["axes"]),12 if norm=="domain_bn" else 8)
                self.assertFalse(proof["candidate_accepted"])
                def mask(_module,_args,output):
                    out = output.clone()
                    out[:,[2,95,126]] = 0
                    return out
                first = v.detach().clone().requires_grad_()
                second = v.detach().clone().requires_grad_()
                with model.get_submodule(ANCHOR).register_forward_hook(mask):
                    reference = model("scene",first,q,a)[:3]
                    sum((head[:,0]**2).mean() for head in reference).backward()
                compared = compact("scene",second,q,a)[:3]
                sum((head[:,0]**2).mean() for head in compared).backward()
                for left,right in zip(reference,compared,strict=True):
                    torch.testing.assert_close(left,right,rtol=0,atol=1e-5)
                    self.assertTrue(torch.equal(left.argmax(1),right.argmax(1)))
                torch.testing.assert_close(first.grad,second.grad,rtol=1e-5,atol=1e-6)
                self.assertEqual(snapshot(model),before)

    def test_checkpoint_resume_and_rejected_topologies(self):
        model,values,_ = fixture("domain_bn")
        model.to(DEVICE)
        v,q,a = [values[name].to(DEVICE) for name in ("x","q","a")]
        witness = torch.rand(1,64,5,6,device=DEVICE)
        external = library()
        compact,_ = compare_stage1(model,witness,[2],depgraph=external)
        resumed,_ = compare_stage1(model,witness,[2],depgraph=external)
        opt = torch.optim.Adam(compact.parameters(),lr=0.001,foreach=False,fused=False)
        after = torch.optim.Adam(resumed.parameters(),lr=0.001,foreach=False,fused=False)
        before = snapshot(compact)
        step(compact,opt,v,q,a)
        self.assertNotEqual(snapshot(compact),before)
        saved = BytesIO()
        torch.save({"model":compact.state_dict(),"optimizer":opt.state_dict()},saved)
        saved.seek(0)
        loaded = torch.load(saved,weights_only=True,map_location=DEVICE)
        resumed.load_state_dict(loaded["model"],strict=True)
        after.load_state_dict(loaded["optimizer"])
        for left,right in zip(step(compact,opt,v,q,a),step(resumed,after,v,q,a),strict=True):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        self.assertEqual(snapshot(compact),snapshot(resumed))
        with self.assertRaisesRegex(ValueError,"Unique stage-1"):
            compare_stage1(model,witness,[2,2],depgraph=external)
        incompatible,_,_ = fixture("group_norm")
        with self.assertRaisesRegex(ValueError,"channel-independent"):
            compare_stage1(incompatible.to(DEVICE),witness,[2],depgraph=external)
        coupled,_,_ = fixture(interaction_mode="add")
        with self.assertRaisesRegex(ValueError,"no intermediate interaction"):
            compare_stage1(coupled.to(DEVICE),witness,[2],depgraph=external)
