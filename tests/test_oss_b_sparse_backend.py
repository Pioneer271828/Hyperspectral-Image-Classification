"""End-to-end fixed-input test of the opt-in PyG COO multiply."""

import copy
from pathlib import Path
import os
import subprocess
import tempfile
from unittest import TestCase, skipUnless

import torch

from joint_supervision import SupervisionRuntime, supervision_settings
from joint_training import _tree_fingerprint, run_training
from model.research_config import ResearchConfig
from scripts.train_joint_m4 import graph_binding
from test_calibrated_selection import items
from test_joint_training import PROVENANCE, SOURCE, checkpoints, hashes, settings
from transfer.oss_b_sparse_backend import coo_product, use_product


PIN = "d1f2f8642387b6e0563d6aee227c39130746c738"
DEVICE = os.environ.get("OSS_B_DEVICE","cpu")
ENABLED = os.environ.get("OSS_B_TEST_PYG")=="1"


def load_scatter():
    import torch_geometric
    from torch_geometric.utils import scatter
    root = Path(torch_geometric.__file__).resolve().parents[1]
    head = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if head!=PIN or subprocess.check_output(["git","-C",str(root),"status","--porcelain"]):
        raise ValueError("Clean fixed PyG source required")
    return scatter


class SparseContractTests(TestCase):
    def test_scope_and_unsupported_inputs(self):
        dense = torch.eye(2)
        with self.assertRaisesRegex(ValueError,"two-dimensional COO"):
            coo_product(dense,dense,None)
        with self.assertRaisesRegex(ValueError,"two-dimensional COO"):
            coo_product(dense.to_sparse(),torch.ones(3,2),None)
        with self.assertRaisesRegex(ValueError,"only supports sum"):
            with use_product(None):
                torch.sparse.mm(dense.to_sparse(),dense,reduce="mean")
        torch.testing.assert_close(torch.sparse.mm(dense.to_sparse(),dense),dense)


@skipUnless(ENABLED,"Fixed PyG acceptance requires OSS_B_TEST_PYG=1")
class PyGTests(TestCase):
    def test_duplicate_empty_rectangular_and_operand_gradients(self):
        scatter = load_scatter()
        indices = torch.tensor([[0,0,0,3],[1,1,2,0]],device=DEVICE)
        operands = (torch.tensor([.1,.2,.3,-.7],device=DEVICE,dtype=torch.float64),
                    torch.arange(15,device=DEVICE,dtype=torch.float64).reshape(5,3).T)
        outcomes = []
        for product in (torch.sparse.mm,lambda a,b:coo_product(a,b,scatter)):
            values,features = [v.detach().clone().requires_grad_() for v in operands]
            a = torch.sparse_coo_tensor(indices,values,(5,3))
            result = product(a,features)
            outcomes.append((result,*torch.autograd.grad(result.square().sum(),(values,features))))
            self.assertEqual(int(result[[1,2,4]].count_nonzero()),0)
            empty = torch.sparse_coo_tensor(indices[:,:0],values[:0],(5,3))
            self.assertEqual(int(product(empty,features).count_nonzero()),0)
        for a,b in zip(*outcomes,strict=True):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_stock_graph_three_heads_update_and_frozen_resume(self):
        scatter = load_scatter()
        data = items()
        opt = supervision_settings(settings(64),"x_residual","legacy_three_heads")
        runtime = SupervisionRuntime("x_residual","legacy_three_heads")
        cfg = ResearchConfig.from_dict(opt["config"])
        model = runtime.build_model(cfg,{name:item["class_num"] for name,item in data.items()}).to(DEVICE)
        candidate = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            native = directory/"native"
            full,split = directory/"full",directory/"split"
            state = _tree_fingerprint(model.state_dict())
            self.assertEqual(state,_tree_fingerprint(candidate.state_dict()))
            data = {name:{k:(v.to(DEVICE) if torch.is_tensor(v) else v) for k,v in item.items()}
                    for name,item in data.items()}
            item = data["paviaU"]
            a,q,x = item["A"],item["Q"],item["net_input"]
            native_out = model("paviaU",x,q,a,wavelengths=item["wavelengths"])
            sum(v.square().mean() for v in native_out[:3]).backward()
            with use_product(scatter) as calls:
                result = candidate("paviaU",x,q,a,wavelengths=item["wavelengths"])
                sum(v.square().mean() for v in result[:3]).backward()
            self.assertGreaterEqual(calls["coo_product"],4)
            for v,w in zip(native_out[:3],result[:3],strict=True):
                torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-6)
                self.assertTrue(torch.equal(v.argmax(1),w.argmax(1)))
            for (name,left),(right_name,right) in zip(model.named_parameters(),candidate.named_parameters(),strict=True):
                self.assertEqual(name,right_name)
                torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-6)
            provenance = {**PROVENANCE,"m4_graph_binding":graph_binding(data)}
            run_training(opt,data,native,dict(provenance,backend="native"),SOURCE,device=DEVICE,runtime_adapter=runtime)
            custom = dict(provenance,backend=f"pyg-coo-sum:{PIN}")
            with use_product(scatter) as witness:
                run_training(opt,data,full,custom,SOURCE,device=DEVICE,runtime_adapter=runtime)
                run_training(opt,data,split,custom,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=5)
                with self.assertRaisesRegex(ValueError,"identity mismatch"):
                    run_training(opt,data,split,dict(provenance,backend="native"),SOURCE,
                                 device=DEVICE,runtime_adapter=runtime,resume=True)
                answer = run_training(opt,data,split,custom,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
                self.assertEqual(answer["updates_this_call"],7)
                frozen = hashes(split)
                done = run_training(opt,data,split,custom,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
                self.assertEqual(done["updates_this_call"],0)
                self.assertEqual(hashes(split),frozen)
            self.assertGreater(witness["coo_product"],100)
            f,s,n = map(checkpoints,(full,split,native))
            self.assertEqual(f["cursor"],12)
            for key in ("model","optimizer","scheduler","rng","cursor","history","trace"):
                self.assertEqual(_tree_fingerprint(f[key]),_tree_fingerprint(s[key]),key)
            self.assertEqual(f["best_step"],n["best_step"])
            for name,value in n["model"].items():
                torch.testing.assert_close(value,f["model"][name],rtol=1e-5,atol=1e-6)
