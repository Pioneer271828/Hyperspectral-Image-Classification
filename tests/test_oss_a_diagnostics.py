"""Acceptance probes for spectral units, irregular routes and the joint matrix."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from joint_training import run_training, _tree_fingerprint
from model.research_config import ResearchConfig
from model.variable_channel_adapter import VariableChannelAdapter
from scripts.train_joint_m4 import graph_binding
from test_calibrated_selection import items
from test_joint_training import PROVENANCE, SOURCE, checkpoints, hashes, settings
from transfer.oss_a_joint_registration import OrthogonalRuntime, options, register, verify_inputs
from transfer.oss_a_region_layout import routed_region_attention
from transfer.oss_a_spectral import diagnostic


class DiagnosticTests(TestCase):
    def test_wavelength_metadata_diagnostics(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(64)
            adapter = VariableChannelAdapter(latent_channels=64,embed_dim=32)
        before = _tree_fingerprint(adapter.state_dict())
        all_rows = {scene:diagnostic(adapter,scene) for scene in ("paviaU","xuzhou","whu-honghu")}
        self.assertTrue(all(row["wavelength_quality"]=="approximate" and row["fwhm_unit"]=="missing"
                            and row["optimizer_updates"]==0 for rows in all_rows.values() for row in rows))
        self.assertTrue(all(rows[1]["active_anchors"]==64 for rows in all_rows.values()))
        self.assertLess(all_rows["paviaU"][0]["active_anchors"],64)
        self.assertEqual(all_rows["xuzhou"][0]["outside_global_nm"],2)
        self.assertEqual(_tree_fingerprint(adapter.state_dict()),before)

    def test_irregular_regions_dense_reference_and_permutation(self):
        ids = torch.tensor([11,2,8,1,6,29,4])
        gen = torch.Generator().manual_seed(31)
        q,k,v,l = [torch.randn(n,4,generator=gen,dtype=torch.float64,requires_grad=True) for n in (7,7,7,2)]
        output,neighbors,selected = routed_region_attention(q,k,v,l,ids,3)
        scale = 4**-.5
        expected = []
        for i in range(len(q)):
            local = [int((ids==region).nonzero()[0]) for region in neighbors[i]]
            global_ = torch.stack([((k@router)*scale).softmax(0)@v for router in l])
            logits = torch.cat(((k[local]@q[i])*scale,(l@q[i])*scale))
            expected.append(logits.softmax(0)@torch.cat((v[local],global_),dim=0))
        torch.testing.assert_close(output,torch.stack(expected),rtol=1e-12,atol=1e-12)
        permutation = torch.tensor([6,2,0,3,1,5,4])
        after,edges,_ = routed_region_attention(q[permutation],k[permutation],v[permutation],l,ids[permutation],3)
        torch.testing.assert_close(after,output[permutation],rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(edges,neighbors[permutation]))
        self.assertTrue(torch.equal(selected,(q@l.T).argmax(1)))
        for grad in torch.autograd.grad(output.square().sum(),(q,k,v,l)):
            self.assertTrue(torch.isfinite(grad).all() and bool(grad.abs().sum()))
        for bad_ids,bad_k in ((ids,8),(torch.ones_like(ids),3)):
            with self.assertRaisesRegex(ValueError,"unique region"):
                routed_region_attention(q,k,v,l,bad_ids,bad_k)

    def test_registered_24_cells_and_synthetic_continuation(self):
        data = items()
        identity = register(settings,data,data_identity="joint-synthetic-fixture",source_commit="fixture")
        self.assertEqual(len(identity["cells"]),24)
        self.assertEqual(identity["registered_updates"],288)
        self.assertFalse(identity["real_training_authorized"])
        verify_inputs(identity,data)
        forged = dict(identity,scope="synthetic_only",provenance_authenticated=True)
        with self.assertRaisesRegex(ValueError,"authorized provenance"):
            verify_inputs(forged,data)
        changed = {name:dict(item,net_input=item["net_input"]+1) for name,item in data.items()}
        with self.assertRaisesRegex(ValueError,"inputs changed"):
            verify_inputs(identity,changed)
        mixed = {name:dict(item,val_samples_gt=item["train_samples_gt"]) for name,item in data.items()}
        with self.assertRaisesRegex(ValueError,"Disjoint train and validation"):
            verify_inputs(identity,mixed)
        for seed in (64,512,2048):
            self.assertEqual(len({row["schedule_sha256"] for row in identity["cells"] if row["seed"]==seed}),1)
        altered = {name:dict(item,test_label_mask=torch.ones(1)) for name,item in data.items()}
        with self.assertRaisesRegex(ValueError,"Train/validation-only"):
            register(settings,altered,data_identity="rejected",source_commit="fixture")
        device = os.environ.get("OSS_A_DEVICE","cpu")
        data = {name:{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in item.items()}
                for name,item in data.items()}
        provenance = {**PROVENANCE,"m4_graph_binding":graph_binding(data),"oss_a_joint_registration":identity}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            shared_initializations = {}
            for index,cell in enumerate(identity["cells"]):
                key = cell["residual"],cell["supervision"],cell["adapter"]
                runtime = OrthogonalRuntime(*key)
                option = options(settings(cell["seed"]),*key)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(cell["seed"])
                    model = runtime.build_model(ResearchConfig.from_dict(option["config"]),
                                                {name:item["class_num"] for name,item in data.items()})
                shared = {name:value for name,value in model.state_dict().items()
                          if not name.startswith("backbone.region_residual.")}
                init_key = cell["seed"]
                fingerprint = _tree_fingerprint(shared)
                self.assertEqual(shared_initializations.setdefault(init_key,fingerprint),fingerprint)
                full = root/f"full-{index}"
                answer = run_training(option,data,full,provenance,SOURCE,device=device,runtime_adapter=runtime)
                self.assertEqual(answer["updates_this_call"],12)
                if cell["seed"]!=64:
                    continue
                split = root/f"split-{index}"
                run_training(option,data,split,provenance,SOURCE,device=device,runtime_adapter=runtime,stop_after=5)
                answer = run_training(option,data,split,provenance,SOURCE,device=device,runtime_adapter=runtime,resume=True)
                self.assertEqual(answer["updates_this_call"],7)
                for field in ("model","optimizer","scheduler"):
                    self.assertEqual(_tree_fingerprint(checkpoints(full)[field]),_tree_fingerprint(checkpoints(split)[field]))
                frozen = hashes(split)
                answer = run_training(option,data,split,provenance,SOURCE,device=device,runtime_adapter=runtime,resume=True)
                self.assertEqual(answer["updates_this_call"],0)
                self.assertEqual(hashes(split),frozen)
