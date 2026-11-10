"""Independent NumPy softmax oracle, immutable recovery, synthetic model capture."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from evaluation.linear_probe import POLICY, advance, capture_pack, fingerprint, gradient, metrics, prepare
from research_runtime import code_snapshot, hash_file, write_json
from scripts.run_frozen_linear_probes import run


def fixture(classes=3, width=4, steps=12):
    g = torch.Generator().manual_seed(64)
    labels = {"train":torch.arange(classes).repeat(3), "val":torch.arange(classes).repeat(2)}
    ids = {"train":torch.arange(3*classes), "val":torch.arange(3*classes,5*classes)}
    features = {name:{role:torch.randn(len(y),size,generator=g,dtype=torch.float64)
                      for role,y in labels.items()} for name,size in (("wide",width*2),("narrow",width))}
    pack = {"features":features,"labels":labels,"row_ids":ids,
            "identity":{"query_labels_accessed":False,"synthetic":True}}
    plan = {"policy":POLICY,"classes":classes,"width":width,"seed":64,"steps":steps,
            "lr":.1,"l2":.01,"scale_floor":1e-12}
    return pack, plan


def oracle(x,y,w,b,l2):
    logits = x@w.T+b; shifted = logits-logits.max(1,keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(1,keepdims=True)); p = np.exp(logp)
    loss = -logp[np.arange(len(y)),y].mean()+.5*l2*(w*w).sum()
    p[np.arange(len(y)),y] -= 1; p /= len(y)
    return loss,p.T@x+l2*w,p.sum(0)


class LinearProbeTests(unittest.TestCase):
    def test_independent_numpy_gradient_finite_difference_and_metrics(self):
        pack,plan=fixture(); x,y,_=prepare(pack,plan); x=x["wide"]["train"]; y=y["train"]
        w=torch.arange(12,dtype=torch.float64).reshape(3,4)/30; b=torch.tensor([.1,-.3,.2],dtype=torch.float64)
        loss,gw,gb=gradient(x,y,w,b,plan["l2"])
        expected=oracle(x.numpy(),y.numpy(),w.numpy(),b.numpy(),plan["l2"])
        self.assertAlmostEqual(loss,expected[0],places=13)
        np.testing.assert_allclose(gw.numpy(),expected[1],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(gb.numpy(),expected[2],rtol=1e-12,atol=1e-14)
        for i,j in ((0,0),(1,2),(2,3)):
            a=w.numpy().copy(); a[i,j]+=1e-6; c=w.numpy().copy(); c[i,j]-=1e-6
            numeric=(oracle(x.numpy(),y.numpy(),a,b.numpy(),.01)[0]-oracle(x.numpy(),y.numpy(),c,b.numpy(),.01)[0])/2e-6
            self.assertAlmostEqual(numeric,float(gw[i,j]),places=8)
        result=metrics(x,y,w,b); z=x.numpy()@w.numpy().T+b.numpy(); z-=z.max(1,keepdims=True)
        pred=(z-np.log(np.exp(z).sum(1,keepdims=True))).argmax(1)
        cm=np.zeros((3,3),dtype=np.int64); np.add.at(cm,(y.numpy(),pred),1)
        self.assertEqual(result["confusion"],cm.tolist())
        self.assertEqual(result["oa"],float(np.trace(cm)/cm.sum()))
        np.testing.assert_allclose(result["recall"],cm.diagonal()/cm.sum(1),rtol=0,atol=0)

    def test_equal_capacity_three_class_counts_exact_resume_complete_zero_and_input_rng(self):
        for classes in (3,9,22):
            pack,plan=fixture(classes); before=fingerprint(pack); rng=torch.get_rng_state().clone()
            full,report=advance(pack,plan); first,_=advance(pack,plan,stop_after=2)
            split,second=advance(pack,plan,state=first); completed,last=advance(pack,plan,state=split)
            self.assertEqual(full["checksum"],split["checksum"]); self.assertEqual(full["checksum"],completed["checksum"])
            self.assertEqual(second["probe_updates_this_call"],20); self.assertEqual(last["probe_updates_this_call"],0)
            self.assertEqual(before,fingerprint(pack)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in full["layers"].items():
                self.assertEqual(report["layers"][name]["normalization"]["parameters"],classes*5)
                self.assertEqual(value["best"]["step"],min(value["history"],key=lambda h:h["val_nll"])["step"])
                self.assertGreater(value["weight"].abs().sum(),0)

    def test_train_only_normalization_and_validation_never_changes_update(self):
        pack,plan=fixture(); changed=copy.deepcopy(pack)
        changed["labels"]["val"]=(changed["labels"]["val"]+1)%3
        for layer in changed["features"].values(): layer["val"]*=100
        a,ra=advance(pack,plan); b,rb=advance(changed,plan)
        for name in a["layers"]:
            for key in ("weight","bias"): torch.testing.assert_close(a["layers"][name][key],b["layers"][name][key],rtol=0,atol=0)
            self.assertEqual(ra["layers"][name]["normalization"],rb["layers"][name]["normalization"])
        with self.assertRaisesRegex(ValueError,"identity"): advance(changed,plan,state=a)

    def test_constant_missing_classes_first_tie_and_reject_illegal_roles(self):
        pack,plan=fixture(); pack["labels"]["val"].zero_()
        for layer in pack["features"].values():
            for role in layer: layer[role].zero_()
        _,report=advance(pack,plan)
        for layer in report["layers"].values():
            self.assertEqual(layer["selected_step"],1); self.assertEqual(layer["normalization"]["near_constant_columns"],4)
            self.assertEqual(layer["metrics"]["val"]["missing_classes"],[1,2]); self.assertIsNone(layer["metrics"]["val"]["aa"])
            self.assertEqual(layer["metrics"]["val"]["exact_tie_rows"],6)
            self.assertEqual(layer["metrics"]["val"]["roundoff_sensitive_rows"],6)
        for mutate in (lambda p:p["labels"].update(query=torch.tensor([0])),
                       lambda p:p["identity"].update(query_labels_accessed=True),
                       lambda p:p["row_ids"].update(val=p["row_ids"]["train"][:6]),
                       lambda p:p["labels"]["train"].zero_(),
                       lambda p:p["features"]["wide"]["train"].fill_(float("nan"))):
            candidate=copy.deepcopy(pack); mutate(candidate)
            with self.assertRaises(ValueError): advance(candidate,plan)

    def test_state_tamper_plan_identity_and_first_min_history_rejected(self):
        pack,plan=fixture(); state,_=advance(pack,plan,stop_after=2)
        bad=copy.deepcopy(state); bad["layers"]["wide"]["weight"][0,0]+=1
        with self.assertRaises(ValueError): advance(pack,plan,state=bad)
        with self.assertRaises(ValueError): advance(pack,{**plan,"steps":13},state=state)
        for mutate in (lambda s:s["layers"]["wide"]["best"].update(step=999),
                       lambda s:s["layers"]["wide"]["history"][0].update(step=99)):
            bad=copy.deepcopy(state); bad.pop("checksum"); mutate(bad); bad["checksum"]=fingerprint(bad)
            with self.assertRaises(ValueError): advance(pack,plan,state=bad)

    def test_actual_initialized_model_capture_four_layers_readonly_and_role_guard(self):
        from test_target_training import fixture as target_fixture
        from target_bn_calibration import OBS_KEYS
        from target_initialization import state_id
        from g4_matched_target_study import initialize, METHODS
        with tempfile.TemporaryDirectory() as tmp:
            (data,_,guard),_=target_fixture(Path(tmp)); item=data["xuzhou"]
            obs={k:v for k,v in item.items() if k in OBS_KEYS}
            model,_=initialize("xuzhou",64,METHODS[0],"none"); model.eval()
            labels={r:item[r+"_labels"] for r in ("train","val")}; ids={r:item[r+"_indices"] for r in labels}
            before=state_id(model); rng=torch.get_rng_state().clone()
            pack=capture_pack(model,"xuzhou",obs,labels=labels,row_ids=ids)
            _,plan=fixture(9,width=64); state,report=advance(pack,plan)
            self.assertEqual(len(state["layers"]),4); self.assertEqual(report["probe_updates_this_call"],48)
            self.assertEqual(before,state_id(model)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(p.grad is None for p in model.parameters())); guard()
            for layer in pack["features"].values(): self.assertTrue(all(not x.requires_grad for x in layer.values()))
            with patch("evaluation.frozen_representation_capture.capture",side_effect=AssertionError("before forward")):
                with self.assertRaises(ValueError): capture_pack(model,"xuzhou",obs,labels={**labels,"query":labels["val"]},row_ids=ids)
                with self.assertRaises(ValueError): capture_pack(model,"xuzhou",obs,labels=labels,row_ids={**ids,"val":ids["train"][:45]})

    def test_actual_cli_pins_resume_and_output_negative_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inputs=root/"inputs"; inputs.mkdir(); pack,plan=fixture()
            archive=inputs/"pack.pt"; settings=inputs/"plan.json"; torch.save(pack,archive); write_json(settings,plan)
            code=code_snapshot(); pins=(str(archive),hash_file(archive),str(settings),hash_file(settings))
            cmd=[sys.executable,"scripts/run_frozen_linear_probes.py","--expected-commit",code["commit"],
                 "--pack",pins[0],"--pack-sha256",pins[1],"--plan",pins[2],"--plan-sha256",pins[3]]
            subprocess.check_output(cmd+["--out",str(root/"first"),"--stop-after","2"],text=True)
            cursor=root/"first/state.pt"
            result=json.loads(subprocess.check_output(cmd+["--out",str(root/"second"),"--resume",str(cursor),"--resume-sha256",hash_file(cursor)],text=True))
            self.assertEqual(result["probe_updates_this_call"],20)
            full,_=advance(pack,plan); actual=torch.load(root/"second/state.pt",weights_only=True)
            self.assertEqual(full["checksum"],actual["checksum"])
            with patch("torch.load",side_effect=AssertionError("No decode")):
                with self.assertRaises(ValueError): run(pins[0],"0"*64,*pins[2:],root/"blocked",code)
                with self.assertRaises(ValueError): run(*pins,inputs/"nested",code)
                with self.assertRaises(FileExistsError): run(*pins,root/"first",code)
            self.assertNotEqual(subprocess.run(cmd+["--out",str(root/"query"),"--query"],capture_output=True).returncode,0)
            self.assertEqual(hash_file(archive),pins[1]); self.assertEqual(hash_file(settings),pins[3])


if __name__ == "__main__": unittest.main()
