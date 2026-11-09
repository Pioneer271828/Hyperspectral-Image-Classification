"""Actual restricted updates, source/context contracts and exact recovery."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from g4_light_target import (ADAPTER_HEADS, CONTROLS, HEAD_ONLY, LightTargetRuntime, allowed,
    initialize, parameter_binding, prepare)
from g4_support_target import MODES
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, write_json
from test_g4_support_context import prepared
from test_joint_training import checkpoints, hashes
from test_m4_full_transfer import fixture as active_source
from test_none_full_transfer import fixture as none_source


def bound(root, manifest, out, method, mode, spec, device="cpu", **extra):
    return prepare(manifest, file_hash(manifest), root/out, target="xuzhou", method=method,
        mode=mode, model_seed=64, support_seed=extra.get("support_seed", 512), per_class=extra.get("k", 1),
        source_spec=spec, device=device)


class LightTargetTests(unittest.TestCase):
    def test_source_initial_parameter_names_and_fresh_heads_are_identical_across_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for mode in MODES:
                _, spec=none_source(root/mode) if mode=="none" else active_source(root/mode, mode)
                values=[]
                for method in CONTROLS:
                    model, record, binding=initialize("xuzhou",method,mode,model_seed=64,head_seed=1064,source_spec=spec)
                    values.append((record["initial_state_sha256"],binding))
                    self.assertEqual(record["method"], method)
                    self.assertEqual(set(n for n,p in model.named_parameters() if p.requires_grad),set(binding["trainable_parameters"]))
                    self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters()
                        if n.startswith(("backbone.CNN", "backbone.HGNN", "backbone.region_residual"))))
                self.assertEqual(values[0][0],values[1][0])
                self.assertLess(values[0][1]["trainable_elements"],values[1][1]["trainable_elements"])

    def test_six_roles_exact_cpu_resume_frozen_invariants_three_reloads_and_completed_zero(self):
        device=os.environ.get("LIGHT_TARGET_DEVICE","cpu")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,catalog,_=prepared(root);manifest=catalog/"manifest.json"
            for mode in MODES:
                _, spec=none_source(root/mode) if mode=="none" else active_source(root/mode,mode)
                for method in CONTROLS:
                    role=mode+"-"+method
                    opts,items,provenance,code,runtime,guard=bound(root,manifest,role+"-full",method,mode,spec,device)
                    full=root/(role+"-full")
                    run_training(opts,items,full,provenance,code,runtime_adapter=runtime,finish_guard=guard,device=device)
                    opts,items,provenance,code,runtime,guard=bound(root,manifest,role+"-split",method,mode,spec,device)
                    split=root/(role+"-split")
                    self.assertEqual(run_training(opts,items,split,provenance,code,runtime_adapter=runtime,
                        finish_guard=guard,device=device,stop_after=2)["updates_this_call"],2)
                    # A new adapter exercises persisted bank/state rather than in-memory continuation.
                    opts,items,provenance,code,runtime,guard=bound(root,manifest,role+"-split",method,mode,spec,device)
                    result=run_training(opts,items,split,provenance,code,runtime_adapter=runtime,
                        finish_guard=guard,device=device,resume=True)
                    self.assertEqual(result["updates_this_call"],10)
                    if device=="cpu":
                        for key in ("model","optimizer","scheduler","history","best_probabilities"):
                            self.assertEqual(_tree_fingerprint(checkpoints(full)[key]),_tree_fingerprint(checkpoints(split)[key]),role+key)
                    trace=checkpoints(split)["trace"]
                    self.assertEqual(len(trace),12)
                    self.assertTrue(all(r["frozen_parameters_unchanged"] and r["frozen_buffers_unchanged"]
                        and r["changed_trainable_elements"]>0 for r in trace))
                    if method==ADAPTER_HEADS:
                        self.assertTrue(all(r["adapter_changed_elements"]>0 and r["gradient_norms"]["adapter"]>0 for r in trace))
                        self.assertFalse((split/"feature-bank.pt").exists())
                    else:
                        self.assertTrue(all(r["adapter_changed_elements"]==0 and r["gradient_norms"]["adapter"]==0 for r in trace))
                        self.assertTrue((split/"feature-bank.pt").exists())
                    cost=json.loads((split/"adaptation-cost.json").read_text())
                    self.assertEqual(sum(v["phase"]=="optimizer_update_with_invariant_checks" for v in cost["events"]),12)
                    self.assertEqual(sum(v["phase"]=="initial_ordered_target_calibration" for v in cost["events"]),2)
                    self.assertFalse(cost["deployment_speedup_claim"])
                    moved={n:{k:v.to(device) if torch.is_tensor(v) else v for k,v in item.items()} for n,item in items.items()}
                    raw=split/f'raw-validation-{result["result"]["selected_step"]:06d}.pt'
                    expected=_tree_fingerprint(torch.load(split/"selected-probabilities.pt",weights_only=True))
                    before=hashes(split)
                    for _ in range(3):
                        model=runtime.load_selected(split/"primary.pt",moved,raw,device=device)
                        self.assertEqual(_tree_fingerprint(runtime.validation_snapshot(model,moved)[1]),expected)
                        self.assertEqual(parameter_binding(model,method),runtime.parameter_contract)
                    self.assertEqual(run_training(opts,items,split,provenance,code,runtime_adapter=runtime,
                        finish_guard=guard,device=device,resume=True)["updates_this_call"],0)
                    self.assertEqual(before,hashes(split))
                    model=load_model_checkpoint(split/"primary.pt")
                    model.training_config["light_target_contract"]["trainable_elements"]+=1
                    save_model_checkpoint(model,root/(role+"-bad.pt"))
                    with self.assertRaisesRegex(ValueError,"Selected light target"):
                        runtime.load_selected(root/(role+"-bad.pt"),moved,raw,device=device)

    def test_policy_permissions_and_frozen_state_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,catalog,_=prepared(root);_,spec=none_source(root/"source")
            opts,items,_,_,runtime,_=bound(root,catalog/"manifest.json","run",ADAPTER_HEADS,"none",spec)
            for key,value in (("optimizer_steps",1500),("profile","full"),("method",HEAD_ONLY)):
                bad=copy.deepcopy(opts);bad[key]=value
                with self.assertRaises(ValueError): runtime.validate(bad)
            runtime.directory.mkdir()
            cfg=ResearchConfig.from_dict(opts["config"]);model=runtime.build_model(cfg,{"xuzhou":9})
            frozen=next(p for n,p in model.named_parameters() if not allowed(n,ADAPTER_HEADS))
            frozen.requires_grad_(True)
            with self.assertRaises(ValueError):runtime.optimizer(model,cfg)
            frozen.requires_grad_(False)
            optimizer=runtime.optimizer(model,cfg)
            original=frozen.detach().clone()
            with torch.no_grad(): frozen.add_(.01)
            with self.assertRaisesRegex(ValueError,"Frozen parameter"):
                runtime.train_step_at(0,model,optimizer,"xuzhou",items["xuzhou"],cfg)
            with torch.no_grad():
                frozen.copy_(original)
                next(model.buffers()).add_(1)
            with self.assertRaisesRegex(ValueError,"Frozen optimizer-phase buffers"):
                runtime.train_step_at(0,model,optimizer,"xuzhou",items["xuzhou"],cfg)
            with self.assertRaises(ValueError):bound(root,catalog/"manifest.json","bad","target_scratch_full","none",spec)
            with self.assertRaises(ValueError):bound(root,catalog/"manifest.json","bad",HEAD_ONLY,"none",{})

    def test_head_only_bank_corruption_and_wrong_method_resume_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,catalog,_=prepared(root);_,spec=none_source(root/"source")
            args=bound(root,catalog/"manifest.json","run",HEAD_ONLY,"none",spec)
            opts,items,prov,code,runtime,guard=args
            run_training(opts,items,root/"run",prov,code,runtime_adapter=runtime,finish_guard=guard,stop_after=2)
            wrong=bound(root,catalog/"manifest.json","run",ADAPTER_HEADS,"none",spec)
            before=hashes(root/"run")
            with self.assertRaisesRegex(ValueError,"Resume identity"):
                run_training(wrong[0],wrong[1],root/"run",wrong[2],wrong[3],runtime_adapter=wrong[4],resume=True)
            self.assertEqual(before,hashes(root/"run"))
            with (root/"run/feature-bank.pt").open("ab") as stream:stream.write(b"changed")
            before=hashes(root/"run")
            with self.assertRaises(ValueError):
                fresh=bound(root,catalog/"manifest.json","run",HEAD_ONLY,"none",spec)
                run_training(fresh[0],fresh[1],root/"run",fresh[2],fresh[3],runtime_adapter=fresh[4],resume=True)
            self.assertEqual(before,hashes(root/"run"))

    def test_cli_actual_two_ten_zero_and_full_or_wrong_sha_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,catalog,_=prepared(root);_,spec=none_source(root/"source")
            specfile=root/"source/source-spec.json";write_json(specfile,spec);manifest=catalog/"manifest.json"
            device=os.environ.get("LIGHT_TARGET_DEVICE","cpu")
            command=[sys.executable,"scripts/train_g4_light_target_short.py","--expected-commit",code_snapshot()["commit"],
                "--manifest",str(manifest),"--manifest-sha256",file_hash(manifest),"--source-spec",str(specfile),
                "--source-spec-sha256",file_hash(specfile),"--out",str(root/"cli"),"--target","xuzhou",
                "--method",ADAPTER_HEADS,"--mode","none","--model-seed","64","--support-seed","2048",
                "--per-class","5","--device",device]
            def invoke(extra):return json.loads(subprocess.check_output(command+extra,text=True))
            self.assertEqual(invoke(["--stop-after","2"])["updates_this_call"],2)
            self.assertEqual(invoke(["--resume"])["updates_this_call"],10)
            before=hashes(root/"cli")
            self.assertEqual(invoke(["--resume"])["updates_this_call"],0)
            self.assertEqual(before,hashes(root/"cli"))
            self.assertNotEqual(subprocess.run(command+["--full"],capture_output=True).returncode,0)
            wrong=command.copy();wrong[wrong.index("--expected-commit")+1]="0"*40
            self.assertNotEqual(subprocess.run(wrong+["--resume"],capture_output=True).returncode,0)


if __name__=="__main__":unittest.main()
