"""Versioned probe arithmetic, recovery, and evidence status regressions."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from evaluation import frozen_probe_audit as audit, frozen_probe_campaign as campaign
from evaluation import frozen_probe_registry as reg, linear_probe, probe_decimal_reference as reference
from evaluation import probe_pair_audit_status as status, stable_probe_standardization as stable


def probe_fixture(classes):
    generator = torch.Generator().manual_seed(64)
    labels = {"train":torch.arange(classes).repeat(10),"val":torch.arange(classes).repeat(5)}
    ids = {"train":torch.arange(10*classes),"val":torch.arange(10*classes,15*classes)}
    features = {}
    for name in reg.LAYERS:
        width = 128 if name.startswith("stage1") else 64
        features[name] = {}
        for role, y in labels.items():
            x = torch.randn(len(y),width,generator=generator)*0.05
            x[torch.arange(len(y)),y] += 2
            features[name][role] = x
    pack = {"features":features,"labels":labels,"row_ids":ids,
            "identity":{"synthetic":True,"query_labels_accessed":False}}
    return pack,{**reg.PLAN,"classes":classes,"steps":7,"policy":linear_probe.POLICY_V2}


def synthetic_pairs(seeds=(64,512,2048)):
    results = []
    for seed in seeds:
        for mode in ("none","capacity_control","x_residual"):
            case = f"synthetic-{seed}-{mode}"
            pair = {"case":case,"role":{"target":"paviaU","seed":seed,"method":"scratch","mode":mode,
                                       "initial_step":0,"terminal_step":500}}
            for stage in ("initial","terminal"):
                layers = {}
                for name in reg.LAYERS:
                    roles = {}
                    for role,n in (("train",10),("val",5)):
                        fail = (seed,mode,stage,name,role)==(64,"x_residual","initial","stage1_cnn","val")
                        metrics = {"oa":0.5,"aa":0.5,"kappa":0.4,"macro_f1":0.5,"nll":1.2,"zero_recall_classes":[],
                                   "recall":[0.6]*9,"f1":[0.5]*9,"support":[n]*9}
                        roles[role] = {"selected":metrics,"terminal":metrics,
                                       "cross_backend":{"strict_decisions":True,"probabilities_close":not fail}}
                    layers[name] = {"selected_step":500,"selected_at_budget_end":True,"roles":roles}
                pair[stage] = {"role":pair["role"],"case":case,"status":"accepted","disposition":"new","layers":layers}
            results.append(pair)
    return results


class StableProbeNumericsTests(unittest.TestCase):
    def test_small_float32_dispersion_reproduces_v1_scale_error_and_v2_matches_decimal(self):
        # Synthetic representable offsets model the magnitude in the frozen CNN failures.
        values = 0.0628640625+((torch.arange(220,dtype=torch.float64)*37 % 211)-105)*2**-27
        train = values.to(torch.float32).to(torch.float64)[:,None]
        rows = {"train":train,"val":train[::2]}
        oracle,stats = reference.standardize({role:x.numpy() for role,x in rows.items()},1e-12)
        mean,sd = train.mean(0),train.std(0,unbiased=False)
        v1 = ((train-mean)/sd).numpy()
        self.assertGreater(float(np.max(np.abs(v1-oracle["train"]))),1e-12)
        v2, metadata = stable.standardize(rows,1e-12)
        for role in rows:
            np.testing.assert_allclose(v2[role].numpy(),oracle[role],atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(metadata["scale"].numpy(),stats["scale"],atol=0,rtol=1e-14)

    def test_constant_exact_floor_near_floor_offsets_and_extreme_amplitudes(self):
        floor = 2**-40
        small = math.nextafter(floor,0)
        large = math.nextafter(floor,math.inf)
        bias = 2.0**41
        ulp = math.ulp(bias)
        train = torch.tensor([[-small,-floor,-large,bias,bias,-1e300,-1e-200],
                              [small,floor,large,bias+ulp,bias,1e300,1e-200]],dtype=torch.float64)
        rows = {"train":train,"val":train.flip(0)}
        oracle,stats = reference.standardize({role:x.numpy() for role,x in rows.items()},floor)
        result,metadata = stable.standardize(rows,floor)
        self.assertEqual(metadata["scaled_columns"],[False,False,True,True,False,True,False])
        self.assertEqual(metadata["scaled_columns"],stats["scaled_columns"])
        self.assertEqual(metadata["near_constant_columns"],4)
        for role in rows:
            np.testing.assert_allclose(result[role].numpy(),oracle[role],atol=1e-14,rtol=1e-14)

    def test_layout_column_order_and_train_only_nonfinite_rejected(self):
        pack,plan = probe_fixture(9)
        variants = deepcopy(pack)
        for value in variants["features"].values():
            for role,x in value.items():
                value[role] = x.T.contiguous().T
        observed,_,meta = linear_probe.prepare(pack,plan)
        reloaded,_,alternate = linear_probe.prepare(variants,plan)
        for name in observed:
            for role in observed[name]:
                self.assertTrue(torch.equal(observed[name][role],reloaded[name][role]))
            self.assertEqual(linear_probe.fingerprint(meta[name]),linear_probe.fingerprint(alternate[name]))
        name = "stage2_cnn"
        roles = {role:x.to(torch.float64) for role,x in pack["features"][name].items()}
        base,_ = stable.standardize(roles,plan["scale_floor"])
        reverse,_ = stable.standardize({role:x.flip(1) for role,x in roles.items()},plan["scale_floor"])
        self.assertTrue(torch.equal(base["train"],reverse["train"].flip(1)))
        variants["features"][name]["val"] = variants["features"][name]["val"]*1000
        variants["labels"]["val"] = (variants["labels"]["val"]+1)%9
        base_state,_ = linear_probe.advance(pack,plan)
        new_state,_ = linear_probe.advance(variants,plan)
        for name in observed:
            self.assertTrue(torch.equal(base_state["layers"][name]["weight"],new_state["layers"][name]["weight"]))
            self.assertTrue(torch.equal(base_state["layers"][name]["bias"],new_state["layers"][name]["bias"]))
        for bad in (float("nan"),float("inf")):
            variants["features"]["stage2_cnn"]["val"][0,0] = bad
            with self.assertRaisesRegex(ValueError,"Finite bounded"):
                linear_probe.prepare(variants,plan)

    def test_four_real_shapes_9_and_22_saved_recovery_and_independent_v2_audit(self):
        for classes in (9,22):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as folder:
                pack,plan = probe_fixture(classes)
                scope = {"probe_plan":plan,"paths":{"full":[7],"split":[2,4,7],"completed":[7]},
                         "columns":{stage:reg.probe_columns(64,64,width)
                                    for stage,width in (("stage1",128),("stage2",64))}}
                case = {"name":"synthetic","capture_receipt_sha256":"0"*64}
                case_dir = Path(folder)/case["name"]
                ledger = []
                sha = campaign.run_case(case_dir,None,case,{},scope,"1"*64,ledger.append,
                    writer=lambda bound,local_only:pack,contract=lambda case:{})
                archived,state,report,probs,counts,_ = campaign.checked_execution(
                    case_dir,sha,scope,"1"*64,case,{},contract=lambda case:{})
                result = audit.audit_probes(archived,scope,state,report,probs)
                self.assertEqual(counts,{"full":28,"split":28,"completed":0})
                self.assertEqual(result["numerical_failures"],[])
                self.assertEqual(result["independent_selected_probability_roles"],8)
                self.assertEqual(report["policy"],linear_probe.POLICY_V2)
                with self.assertRaisesRegex(ValueError,"identity"):
                    linear_probe.advance(pack,{**plan,"policy":linear_probe.POLICY},state=state)

    def test_v1_uses_original_formula_metadata_and_policy(self):
        pack,plan = probe_fixture(9)
        plan["policy"] = linear_probe.POLICY
        layers,_,metadata = linear_probe.prepare(pack,plan)
        for name in layers:
            columns = metadata[name]["columns"]
            train = pack["features"][name]["train"].to(torch.float64)[:,columns]
            sd = train.std(0,unbiased=False)
            scale = torch.where(sd > plan["scale_floor"],sd,torch.ones_like(sd))
            self.assertTrue(torch.equal(layers[name]["train"],(train-train.mean(0))/scale))
            self.assertNotIn("anchor",metadata[name])
        state,report = linear_probe.advance(pack,plan)
        self.assertEqual(report["policy"],linear_probe.POLICY)
        self.assertEqual(linear_probe.advance(pack,plan,state=state)[1]["probe_updates_this_call"],0)

    def test_numerical_failure_and_unaudited_endpoint_propagate_to_pair_class_and_group(self):
        contrasts,classes,groups,class_groups,protection = status.build(synthetic_pairs())
        def selected(row):
            return (row["mode"],row["layer"],row["role"],row["checkpoint"],row["contrast"],row["stage_comparison"])==(
                "x_residual","stage1_cnn","val","selected","residual_minus_capacity_control","change")
        failed = next(row for row in contrasts if row["seed"]==64 and selected(row))
        self.assertEqual(failed["audit_status"],"failed")
        self.assertEqual(failed["endpoint_count"],4)
        self.assertEqual(failed["failed_endpoint_count"],1)
        self.assertTrue(failed["strict_decisions"])
        self.assertEqual(failed["probability_failed_endpoint_count"],1)
        self.assertEqual(failed["decision_failed_endpoint_count"],0)
        self.assertEqual(next(row for row in classes if row["seed"]==64 and selected(row))["recall"],0)
        group = next(row for row in groups if selected(row))
        self.assertEqual(group["endpoint_count"],12)
        self.assertEqual(group["dependency_audits"]["initial_candidate"]["failed_endpoint_count"],1)
        self.assertFalse(group["eligible_for_formal_numeric_claim"])
        class_group = next(row for row in class_groups if selected(row))
        self.assertEqual(class_group["audit_status"],"failed")
        self.assertEqual(class_group["support"],5)
        guarded = next(row for row in protection if selected(row))
        self.assertTrue(guarded["passes_2pp"])
        self.assertIsNone(guarded["formal_passes_2pp"])
        for row in contrasts:
            if row["checkpoint"]=="terminal":
                self.assertEqual(row["audit_status"],"not_audited")
                self.assertEqual(row["endpoint_count"],row["not_audited_endpoint_count"])
                self.assertFalse(row["eligible_for_formal_numeric_claim"])
        partial = status.build(synthetic_pairs((512,2048)))[2]
        self.assertEqual(next(row for row in partial if selected(row))["audit_status"],"incomplete")


if __name__ == "__main__":
    unittest.main()
