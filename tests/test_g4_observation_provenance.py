"""Frozen observer provenance, no label decoding and conservative claim permissions."""
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
import g4_observation_provenance as module
from g4_observation_provenance import CATALOG, INPUT_POLICY, audit, inspect_observation, require_claim
from g4_support_context import prepare_contexts, read
from joint_baseline import _item_fingerprint
from loadData.data_read import read_band_metadata, PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, write_json
from test_g4_support_context import fixture as context_inputs


def fixture(root):
    inputs,spec=context_inputs(root/"inputs")
    for target,ref in spec["observations"].items():
        item=torch.load(ref["observation"],weights_only=True)
        count={"paviaU":103,"xuzhou":436,"whu-honghu":270}[target]
        bands=read_band_metadata(target,count)
        item.update(bands=count,net_input=torch.rand(20,20,count+2,generator=torch.Generator().manual_seed(64)),
                    wavelengths=torch.tensor(bands.wavelengths,dtype=torch.float32))
        torch.save(item,ref["observation"])
        record=read(ref["record"])
        record.update(observation_sha256=file_hash(ref["observation"]),prepared_fields=_item_fingerprint(item))
        write_json(ref["record"],record)
        ref.update(observation_sha256=file_hash(ref["observation"]),record_sha256=file_hash(ref["record"]))
    write_json(inputs,spec)
    folder=root/"contexts";manifest=prepare_contexts(inputs,file_hash(inputs),folder,code_snapshot())
    accepted={"status":"passed","issue":188,"code":code_snapshot(),
        "producer_receipt_sha256":file_hash(folder/"receipt.json"),"manifest_sha256":file_hash(folder/"manifest.json"),
        "optimizer_updates":0,"query_labels_accessed":False,"raw_gt_accessed":False,"training_executable":False,
        "phase_checks":54,"label_tensor_checks":108,"graph_observation_checks":27,
        "contexts":[{"target":r["target"],"support_seed":r["support_seed"],"k":r["support_per_class"]} for r in manifest["contexts"]]}
    ap=root/"inputs/accepted-context-audit.json";write_json(ap,accepted)
    docs=root/"evidence";docs.mkdir()
    x,h=docs/"xuzhou.txt",docs/"honghu.json"
    x.write_text("HYSPEX retained436 bands range415 to2508 nm",encoding="utf-8")
    write_json(h,{"has_true_center_wavelengths":False,"bands":270,"spectral_range_nm":[400.,1000.],
                  "sensor":"Headwall Nano-Hyperspec"})
    facts=read(CATALOG)
    facts["local_documents"]["xuzhou_package_readme"]["sha256"]=file_hash(x)
    facts["local_documents"]["honghu_local_metadata"]["sha256"]=file_hash(h)
    catalog=docs/"catalog.json";write_json(catalog,facts)
    data={"policy":INPUT_POLICY,"context_manifest":{"path":str(folder/"manifest.json"),"sha256":file_hash(folder/"manifest.json")},
        "context_audit":{"path":str(ap),"sha256":file_hash(ap)},"evidence_files":{
            "xuzhou_package_readme":{"path":str(x),"sha256":file_hash(x)},
            "honghu_local_metadata":{"path":str(h),"sha256":file_hash(h)}}}
    path=docs/"inputs.json";write_json(path,data)
    return path,catalog,spec


def run(root,path,catalog,out="audit"):
    return audit(path,file_hash(path),root/out,code_snapshot(),catalog_path=catalog,catalog_sha256=file_hash(catalog))


def repin_context(path,change):
    spec=read(path);mp=Path(spec["context_manifest"]["path"]);value=read(mp);change(value);write_json(mp,value)
    ph=file_hash(mp);producer=read(mp.parent/"receipt.json");producer["manifest_sha256"]=ph
    write_json(mp.parent/"receipt.json",producer)
    ap=Path(spec["context_audit"]["path"]);accepted=read(ap)
    accepted.update(manifest_sha256=ph,producer_receipt_sha256=file_hash(mp.parent/"receipt.json"));write_json(ap,accepted)
    spec["context_manifest"]["sha256"]=ph;spec["context_audit"]["sha256"]=file_hash(ap);write_json(path,spec)


class ObservationProvenanceTests(unittest.TestCase):
    def test_all27_three_observations_and_only_three_pt_deserializations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,catalog,spec=fixture(root)
            expected={ref["observation"] for ref in spec["observations"].values()};seen=[];original=torch.load
            def only_observations(file,*args,**kwargs):
                self.assertIn(str(file),expected);seen.append(str(file));return original(file,*args,**kwargs)
            with patch.object(module.torch,"load",side_effect=only_observations), \
                 patch("loadData.data_read.readData",side_effect=AssertionError("No raw data loader")):
                result=run(root,path,catalog)
            self.assertEqual(set(seen),expected);self.assertEqual(len(seen),3)
            self.assertEqual((result["observations"],result["contexts"],result["label_tensors_deserialized"]),(3,27,0))
            self.assertTrue(require_claim(result,"few_shot_scene_transductive"))
            self.assertFalse(result["training_executable"])
            for claim in module.CLAIMS:
                if claim!="few_shot_scene_transductive":
                    with self.assertRaises(ValueError):require_claim(result,claim)
            self.assertTrue(all(r["wavelength_quality"]=="approximate" and not r["fwhm_present"] for r in result["rows"]))
            self.assertEqual(sorted(set(result["context_annotation_labels"].values())),[54,90,132,135,220,330])
            for p,h in result["protected_files"].items():self.assertEqual(file_hash(p),h)

    def test_numeric_vectors_match_independent_reference_and_reject_self_consistent_bad_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,catalog,spec=fixture(root);facts=read(catalog)
            for target,ref in spec["observations"].items():
                item=torch.load(ref["observation"],weights_only=True);record=read(ref["record"])
                expected=PAVIAU_ROSIS03_CENTER_WAVELENGTHS_NM if target=="paviaU" else np.linspace(
                    *( (415.,2508.,436) if target=="xuzhou" else (400.,1000.,270) ),dtype=np.float32)
                np.testing.assert_array_equal(item["wavelengths"].numpy(),expected)
                inspect_observation(target,item,record,facts["scenes"][target])
                for kind in ("vector","dtype","fwhm","labels"):
                    bad=copy.deepcopy(item)
                    if kind=="vector":bad["wavelengths"]=bad["wavelengths"].roll(1)
                    if kind=="dtype":bad["wavelengths"]=bad["wavelengths"].double()
                    if kind=="fwhm":bad["fwhm"]=torch.ones(bad["bands"])
                    if kind=="labels":bad["query_labels"]=torch.zeros(1)
                    current={**record,"prepared_fields":_item_fingerprint(bad)}
                    with self.assertRaises(ValueError):inspect_observation(target,bad,current,facts["scenes"][target])

    def test_no_forged_quality_or_claim_upgrade(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,catalog,_=fixture(root);result=run(root,path,catalog)
            bad=copy.deepcopy(result);bad["claims"]["physical_FWHM"]["allowed"]=True
            with self.assertRaises(ValueError):require_claim(bad,"physical_FWHM")
            facts=read(catalog);facts["scenes"]["paviaU"]["wavelength_quality"]="verified";write_json(catalog,facts)
            with self.assertRaisesRegex(ValueError,"Unproved physical"):
                run(root,path,catalog,"bad")
            self.assertFalse((root/"bad").exists())

    def test_partial_or_wrong_graph_even_with_consistent_hashes_rejected(self):
        for mutation in (lambda m:m["contexts"].pop(),
                         lambda m:m["contexts"][0].update(graph_identity="f"*64),
                         lambda m:m["contexts"][0].update(file="../escape.pt"),
                         lambda m:m.update(spatial_isolation=True)):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path,catalog,_=fixture(root);repin_context(path,mutation)
                with self.assertRaises(ValueError):run(root,path,catalog)
                self.assertFalse((root/"audit").exists())

    def test_changed_evidence_unsafe_output_existing_outputs_and_end_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,catalog,_=fixture(root)
            for out in (root/"contexts/run",root/"evidence/run",root/"inputs/run",root):
                with self.assertRaises(ValueError):run(root,path,catalog,out)
            result=run(root,path,catalog);before=file_hash(root/"audit/receipt.json")
            with self.assertRaises(ValueError):run(root,path,catalog)
            self.assertEqual(before,file_hash(root/"audit/receipt.json"))
            source=Path(read(path)["evidence_files"]["xuzhou_package_readme"]["path"])
            real=module.inspect_observation
            def mutate(*args):
                value=real(*args);source.write_bytes(source.read_bytes()+b" ");return value
            with patch.object(module,"inspect_observation",side_effect=mutate):
                with self.assertRaises(ValueError):run(root,path,catalog,"changed")
            self.assertFalse((root/"changed").exists())

    def test_cli_complete_audit_and_wrong_sha_before_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,catalog,_=fixture(root)
            command=[sys.executable,"scripts/audit_g4_observation_provenance.py","--expected-commit",code_snapshot()["commit"],
                "--inputs",str(path),"--inputs-sha256",file_hash(path),"--catalog",str(catalog),
                "--catalog-sha256",file_hash(catalog),"--out",str(root/"cli")]
            result=json.loads(subprocess.check_output(command,text=True));self.assertEqual(result["contexts"],27)
            wrong=command.copy();wrong[wrong.index("--expected-commit")+1]="0"*40
            wrong[wrong.index("--inputs")+1]="missing-input"
            bad=subprocess.run(wrong,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertIn("Expected provenance audit commit",bad.stderr)


if __name__=="__main__":unittest.main()
