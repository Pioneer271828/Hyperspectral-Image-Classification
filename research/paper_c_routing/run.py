"""Zero-update frozen WP11 scratch or authenticated G3 transfer development.

Separate direction admission precedes model loading. Synthetic spectral views
and released V5 readout never qualify as independent device confirmation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from policy import ARMS, admit, digest, execute, metrics, verify_completed

DEPENDENCY = "395caef57fba8ab6a09223538f748179dab552f2"
RELEASE_SHA = "9a4493d2c25c6cff64d6b8ee8958e467546985865b72df41540327262d24aacb"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def clean(root, expected):
    def git(*args):return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    if git("rev-parse","HEAD") != expected or git("status","--porcelain"):
        raise ValueError("Frozen code identity differs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--dependency",required=True)
    p.add_argument("--runs",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after",type=int)
    p.add_argument("--profile",choices=("wp11_scratch","g3_pavia_xuzhou"),default="wp11_scratch")
    p.add_argument("--g3-root")
    p.add_argument("--g3-seed",type=int,choices=(2036,2046,2056),default=2036)
    p.add_argument('--g3-method',choices=('source_pretrained_full_finetune','source_pretrained_head_only','target_scratch_full'),default='source_pretrained_full_finetune')
    p.add_argument("--range-view",action="store_true")
    p.add_argument("--descriptor",choices=("retained_fraction","approximate_span"),default="retained_fraction")
    a=p.parse_args()
    started=perf_counter()
    if sys.platform!="linux" or os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Authorized Linux CPU only")
    if a.profile=='wp11_scratch' and (a.g3_root is not None or a.g3_seed!=2036 or a.g3_method!='source_pretrained_full_finetune'):
        raise ValueError('G3 options cannot be silently ignored by the WP11 scratch profile')
    root=Path(__file__).resolve().parents[2]
    clean(root,a.expected_commit);clean(a.dependency,DEPENDENCY)
    torch.set_num_threads(1)
    runs=Path(a.runs).resolve();out=Path(a.out).resolve()
    if not out.is_relative_to(runs) or out==runs:raise ValueError("New external output required")
    table=root/"research/paper1_c/ordered-pair-feasibility.csv"
    model_pins={}
    if a.profile=="wp11_scratch":
        qualification=admit(table,"xuzhou","xuzhou",scope="target_local_development_control",role="devval",
                            ancestry={"source":"xuzhou","target":"xuzhou","kind":"target_scratch"})
    else:
        g3=Path(a.g3_root).resolve()
        profiles=json.loads((Path(__file__).parent/'g3-profiles.json').read_text())
        expected={**profiles['profiles'][str(a.g3_seed)],'support-10.json':profiles['support_sha256']}
        model_name='model.pt'
        if a.g3_method!='source_pretrained_full_finetune':
            expected.pop('model.pt');expected.pop('model.pt.json')
            expected.update(profiles['variants'][str(a.g3_seed)][a.g3_method])
            model_name=a.g3_method+'/model.pt'
        if not g3.is_relative_to(runs.parent/"references"):raise ValueError("Imported frozen models must stay in project references")
        model_pins={str(g3/n):h for n,h in expected.items()}
        if any(sha(path)!=h for path,h in model_pins.items()):raise ValueError("Imported original model/metadata identity differs")
        task=json.loads((g3/"frozen-task.json").read_text());cfg=json.loads((g3/(model_name+'.json')).read_text())
        original_support=json.loads((g3/'support-10.json').read_text())
        frozen=json.loads((g3/"target-checkpoints-frozen.json").read_text())[a.g3_method]
        if (task["config"]["source_dataset"]!="paviaU" or task["config"]["target_dataset"]!="xuzhou"
                or task["task_hash"]!=cfg["training_config"]["task_hash"]
                or original_support['support_hash']!=task['support_hash']
                or cfg['training_config']['optimizer_steps']!=1000 or task['config']['model_seed']!=a.g3_seed
                or cfg["training_config"]["method"]!=a.g3_method
                or frozen["checkpoint_sha256"]!=expected[model_name] or frozen["sidecar_sha256"]!=expected[model_name+'.json']):
            raise ValueError("Single-source task/selected model lineage differs")
        if a.g3_method=='target_scratch_full':
            qualification=admit(table,'xuzhou','xuzhou',scope='target_local_development_control',role='devval',
                                ancestry={'source':'xuzhou','target':'xuzhou','kind':'target_scratch'})
        else:
            qualification=admit(table,"paviaU","xuzhou",scope="cross_device_development",role="devval",
                                ancestry={"source":"paviaU","target":"xuzhou","kind":"verified_source_pretrain_target_adaptation"})
        qualification.update(original_target_training_K=10,original_target_validation_V=0,
                             original_target_updates=1000,original_source_updates=0 if a.g3_method=='target_scratch_full' else 10000,
                             scoring_role="existing bank V5; development recontextualization, not independent confirmation",
                             light="exact frozen trained CNN branch copy; no new light-head fitting")
        qualification['original_method']=a.g3_method
    contract={"schema":"paper_c_routing_dev_zero_v1","code":a.expected_commit,"dependency":DEPENDENCY,
              "qualification":qualification,"table_sha256":sha(table),"release_sha256":RELEASE_SHA,
              "views":[1,2,4],"full_fraction":.5,"seed":64,"rounds":6,"warmups":1,
              "scope":"development only; synthetic retained-column views separate from native full",
              "source_pretraining":"none: target-scratch ancestry","optimizer_updates":0,
              "calibration":False,"confirmation":False}
    contract.update(profile=a.profile,model_pins=model_pins,frozen_model_seed=a.g3_seed if a.profile!='wp11_scratch' else None,
                    original_method=a.g3_method if a.profile!='wp11_scratch' else 'WP11 target scratch5000 and light fit')
    contract.update(range_view=a.range_view,descriptor=a.descriptor,
                    descriptor_quality="approximate delivered centers; no certified SRF or device calibration")
    if a.profile!="wp11_scratch":contract["source_pretraining"]="original Pavia source10000 + target1000, no new updates"
    if a.profile!='wp11_scratch' and a.g3_method=='target_scratch_full':
        contract['source_pretraining']='none; original matched target scratch1000, no new updates'
    pin=digest(contract)
    if out.exists():
        if not a.resume or json.loads((out/"contract.json").read_text())!=contract:raise ValueError("Explicit exact resume required")
        if (out/"summary.json").exists():
            verify_completed(out)
            print(json.dumps({"completed_reentry":True,"neural_forwards":0,"optimizer_updates":0,"contract":pin}));return
    else:
        out.mkdir();new(out/"contract.json",contract)
    sys.path.insert(0,str(Path(a.dependency).resolve()))
    from inference.wp11_execution import release_context
    from inference.native_deployment import Request,Runtime
    from evaluation.wp11_static_cost import prepare_requests
    from transfer.g3_training import state_hash
    from research_runtime import tensor_hash
    load_start=perf_counter()
    release={"path":str(runs/"wp11-matched-xuzhou-frozen01/release.json"),"sha256":RELEASE_SHA}
    if a.profile=="wp11_scratch":
        released,origin,scope,scene,record,full,light,history,pins=release_context(release)
        light.validate(full)
    else:
        from transfer.registered_development import load_development,_read
        released=json.loads(_read(release))
        scene,record,files=load_development(released["development"])
        pins=[release,*files]
    if scene.dataset!="xuzhou" or "target_scratch_full" not in released["source_weights"]["path"]:
        raise ValueError("The declared target-scratch ancestry no longer matches")
    protected={p["path"]:p["sha256"] for p in pins}
    protected.update(model_pins)
    if a.profile!="wp11_scratch":
        from transfer.registered_development import _load_source
        from inference.light_cnn import LightCNN
        from types import SimpleNamespace
        full=_load_source({"path":str(g3/model_name),"sha256":expected[model_name]},
                          {"path":str(g3/(model_name+'.json')),"sha256":expected[model_name+'.json']})
        light=SimpleNamespace(model=LightCNN(full))
    for path,h in protected.items():
        if sha(path)!=h:raise ValueError("Frozen model or development input differs")
    requests=[];indices=[];labels=[]
    full_requests=prepare_requests(scene,record)
    support=dict(zip(record["validation_support"]["indices"],record["validation_support"]["labels"]))
    width=scene.plan["shape"][1]
    for r in full_requests:
        tile=scene.plan["tiles"][int(r["id"])];y0,y1,x0,x1=tile["box"]
        local=[(y-y0)*(x1-x0)+x-x0 for y in range(y0,y1) for x in range(x0,x1) if y*width+x in support]
        lab=[support[y*width+x] for y in range(y0,y1) for x in range(x0,x1) if y*width+x in support]
        n=len(r["wavelengths"])
        views=[(f"stride{s}",torch.arange(0,n,s)) for s in (1,2,4)]
        if a.range_view:views.append(("range_le1000nm",torch.where(r["wavelengths"]<=1000)[0]))
        for view_name,pos in views:
            if len(pos)<4:raise ValueError("Range has insufficient retained bands")
            x=torch.cat((r["x"][:,:,pos],r["x"][:,:,n:]),-1)
            key=f"{r['id']}-{view_name}"
            span=float((r["wavelengths"][pos[-1]]-r["wavelengths"][pos[0]])/(r["wavelengths"][-1]-r["wavelengths"][0]))
            coverage=len(pos)/n if a.descriptor=="retained_fraction" else span
            requests.append(Request(key,"xuzhou",tensor_hash(x),str(r["id"]),view_name,x,
                                    r["wavelengths"][pos],None,None,r["scale"],coverage,True))
            indices.append(local);labels.append(lab)
    before={"full":state_hash(full),"light":state_hash(light.model)}
    input_hash=digest([r.identity for r in requests])
    runtime=Runtime(device="cpu",full=full,light=light.model,identity=before,graph_policy="rebuild")
    loading_ms=(perf_counter()-load_start)*1000
    if (out/"setup.json").exists():
        setup=json.loads((out/"setup.json").read_text())
        if setup["input_hash"]!=input_hash or setup["models"]!=before:raise ValueError("Resumed input/model drift")
        reference=setup["reference"]
    else:
        warm=execute(runtime,requests,"content_coverage")
        reference=warm["full_plan"]
        setup={"input_hash":input_hash,"models":before,"reference":reference,"loading_ms":loading_ms,
               "reference_counts":warm["counts"],"reference_ms":warm["pipeline_ms"],"protected":protected,
               "released_support":len(support),"new_labels":0,"class_order":list(range(1,10))}
        setup["requests"]=[{"key":r.key,"parent":r.parent,"band":r.band,"coverage":r.coverage,
                            "bands":len(r.waves),"wavelength_min":float(r.waves.min()),
                            "wavelength_max":float(r.waves.max())} for r in requests]
        new(out/"setup.json",setup)
    completed=0;results=[]
    for repeat in range(6):
        order=ARMS[repeat%6:]+ARMS[:repeat%6]
        for arm in order:
            name=f"round-{repeat}-{arm}"
            path=out/(name+".json");prob_path=out/(name+".npz")
            if path.exists():
                row=json.loads(path.read_text())
                if row["contract"]!=pin or row["probabilities_sha256"]!=sha(prob_path):raise ValueError("Completed stage drift")
            else:
                if prob_path.exists():raise ValueError("Uncommitted probability artifact; retain and use new attempt")
                actual=execute(runtime,requests,arm,reference=reference)
                arrays={f"p{i}":p[0].cpu().numpy() for i,p in enumerate(actual["predictions"])}
                with prob_path.open("xb") as f:np.savez(f,**arrays)
                grouped={}
                for i,req in enumerate(requests):
                    g=grouped.setdefault(req.band,{"y":[],"p":[]})
                    g["y"].extend(labels[i]);g["p"].extend(arrays[f"p{i}"][indices[i]])
                scored={band:metrics(g["y"],np.asarray(g["p"]),9) for band,g in grouped.items()}
                row={k:v for k,v in actual.items() if k!="predictions"}
                row.update(contract=pin,round=repeat,input_hash=input_hash,metrics=scored,
                           probabilities_sha256=sha(prob_path),classification_scope="existing devval only")
                new(path,row);completed+=1
            results.append(row)
            if a.stop_after is not None and completed>=a.stop_after:
                print(json.dumps({"status":"PARTIAL","new_stages":completed,"optimizer_updates":0}));return
    if before!={"full":state_hash(full),"light":state_hash(light.model)}:raise ValueError("Frozen weights changed")
    if digest([r.identity for r in requests])!=input_hash:raise ValueError("Frozen input tensors changed")
    if protected!={path:sha(path) for path in protected}:raise ValueError("Frozen inputs changed")
    for arm in ARMS:
        rows=[r for r in results if r["arm"]==arm]
        if any(r["full_plan"]!=rows[0]["full_plan"] or r["metrics"]!=rows[0]["metrics"] for r in rows):
            raise ValueError("Frozen repeat mismatch")
    files={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    new(out/"summary.json",{"status":"PASSED_DEVELOPMENT_CONTROL","contract":pin,"files":files,"results":results,
                            "protected_inputs_unchanged":True,"models_unchanged":True,"optimizer_updates":0,
                            "query_reads":0,"raw_gt_reads":0,"confirmation":False,
                            "elapsed_this_invocation_seconds":perf_counter()-started})
    print(json.dumps({"status":"PASSED_DEVELOPMENT_CONTROL","new_stages":completed,"optimizer_updates":0}))


if __name__=="__main__":main()
