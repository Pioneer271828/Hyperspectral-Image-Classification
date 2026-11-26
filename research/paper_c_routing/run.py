"""Zero-optimizer WP11 target-local development control with resumable outputs.

It cannot claim any ordered cross-device direction: its authenticated source
model is target scratch. Separate direction admission precedes model loading.
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

from policy import ARMS, admit, digest, execute, metrics

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
    a=p.parse_args()
    started=perf_counter()
    if sys.platform!="linux" or os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Authorized Linux CPU only")
    root=Path(__file__).resolve().parents[2]
    clean(root,a.expected_commit);clean(a.dependency,DEPENDENCY)
    torch.set_num_threads(1)
    runs=Path(a.runs).resolve();out=Path(a.out).resolve()
    if not out.is_relative_to(runs) or out==runs:raise ValueError("New external output required")
    table=root/"research/paper1_c/ordered-pair-feasibility.csv"
    qualification=admit(table,"xuzhou","xuzhou",scope="target_local_development_control",role="devval",
                        ancestry={"source":"xuzhou","target":"xuzhou","kind":"target_scratch"})
    contract={"schema":"paper_c_routing_dev_zero_v1","code":a.expected_commit,"dependency":DEPENDENCY,
              "qualification":qualification,"table_sha256":sha(table),"release_sha256":RELEASE_SHA,
              "views":[1,2,4],"full_fraction":.5,"seed":64,"rounds":6,"warmups":1,
              "scope":"development only; synthetic retained-column views separate from native full",
              "source_pretraining":"none: target-scratch ancestry","optimizer_updates":0,
              "calibration":False,"confirmation":False}
    pin=digest(contract)
    if out.exists():
        if not a.resume or json.loads((out/"contract.json").read_text())!=contract:raise ValueError("Explicit exact resume required")
        if (out/"summary.json").exists():
            s=json.loads((out/"summary.json").read_text())
            for name,h in s["files"].items():
                if sha(out/name)!=h:raise ValueError("Completed evidence changed")
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
    released,origin,scope,scene,record,full,light,history,pins=release_context(release)
    light.validate(full)
    if scene.dataset!="xuzhou" or "target_scratch_full" not in released["source_weights"]["path"]:
        raise ValueError("The declared target-scratch ancestry no longer matches")
    protected={p["path"]:p["sha256"] for p in pins}
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
        for stride in (1,2,4):
            pos=torch.arange(0,n,stride);x=torch.cat((r["x"][:,:,pos],r["x"][:,:,n:]),-1)
            key=f"{r['id']}-stride{stride}"
            requests.append(Request(key,"xuzhou",tensor_hash(x),str(r["id"]),f"stride{stride}",x,
                                    r["wavelengths"][pos],None,None,r["scale"],len(pos)/n,True))
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
