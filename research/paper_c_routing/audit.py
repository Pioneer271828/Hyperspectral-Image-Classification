"""Independent saved-probability, call-budget, route and provenance audit; no NN."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from statistics import median

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True);p.add_argument("--development",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args();root=Path(a.run)
    summary=json.loads((root/"summary.json").read_text());setup=json.loads((root/"setup.json").read_text())
    contract=json.loads((root/"contract.json").read_text())
    for name,h in summary["files"].items():
        if sha(root/name)!=h:raise ValueError("Committed evidence changed")
    for name,h in setup["protected"].items():
        if sha(name)!=h:raise ValueError("Original weight or development file changed")
    dev=json.loads(Path(a.development).read_text());rec=dev["record"]
    lookup=dict(zip(rec["validation_support"]["indices"],rec["validation_support"]["labels"]))
    tiles=[t for t in rec["roles"] if t["role"]=="devval"]
    points=[];width=dev["plan"]["shape"][1]
    for tile in tiles:
        y0,y1,x0,x1=tile["box"]
        ids=[(y*width+x,(y-y0)*(x1-x0)+x-x0) for y in range(y0,y1) for x in range(x0,x1) if y*width+x in lookup]
        for stride in (1,2,4):points.append((f"{tile['id']}-stride{stride}",stride,ids))
    audited=[];baselines={}
    for repeat in range(6):
        light=np.load(root/f"round-{repeat}-light.npz",allow_pickle=False)
        full=np.load(root/f"round-{repeat}-full.npz",allow_pickle=False)
        for arm in ("light","full","content","content_coverage","permuted_coverage","matched_random"):
            row=json.loads((root/f"round-{repeat}-{arm}.json").read_text())
            data=np.load(root/f"round-{repeat}-{arm}.npz",allow_pickle=False)
            selected=set(row["full_plan"]);conf={s:np.zeros((9,9),dtype=np.int64) for s in (1,2,4)}
            if row["counts"]["full_calls"]!=len(selected) or row["counts"]["graph_builds"]!=len(selected):
                raise ValueError("Graph was computed for skipped heavy calls")
            if row["counts"]["light_calls"]!=(0 if arm=="full" else len(points)):
                raise ValueError("Cheap work omitted from cost")
            for i,(key,stride,ids) in enumerate(points):
                observed=data[f"p{i}"];expected=(full if key in selected else light)[f"p{i}"]
                if not np.array_equal(observed,expected):raise ValueError("Routed output differs from frozen endpoints")
                for global_id,local_id in ids:
                    conf[stride][lookup[global_id]-1, int(np.argmax(observed[local_id]))]+=1
            for stride,c in conf.items():
                old=row["metrics"][f"stride{stride}"]
                if c.tolist()!=old["confusion"] or c.sum(1).tolist()!=old["support"]:
                    raise ValueError("Independent support/confusion differs")
                recall=np.diag(c)/c.sum(1)
                if not np.allclose(recall,old["recall"],rtol=0,atol=1e-14):raise ValueError("Recall differs")
            if arm=="matched_random":
                reference=json.loads((root/f"round-{repeat}-content_coverage.json").read_text())["full_plan"]
                for stride in (1,2,4):
                    cohort={k for k,s,_ in points if s==stride}
                    if len(selected&cohort)!=len(set(reference)&cohort):raise ValueError("Random call-rate strata differ")
            baselines.setdefault(arm,[]).append(row["pipeline_ms"])
            audited.append({"round":repeat,"arm":arm,"full_calls":len(selected),"graph_skips":len(points)-len(selected),
                            "recall_values":27,"confusion_cells":243,"endpoint_probabilities_exact":True})
    value={"status":"PASSED","producer":contract["code"],"scope":contract["qualification"],
           "stages":audited,"actual_optimizer_updates":0,"neural_forwards_in_this_audit":0,
           "independent_confusion_cells":sum(x["confusion_cells"] for x in audited),
           "independent_recalls":sum(x["recall_values"] for x in audited),
           "latency_p50_ms":{k:median(v) for k,v in baselines.items()},
           "independent_confirmation":False,"original_inputs_unchanged":True}
    with Path(a.out).open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({k:value[k] for k in ("status","producer","latency_p50_ms","independent_confusion_cells","independent_recalls")}))


if __name__=="__main__":main()
