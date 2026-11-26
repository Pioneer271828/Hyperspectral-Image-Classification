"""Independent saved-probability, call-budget, route and provenance audit; no NN."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from statistics import median

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def independent_plans(roster, light, contract):
    """Reconstruct all scores and routes without importing producer policy/torch."""
    ids=[r['key'] for r in roster]
    margins={}
    for i,key in enumerate(ids):
        probability=light[f'p{i}']
        top=np.sort(probability,axis=1)[:,-2:]
        margins[key]=float(np.min(top[:,1]-top[:,0]))
    coverage={r['key']:r['coverage'] for r in roster}
    permuted=dict(coverage)
    for parent in sorted({r['parent'] for r in roster}):
        keys=[r['key'] for r in roster if r['parent']==parent]
        def token(k):
            return hashlib.sha256(json.dumps([contract['seed'],k],sort_keys=True,allow_nan=False).encode()).hexdigest()
        ordered=sorted(keys,key=lambda k:(token(k),k))
        values=[coverage[k] for k in ordered]
        permuted.update(zip(ordered,values[1:]+values[:1]))
    plans={'light':[],'full':sorted(ids)}
    quota=int(np.ceil(len(ids)*contract['full_fraction']))
    for arm,cov in (('content',None),('content_coverage',coverage),('permuted_coverage',permuted)):
        scores={k:1-margins[k]+(0 if cov is None else 1-cov[k]) for k in ids}
        plans[arm]=sorted(sorted(ids,key=lambda k:(-scores[k],k))[:quota])
    # WP11 random draws light identities within actual view/shape/coverage strata.
    rng=np.random.default_rng(contract['seed']);cheap=set(ids)-set(plans['content_coverage'])
    strata={}
    for row in roster:
        strata.setdefault((row['band'],row['coverage']),[]).append(row['key'])
    accepted=[]
    for keys in strata.values():
        accepted.extend(rng.permutation(keys)[:len(cheap.intersection(keys))].tolist())
    plans['matched_random']=sorted(set(ids)-set(accepted))
    return plans,margins,coverage,permuted


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
        bands=["stride1","stride2","stride4"]+(["range_le1000nm"] if contract.get("range_view") else [])
        for band in bands:points.append((f"{tile['id']}-{band}",band,ids))
    roster=setup['requests']
    if [x['key'] for x in roster]!=[x[0] for x in points]:raise ValueError('Request order differs')
    if summary['optimizer_updates']!=0 or contract['confirmation'] or contract['calibration']:
        raise ValueError('Development-only scope changed')
    audited=[];baselines={};graph_times={};planning_times={};compact={}
    for repeat in range(6):
        light=np.load(root/f"round-{repeat}-light.npz",allow_pickle=False)
        full=np.load(root/f"round-{repeat}-full.npz",allow_pickle=False)
        plans,margins,coverage,permuted=independent_plans(roster,light,contract)
        if setup['reference']!=plans['content_coverage']:raise ValueError('Setup route reference differs')
        for arm in ("light","full","content","content_coverage","permuted_coverage","matched_random"):
            row=json.loads((root/f"round-{repeat}-{arm}.json").read_text())
            data=np.load(root/f"round-{repeat}-{arm}.npz",allow_pickle=False)
            selected=set(row["full_plan"]);conf={s:np.zeros((9,9),dtype=np.int64) for s in bands}
            if row['full_plan']!=plans[arm]:raise ValueError('Independent route reconstruction differs')
            if arm in ('content','content_coverage','permuted_coverage'):
                if row['margin']!=margins:raise ValueError('Independent whole-tile margin differs')
                expected_cov=permuted if arm=='permuted_coverage' else coverage
                if row['descriptor_values']!=expected_cov:raise ValueError('Independent descriptor permutation differs')
            if row["counts"]["full_calls"]!=len(selected) or row["counts"]["graph_builds"]!=len(selected):
                raise ValueError("Graph was computed for skipped heavy calls")
            if row["counts"]["light_calls"]!=(0 if arm=="full" else len(points)):
                raise ValueError("Cheap work omitted from cost")
            if (row['counts']['graph_reuses'] or row['counts']['gate_calls']!=(len(points) if arm in
                    ('content','content_coverage','permuted_coverage') else 0)):
                raise ValueError('Graph reuse or confidence-work accounting differs')
            for i,(key,stride,ids) in enumerate(points):
                observed=data[f"p{i}"];expected=(full if key in selected else light)[f"p{i}"]
                if (observed.shape!=(784,9) or not np.isfinite(observed).all() or np.any(observed<0)
                        or not np.allclose(observed.sum(1),1,atol=1e-6,rtol=0)):
                    raise ValueError('Invalid dense endpoint probabilities')
                if not np.array_equal(observed,expected):raise ValueError("Routed output differs from frozen endpoints")
                if arm in ('light','full') and repeat:
                    with np.load(root/f'round-0-{arm}.npz',allow_pickle=False) as original:
                        if not np.array_equal(observed,original[f'p{i}']):raise ValueError('Frozen endpoint repeat changed')
                for global_id,local_id in ids:
                    conf[stride][lookup[global_id]-1, int(np.argmax(observed[local_id]))]+=1
            for stride,c in conf.items():
                old=row["metrics"][stride]
                if c.tolist()!=old["confusion"] or c.sum(1).tolist()!=old["support"]:
                    raise ValueError("Independent support/confusion differs")
                recall=np.diag(c)/c.sum(1)
                if not np.allclose(recall,old["recall"],rtol=0,atol=1e-14):raise ValueError("Recall differs")
                if old['oa']!=float(c.trace()/c.sum()) or old['aa']!=float(recall.mean()):
                    raise ValueError('OA or AA differs')
            if arm=="matched_random":
                reference=json.loads((root/f"round-{repeat}-content_coverage.json").read_text())["full_plan"]
                for stride in bands:
                    cohort={k for k,s,_ in points if s==stride}
                    if len(selected&cohort)!=len(set(reference)&cohort):raise ValueError("Random call-rate strata differ")
            baselines.setdefault(arm,[]).append(row["pipeline_ms"])
            graph_times.setdefault(arm,[]).append(row['graph_ms'])
            planning_times.setdefault(arm,[]).append(row['planning_ms'])
            if not all(np.isfinite(row[k]) and row[k]>=0 for k in ('pipeline_ms','graph_ms','planning_ms')):
                raise ValueError('Invalid actual elapsed cost')
            if repeat==0:compact[arm]={'metrics':row['metrics'],'counts':row['counts'],'full_plan':row['full_plan'],
                                      'per_view_full_calls':{band:sum(k.endswith('-'+band) for k in selected) for band in bands}}
            audited.append({"round":repeat,"arm":arm,"full_calls":len(selected),"graph_skips":len(points)-len(selected),
                            "recall_values":9*len(bands),"confusion_cells":81*len(bands),"endpoint_probabilities_exact":True,
                            "route_reconstructed":True,"descriptors_and_confidence_reconstructed":True})
    value={"status":"PASSED","producer":contract["code"],"scope":contract["qualification"],
           "stages":audited,"actual_optimizer_updates":0,"neural_forwards_in_this_audit":0,
           "independent_confusion_cells":sum(x["confusion_cells"] for x in audited),
           "independent_recalls":sum(x["recall_values"] for x in audited),
           "latency_p50_ms":{k:median(v) for k,v in baselines.items()},
           "latency_p95_ms":{k:float(np.percentile(v,95)) for k,v in baselines.items()},
           "graph_p50_ms":{k:median(v) for k,v in graph_times.items()},
           "planning_p50_ms":{k:median(v) for k,v in planning_times.items()},
           "compact_results":compact,"requests":roster,"contract":contract,
           "setup_reference_counts":setup['reference_counts'],'setup_reference_ms':setup['reference_ms'],
           "loading_ms":setup['loading_ms'],
           "independent_confirmation":False,"original_inputs_unchanged":True}
    with Path(a.out).open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({k:value[k] for k in ("status","producer","latency_p50_ms","independent_confusion_cells","independent_recalls")}))


if __name__=="__main__":main()
