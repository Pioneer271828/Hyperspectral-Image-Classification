"""Recompute development-only metrics from the 21 frozen historical joint runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch


CAMPAIGNS = {
    "m4s2-issue135-full01":(12,"b36f1c976120eb8abd7d74e7f44295227550e9ea"),
    "m2j2-issue139-full01":(9,"d945daa15ad8752cceaf3898ebed6fca5d8415d0"),
}
PREPARATION = "base97a-honghu-issue97-20261103-prepared01"
SCENES = ("paviaU","xuzhou","whu-honghu")
PHASES = ("train","val")


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def labels(runs, manifest_sha256):
    directory = runs / PREPARATION
    manifest = directory / "manifest.json"
    if digest(manifest)!=manifest_sha256:
        raise ValueError("Frozen historical preparation manifest differs")
    result = {}
    entries = read(manifest)["datasets"]
    if set(entries)!=set(SCENES):
        raise ValueError("Historical preparation scene roster differs")
    for scene in SCENES:
        entry = entries[scene]
        path = directory / (scene+"-prepared.pt")
        if entry["prepared_file"]!=path.name or digest(path)!=entry["prepared_sha256"]:
            raise ValueError(f"Historical train/val payload differs: {scene}")
        payload = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        if any(key.startswith("test_") for key in payload):
            raise ValueError("Evaluation fields outside train/validation")
        result[scene] = {phase:payload[f"{phase}_samples_gt"].long() for phase in PHASES}
    return result


def metrics(probabilities, truth):
    truth = truth[truth>0]-1
    correct = {}
    by_head = {}
    for head,p in probabilities.items():
        if p.ndim!=2 or p.shape[0]!=len(truth) or not torch.isfinite(p).all():
            raise ValueError("Frozen probability and label alignment differs")
        predicted = p.argmax(1)
        valid = predicted == truth
        cm = torch.bincount(truth*p.shape[1]+predicted,minlength=p.shape[1]**2).reshape(p.shape[1],p.shape[1])
        support = cm.sum(1)
        counts = cm.diagonal()
        total = int(cm.sum())
        expected = float(support.double()@cm.sum(0).double())/(total**2)
        oa = float(counts.sum())/total
        by_head[head] = dict(oa=oa,aa=float((counts/support).mean()),
                            kappa=(oa-expected)/(1-expected),
                            macro_f1=float((2*counts/(support+cm.sum(0))).mean()),
                            recall=(counts/support).tolist(),confusion=cm.tolist())
        correct[head] = valid
    overlap = {}
    if set(correct)=={"fused","graph","cnn"}:
        for label in range(-1,int(truth.max())+1):
            eligible = torch.ones_like(truth,dtype=torch.bool) if label<0 else truth==label
            found = {key:mask[eligible] for key,mask in correct.items()}
            cnn,graph,fused = (found[key] for key in ("cnn","graph","fused"))
            probs = probabilities["fused"][eligible]
            first = probs.topk(2,dim=1).values
            overlap[str(label+1)] = dict(support=len(cnn),cnn_only=int((cnn&~graph).sum()),
                graph_only=int((graph&~cnn).sum()),both=int((graph&cnn).sum()),neither=int((~graph&~cnn).sum()),
                rescued_by_fusion=int((fused&~graph&~cnn).sum()),fusion_misses_both_correct=int((~fused&graph&cnn).sum()),
                fused_mean_margin=float((first[:,0]-first[:,1]).mean()))
    return by_head,overlap


def parameter_delta(first,last):
    result = {}
    for name in ("anchor_offsets","query_encoder","key_encoder","value_proj","region_residual"):
        keys = [key for key in first if name in key]
        result[name] = dict(tensors=len(keys),changed=sum(int(torch.count_nonzero(first[key]!=last[key])) for key in keys),
                            max_abs=max((float((first[key]-last[key]).abs().max()) for key in keys),default=0.))
    return result


def campaign(path,count,producer,truth):
    rows = []
    for case in sorted((path/"cases").iterdir()):
        result = read(case/"result.json")
        history = read(case/"history.json")
        if (result["start_code"]["commit"]!=producer or result["test_scoring"] is not False
                or result["optimizer_steps"]!=1503 or [row["step"] for row in history]!=list(range(3,1504,30))):
            raise ValueError(f"Historical case identity differs: {case.name}")
        for filename in ("history.json","selected-probabilities.pt","state-000003.pt","state-001503.pt"):
            entry = result["artifacts"].get(filename)
            if not isinstance(entry,str) or digest(case/filename)!=entry:
                raise ValueError(f"Historical artifact differs: {case.name}/{filename}")
        selected = max(history,key=lambda x:x["macro_oa"])
        if selected["step"]!=result["selected_step"]:
            raise ValueError("First maximum selection differs")
        probabilities = torch.load(case/"selected-probabilities.pt",map_location="cpu",weights_only=True,mmap=True)
        observed = {}
        for scene in SCENES:
            if set(probabilities[scene])!=set(PHASES):
                raise ValueError("Unexpected evaluation role")
            observed[scene] = {}
            for phase in PHASES:
                actual,overlap = metrics(probabilities[scene][phase],truth[scene][phase])
                expected = result["selected_validation"]["metrics"][scene][phase]
                if set(actual)!=set(expected) or any(row["confusion"]!=expected[key]["confusion"] for key,row in actual.items()):
                    raise ValueError(f"Frozen confusion differs: {case.name}/{scene}/{phase}")
                observed[scene][phase] = dict(heads=actual,head_overlap=overlap)
        first,last = [torch.load(case/f"state-{step:06d}.pt",map_location="cpu",weights_only=True,mmap=True)["model"]
                      for step in (3,1503)]
        rows.append(dict(case=case.name,selected=selected["step"],observations=observed,
                         macro_oa=[row["macro_oa"] for row in history],
                         val_nll=[row["val_loss"] for row in history],
                         tail_50_delta=history[-1]["macro_oa"]-history[-6]["macro_oa"],
                         parameter_deltas_3_to_1503=parameter_delta(first,last)))
    if len(rows)!=count:
        raise ValueError(f"Historical case count differs: {path.name}")
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs",required=True,type=Path)
    cli.add_argument("--prepared-sha256",required=True)
    cli.add_argument("--out",required=True,type=Path)
    args = cli.parse_args()
    truth = labels(args.runs,args.prepared_sha256)
    result = {name:campaign(args.runs/name,count,producer,truth) for name,(count,producer) in CAMPAIGNS.items()}
    with args.out.open("x") as stream:
        json.dump(dict(campaigns=result,history_cases=21,events=1071,
                       optimizer_updates=0,query_reads=0,new_label_access=0),stream,indent=2,allow_nan=False)
        stream.write("\n")
    print("Verified 21 historical cases and 1,071 train/val history events")


if __name__=="__main__":
    main()
