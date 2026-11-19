"""Run and freeze the registered band-by-graph matrix through native training."""

from collections import defaultdict
from itertools import product
from pathlib import Path
import os
import resource
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import case_lock
from transfer import development_bank as bank, native_scale_training as native
from transfer.band_graph_control import ARMS, FIXED_ARMS
from transfer.band_graph_fixture import BandGraphFixture, SCHEMA, SEEDS, LAYOUTS, TASKS, VIEWS, STEPS, generated
from transfer.native_scale_observation import forward, head_hashes
from transfer.spatial_coverage_training import load

SCENES = tuple(f"{t}-{l}" for t,l in product(TASKS,LAYOUTS))
PHASES = ("full","prefix","resume","completed")


def registration(code):
    return dict(schema=SCHEMA, producer=code["commit"],source_hash=code["source_hash"],scenes=list(SCENES),
                seeds=list(SEEDS), views=list(VIEWS), arms=list(ARMS), updates=STEPS, prefix=4, events=[0,6,12],
                real_updates=0,query_accessed=False,raw_gt_loaded=False,
                criteria=dict(pp=.10, required_pairs=4, budget_locked=True))


def scope(folder,code):
    link=pin(folder/"band-graph-registration.json")
    check(_read(link)==registration(code),"Band graph campaign registration differs")
    return link


def admitted(folder,task,layout,seed):
    release,values,ground=generated(task,layout,seed)
    path=folder/"bank"
    if (path/"bank.json").exists():
        link=dict(root=str(path),sha256=pin(path/"bank.json")["sha256"],registration=pin(path/"registration.json")["sha256"])
    else:
        answer=bank.publish(path,folder,release,lambda row:values[row["id"]],resume=path.exists())
        link={k:answer[k] for k in ("root","sha256","registration")}
    return bank.Bank(link,folder,release),ground


@torch.no_grad()
def endpoint(provider, model):
    model.eval()
    predictions, traces, labels = defaultdict(list), [], []
    started=perf_counter()
    for tile in (4,5):
        actual,_,targets=provider.materialize("devval",{"band":provider.band,"tile":tile},provider.producer)
        timed={}
        out,graph=forward(model,actual,provider.producer,"devval",timed=timed,
                          graph_override=provider.graph_override(actual))
        labels.extend(int(label) for _,label in targets.points)
        report=provider.scores(out,targets,detailed=True)
        for head,name in zip(out[:3],("fused","graph","cnn"),strict=True):
            predictions[name].extend(head[targets.cells.tolist()].tolist())
        traces.append(dict(tile=tile,heads=head_hashes(out,actual),probability_hashes={name:value["point_prediction_sha256"] for name,value in report.items()},
                           nnz=graph["a0_nnz"],q_nnz=graph["q_nnz"],a=graph["a0"],q=graph["q"],
                           inference_seconds=timed["forward_seconds"],graph_seconds=sum(timed["graph_seconds"]),
                           band_ids=actual.receipt["source"]["band_ids"]))
    return dict(scores=dict(predictions),traces=traces,labels=labels,inference_seconds=perf_counter()-started)


def train_case(path,provider,reg_sha):
    started=perf_counter()
    code=provider.producer
    spec=provider.contract(9,provider.seed,provider.arm,1,code)
    phases, artifacts = {}, {}
    parents={"resume":"prefix","completed":"resume"}
    for tag in PHASES:
        file=path/f"{tag}.pt"
        prior=load(path/f"{parents[tag]}.pt") if tag in parents else None
        phases[tag]=native.execute(spec,tag,file,code,prior,released=provider)
        artifacts[tag]=pin(file)["sha256"]
    full,sliced,done=(load(path/f"{name}.pt") for name in ("full","resume","completed"))
    check(native._exact(full,sliced) and native._exact(full,done),"Continuous and interrupted graph fits differ")
    selected=min(full["validation"],key=lambda x:x["fused_nll"])
    check(full["selected"]["step"]==selected["step"],"Native first-min selection changed")
    model=provider.baseline(provider.seed)
    model.load_state_dict(full["selected"]["state"],strict=True)
    scored=endpoint(provider,model)
    original=selected["rows"]
    check([x["heads"] for x in original]==[t["heads"] for t in scored["traces"]],
          "Selected native forward differs from the recorded optimizer validation")
    outputs=dict(status="passed",arm=provider.arm,scope=reg_sha, source=spec["initialization"]["source"],
                 stage=phases, checkpoint_sha=artifacts,budget=STEPS, selected=selected["step"],
                 train_roster=digest(spec["training"]),exposure=digest([item["tile"] for item in spec["training"]]),classes=9,equal=True,inputs=digest(spec["inputs"]),
                 endpoint=scored,seconds=perf_counter()-started,peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return _write(path/"receipt.json",outputs)


def compatibility(full,short):
    same=next(iter(full.views.values()))[0]
    different=next(iter(short.views.values()))[0]
    try:
        full[same].actual(different,"fixed_none")
    except ValueError:
        return True
    raise ValueError("Different retained band IDs admitted stale graph support")


def one_view(folder,admit,seed,layout,band,reg,code):
    path=folder/band
    path.mkdir(exist_ok=True)
    core=BandGraphFixture(admit,seed,layout,band,"fixed_none",code["commit"])
    matched=core.match_inventory()
    unavailable={}
    for arm in FIXED_ARMS[2:]:
        bad=[tag for tag,row in matched.items() if row["controls"][arm]["status"]!="optimal"]
        if bad:
            unavailable[arm]=bad
    receipts={}
    for arm in ARMS:
        if arm in unavailable:
            continue
        tag=arm.replace(":","-")
        snapshot=path/tag
        if snapshot.exists():
            link=pin(snapshot/"receipt.json")
            previous=_read(link)
            check(previous["scope"]==reg["sha256"] and previous["arm"]==arm,"Previous completed fit differs")
            receipts[arm]=link["sha256"]
        else:
            with TemporaryDirectory(dir=path,prefix=f".{tag}-") as temp:
                stage=Path(temp)/"case"
                stage.mkdir()
                link=train_case(stage,core.variant(arm),reg["sha256"])
                os.rename(stage,snapshot)
                receipts[arm]=link["sha256"]
    reference=load(path/"fixed_none/full.pt")
    same=core.baseline(seed)
    same.load_state_dict(reference["selected"]["state"],strict=True)
    frozen={arm:endpoint(core.variant(arm),same) for arm in FIXED_ARMS if arm not in unavailable}
    check(all(row["scores"]["cnn"]==frozen["fixed_none"]["scores"]["cnn"] for row in frozen.values()),
          "Frozen graph intervention changed the CNN head")
    compare=BandGraphFixture(admit,seed,layout,"continuous" if band=="full" else "full","fixed_none",code["commit"])
    negative=compatibility(core,compare)
    return _write(path/"receipt.json",dict(band=band,scope=reg["sha256"],matches=matched,unavailable=unavailable,
                  fits=receipts,frozen=frozen,foreign_band_rejected=negative))["sha256"]


def run(folder,code,select):
    value=registration(code)
    marker=folder/"band-graph-registration.json"
    with case_lock(folder/".band-graph.lock"):
        if not marker.exists():
            _write(marker,value)
    reg=scope(folder,code)
    wanted=[(task,layout,seed) for task,layout,seed in product(TASKS,LAYOUTS,SEEDS)
            if select in ("all",task,f"{task}-{layout}",f"{task}-{layout}-{seed}")]
    check(bool(wanted),"No registered synthetic band graph cases")
    for task,layout,seed in wanted:
        name=f"{task}-{layout}-{seed}"
        path=folder/name
        path.mkdir(exist_ok=True)
        with case_lock(path/".lock"):
            if (path/"receipt.json").exists():
                old=_read(pin(path/"receipt.json"))
                check(old["scope"]==reg["sha256"] and old["task"]==task,"Foreign completed campaign case")
                print(f"reused {name}",flush=True)
                continue
            admit, truth=admitted(path,task,layout,seed)
            views={}
            for band in VIEWS:
                link=path/band/"receipt.json"
                views[band]=(pin(link)["sha256"] if link.exists() else one_view(path,admit,seed,layout,band,reg,code))
            require_frozen_code(code)
            _write(path/"receipt.json",dict(scope=reg["sha256"],task=task,layout=layout,seed=seed,
                    views=views,truth=truth,real_updates=0,query_accessed=False,raw_gt_loaded=False))
            print(f"completed {name}",flush=True)
