"""Exercise registered calibration, confirmation and fault recovery on independent Banks."""

from copy import deepcopy
from itertools import product
from statistics import fmean, stdev
from time import perf_counter_ns
import resource

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.representation_localization import taps
from evaluation.wp12_cost_primitives import timing_summary
from inference.gate_calibration import GateProtocol, SPARSE_PROTOCOL, calibrate, confirm
from inference.light_cnn import LightCNN
from inference.native_deployment import RouteFailure, Runtime, clock_ms, matched_random, output_hash, prepare
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _read, _write, pin
from transfer.g3_training import state_hash
from transfer.native_independent_release import (AUTHORITY, MANIFEST, USES, materialize_reviewed,
                                                open_independent, synthetic_bank)

SEEDS=(64,512,2048)
SUPPORT=(512,2048)
BANDS=("full","central_gap","missing")
BUDGETS={"strict":dict(floor=.8,max_full=4),"broad":dict(floor=.4,max_full=2)}
THRESHOLDS=[.15,.065,.025,0.]


def need(condition,message):
    if not condition:
        raise ValueError(message)


def models(seed,scene):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        full=MultiDatasetVCASAHGCN({scene:9},graph_mode="learned",norm_type="group_norm",
                                  latent_channels=128,adapter_embed_dim=64,alph=1.).eval()
    full.backbone.adapter.cache_attention=False
    # The fixture intentionally ties fused and light classification; the full
    # route still executes the original learned Q/A and graph branch.
    full.heads[scene].load_state_dict(full.cnn_heads[scene].state_dict(),strict=True)
    full.requires_grad_(False)
    return full,LightCNN(full)


def approved_banks(root,out,code,support):
    scope=out/f"support-{support}"
    scope.mkdir()
    history,_=synthetic_bank(scope,"historical",0,support_seed=support)
    bank,entry=synthetic_bank(scope,"candidate",40,support_seed=support)
    val=sorted(r["id"] for r in bank.role_rows("devval"))
    roles=dict(support=sorted(r["id"] for r in bank.role_rows("devtrain")),calibration=val[::2],evaluation=val[1::2])
    manifest=_write(scope/"manifest.json",dict(schema=MANIFEST,candidate=entry,roles=roles,minimum_buffer=3,history=[]))
    authority=_write(scope/"authority.json",dict(schema=AUTHORITY,approved_manifest_sha256=manifest["sha256"],
                     historical_bank_sha256=[history.pin["sha256"]],uses=dict.fromkeys(USES,True),source_mode="synthetic"))
    bank,review,roles=open_independent(root,manifest,authority,test_history=[history])
    need(review["uses"]["independent_accuracy"]["eligible"],"Independent synthetic authority differs")
    return bank,review,roles,manifest["sha256"]


def label_metadata(bank, roles):
    blocks, per_role = [], {}
    for role,internal in (("calibration","devval"),("evaluation","devval")):
        registered="confirmation" if role=="evaluation" else role
        entries={}
        for tile in roles[role]:
            row=bank.role_row(internal,tile)
            y0,y1,x0,x1=row["box"]
            width=bank.release.shape[1]
            locations=[((i//width-y0)*(x1-x0)+i%width-x0,label) for i,label in row["samples"]]
            for band in BANDS:
                key=f"{tile}/{band}"
                blocks.append(dict(id=key,scene=bank.release.scene,condition=band,role=registered,
                                   context=row["box"],label_indices=[i for i,_ in locations]))
                truth=np.zeros((y1-y0)*(x1-x0),dtype=np.int64)
                for offset,label in locations:
                    truth[offset]=label
                entries[key]=truth
        per_role[registered]=entries
    # Independent historical placement already verified by the authorized Bank.
    # Two distinct support parents represent the train/selection identity in the gate protocol.
    for role,tile in zip(("training","selection"),roles["support"][:2],strict=True):
        row=bank.role_row("devtrain",tile)
        blocks.append(dict(id=f"{role}/{tile}",scene=bank.release.scene,condition="full",role=role,
                           context=row["box"],label_indices=[]))
    commitments={role:digest({name:labels[labels>0].tolist() for name,labels in entries.items()})
                 for role,entries in per_role.items()}
    return blocks,per_role,commitments


def requests(bank,review,roles,role,code,device,*,use="independent_accuracy"):
    output=[]
    for tile,band in product(roles[role],BANDS):
        actual,_,_=materialize_reviewed(bank,review,roles,role,tile,code["commit"],band,use=use)
        output.append(prepare(actual,code["commit"],key=f"{tile}/{band}",device=device,
                              original_band_count=bank.release.shape[-1]))
    return output


def endpoint_records(engine,reqs,labels):
    # Unknown wavelength tiles never exercise the light path.
    full=engine.run(reqs,"full")
    fast=engine.run(reqs,"random",random_fast=[r.key for r in reqs if r.known])
    records=[]
    for req,f,l in zip(reqs,full["predictions"],fast["predictions"],strict=True):
        records.append(dict(id=req.key,labels=labels[req.key],fast=l[0].detach().cpu().numpy(),
                            full=f[0].detach().cpu().numpy(),coverage_known=req.known,coverage_fraction=req.coverage))
    return full,fast,records


def quality(labels,reqs,raw):
    truth,guess=[],[]
    for req,output in zip(reqs,raw["predictions"],strict=True):
        truth.extend(labels[req.key].tolist())
        guess.extend((output[0].argmax(1)+1).cpu().tolist())
    return classification_metrics(truth,guess,range(1,10))


def evidence(reqs,labels,raw):
    values=quality(labels,reqs,raw)
    adopted={x["key"] for x in raw["routes"] if x["route"]=="light"}
    return dict(counts=raw["counts"],routes=raw["routes"],predictions=output_hash(raw),metrics=values,
                graph_ms=raw["graph_ms"],forward_with_routing_ms=raw["elapsed_ms"],gpu_event_ms=raw["gpu_event_ms"],
                early_exit_points=sum(int((labels[key]>0).sum()) for key in adopted))


def repeat_route(engine, requests, expected, mode, options, prepare_warm):
    result={}
    for warm in (False,True):
        scope="warm_end_to_end" if warm else "resident_with_route"
        samples=[]
        if engine.device.type=="cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        for index in range(6):
            clock=perf_counter_ns()
            inputs=prepare_warm() if warm else requests
            outcome=engine.run(inputs,mode,**options)
            observed=output_hash(outcome)
            duration=clock_ms(clock) if warm else outcome["elapsed_ms"]
            need(observed==expected and [r.observation for r in inputs]==[r.observation for r in requests],
                 "Timing replay changed data or frozen predictions")
            if index>=2:
                samples.append(duration)
        result[scope]=dict(**timing_summary(samples,repeats=4,blocks=len(requests)),warmups=2,
                           rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           cuda_peak_bytes=torch.cuda.max_memory_allocated(engine.device)
                           if engine.device.type=="cuda" else None)
    return result


def fault_and_cache(full,light,eval_inputs,device):
    engine=Runtime(device=device,full=deepcopy(full),light=deepcopy(light),graph_policy="reuse_frozen")
    before=engine.run(eval_inputs,"full")
    after=engine.run(eval_inputs,"full")
    need(before["counts"]["graph_builds"]==len(eval_inputs) and after["counts"]["graph_reuses"]==len(eval_inputs)
         and output_hash(before)==output_hash(after),"Reused graph altered fixed predictions")
    key=next(r.key for r in eval_inputs if r.known)
    failed=engine.run(eval_inputs,"adaptive",threshold=0.,faults={key:"light"})
    row=next(i for i in failed["routes"] if i["key"]==key)
    need(row["route"]=="full" and failed["counts"]["light_failures"]==1,"Recoverable route failed open")
    try:
        engine.run(eval_inputs,"full",faults={key:"full"})
    except RouteFailure as error:
        hard=error.stats
    else:
        raise RuntimeError("Required full failure was incorrectly accepted")
    host=deepcopy(full)
    try:
        with taps(host,"xuzhou"):
            Runtime(device=device,full=host)
    except ValueError as error:
        need("diagnostic" in str(error),"Wrong probe exclusion")
    else:
        raise RuntimeError("Deployment allowed active layer probes")
    with torch.no_grad():
        next(engine.full.parameters()).add_(1)
    try:
        engine.run(eval_inputs,"full")
    except ValueError as error:
        need("changed" in str(error),"Wrong frozen state change rejection")
    else:
        raise RuntimeError("Mutated states reused a cached graph")
    return dict(first_builds=before["counts"]["graph_builds"],matched_reuses=after["counts"]["graph_reuses"],
                recoverable_light_failed_ms=failed["elapsed_ms"],light_failure=failed["counts"],full_failure=hard,
                changed_weight_rejected=True,diagnostic_probe_rejected=True)


def exercise(root,out,code,device="cpu",*,physical_admission=None):
    out.mkdir()
    reviews={support:approved_banks(root,out,code,support) for support in SUPPORT}
    rows=[]
    for support,seed in product(SUPPORT,SEEDS):
        bank,review,roles,manifest=reviews[support]
        blocks,labels,commitments=label_metadata(bank,roles)
        done=out/f"s{support}-m{seed}"
        done.mkdir()
        full,light=models(seed,bank.release.scene)
        engine=Runtime(device=device,full=deepcopy(full),light=deepcopy(light),
                       identity={"full":state_hash(full),"light":state_hash(light)})
        calibration=requests(bank,review,roles,"calibration",code,device)
        _,_,cal_rows=endpoint_records(engine,calibration,labels["calibration"])
        # The entire threshold grid, quality boundary and resource policy are
        # fixed before constructing any confirmation requests.
        protocols={}
        for name,budget in BUDGETS.items():
            specification=dict(schema=SPARSE_PROTOCOL,name=name,full_state=engine.identity["full"],
                  light_artifact=engine.identity["light"],criterion="margin",thresholds=THRESHOLDS,
                  min_coverage=budget["floor"],classes=9,min_class_support=1,max_drop=dict.fromkeys(("oa","aa","recall"),0.),
                  blocks=blocks,data_hashes=commitments,admission_hash=manifest,
                  calibration_request_hash=digest([r.identity for r in calibration]))
            gate=GateProtocol(specification,digest(specification))
            value=calibrate(gate,cal_rows)
            decision=_write(done/f"{name}-freeze.json",dict(protocol=specification,policy=value))
            protocols[name]=(gate,decision)
        evaluation=requests(bank,review,roles,"evaluation",code,device)
        need(not {r.parent for r in calibration}&{r.parent for r in evaluation},"Confirmation shares a calibration parent")
        base,fast,test_rows=endpoint_records(engine,evaluation,labels["confirmation"])
        for name,budget in BUDGETS.items():
            gate,decision=protocols[name]
            frozen=_read(decision)["policy"]
            assess=confirm(gate,frozen,test_rows,expected_freeze_hash=digest(frozen))
            opts=assess["inference_kwargs"]
            mode="adaptive" if opts["mode"]=="adaptive" else "full"
            options=({"threshold":opts["threshold"],"min_coverage":budget["floor"]} if mode=="adaptive" else {})
            routed=engine.run(evaluation,mode,**options)
            matched=matched_random(evaluation,routed,seed+support,budget["floor"])
            random=engine.run(evaluation,"random",random_fast=matched,min_coverage=budget["floor"])
            if mode=="adaptive":
                for key in ("light_calls","full_calls","graph_builds","coverage_fallbacks"):
                    need(routed["counts"][key]==random["counts"][key],"Random control differs in matched work")
            else:
                # Fail-closed retains the true static full endpoint without extra light calls.
                random=base
            treatments={key:evidence(evaluation,labels["confirmation"],value)
                        for key,value in (("full",base),("light_with_required_fallback",fast),("adaptive",routed),("matched_random",random))}
            warm=lambda:requests(bank,review,roles,"evaluation",code,device,use="cost")
            modes={"full":("full",{}),"adaptive":(mode,options),
                   "light_with_required_fallback":("random",{"random_fast":[r.key for r in evaluation if r.known]}),
                   "matched_random":(("random",{"random_fast":matched,"min_coverage":budget["floor"]})
                                     if mode=="adaptive" else ("full",{}))}
            for key,(kind,arguments) in modes.items():
                treatments[key]["timing"]=repeat_route(engine,evaluation,treatments[key]["predictions"],kind,arguments,warm)
            selected=routed["counts"]["full_calls"]
            rows.append(dict(support=support,seed=seed,budget=name,limit=budget["max_full"],overrun=max(0,selected-budget["max_full"]),
                             frozen=decision["sha256"],policy=digest(frozen),status=assess["status"],
                             threshold=opts.get("threshold"),input_hash=digest([r.identity for r in evaluation]),
                             matched_random_keys=matched,model_identity=engine.identity,output=treatments))
        if support==SUPPORT[0] and seed==SEEDS[0]:
            failures=fault_and_cache(full,light,evaluation,device)
    summaries={}
    for name in BUDGETS:
        candidates=[r for r in rows if r["budget"]==name]
        paired=[r["output"]["adaptive"]["metrics"]["oa"]-r["output"]["full"]["metrics"]["oa"] for r in candidates]
        summaries[name]=dict(cases=len(candidates),mean_oa_delta=fmean(paired),sample_sd=stdev(paired),
                             overrun=sum(r["overrun"] for r in candidates),
                             accepted=sum(r["status"]=="empirical_pass" for r in candidates))
    require_frozen_code(code)
    return _write(out/"routing.json",dict(producer=code["commit"],device=device,rows=rows,summaries=summaries,failures=failures,
                  scope="isolated_synthetic_engineering",physical_admission=physical_admission,optimizer_updates=0,query_accessed=False,
                  formal_quality_cost_claim=False))
