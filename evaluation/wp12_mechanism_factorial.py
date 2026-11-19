"""Frozen equal-band placement/metadata/geometry contrasts on prior production weights."""

from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from evaluation.wp12_native_production import endpoints, selected_model
from model.research_config import digest
from transfer import native_production_campaign as campaign, native_production_model as model_api
from transfer.g3_training import state_hash
from transfer.spatial_coverage_protocol import bank_view

SCHEMA = "wp12_retained_factorial_frozen_v1"
PRODUCER = "dd5cc367a6e0f04cf0278844e95600be8990e2cf"
PARENT = ("wp05-wp06-production-20261119-qa/program-v2", "5d503cbd25f89c284431a4e77a8059421e675eb678846c9f9b2db02f24c1ff18")
PREVIOUS = ("wp05-wp06-production-20261119-qa/accepted/production.json",
            "f432c0e4862ac6ae3def6c639697e688ef2c6c74a7487836753c55fb8575cc52")
VIEWS = ("central_gap","matched_dispersed")
METADATA = ("declared","missing")
FACTORS = (1,2)
COVERAGE = ("none","anchor_only","binary","continuous","shuffled")


def registration(root):
    return dict(parent_sha256=PARENT[1],previous_acceptance=PREVIOUS[1],producer=PRODUCER,
                design=[list(i) for i in product(VIEWS,METADATA,FACTORS)],
                coverage=list(COVERAGE),scope="previous_independent_synthetic_confirmation_only",
                old_joint_stress=("missing",2),retained_count=13,confirmed_original_points_only=True)


def point_trial(p, model, tile, band, meta, factor, execution, *, config=None):
    execution.sync()
    start=perf_counter()
    actual,_,targets=bank_view(p.bank,p.bank.role_row("devval",tile),p.producer,band,factor=factor,metadata=meta)
    prepared=perf_counter()-start
    output,graph,timing=model_api.run_forward(model,actual,p.producer,"devval",config or p.config,p.axis,execution=execution)
    evidence={}
    for head, values in zip(HEADS,output[:3],strict=True):
        full=actual.lattice.project(values).detach().cpu().numpy()
        width=actual.lattice.scene_shape[1]
        y0,_,x0,x1=actual.lattice.box
        indexes=[(x//width-y0)*(x1-x0)+x%width-x0 for x,_ in targets.points]
        evidence[head]=full[indexes].tolist()
    execution.sync()
    raw=actual.receipt
    info={} if len(output)<4 else (output[-1].get("coverage") or {})
    result=dict(band=band, metadata=raw["source"]["spectral_metadata_mode"],factor=factor,
        truth=[y for _,y in targets.points],point_set=targets.summary()["point_set"],
        bands=list(actual.view.bands.band_ids),spectral=raw["spectral"],
        wavelength_quality=actual.view.bands.wavelength_quality,metadata_fingerprint=actual.view.bands.fingerprint,
        grid=raw["grid_shape"],q=graph["q_shape"],head=evidence,coverage_reason=info.get("fallback_reason"))
    return result,dict(total_seconds=perf_counter()-start,prepare_seconds=prepared,
                       forward_seconds=timing["forward_seconds"],graph_seconds=sum(timing["graph_seconds"]))


def verify_input_pair(a,b):
    require(all(a[k]==b[k] for k in ("truth","point_set","bands","spectral","grid","factor"))
            and a["wavelength_quality"]!="index_only" and b["wavelength_quality"]=="index_only"
            and a["metadata_fingerprint"]!=b["metadata_fingerprint"],
            "Metadata factorial changed actual retained observations or original-pixel labels")


def prior(root):
    program=root/PARENT[0]
    plan, link=campaign.plan_for(root,program,{"commit":PRODUCER})
    require(link["sha256"]==PARENT[1],"Prior frozen training camp differs")
    old=json_pin(root,dict(path=str(root/PREVIOUS[0]),sha256=PREVIOUS[1]))
    require(old["plan"]==PARENT[1] and old["scope"]=="independent_synthetic_confirmation","Prior production approval differs")
    return program,plan,link,old


@torch.inference_mode()
def run(root,case,execution):
    program,plan,link,old=prior(root)
    require(case in old["cohorts"] and case!="residual-continuous","Factorial uses only paired frozen production cases")
    p,received,_,packet=endpoints(root,program,plan,link,PRODUCER,case)
    model=selected_model(p,received,packet,execution.device)
    frozen=state_hash(model)
    proof=old["cohorts"][case]["receipt"]
    historic=json_pin(root,dict(path=str(program.parent/"capture-cpu"/(case+".json")),sha256=proof))
    rows=[]
    for tile in p.roles["evaluation"]:
        block=plan["inputs"][plan["design"][case]["input"]]["blocks"][str(tile)]
        if block["domain"]!="out":
            continue
        values={}
        for band,factor in product(VIEWS,FACTORS):
            items=[point_trial(p,model,tile,band,meta,factor,execution) for meta in METADATA]
            a,b=[row[0] for row in items]
            verify_input_pair(a,b)
            require(len(a["bands"])==13,"Factorial requires equal retained-band count")
            if p.config.arm=="pixel":
                require(a["head"]==b["head"],"Metadata leaked into the pure spectral reference")
            values[band,factor]=items
        original,timing=point_trial(p,model,tile,"missing",None,2,execution)
        accepted,=[row for row in historic["rows"] if (row["tile"],row["band"],row["factor"])==(tile,"missing",2)]
        require(accepted["retained_ids"]==original["bands"] and accepted["truth"]==original["truth"]
                and accepted["retained_sha256"]==original["spectral"],"Original joint-stress observation differs")
        for head in HEADS:
            a,b=map(np.asarray,(accepted["heads"][head],original["head"][head]))
            require(np.allclose(a,b,rtol=1e-5,atol=5e-5) and np.array_equal(a.argmax(1),b.argmax(1)),
                    "Old joint-stress decisions changed in the new audit")
        rows.append(dict(tile=tile,mode=block["mode"],cells=[dict(result=result,timing=clock)
                  for items in values.values() for result,clock in items],old_joint_stress=original,
                  old_joint_stress_time=timing))
    require(state_hash(model)==frozen,"Factorial altered frozen checkpoint")
    return dict(schema=SCHEMA,scope="independent_synthetic_confirmation",case=case,state=frozen,
                checkpoint=received["full"]["sha256"],source_capture=proof,device=execution.numeric,rows=rows,
                registration=digest(registration(root)),optimizer_updates=0,query_accessed=False,raw_gt_loaded=False,
                approximate_real_physical_axis_promoted=False,independent_real_accuracy=None)


@torch.inference_mode()
def coverage(root,execution,working):
    program,plan,link,_=prior(root)
    case="s64-k211-native"
    p,received,_,packet=endpoints(root,program,plan,link,PRODUCER,case)
    old=selected_model(p,received,packet,"cpu")
    inherited=old.state_dict()
    notes={}
    for name in COVERAGE:
        config=replace(p.config,spectral=name)
        model=model_api.configured_model(p.baseline(p.seed),config,len(p.axis),p.seed+7321)
        model.load_state_dict({**model.state_dict(),**inherited},strict=True)
        require(all(torch.equal(model.state_dict()[key],value) for key,value in inherited.items()),
                "Coverage port altered legacy shared tensors")
        with TemporaryDirectory(dir=working) as temp:
            path=Path(temp)/"coverage.pt"
            model_api.export_production(model.eval(),path,config,p.axis)
            loaded,exact,axis=model_api.open_production(path,execution.device)
            require(exact==config and axis==p.axis and state_hash(loaded)==state_hash(model),"Coverage checkpoint roundtrip differs")
            model.to(execution.device)
            held=p.roles["evaluation"][0]
            rows=[]
            for band,meta in product(("full","central_gap"),METADATA):
                a,_=point_trial(p,model,held,band,meta,1,execution,config=config)
                b,clock=point_trial(p,loaded,held,band,meta,1,execution,config=config)
                require(a==b,"Coverage restored output differs")
                rows.append(dict(band=band,metadata=meta,positions=a["bands"],reason=a["coverage_reason"],
                                 grid=a["grid"],seconds=clock["total_seconds"]))
        notes[name]=dict(shared_count=len(inherited),checkpoint_restored=True,
                         initialization_only=name!="none",views=rows)
    return dict(schema=SCHEMA,scope="coverage_checkpoint_and_fallback_engineering_only",device=execution.numeric,
                modes=notes,optimizer_updates=0,raw_gt_loaded=False,query_accessed=False)
