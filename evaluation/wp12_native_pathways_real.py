"""Consume only the existing frozen role/observation roster, one durable block at a time."""

from time import perf_counter

import numpy as np

from evaluation import frozen_training_stages as auth, wp12_native_pathways as path
from evaluation import wp12_real_attribution as prior
from evaluation.wp12_linear_attribution import FOCUS, RIVALS
from research_runtime import require_frozen_code, tensor_hash
from scripts import run_frozen_band_region as archive
from scripts.run_wp12_retained_evidence import RELEASE, pin
from train_spatial import SpatialConfig
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.registered_development import load_development

SCHEMA = "wp12_native_frozen_signal_path_v1"
CASES = ("xz-initial","xz-selected","hh-short-shared","hh-short-scratch","hh-long-scratch","hh-long-shared")
CONTROL = (2,4)
require = auth.require


def xuzhou(root,case):
    slot = case.split("-")[1]
    plan = auth.stage_plan(root,pin(root,RELEASE))
    release = auth.json_pin(root,plan["release"])
    ref = plan["stages"][slot]
    model = auth.load_stage(root,release,ref,slot)
    scene,record,_ = load_development(plan["development"])
    layout = prior.xz_layout(scene,record)
    blocks = []
    for role,tiles in layout.items():
        declared = ("full",) if role=="train" else tuple(plan["views"])
        for ordinal,(tile,labels) in enumerate(tiles,1):
            row = labels.ravel()
            if not np.isin(row,(*FOCUS["xuzhou"],*CONTROL)).any():
                continue
            for view in declared:
                blocks.append((role,ordinal,view,tile,row,
                    lambda t=tile,n=view:auth.checked_view(scene,t,n,plan["bands"][n])))
    return model,scene,dict(checkpoint=ref["state"],cursor=ref["step"],population=plan["identities"],
              assumption="published_Xuzhou_support_full_and_development_three_views",support_seed=None,
              model_seed=64,k=10),blocks


def honghu(root,case):
    short = "-short-" in case
    scratch = case.endswith("scratch")
    identifier = ("s512-k5-"+("scratch" if scratch else "shared_transfer")+"-none") if short else (
        "s2048-k5-scratch-m512-none" if scratch else "s2048-k1-shared_transfer-m512-none")
    if short:
        reg,plan,geometry,allowed,scene,base,scope,_ = prior.short_context(root)
        item, = [r for r in scope["cases"] if r["case"]["id"]==identifier]
        spec = item["case"]
        src = (reg["source"]["weight"],reg["source"]["result"]) if not scratch else ()
        model,_ = archive.init_honghu(SpatialConfig(reg["config"]),*src,coverage_arm=spec["arm"])
        ref = item["state"]
        checkpoint = dict(path=ref["path"],sha256=ref["state_sha256"])
        expected_state,expected_cursor = ref["selected"],ref["selected_step"]
        val_conditions = {"stride":base["records"]["val"],"full":scope["conditions"]["full"]["sampling"]}
    else:
        parent,reg,plan,geometry,allowed,scene,base = prior.long_context(root)
        item, = [r for r in parent["cases"] if r["case"]["id"]==identifier]
        spec = item["case"]
        audit = auth.json_pin(root,item["audit"])
        member, = [name for name in audit["artifact_pins"] if name.endswith("/split/state.pt")]
        checkpoint = dict(path=member,sha256=audit["artifact_pins"][member])
        expected_state,expected_cursor = audit["state_sha256"]["selected"],audit["selected_step"]
        model,_ = prior.long.initialize_case(reg,spec)
        val_conditions = {"stride":base["records"]["val"]}
    state = prior.checkpoint(root,checkpoint)
    model.load_state_dict(state["best"]["model"],strict=True)
    require(state_hash(model)==expected_state and state["best"]["step"]==expected_cursor
            and state["cursor"]==(24 if short else 60),"Wrong frozen HongHu checkpoint")
    roles = prior.hh.prior.released_roles(reg,plan,geometry,allowed,spec)
    populations = prior.prepared_roles(scene,allowed,roles,spec,long_fit=not short)
    require({role:value[2] for role,value in populations.items()}==item["populations"],"Foreign guarded population")
    kept = {"train":{"stride":base["records"]["train"]},"val":val_conditions}
    blocks = []
    for role,tiles in roles.items():
        allowed_ids = {t["id"]:t for t in allowed[role]}
        for ordinal,(tile,local) in enumerate(tiles,1):
            row = local.ravel()
            if not np.isin(row,(*FOCUS[scene.dataset],*CONTROL)).any():
                continue
            for view,sampling in kept[role].items():
                condition = dict(sampling=sampling,metadata="nominal")
                def materialize(role=role,tile=tile,condition=condition,allowed_ids=allowed_ids):
                    return (prior.hh.selected_view(scene,tile,allowed_ids,condition) if role=="val" else
                            prior.hh.guarded.retained_view(scene,tile,condition))
                blocks.append((role,ordinal,view,tile,row,materialize))
    basis = dict(checkpoint=checkpoint,cursor=expected_cursor,population=item["populations"],training=identifier,
                 archived_path_cursor=state["cursor"],approved_combined_updates=120 if not short else None,
                 support_seed=spec["support_seed"],model_seed=spec.get("model_seed",64),k=spec["k"],method=spec["method"])
    return model.eval().requires_grad_(False),scene,basis,blocks


def graph_frozen(model,scene,tile,view,labels,case):
    points = np.flatnonzero(np.isin(labels,(*FOCUS[scene.dataset],*CONTROL)))
    alternative = {15:21,21:20} if case=="hh-long-scratch" else {}
    rivals = RIVALS[scene.dataset]
    candidates = [(int(i),int(labels[i]),alternative.get(int(labels[i]),rivals.get(int(labels[i]),(1,))[0])) for i in points]
    t = perf_counter()
    clean,q,a,reference = path.capture(model,scene,tile,view)
    captured = perf_counter()-t
    result = path.measure(model,scene.dataset,clean,q,a,candidates,reference=reference)
    arrays = result.pop("features")
    features = {name:dict(shape=[len(points),value.shape[-1]],summary_rms=float(value[points].square().mean().sqrt()),
                          digest=tensor_hash(value[points])) for name,value in arrays.items()}
    for ordinal,row in enumerate(result["rows"],1):
        row.pop("selected_cell")
        row["ordinal"] = ordinal
    return dict(**result,feature_receipts=features,native_capture_seconds=captured,
                band_ids=list(view.bands.band_ids),role_input=view.observation.partition,
                scope="read_only_natural_sensitivity_no_intervention")


def run(root,out,code,case,execution,*,drain_after=None):
    require(case in CASES and (drain_after is None or type(drain_after) is int and drain_after>0),
            "Exact frozen case and checkpoint cursor required")
    t = perf_counter()
    setup = xuzhou if case.startswith("xz-") else honghu
    model,scene,basis,blocks = setup(root,case)
    model.to(execution.device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention=False
    model.backbone.adapter.clear_attention_cache()
    setup_seconds = perf_counter()-t
    identity = state_hash(model)
    cases = [(f"{role}-{block:03d}-{view}",role,block,view,tile,labels,open_view)
             for role,block,view,tile,labels,open_view in blocks]
    plan = dict(schema=SCHEMA,case=case,scene=scene.dataset,producer=code["commit"],basis=basis,model=identity,
                device=execution.numeric,roster=[k for k,*_ in cases],controls=CONTROL,approved_views_only=True,
                classification_role="historical_development",query_accessed=False,raw_gt_loaded=False,optimizer_updates=0)
    # JSON round trip uses an ordered list for the plan's serialized control classes.
    plan["controls"] = list(CONTROL)
    out.mkdir(parents=True,exist_ok=True)
    with case_lock(out):
        placement = out/"plan.json"
        if placement.exists():
            plan_pin = auth.file_pin(placement)
            require(auth.json_pin(root,plan_pin)==plan,"Resume changed model, roles, retained views or device")
        else:
            plan_pin = archive._write(placement,plan)
        ledger = {}
        new = 0
        for key,role,block,view,tile,labels,read in cases:
            destination = out/(key+".json")
            if destination.exists():
                link = auth.file_pin(destination)
                item = auth.json_pin(root,link)
                require(item["plan"]==plan_pin and item["key"]==key and item["model"]==identity,
                        "Committed block did not arise from the exact frozen work")
            elif drain_after is not None and new>=drain_after:
                continue
            else:
                started = perf_counter()
                observed = read()
                read_time = perf_counter()-started
                stats = graph_frozen(model,scene,tile,observed,labels,case)
                execution.sync()
                require(state_hash(model)==identity,"Native frozen diagnostic mutated the endpoint")
                require_frozen_code(code)
                link = archive._write(destination,dict(schema=SCHEMA,plan=plan_pin,key=key,model=identity,
                    role="support" if role=="train" else "calibration_development",block=block,view=view,
                    class_counts=np.bincount(labels.astype(int),minlength=scene.classes+1)[1:].tolist(),
                    materialize_seconds=read_time,measurement=stats))
                new += 1
            ledger[key] = link
        result = dict(schema=SCHEMA,plan=plan_pin,producer=code["commit"],cases=ledger,
                      query_accessed=False,raw_gt_loaded=False,optimizer_updates=0)
        if len(ledger)==len(cases):
            path_ = out/"complete.json"
            if path_.exists():
                link = auth.file_pin(path_)
                require(auth.json_pin(root,link)==result,"Existing completion differs")
            else:
                link = archive._write(path_,result)
            return dict(status="complete",receipt=link,new_blocks=new,load_seconds=setup_seconds,
                        driver_wall_seconds=perf_counter()-t)
        return dict(status="drained",committed=len(ledger),new_blocks=new,expected=len(cases),
                    plan=plan_pin,load_seconds=setup_seconds,driver_wall_seconds=perf_counter()-t)


def branches(root,case,execution):
    """Project the actual CNN and graph endpoint features onto one frozen fused margin."""
    import torch

    require(case in ("xz-initial","xz-selected","hh-short-shared","hh-long-scratch"),"Unknown approved branch comparison")
    setup = xuzhou if case.startswith("xz-") else honghu
    model,scene,basis,blocks = setup(root,case)
    model.to(execution.device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention=False
    model.backbone.adapter.clear_attention_cache()
    original = state_hash(model)
    measurements = {}
    tick = perf_counter()
    for role,block,band,tile,labels,read in blocks:
        if case.startswith("xz-") and role!="val":
            continue
        focus = np.flatnonzero(np.isin(labels,FOCUS[scene.dataset]))
        if not len(focus):
            continue
        observation = read()
        clean,q,a,reference = path.capture(model,scene,tile,observation)
        with torch.no_grad():
            logits,features,_,_ = path.spatial_forward(model,scene.dataset,clean,q,a,features=True)
            left = features["stage2_cnn"]
            right = features["stage2_graph"]
            projected = (model.backbone.fuse_features(left,torch.zeros_like(right)),
                         model.backbone.fuse_features(torch.zeros_like(left),right))
            linear = model.heads[scene.dataset]
            rows = []
            for ordinal,cell in enumerate(focus,1):
                true = int(labels[cell])
                rivals = (1,) if scene.dataset=="xuzhou" else ((21 if true==15 else 20,) if case=="hh-long-scratch"
                         else RIVALS[scene.dataset][true])
                for rival in rivals:
                    delta = linear.weight[true-1]-linear.weight[rival-1]
                    bias = float(linear.bias[true-1]-linear.bias[rival-1])
                    parts = [float(delta@v[cell]) for v in projected]
                    actual = float(logits["fused"][cell,true-1]-logits["fused"][cell,rival-1])
                    require(abs(sum(parts)+bias-actual)<2.5e-5,"Frozen fused logit reconstruction differs by branch")
                    rows.append(dict(ordinal=ordinal,truth=true,rival=rival,margin=actual,bias=bias,
                                     cnn_dot=parts[0],graph_dot=parts[1],
                                     predictions={h:int(p[cell].argmax())+1 for h,p in zip(auth.HEADS,reference,strict=True)}))
        measurements[f"{role}-{block:03d}-{band}"] = dict(role=role,block=block,band=band,rows=rows)
    execution.sync()
    require(state_hash(model)==original,"Frozen branch decomposition changed the model")
    return dict(schema="wp12_native_fused_branch_projection_v1",case=case,model=original,basis=basis,
                measurement=measurements,numeric=execution.numeric,seconds=perf_counter()-tick,
                optimizer_updates=0,raw_gt_loaded=False,query_accessed=False)
