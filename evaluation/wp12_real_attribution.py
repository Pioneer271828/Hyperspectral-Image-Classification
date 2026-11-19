"""Read approved checkpoints and connect observation, exposure and native-head outputs."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import frozen_training_stages as stages, wp12_linear_attribution as linear
from evaluation.wp12_release_results import proof
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from scripts import run_frozen_band_region as archived
from scripts.run_wp12_retained_evidence import RELEASE, pin
from train_spatial import SpatialConfig
from transfer import p4_frozen_diagnostics as hh, p4_full_budget_registration as long
from transfer.development_training import sparse_labels
from transfer.g3_training import state_hash
from transfer.registered_development import load_development

SCHEMA = "wp12_real_development_error_attribution_v1"
XZ_CURSORS = {"initial": 0, "prefix": 7, "selected": 347, "last": 512}
SHARD_GROUPS = ("honghu_short","honghu_120","xuzhou")
require = stages.require


def checkpoint(root, link):
    return torch.load(stages.pinned(root,link),map_location="cpu",weights_only=True)


def confusion(rows, classes):
    result = {}
    truth = np.asarray([r["truth"] for r in rows],dtype=np.int64)-1
    for head in stages.HEADS:
        predicted = np.asarray([r["predicted"][head] for r in rows],dtype=np.int64)-1
        values = np.bincount(classes*truth+predicted,minlength=classes**2).reshape(classes,classes)
        result[head] = values.tolist()
    return result


def former_confusion(rows, historic, classes):
    actual = confusion(rows,classes)
    require(actual=={head:historic[head]["confusion"] for head in stages.HEADS},
            "Attribution replay does not match authorized saved point scores")


def prepared_roles(scene, allowed, release, case, *, long_fit):
    population = long.fixed_population if long_fit else hh.fixed_population
    return {role:population(release[role],allowed,role,scene.plan["shape"][1],budget)
            for role,budget in (("train",case["k"]),("val",5))}


def hh_endpoint(model, scene, role_views, role_meta, reference, *, device):
    values = {}
    for role, views in role_views.items():
        _, labels, population = role_meta[role]
        rows, offset = [], 0
        for block,(tile,cells,view) in enumerate(views,1):
            n = len(cells)
            if n:
                derived,_ = linear.tile_rows(model,scene,tile,view,cells,labels[offset:offset+n].tolist(),block,reference)
                rows.extend(derived)
            offset += n
        require(offset==len(labels),"Role-native blocks differ from the authorized ordered point population")
        values[role] = dict(population=population,rows=rows,classes=linear.class_rows(rows,22))
    return values


def hh_fit(root, group, case, reg, scene, allowed, frozen, release, weights_pin, state, saved,
           population_pins, initialize, *, device, extra_full=None):
    model, initial = initialize()
    role_meta = prepared_roles(scene,allowed,release,case,long_fit=group=="honghu_120")
    require({role:data[2] for role,data in role_meta.items()}==population_pins,"HongHu role/source changed")
    role_views = {role:hh.fixed_views(scene,allowed,data[0],
                   {"sampling":frozen["records"][role],"metadata":"nominal"},role)
                  for role,data in role_meta.items()}
    model.to(device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention=False
    model.backbone.adapter.clear_attention_cache()
    reference = linear.weights(model,scene.dataset)
    step = state["best"]["step"]
    end = state["cursor"]
    require(0<step<=end and end==(24 if group=="honghu_short" else 120)
            and len(state["trace"])==end and [x["step"] for x in state["trace"]]==list(range(1,end+1)),
            "Foreign training cursor or incomplete trajectory")
    observed = linear.paired_roles(role_meta["train"][1],release["train"],[x["tile"] for x in state["trace"]],
                                   {0,step,end,*[r["step"] for r in state["history"]]},22)
    fit = {}
    identities = {"initial":state_hash(model)}
    for name,cursor,parameters in (("initial",0,None),("selected",step,state["best"]["model"]),("last",end,state["model"])):
        if parameters is not None:
            model.load_state_dict(parameters,strict=True)
        current = state_hash(model)
        identities[name] = current
        if name=="last" and current==identities["selected"]:
            require(cursor==step,"Repeated physical weights at different cursors require explicit provenance")
            fit[name] = dict(reuse="selected",cursor=cursor)
            continue
        roles = hh_endpoint(model,scene,role_views,role_meta,reference,device=device)
        for role,body in roles.items():
            former_confusion(body["rows"],saved[name][role],22)
            if name!="initial":
                body["max_trajectory_error"] = linear.evolution(fit["initial"]["roles"][role]["rows"],body["rows"])
        full = None
        if extra_full is not None and name=="selected":
            held = hh.fixed_views(scene,allowed,role_meta["val"][0],extra_full[0],"val")
            full = hh_endpoint(model,scene,{"val":held},role_meta,reference,device=device)["val"]
            former_confusion(full["rows"],extra_full[1],22)
        fit[name] = dict(cursor=cursor,state=current,roles=roles,selected_full_development=full,
                         parameter_delta=linear.parameter_deltas(reference,linear.weights(model,scene.dataset)))
        require(state_hash(model)==current,"Frozen attribution changed a model tensor")
    return dict(group=group,case=case["id"],support_seed=case["support_seed"],model_seed=case.get("model_seed",64),
                k=case["k"],method=case["method"],arm=case["arm"],fit=fit,exposure=observed,
                checkpoint=weights_pin,populations=population_pins,states=identities,
                historic_event_margins=[dict(step=row["step"],heads=row["metrics"]) for row in state["history"]],
                tensor_gradient_record=state.get("parameter_audit",{}).get("by_category"),
                per_class_gradient_norm_available=False,actual_optimizer_updates=0)


def short_context(root):
    links,docs = archived.source_documents(root)
    scope = docs["hh_scope"]
    receipt = docs["hh_predictions"]
    _,reg,plan,_,_,_ = archived._honghu_context(scope)
    exposure = stages.json_pin(root,dict(path=reg["exposure"]["plan"],sha256=reg["pins"][reg["exposure"]["plan"]]))
    geometry,allowed,scene,frozen = hh.prior.inputs(reg,plan,exposure)
    require(scope["raw_gt_loaded"] is scope["query_accessed"] is False and receipt["scope_sha256"]==links["hh_scope"]["sha256"],
            "HongHu short frozen readout and published roles differ")
    return reg,plan,geometry,allowed,scene,frozen,scope,receipt


def short_cases(root, keys, device):
    reg,plan,geometry,allowed,scene,frozen,scope,receipt = short_context(root)
    for saved in scope["cases"]:
        case = saved["case"]
        if case["id"] not in keys:
            continue
        release = hh.prior.released_roles(reg,plan,geometry,allowed,case)
        entry = saved["state"]
        state_pin = dict(path=entry["path"],sha256=entry["state_sha256"])
        state = checkpoint(root,state_pin)
        filename = case["id"]+".json"
        link = dict(path=str(root/Path(archived.PINS["hh_predictions"][0]).parent/filename),
                    sha256=receipt["artifacts"][filename])
        readouts = stages.json_pin(root,link)["readouts"]
        old = {stage:{role:readouts[f"fit__{stage}__{role}"]["heads"] for role in ("train","val")}
               for stage in ("initial","selected","last")}
        source_args = (reg["source"]["weight"],reg["source"]["result"]) if case["method"]=="shared_transfer" else ()
        row = hh_fit(root,"honghu_short",case,reg,scene,allowed,frozen,release,state_pin,state,old,
                    saved["populations"],lambda: archived.init_honghu(SpatialConfig(reg["config"]),*source_args,
                    coverage_arm=case["arm"]),device=device,
                    extra_full=(scope["conditions"]["full"],readouts["obs__full"]["heads"]))
        require(row["states"]=={key:entry[key] for key in ("initial","selected","last")}
                and row["fit"]["selected"]["cursor"]==entry["selected_step"],
                "HongHu physical short weights differ from archived state identity")
        for cursor,counts in saved["exposure"]["label_exposure_at_events"].items():
            require(row["exposure"][cursor]["labeled_visits"]==counts,"Short recorded exposure differs")
        yield row


def long_cases(root, keys, device):
    parent = stages.json_pin(root,proof(root,"p4"))
    scope = stages.json_pin(root,parent["scientific_scope"])
    reg = stages.json_pin(root,dict(path=scope["registration"],sha256=scope["protected_files"][scope["registration"]]))
    plan = stages.json_pin(root,dict(path=reg["plan"],sha256=reg["pins"][reg["plan"]]))
    exposure = stages.json_pin(root,dict(path=reg["exposure"]["plan"],sha256=reg["pins"][reg["exposure"]["plan"]]))
    geometry,allowed,scene,frozen = hh.prior.inputs(reg,plan,exposure)
    for item in parent["cases"]:
        case = item["case"]
        if case["id"] not in keys:
            continue
        audit = stages.json_pin(root,item["audit"])
        require(audit["case"]==case and audit["audit_optimizer_updates"]==0
                and audit["raw_gt_loaded"] is audit["query_accessed"] is False,
                "Foreign HP4 custody or observed validation")
        contract_path, = [p for p in audit["artifact_pins"] if p.endswith("/split/state.pt")]
        state_pin = dict(path=contract_path,sha256=audit["artifact_pins"][contract_path])
        release = hh.prior.released_roles(reg,plan,geometry,allowed,case)
        if not Path(contract_path).exists():
            yield dict(group="honghu_120",case=case["id"],history="confusion_only",fit=audit["fit"],
                       checkpoint=state_pin,k=case["k"],model_seed=case["model_seed"],support_seed=case["support_seed"],
                       populations=audit["populations"],exposure=audit["exposure"])
            continue
        state = checkpoint(root,state_pin)
        old = {stage:{role:entry["heads"] for role,entry in readouts.items()}
               for stage,readouts in audit["fit"].items()}
        row = hh_fit(root,"honghu_120",case,reg,scene,allowed,frozen,release,state_pin,state,old,
                    audit["populations"],lambda: long.initialize_case(reg,case),device=device)
        require(row["states"]==audit["state_sha256"] and row["fit"]["selected"]["cursor"]==audit["selected_step"],
                "Reviewed 120-step endpoint differs from exact approved checkpoint")
        yield row


def xz_layout(scene,record):
    result = {}
    for role,field in (("train","train_support"),("val","validation_support")):
        dense = sparse_labels(record,scene.plan,field)
        result[role] = [(tile,dense[tile_slice(tile)]) for tile,binding in zip(scene.plan["tiles"],record["roles"],strict=True)
                        if binding["role"]==("devtrain" if role=="train" else "devval")]
    return result


def xz_exposure(scene,layout,history):
    truth = {}
    for tile,values in layout["train"]:
        y0,_,x0,_ = tile["box"]
        for y,x in np.argwhere(values):
            truth[(y0+int(y))*scene.plan["shape"][1]+x0+int(x)] = (tile["id"],int(values[y,x]))
    output = {"0":dict(labeled_visits=[0]*9,labeled_steps=[0]*9,unique_labels=[0]*9,head_tensor_gradient_steps=[0]*3)}
    visits, steps, totals = Counter(),Counter(),Counter()
    distinct = defaultdict(set)
    for position,row in enumerate(history,1):
        require(row["step"]==position and row["support_ids"],"Foreign Xuzhou update cursor or missing supervision")
        seen = set()
        for point in row["support_ids"]:
            tile,label = truth[point]
            require(tile==row["tile_id"],"Recorded gradient access crossed the released support tile")
            visits[label] += 1
            seen.add(label)
            distinct[label].add(point)
        steps.update(seen)
        for key,prefix in zip(stages.HEADS,("heads","hgnn_heads","cnn_heads"),strict=True):
            if f"{prefix}.xuzhou.weight" in row["gradient_parameters"] and f"{prefix}.xuzhou.bias" in row["gradient_parameters"]:
                totals[key] += 1
        if position in XZ_CURSORS.values():
            output[str(position)] = dict(labeled_visits=[visits[c] for c in range(1,10)],
                labeled_steps=[steps[c] for c in range(1,10)],unique_labels=[len(distinct[c]) for c in range(1,10)],
                head_tensor_gradient_steps=[totals[h] for h in stages.HEADS])
    require(len(history)==512 and set(output)=={str(c) for c in XZ_CURSORS.values()},"Missing Xuzhou training trace")
    return output


def xz_spectra(scene,tile,labels,view,plan,store):
    # Original delivered columns and frozen normalization statistics, restricted to published support.
    original = np.asarray(scene.cube[tile_slice(tile)],dtype=np.float64).reshape(-1,len(scene.bands.band_ids))
    kept = {name:set(values) for name,values in plan["bands"].items()}
    groups = {"shared":kept["central_gap"]&kept["matched_dispersed"],
              "central_only":kept["central_gap"]-kept["matched_dispersed"],
              "dispersed_only":kept["matched_dispersed"]-kept["central_gap"]}
    require(len(groups["central_only"])==len(groups["dispersed_only"]) and all(groups.values()),
            "Historical retained groups are not matched by usable column count")
    ids = {bid:i for i,bid in enumerate(scene.bands.band_ids)}
    loc = {bid:i for i,bid in enumerate(view.bands.band_ids)}
    norm = view.observation.spectral.reshape(-1,len(loc))
    flat = labels.ravel()
    for cls in set(flat.tolist())-{0}:
        samples = flat==cls
        for group,bands in groups.items():
            seq = sorted(bands)
            raw_values = original[samples][:,[ids[b] for b in seq]]
            normalized_values = norm[samples][:,[loc[b] for b in seq]]
            store[cls,group].append((raw_values,normalized_values))
    return {group:sorted(bands) for group,bands in groups.items()}


def spectral_contrasts(arrays, groups):
    output = {}
    for cls in (5,6):
        for rival in linear.RIVALS["xuzhou"][cls]:
            scores = {}
            for group,bands in groups.items():
                record = {}
                a,b = arrays[cls,group],arrays[rival,group]
                for i,scale in enumerate(("delivered_raw","frozen_train_normalized")):
                    aa,bb = np.concatenate([x[i] for x in a]), np.concatenate([x[i] for x in b])
                    delta = aa.mean(0)-bb.mean(0)
                    block_delta = [float(np.sqrt(np.mean((v[i].mean(0)-bb.mean(0))**2))) for v in a]
                    record[scale] = dict(rms_class_mean_difference=float(np.sqrt(np.mean(delta**2))),
                         equal_weight_signed_mean=float(delta.mean()),class_n=len(aa),rival_n=len(bb),
                         class_blocks=len(a),rival_blocks=len(b),block_to_rival_centroid_rms=block_delta)
                scores[group] = dict(columns=bands,**record)
            output[f"{cls}/vs/{rival}"] = scores
    return output


def xz_cases(root, device):
    plan = stages.stage_plan(root,pin(root,RELEASE))
    release = stages.json_pin(root,plan["release"])
    scene,record,_ = load_development(plan["development"])
    layout = xz_layout(scene,record)
    train = checkpoint(root,plan["stages"]["last"]["state"])
    exposure = xz_exposure(scene,layout,train["history"])
    reference = None
    history = {}
    spectra = defaultdict(list)
    groups = None
    layer_delta = {}
    views = ("full","central_gap","matched_dispersed")
    for name,expected in XZ_CURSORS.items():
        spec = plan["stages"][name]
        require(spec["step"]==expected,"Xuzhou checkpoint was selected from another protocol")
        model = stages.load_stage(root,release,spec,name).to(device)
        model.backbone.adapter.cache_attention=False
        model.backbone.adapter.clear_attention_cache()
        w = linear.weights(model,scene.dataset)
        reference = w if reference is None else reference
        identity = state_hash(model)
        fit = {}
        for role,tiles in layout.items():
            case_views = ("full",) if role=="train" else views
            rows = {view:[] for view in case_views}
            norms = []
            for block,(tile,labels) in enumerate(tiles,1):
                cells = np.flatnonzero(labels.ravel())
                if not cells.size:
                    continue
                truth = labels.ravel()[cells].tolist()
                current = {}
                for view in case_views:
                    datum = stages.checked_view(scene,tile,view,plan["bands"][view])
                    result, tensors = linear.tile_rows(model,scene,tile,datum,cells,truth,block,reference,layers=role=="val")
                    rows[view].extend(result)
                    if role=="train" and name=="initial":
                        groups = xz_spectra(scene,tile,labels,datum,plan,spectra)
                    if role=="val":
                        current[view] = tensors
                if role=="val":
                    norms.append(dict(block=block,n=len(cells),view_comparisons={
                        alt:{site:float((current[alt][site]-current["full"][site]).float().square().mean().sqrt())
                             for site in ("vca","pre_pool","cnn","graph","fused")
                             if current[alt][site].shape==current["full"][site].shape} for alt in views[1:]}))
            if role=="val":
                layer_delta[name] = norms
            fit[role] = {}
            for view,values in rows.items():
                maximum = None if history=={} else linear.evolution(history["initial"]["roles"][role][view]["rows"],values)
                fit[role][view] = dict(rows=values, classes=linear.class_rows(values,9),max_trajectory_error=maximum)
        require(state_hash(model)==identity,"Xuzhou frozen instrument or head state changed")
        history[name] = dict(cursor=expected,checkpoint=spec["state"],state=identity,roles=fit,
                             parameter_delta=linear.parameter_deltas(reference,w))
    return dict(group="xuzhou",case="wp11-full-k10",k=10,model_seed=64,source=plan["development"],
                fit=history,exposure=exposure,identities=plan["identities"],bands=plan["bands"],
                normalized_before_after=spectral_contrasts(spectra,groups),layer_delta=layer_delta,
                new_view_admitted=False,actual_optimizer_updates=0)
