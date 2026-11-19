"""Score existing frozen equal-capacity readouts within explicit role and block identities."""

from collections import defaultdict

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import json_pin, pinned, require
from evaluation.linear_probe import validate_plan, fingerprint
from model.research_config import digest

ROLES = ("support","calibration","evaluation")
SEEDS = (64,512,2048)
LAYERS = ("stage1_cnn","stage1_graph","stage2_cnn","stage2_graph")
PAIRED = "wp06-v2-paired-review01"
EVIDENCE = {
    "full54_acceptance": (PAIRED+"/acceptance.json","eb0ba5878acd896bbebd1fd1b7c7a8c987232aa4827ae239ddaba16a4a6017a3"),
    "locked_support": ("wp05-wp06-production-20261119-qa/accepted/locked-support.json",
                       "3d38301f9aeaf090c808fdc9e8a9c984cfd3eba769d74deb654bf9f1c0342f2e"),
    "old_control": ("wp06-controlled-signal-evidence01/receipt.json",
                    "187cd55afdb623096f7392f52ea18f11ae25b3f70a4821ea1dfcb776696cafec"),
    "two_hop_learned": ("wp06-learned-channel02/experiment/accepted.json",
                        "0a0375f67f8369bf4131b0c420caa57edd261e58cc2996c736b82c947becfe74"),
}


def block_metrics(truth,pred,blocks,classes):
    require(len(truth)==len(pred)>0,"Aligned nonempty released truth and decisions required")
    metrics = classification_metrics(truth,pred,range(1,classes+1))
    result = dict(total=len(truth),micro_oa=metrics["oa"],class_macro_aa=metrics["aa"],
                  per_class=metrics["per_class"],block_macro_oa=None,blocks=None,leave_one_block=None)
    if blocks is None:
        result["geometry_scope"] = "archived_roster_lacks_authenticated_parent_blocks"
        return result
    require(len(truth)==len(blocks),"Spatial parents do not bind the classification roster")
    mapping = defaultdict(list)
    for i,tile in enumerate(blocks):
        mapping[tile].append(i)
    left_out = {}
    summary = {}
    for tile,indices in sorted(mapping.items()):
        correct = sum(truth[i]==pred[i] for i in indices)
        summary[str(tile)] = dict(n=len(indices),correct=correct,oa=correct/len(indices),
              per_class=np.bincount([truth[i] for i in indices],minlength=classes+1)[1:].tolist())
        kept = [i for name,positions in mapping.items() if name!=tile for i in positions]
        left_out[str(tile)] = None if not kept else dict(n=len(kept),
                           oa=sum(truth[i]==pred[i] for i in kept)/len(kept))
    result.update(blocks=summary,block_macro_oa=sum(r["oa"] for r in summary.values())/len(summary),
                  leave_one_block=left_out,geometry_scope="fixed_readout_descriptive_removal")
    return result


def score_layer(x,layer,normalization,*,width,classes):
    columns = normalization["columns"]
    mean = torch.as_tensor(normalization["mean"],dtype=torch.float64)
    scale = torch.as_tensor(normalization["scale"],dtype=torch.float64)
    w,b = layer["weight"].cpu().double(),layer["bias"].cpu().double()
    require(len(columns)==width and sorted(set(columns))==columns and x.ndim==2
            and mean.shape==scale.shape==b.new_zeros(width).shape and w.shape==(classes,width)
            and b.shape==(classes,) and x.shape[1]==normalization["original_width"] and columns[-1]<x.shape[1]
            and normalization["parameters"]==classes*(width+1) and (scale>0).all().item(),
            "Readout does not have the fixed fitted width, original channels or class axis")
    values = (x.detach().cpu().double()[:,columns]-mean)/scale
    logp = (values @ w.T + b).log_softmax(1)
    require(torch.isfinite(logp).all().item(),"Frozen readout yielded nonfinite results")
    return logp


def role_score(role, view, expected, layers, truth, row_ids, blocks, weights, normalization, plan):
    """A readout can score a new approved view, carrying the original support fit unchanged."""
    require(role in ROLES and view["role"]==role and view["observation"]==expected["observation"]
            and view["model"]==expected["model"] and view["fit"]==expected["fit"]
            and view["population"]==expected["population"] and view.get("blocks")==expected.get("blocks")
            and view.get("approved_independent")==expected.get("approved_independent"),
            "Frozen readout crosses model, fit, role or observed release")
    require(len(truth)==len(row_ids)==len(set(row_ids)) and set(layers)==set(weights)==set(normalization)
            and digest([list(row_ids),list(truth)])==view["population"]
            and (digest(list(blocks)) if blocks is not None else None)==view.get("blocks"),
            "Unbound readout labels, block parents or feature roster")
    result,logp = {},{}
    for name,x in layers.items():
        require(len(x)==len(truth),"Named layer rows differ from the released class roster")
        logp[name] = score_layer(x,weights[name],normalization[name],width=plan["width"],classes=plan["classes"])
        prediction = (logp[name].argmax(1)+1).tolist()
        result[name] = dict(scores=block_metrics(truth,prediction,blocks,plan["classes"]),
                            decision=prediction,probability_sha256=fingerprint(logp[name]))
    return dict(role=role,independent=role=="evaluation" and view.get("approved_independent") is True,
                binding=view,layers=result),logp


def paired(a,b):
    left,right=a["binding"],b["binding"]
    require(a["role"]==b["role"] and all(left.get(k)==right.get(k) for k in ("model","fit","population","blocks"))
            and set(a["layers"])==set(b["layers"]),"Readout comparison needs an identical model, fit and ordered role")
    return {key:dict(before=info["scores"]["micro_oa"],after=b["layers"][key]["scores"]["micro_oa"],
        changes=sum(x!=y for x,y in zip(info["decision"],b["layers"][key]["decision"],strict=True)))
        for key,info in a["layers"].items()}


def old_readouts(root):
    links = {name:dict(path=str(root/file),sha256=sha) for name,(file,sha) in EVIDENCE.items()}
    accepted = json_pin(root,links["full54_acceptance"])
    legacy = json_pin(root,links["locked_support"])
    require(accepted["status"]=="passed" and accepted["accepted_pairs"]==54 and accepted["accepted_heads"]==432
            and legacy["query_accessed"] is legacy["raw_gt_loaded"] is False and legacy["real_updates"]==0,
            "Historical equal-capacity and view-specific sources changed")
    historical_signals = {name:json_pin(root,links[name]) for name in ("old_control","two_hop_learned")}
    source = dict(path=str(root/PAIRED/"receipt.json"),sha256=accepted["review_sha256"])
    review = json_pin(root,source)
    require(review["new_model_forwards"]==review["audit_probe_updates"]==0
            and review["raw_gt_deserialized"] is review["query_labels_accessed"] is False,"Foreign probe schedule")
    links["full54_review"] = source
    results = {}
    case_names = [f"{scene}-s{seed}-none-{method}" for scene in ("xuzhou","whu-honghu") for seed in SEEDS
                  for method in ("target_scratch_full","source_pretrained_full_finetune")]
    for case in case_names:
        require(case in review["outputs"] or case+".json" in review["outputs"],"Exact historical accepted probe case missing")
        outcomes = {}
        for stage in ("initial","terminal"):
            folder = root/"wp06-v2-paired-execution01"/case/stage
            files = {key:dict(path=str(folder/item),sha256=review["protected_files"][str(folder/item)]) for key,item in (
                ("pack","pack.pt"),("state","completed-0500/state.pt"),("report","completed-0500/report.json"),
                ("logp","selected-logp.pt"))}
            pack,state,expected = (torch.load(pinned(root,files[key]),map_location="cpu",weights_only=True)
                                   for key in ("pack","state","logp"))
            report = json_pin(root,files["report"])
            plan = validate_plan(report["plan"])
            require(plan["width"]==64 and plan["seed"] in SEEDS and plan["l2"]==1e-4
                    and plan["lr"]==.1 and state["cursor"]==plan["steps"]==500
                    and pack["identity"]["query_labels_accessed"] is False,"Different historical fit hyperparameters/roles")
            norms = {layer:report["layers"][layer]["normalization"] for layer in LAYERS}
            weights = {layer:state["layers"][layer]["best"] for layer in LAYERS}
            actual = {}
            for legacy_role,role in (("train","support"),("val","calibration")):
                y = (pack["labels"][legacy_role]+1).tolist()
                ix = pack["row_ids"][legacy_role].tolist()
                binding = dict(role=role, observation=files["pack"]["sha256"],
                    model=digest(pack["identity"]),fit=files["state"]["sha256"],population=digest([ix,y]))
                rows,logp = role_score(role,binding,binding,{k:pack["features"][k][legacy_role] for k in LAYERS},
                                       y,ix,None,weights,norms,plan)
                errors = {k:float((p-expected[k][legacy_role]).abs().max()) for k,p in logp.items()}
                require(max(errors.values())<=1e-12 and all(torch.equal(p.argmax(1),expected[k][legacy_role].argmax(1))
                        for k,p in logp.items()),"Independent frozen scores disagree with the accepted historic logits")
                actual[role] = dict(role=role,maximum_error=errors,layer_capacity=plan["classes"]*65,
                    descriptives={k:v["scores"] for k,v in rows["layers"].items()},held_out_claim=False)
            outcomes[stage] = dict(roles=actual,artifacts=files,plan=plan)
        results[case] = outcomes
    return dict(accepted_sources=links,selected_readouts=results,fixed_and_per_view={
                    case:legacy["cases"][case] for case in ("initial","selected","whu-honghu")},
                new_probe_updates=0,source_stage_differences="no_cross_model_subtraction",
                held_spatial_claim=False,approved_control_sources={k:list(v) for k,v in historical_signals.items()})
