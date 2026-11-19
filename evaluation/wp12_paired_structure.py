"""Independently join train, recovered selected weights and WP12 confirmation receipts."""

from collections import defaultdict

import numpy as np

from evaluation import wp12_cross_backend as comparator, wp12_scientific_claim as claims
from evaluation.wp12_support_contrasts import summarize as support_summary
from evaluation.frozen_training_stages import file_pin, json_pin, require
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent, wp_cross_backend as model_io
from transfer.wp_paired_structure import StructuralProvider, open_plan
from transfer.wp_paired_structure_run import PUBLICATION, completed
from transfer.xuzhou_multiscale_audit import _exact

SCHEMA = "wp12_paired_structure_acceptance_v1"


def observed_parity(root,campaign,plan,code,group,arm,spec,packet,raw,execution,*,test):
    provider = StructuralProvider(root,campaign,plan,group,arm,code,test=test)
    model = provider.model_factory(provider.baseline(spec["seed"]),arm,spec["module_seed"])
    model.load_state_dict(packet["selected"]["state"],strict=True)
    model.to(execution.device).eval().requires_grad_(False)
    entry = plan["inputs"][plan["groups"][group]["release"]]
    n,error,changes = 0,0.,0
    for stored in raw["rows"]:
        capture = stored["accuracy"]
        if capture is None:
            continue
        actual,_,targets = independent.materialize_reviewed(provider.bank,entry["authority"],provider.roles,
            "evaluation",stored["tile"],code["commit"],stored["view"],use="independent_accuracy",factor=spec["factor"])
        heads,_,_ = model_io.run_block(model,actual,code["commit"],"devval",execution)
        replay = comparator.emitted_head_scores(heads,targets,provider.classes)
        require(capture["observation"]==actual.receipt["observation"],"Independent reload observed a different Bank")
        for name,value in replay.items():
            a,b = np.asarray(value),np.asarray(capture["heads"][name])
            error = max(error,float(np.max(np.abs(a-b))))
            changes += int((a.argmax(1)!=b.argmax(1)).sum())
            n += len(a)
    require(n>0 and error<=3e-5 and changes==0,
            "Selected training checkpoint and the cross-model consumer differ on released points")
    return dict(point_heads=n,max_abs_probability=error,changed_predictions=changes)


def results_by_pair(cohort, descriptors, scene, *, protocol, policy, expected_views):
    for pair in descriptors:
        yield claims.paired_result(pair,cohort,scene,protocol,policy,expected_views)


def review(root,campaign,out,code,captures,execution,*,test):
    plan,reg = open_plan(root,campaign,code,test=test)
    marker = file_pin(campaign/"publication.json")
    publication = json_pin(root,marker)
    require(publication["schema"]==PUBLICATION and publication["producer"]==code["commit"] and
            publication["campaign"]==reg and publication["scope"]==plan["scope"]
            and publication["optimizer_updates"]==plan["optimizer_updates"]
            and set(publication["releases"])==set(plan["releases"])==set(captures)
            and set(publication["cases"])==set(plan["groups"]),"Need complete frozen publication and captures")
    out.mkdir(parents=True)
    reviews,readouts,raw_captures = {},{},{}
    for release,claim in captures.items():
        raw = json_pin(root,claim)
        require(raw["request"]==publication["releases"][release],"Capture refers to another trained publication")
        raw_captures[release] = raw
        reviews[release] = comparator.review_case(root,claim,out/(release+".json"),code,
                                                  test_history=[] if test else None)
        readouts[release] = json_pin(root,reviews[release])
    all_cases,parity,recovery,closed = {},{}, {},{}
    candidates = defaultdict(list)
    coverage = set()
    for name,group in plan["groups"].items():
        release = group["release"]
        raw = raw_captures[release]["cohorts"][name]
        result = readouts[release]["cohorts"][name]
        proofs = {}
        require(set(publication["cases"][name])==set(raw["arms"])==set(group["arms"]),"Frozen arm inventory differs")
        for arm,descriptor in group["arms"].items():
            link,record,spec,packet = completed(root,campaign,name,arm,reg,group)
            frozen_raw = raw["arms"][arm]
            frozen = json_pin(root,frozen_raw["endpoint"])
            require(publication["cases"][name][arm]==link and frozen["step"]==record["selected"]
                    and frozen["calibration"]==record["calibration"]
                    and frozen["checkpoint"]["state"]==next(v["state_sha256"] for v in packet["validation"]
                                                            if v["step"]==record["selected"]),
                    "WP12 did not consume the selected state from this training budget")
            identity = (descriptor["backend"],group["factor"],descriptor.get("policy"))
            if identity not in coverage and group["purpose"]=="science":
                coverage.add(identity)
                parity[f"{name}/{arm}"] = observed_parity(root,campaign,plan,code,name,arm,spec,packet,frozen_raw,
                                                          execution,test=test)
            if test and name in ("xz-recovery","xz-column-recovery") and arm=="combined":
                other = completed(root,campaign,name,arm,reg,group,reference=True)
                require(_exact(packet,other[3]) and len(record["timings"])>0,"Full state differed after process recovery")
                recovery[f"{name}/{arm}"] = dict(reference=other[0],recovered=link,
                                                exact=["model","adam","rng","history","first_minimum","selected_state"])
            if test:
                require(record["weights_changed"]>0,"Engineering optimization did not change any registered weights")
            proofs[arm] = record
        if "fixed1" in proofs:
            a,b = (completed(root,campaign,name,arm,reg,group)[3] for arm in ("off","fixed1"))
            require(_exact(a["model"],b["model"]) and _exact(a["optimizer"],b["optimizer"])
                    and a["selected"]["step"]==b["selected"]["step"]
                    and _exact(a["selected"]["state"],b["selected"]["state"]),
                    "Fixed dilation one did not preserve the original training trajectory")
            closed[name] = dict(pair=["off","fixed1"],exact_model_and_adam=True,selected=proofs["off"]["selected"])
        all_cases[name] = proofs
        if group["purpose"]=="science":
            intake = plan["inputs"][release]["authority"]
            science = plan["releases"][release]
            observation = science.get("observation_protocol",claims.PHYSICAL)
            policy = science.get("claim_policy")
            expected_views = tuple(intake["observation_views"])
            by_pair = {f'{p["baseline"]}/{p["candidate"]}':p for p in group["pairs"]}
            for key,value in results_by_pair(result,group["pairs"],readouts[release]["scene"],
                                            protocol=observation,policy=policy,expected_views=expected_views):
                pair = by_pair[key]
                experiment = digest([group["steps"],group["clock"],observation,expected_views,
                    group["arms"][pair["baseline"]],group["arms"][pair["candidate"]],pair["comparison"]])
                candidates[(release,group["factor"],key,experiment,expected_views)].append(
                    dict(seed=group["seed"],**value))
    decisions = {}
    for (release,factor,key,experiment,expected_views),scores in candidates.items():
        authentic = (not test and plan["scope"]=="external_real_execution"
                     and readouts[release]["scope"]=="external_real"
                     and readouts[release]["uses"]["independent_accuracy"]["eligible"])
        decisions[f"{release}/factor{factor}/{key}/{experiment}"] = claims.conclusion(
            scores,expected_views,authentic_real=authentic)
    science_steps = sum(g["steps"]*len(g["arms"]) for g in plan["groups"].values() if g["purpose"]=="science")
    controls = None
    if test:
        from transfer.wp_paired_structure_fixture import controls as planted

        controls = {name:planted(root,record,code["commit"]) for name,record in plan["releases"].items()}
    require_frozen_code(code)
    return _write(out/"review.json",dict(schema=SCHEMA,producer=code["commit"],registration=reg,publication=marker,
        scope=plan["scope"],classification=reviews,arms=all_cases,parity=parity,recovery=recovery,closed=closed,
        criteria=decisions,controls=controls,support_contrasts=support_summary(plan,raw_captures,readouts,all_cases),
        planned_updates=plan["optimizer_updates"],science_updates=science_steps,
        qualification_updates=plan["optimizer_updates"]-science_steps,
        real_optimizer_updates=0 if test else plan["optimizer_updates"],
        query_accessed=False,raw_gt_loaded=False,wp04_executed=False))
