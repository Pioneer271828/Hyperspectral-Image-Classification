"""Independent update, exposure and 2x2 effect checks for synthetic networks."""

from collections import defaultdict
from copy import deepcopy
import math

from compression.precision_recovery_audit import HEADS, independent_predictions
from compression.registered_precision_recovery import stages
from model.research_config import digest
from research_runtime import require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_factorial import ARMS, FactorialRun, REAL_PROPOSAL, load_registration
from transfer.curriculum_real import PACE
from transfer.curriculum_real_audit import audit_exposure
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import read_stage
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact, _packet, _verify_history


def effects(rows):
    """Paired contrasts within one fixed checkpoint protocol and metric."""
    keys = {"arm","source","labels","exposure","budget","metric","seed","condition","value"}
    if (not isinstance(rows,list) or len(rows) != 4 or any(not isinstance(r,dict) or set(r) != keys for r in rows)
            or {r["arm"] for r in rows} != ARMS.keys()):
        raise ValueError("Exactly four complete factorial arms required")
    identity = {k:rows[0][k] for k in keys-{"arm","value"}}
    if (type(identity["budget"]) is not int or identity["budget"] < 1
            or any({k:r[k] for k in identity} != identity or type(r["value"]) not in (float,int)
                   or not math.isfinite(r["value"]) for r in rows)):
        raise ValueError("Factorial sources, labels, support sequence, budgets or metric differ")
    a,b,c,d = (next(r["value"] for r in rows if r["arm"] == k) for k in ARMS)
    return dict(sample_at_full=b-a,sample_at_restricted=d-c,observation_without_sample=c-a,
                observation_with_sample=d-b,interaction=(d-c)-(b-a))


def condition_exposure(history,policy,record,full_bands):
    observations = None if policy is None else policy.to_dict()["training"]
    support = dict(zip(record["train_support"]["indices"],record["train_support"]["labels"],strict=True))
    cells = {}
    for index,row in enumerate(history):
        if observations is None:
            key, bands = "full_unrestricted", full_bands
        else:
            item = observations[index]
            if row["observation"] != policy.training_record(index+1) or row["forward_evidence"]["band_ids"] != item["selected_band_ids"]:
                raise ValueError("Actual factorial observation differs from the registered execution")
            key,bands = digest(item["config"]),row["forward_evidence"]["band_count"]
        if key not in cells:
            cells[key] = dict(forwards=0,band_exposures=0,by_class=defaultdict(list),by_support=defaultdict(list))
        entry = cells[key]
        entry["forwards"] += 1
        entry["band_exposures"] += bands
        for i,w in zip(row["support_ids"],row["applied_weights"],strict=True):
            entry["by_support"][str(i)].append(w)
            entry["by_class"][str(support[i])].append(w)
    for cell in cells.values():
        for kind in ("by_class","by_support"):
            cell[kind] = {key:dict(raw=len(weights),effective=math.fsum(weights)) for key,weights in cell[kind].items()}
    return cells


def audit(pin,completion,code):
    root,reg,source,scene,record,plan,trials = load_registration(pin,code)
    end = admission.read(completion)
    if (completion["path"] != str(root/"batch-complete.json") or end["registration"] != pin
            or end["updates"] != reg["maximum_updates"] or any(end[k] for k in ("optimizer_updates","query","gt"))
            or end["real_training"] != "NOT_RUN"):
        raise ValueError("Expected exact completed synthetic factorial receipt")
    intervals = stages(reg["budget"],reg["prefix"])
    teacher = admission.read(reg["teacher"])
    results, joint = {}, {}
    optimizer = 0
    for arm,(_,sample,obs,_) in trials.items():
        packets,receipts = {},{}
        for stage,(_,stop,previous) in intervals.items():
            s,r = read_stage(root,pin,arm,stage,receipts.get(previous),arms=trials,stages=intervals)
            if end["attempts"][arm][stage] != r:
                raise ValueError("Recorded factorial stage differs from completion")
            receipts[stage] = r
            packets[stage] = _packet(s,reg["contracts"][arm],stop,packet_format=FactorialRun.FORMAT)
            optimizer += s["new_updates"]
        full = packets["full"]
        if (not all(_exact(full[k],packets[other][k]) for k in SCIENCE for other in ("resume","completed"))
                or not _exact(full["history"][:reg["prefix"]],packets["prefix"]["history"])):
            raise ValueError("Factorial continuation differs")
        step = _verify_history(full,plan)
        exposure = audit_exposure(full["history"],plan,teacher,"easy_to_hard" if sample else "off",
                                  dict(PACE,full_at=plan.updates))
        joint[arm] = condition_exposure(full["history"],obs,record,len(scene.bands.band_ids))
        model = deepcopy(source).eval()
        model.load_state_dict(full["selected_state"],strict=True)
        selected = independent_predictions(model,scene,record)
        fused = selected["metrics"]["fused"]
        trained = full["history"][step-1]["validation"]
        if (state_hash(model) != full["history"][step-1]["state_hash"] or fused["confusion_matrix"] != trained["confusion"]
                or not math.isclose(fused["nll"],trained["nll"],abs_tol=1e-7)):
            raise ValueError("Factorial selected model differs from full V5 history")
        results[arm] = dict(step=step,metrics=selected["metrics"],scores=selected["scores"],
                            exposure=exposure,state=state_hash(model),history_sha256=digest(full["history"]))
    if optimizer != reg["maximum_updates"]:
        raise ValueError("Synthetic factorial update count differs")
    def raw(value):
        return {k:{**c,**{g:{i:w["raw"] for i,w in c[g].items()} for g in ("by_class","by_support")}}
                for k,c in value.items()}
    if raw(joint["o_only"]) != raw(joint["s_plus_o"]) or raw(joint["none"]) != raw(joint["s_only"]):
        raise ValueError("Factorial sample weighting changed original support or observation access")
    contrasts = {}
    for head in HEADS:
        contrasts[head] = {}
        for metric in ("oa","aa","kappa","macro_f1","nll"):
            contrasts[head][metric] = effects([dict(arm=arm,source=reg["source_state"],labels=digest(record["validation_support"]),
                 exposure=reg["exposure_hash"],budget=reg["budget"],metric=metric,seed=plan.to_dict()["seed"],
                 condition="full_development_v5",value=value["metrics"][head][metric]) for arm,value in results.items()])
    admission.check_pins([pin,completion,reg["teacher"],*reg["protected"]])
    require_frozen_code(code)
    return dev._json_new(root/"audit.json",dict(status="passed_synthetic_factorial",scope="synthetic",producer=reg["commit"],
        registration=pin,completion=completion,fixture_classes=scene.classes,budget=reg["budget"],
        synthetic_updates=optimizer,optimizer_updates=0,query=0,gt=0,results=results,joint_exposure=joint,effects=contrasts,
        restricted_effective_exposure_differs=joint["s_plus_o"] != joint["o_only"],real_proposal=REAL_PROPOSAL))
