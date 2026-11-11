"""External registration, frozen review and bounded recovery of matched route arms."""
from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from evaluation import matched_edge_review as review
from evaluation.linear_probe import fingerprint
from evaluation.registered_matched_route_audit import Population
from model.research_config import canonical_json, digest
from research_runtime import external_directory, require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.graph_route_context import route_state_hash
from transfer.matched_edge_retraining import MatchedRetrainingPlan


POLICY = "wp06_registered_matched_dynamic_route_v1"
ARMS = ("none","important","low_weight","random:64","random:512","random:2048")
SETTINGS = {"per_row":1,"distance_bins":[2.,4.,100.],"seeds":[64,512,2048],"time_limit":30.}
SEGMENTS = {"full":(0,24,None),"split-first":(0,7,None),"split-last":(7,24,"split-first"),
            "completed":(24,24,"split-last")}


def fresh(out,pins,*,create=True):
    out = external_directory(out)
    for pin in pins:
        parent = Path(pin["path"]).resolve(strict=True).parent
        admission.require(out!=parent and out not in parent.parents and parent not in out.parents,
                          "Output overlaps a protected input directory")
    if create:
        out.mkdir(parents=True,exist_ok=False)
    elif out.exists():
        raise FileExistsError("Fresh external attempt required")
    return out


def snapshot(run):
    return {"source":route_state_hash(run.source),"last":route_state_hash(run.model),
        "selected":fingerprint(run.selected_state),"history":digest(run.history),
        "optimizer":fingerprint(run.optimizer.state_dict()),"numpy_rng":digest(run.rng.bit_generator.state),
        "local_torch_rng":fingerprint(run.torch_rng),"global_torch_rng":fingerprint(torch.get_rng_state())}


def catalog_inventory(catalog_pin,out,code,*,allow_synthetic=False):
    require_frozen_code(code)
    candidates = admission.catalog(catalog_pin)
    rows = []
    for candidate in candidates:
        row = {"candidate":candidate}
        try:
            summary,*_ = admission.metadata(candidate)
            row["metadata"] = summary
            accepted,*_ = admission.authorize(candidate,code,allow_synthetic=allow_synthetic)
            row.update(status="eligible_for_controlled_load",metadata=accepted)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            row.update(status="rejected_before_checkpoint_load",reason=str(exc))
        rows.append(row)
    out = fresh(out,[catalog_pin])
    require_frozen_code(code)
    return dev._json_new(out/"inventory.json",{"policy":POLICY,"catalog":catalog_pin,"consumer":code["commit"],
        "candidates":rows,"checkpoint_loads":0,"model_forwards":0,"optimizer_updates":0,"query":0})


def choose(catalog_pin,name):
    rows = [row for row in admission.catalog(catalog_pin) if row["name"]==name]
    admission.require(len(rows)==1,"One named pinned catalog candidate required")
    return rows[0]


def arm_directory(arm):
    admission.require(arm in ARMS,"Unknown matched arm")
    return arm.replace(":","-")


def prepare(catalog_pin,name,out,code,*,allow_synthetic=False,settings=None):
    require_frozen_code(code)
    candidate = choose(catalog_pin,name)
    donor,run,original,pins = admission.restore(candidate,code,allow_synthetic=allow_synthetic)
    controls = deepcopy(SETTINGS if settings is None else settings)
    admission.require(set(controls)==set(SETTINGS) and controls["seeds"]==SETTINGS["seeds"],
                      "Six fixed control seeds and explicit geometry settings required")
    bins = controls["distance_bins"]
    admission.require(type(controls["per_row"]) is int and controls["per_row"]>=0
        and isinstance(bins,list) and 1<=len(bins)<=32
        and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in bins)
        and bins==sorted(set(bins)) and type(controls["time_limit"]) in (int,float)
        and math.isfinite(controls["time_limit"]) and 0<controls["time_limit"]<=60,
        "Register finite positive pixel-distance bins and a bounded solver deadline before any forward")
    out = fresh(out,[catalog_pin,*pins])
    # This record precedes the first model forward or solver call.
    all_tiles = len(donor["devtrain_tiles"])+len(donor["devval_tiles"])
    val_tiles = len(donor["devval_tiles"])
    budget = donor["budget"]
    prep = dev._json_new(out/"preflight.json",{"policy":POLICY,"catalog":catalog_pin,"candidate":name,"donor":donor,
        "consumer":code,"settings":controls,"arms":list(ARMS),"inputs":pins,
        "registered_limits":{"train_and_val_tiles":all_tiles,"devval_tiles":val_tiles,
            "planning_forwards":4*all_tiles+val_tiles,"review_forwards_maximum":7*val_tiles,
            "solver_calls_maximum":4*all_tiles,"solver_seconds_maximum":4*all_tiles*controls["time_limit"]},
        "retraining_draft":{"paths":{"full":[budget],"split":[7,budget] if budget>7 else None,"completed":[budget]},
            "continuous_updates":6*budget,"segmented_updates":6*budget,"completed_updates":0,
            "donor_original_updates":budget,"real_retraining_authorized":False,
            "estimated_time_basis":"Use the admitted donor's positive-update run and measured whole-context forward cost before a real allocation.",
            "optimization":"Same original source, exposure, optimizer, LR, seed, validation and first minimum NLL."},
        "optimizer_updates":0,"query":0})
    before = snapshot(run)
    started = time.monotonic()
    try:
        plan = MatchedRetrainingPlan.freeze(run,**controls)
        contents = plan.to_dict()
        review_plan = review.MatchedReviewPlan.freeze(run,**controls,
            frozen_controls={row["context"]["tile_id"]:row for row in contents["contexts"]})
        banks, unavailable = plan.banks(run,expected_hash=plan.plan_hash)
        arms = {}
        duplicates = {}
        for arm in ARMS:
            if arm in unavailable:
                arms[arm] = {"status":"unavailable","reasons":unavailable[arm]}
                continue
            folder = out/arm_directory(arm)
            folder.mkdir()
            bank = banks[arm]
            bank_pin = dev._json_new(folder/"bank.json",bank.to_dict())
            registration = dev.register_route(original["source"]["path"],original["development"],run.plan,
                folder/"registration.json",lr=run.spec["lr"],torch_seed=run.spec["torch_seed"],expected_commit=code["commit"],
                bank_path=bank_pin["path"],expected_bank_hash=bank.bank_hash)
            arms[arm] = {"status":"registered","bank":bank_pin,"bank_hash":bank.bank_hash,"registration":registration}
            if arm.startswith("random:"):
                duplicates.setdefault(bank.bank_hash,[]).append(arm)
        admission.require(before==snapshot(run),"Planning changed source, donor state, optimizer or RNG")
        admission.check_pins(pins)
        require_frozen_code(code)
        review_pin = dev._json_new(out/"review-plan.json",review_plan.to_dict())
        retraining_pin = dev._json_new(out/"retraining-plan.json",contents)
        return dev._json_new(out/"scope.json",{"policy":POLICY,"status":"registered" if not unavailable else "registered_incomplete",
            "consumer_commit":code["commit"],"consumer_source_hash":code["source_hash"],"catalog":catalog_pin,
            "candidate":name,"preflight":prep,"review_plan":review_pin,"review_plan_hash":review_plan.plan_hash,
            "retraining_plan":retraining_pin,"retraining_plan_hash":plan.plan_hash,"arms":arms,
            "duplicate_random_banks":[names for names in duplicates.values() if len(names)>1],
            "synthetic":donor["synthetic"],"budget":budget,"query":0,"optimizer_updates":0,
            "source_and_donor_unchanged":True,"wall_seconds":time.monotonic()-started})
    except BaseException as error:
        dev._json_new(out/"preparation-failure.json",{"status":"failed","error":repr(error),"preflight":prep,
            "optimizer_updates":0,"query":0,"wall_seconds":time.monotonic()-started})
        raise


def load_scope(scope_pin,code):
    require_frozen_code(code)
    scope = admission.read(scope_pin)
    preflight = admission.read(scope["preflight"])
    admission.require(scope["policy"]==preflight["policy"]==POLICY
        and scope["consumer_commit"]==code["commit"] and scope["consumer_source_hash"]==code["source_hash"]
        and scope["catalog"]==preflight["catalog"] and scope["candidate"]==preflight["candidate"]
        and set(scope["arms"])==set(ARMS) and scope["optimizer_updates"]==scope["query"]==0
        and scope["budget"]==preflight["donor"]["budget"] and scope["synthetic"] is preflight["donor"]["synthetic"],
        "Prepared scope, candidate, budget or consumer changed")
    for key in ("review_plan","retraining_plan"):
        admission.check_pins([scope[key]])
    for arm in scope["arms"].values():
        if arm["status"]=="registered":
            registration = admission.read(arm["registration"])
            original = preflight["donor"]
            admission.require(registration["bank"]==arm["bank"] and registration["bank_hash"]==arm["bank_hash"]
                and registration["source"]==original["source"] and registration["metadata"]==original["source_metadata"]
                and registration["development"]==original["development"] and registration["plan"]["plan_hash"]
                    ==original["exposure_plan_hash"] and registration["torch_seed"]==original["model_seed"]
                and registration["expected_commit"]==code["commit"] and registration["lr"]==original["lr"]
                and digest({k:v for k,v in registration["spec"].items() if k!="graph_route"})
                    ==original["fixed_training_spec_sha256"],"Arm source, exposure, bank or consumer changed")
            admission.check_pins([arm["bank"]])
        else:
            admission.require(arm["status"]=="unavailable" and arm["reasons"],"Unavailable arm requires a reason")
    return scope,preflight


def frozen_review(scope_pin,out,code):
    scope,preflight = load_scope(scope_pin,code)
    candidate = choose(scope["catalog"],scope["candidate"])
    donor,run,_,pins = admission.restore(candidate,code,allow_synthetic=scope["synthetic"])
    admission.require(donor==preflight["donor"],"Donor admission differs from the preflight")
    out = fresh(out,[scope_pin,*pins])
    before = snapshot(run)
    plan = review.MatchedReviewPlan(admission.read(scope["review_plan"]),expected_hash=scope["review_plan_hash"])
    observer = Population(run)
    try:
        original = review.review(run,plan,expected_plan_hash=plan.plan_hash,observe=observer.observe)
        independent = observer.finish(original)
        admission.require(before==snapshot(run),"Frozen review changed source, donor state, optimizer or RNG")
        admission.check_pins(pins)
        require_frozen_code(code)
        complete = all(arm["status"]=="evaluated" for arm in independent.values())
        return dev._json_new(out/"review.json",{"policy":POLICY,
            "status":"passed" if complete and scope["status"]=="registered" else "incomplete_or_failed",
            "validation_status":"passed" if complete else "incomplete_or_failed",
            "scope":scope_pin,"synthetic":scope["synthetic"],"donor":donor,"original":original,"independent":independent,
            "training_bank_status":{arm:value["status"] for arm,value in scope["arms"].items()},
            "duplicate_random_banks":scope["duplicate_random_banks"],"state_and_rng_unchanged":True,
            "optimizer_updates":0,"query":0})
    except BaseException as error:
        dev._json_new(out/"review-failure.json",{"status":"failed","scope":scope_pin,"error":repr(error),
            "optimizer_updates":0,"query":0})
        raise


def claim_path(scope_pin,arm,segment):
    directory = Path(scope_pin["path"]).resolve().parent.parent/".wp06-matched-route-claims"/scope_pin["sha256"]
    target = directory/(arm_directory(arm)+"-"+segment+".json")
    admission.require(target.resolve()==target,"Allocation path escapes registered directory")
    return target


def result_pin(claim):
    return dev._pin(claim.with_suffix(".result.json"))


def read_segment(scope_pin,arm,segment,scope):
    claim = claim_path(scope_pin,arm,segment)
    proof = admission.read(result_pin(claim))
    allocation = admission.read(proof["allocation"])
    admission.require(proof["allocation"]["path"]==str(claim)
        and allocation["scope"]==scope_pin and (allocation["arm"],allocation["segment"])==(arm,segment),
        "Cross-arm, cross-bank or cross-scope restore rejected")
    start,stop,_ = SEGMENTS[segment]
    status = admission.read(proof["attempt"])
    admission.require(status["registration"]==scope["arms"][arm]["registration"]
        and status["initial_updates"]==start and status["completed_updates"]==status["target_updates"]==stop
        and status["new_updates"]==stop-start and status["query"]==0,
        "Stored attempt has wrong arm or update interval")
    return status,proof


def execute(scope_pin,arm,segment,out,code,*,resume=None):
    scope,preflight = load_scope(scope_pin,code)
    admission.require(scope["synthetic"] is True and scope["budget"]==24,
                      "This scope authorizes only the explicit synthetic 24-step recovery")
    admission.require(segment in SEGMENTS and arm in ARMS and scope["arms"][arm]["status"]=="registered",
                      "A registered feasible arm and exact segment required")
    start,stop,previous = SEGMENTS[segment]
    if previous is not None:
        earlier,proof = read_segment(scope_pin,arm,previous,scope)
        admission.require(resume==proof["attempt"],"Cross-arm, cross-bank or cross-scope restore rejected")
        checkpoint = earlier["checkpoint"]
    else:
        admission.require(resume is None,"Fresh full or split-first segment must restart from original source")
        checkpoint = None
    pins = [scope_pin,*preflight["inputs"],scope["arms"][arm]["registration"]]
    out = fresh(out,pins,create=False)
    claim = claim_path(scope_pin,arm,segment)
    claim.parent.mkdir(parents=True,exist_ok=True)
    allocation = dev._json_new(claim,{"scope":scope_pin,"arm":arm,"segment":segment,"out":str(out),
        "registration":scope["arms"][arm]["registration"],"bank_hash":scope["arms"][arm]["bank_hash"],
        "start":start,"stop":stop,"resume_attempt":resume})
    try:
        attempt = dev.execute_route(scope["arms"][arm]["registration"],out=out,to_step=stop,
            expected_commit=code["commit"],resume=checkpoint)
    except BaseException as error:
        path = out/"status.json"
        dev._json_new(claim.with_suffix(".result.json"),{"allocation":allocation,"status":"failed",
            "attempt":dev._pin(path) if path.is_file() else None,"error":repr(error)})
        raise
    proof = dev._json_new(claim.with_suffix(".result.json"),{"allocation":allocation,"status":attempt["status"],
        "attempt":dev._pin(out/"status.json")})
    read_segment(scope_pin,arm,segment,scope)
    return proof


def audit_recovery(scope_pin,out,code):
    scope,preflight = load_scope(scope_pin,code)
    admission.require(scope["synthetic"],"This module has no real retraining allocation")
    donor,run,_,pins = admission.restore(choose(scope["catalog"],scope["candidate"]),code,allow_synthetic=True)
    admission.require(donor==preflight["donor"],"Donor admission differs")
    plan = MatchedRetrainingPlan(admission.read(scope["retraining_plan"]),expected_hash=scope["retraining_plan_hash"])
    out = fresh(out,[scope_pin,*pins])
    rows = {}
    started = time.monotonic()
    for arm,value in scope["arms"].items():
        if value["status"]!="registered":
            rows[arm] = {"status":"unavailable","reasons":value["reasons"]}
            continue
        try:
            attempts = {key:read_segment(scope_pin,arm,key,scope)[0] for key in SEGMENTS}
            prefix = attempts["split-first"]["checkpoint"]
            complete = {}
            for key in ("full","split-last","completed"):
                ckpt = attempts[key]["checkpoint"]
                restored = plan.restore(run,arm,ckpt["path"],expected_hash=plan.plan_hash,expected_sha256=ckpt["sha256"])
                complete[key] = {"state":snapshot(restored),"acceptance":restored.finish()}
            admission.require(len({digest(value) for value in complete.values()})==1,
                              "Full, split and complete trajectories differ")
            rows[arm] = {"status":"accepted","updates":{key:value["new_updates"] for key,value in attempts.items()},
                "selected_step":complete["full"]["acceptance"]["selected_step"],"prefix":prefix,
                "scientific_fingerprint":digest(complete["full"]),"attempts":{key:value["checkpoint"] for key,value in attempts.items()}}
        except (ValueError,KeyError,TypeError,OSError) as error:
            rows[arm] = {"status":"incomplete_or_failed","reason":str(error)}
    accepted = all(value["status"]=="accepted" for value in rows.values())
    admission.check_pins(pins)
    require_frozen_code(code)
    receipt = dev._json_new(out/"receipt.json",{"policy":POLICY,"status":"passed" if accepted else "incomplete_or_failed",
        "scope":scope_pin,"synthetic":True,"arms":rows,"new_optimizer_updates_during_audit":0,
        "query":0,"wall_seconds":time.monotonic()-started})
    if accepted:
        dev._json_new(out/"acceptance.json",{"policy":POLICY,"status":"passed","scope":scope_pin,
            "receipt":receipt,"synthetic":True,"new_optimizer_updates":288,"completed_reload_updates":0,"query":0})
    return receipt
