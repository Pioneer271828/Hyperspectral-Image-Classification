"""Real Xuzhou route registration, donor acceptance and all-arm admission."""

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.graph_edge_intervention import _selected
from loadData.spatial_tiles import tile_slice
from model.research_config import canonical_json, digest
from research_runtime import external_directory, require_frozen_code
from scripts.run_wp02_route_preflight import _source
from transfer import matched_route_admission as admission, registered_development as dev
from transfer import registered_matched_route as matched, route_campaign_allocation as allocation
from transfer import wp02_route_development as wp02
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import sparse_labels
from transfer.graph_route_context import route_state_hash
from transfer.graph_route_training import GraphRouteRun


POLICY = "wp06_xuzhou_real_route_campaign_v1"
DESIGN = "bb1996c124de03a1d8a819aa21147002913bb88e7778c26a915b9d19a57164b3"
PREFIX_AUDIT = "a5af1055c5a4d3711eecefc44e3e7d18d957a766d4cf2c458e9f6bce8e458276"
BUDGETS = (64,128,256)
CONTROL = "wp06-xuzhou-real-campaign-control01"
ROOT = "wp06-xuzhou-real-route-campaign01"
PRIOR = "wp06-xuzhou-route-preflight01"
MODEL_SEED = EXPOSURE_SEED = 64


def read(pin):
    return admission.read(pin)


def budget_prefixes(scene, record):
    longest = ExposurePlan.freeze(scene,record,updates=BUDGETS[-1],seed=EXPOSURE_SEED,
                                  minimum_class_blocks={str(c):1 for c in record["classes"]},
                                  minimum_support_visits=1)
    rows = []
    for count in BUDGETS:
        progress = longest.progress(count)
        stats = progress["exposure"]
        rows.append({"budget":count,"visited_tiles":sum(n>0 for n in stats["tile_counts"].values()),
                     "missing_support":sum(n>0 for n in stats["support_deficits"].values()),
                     "missing_classes":sum(n>0 for n in stats["class_deficits"].values()),
                     "minimum_visits":min(stats["support_visits"].values()),
                     "prefix_hash":progress["prefix_hash"],"eligible":stats["meets_minimum"]})
    chosen = next((r["budget"] for r in rows if r["eligible"]),None)
    if chosen is None:
        return rows,None
    plan = ExposurePlan.freeze(scene,record,updates=chosen,seed=EXPOSURE_SEED,
                               minimum_class_blocks={str(c):1 for c in record["classes"]},minimum_support_visits=1)
    admission.require(plan.to_dict()["events"]==longest.to_dict()["events"][:chosen],
                      "Budget must use the unchanged original sampler prefix")
    plan.require_eligible()
    return rows,plan


def _prior_inputs(design_pin):
    path = Path(design_pin["path"]).resolve(strict=True)
    admission.require(path==Path(design_pin["path"]) and path.parent.name==CONTROL and path.name=="design.json"
                      and design_pin["sha256"]==DESIGN,"This campaign requires the frozen external design")
    design = read(design_pin)
    runs = path.parent.parent
    prior = runs/PRIOR
    pins = {"design":design_pin,
            "bridge":{"path":str(prior/"preflight.json"),"sha256":design["previous_preflight"]},
            "bridge_audit":{"path":str(prior/"audit.json"),"sha256":design["previous_audit"]},
            "bridge_blocked":{"path":str(prior/"blocked.json"),"sha256":design["previous_blocked"]}}
    bridge,audit,blocked = [read(pins[name]) for name in ("bridge","bridge_audit","bridge_blocked")]
    admission.require(audit["status"]=="passed_bridge_and_exposure_block" and audit["blocked"]==pins["bridge_blocked"]
        and blocked["preflight"]==pins["bridge"] and blocked["probe_forwards"]==2
        and audit["labels_compared"]==90 and audit["optimizer_updates"]==blocked["optimizer_updates"]==0
        and audit["query_accessed"] is audit["raw_gt_accessed"] is False
        and bridge["runtime"]==admission.runtime_identity(), "Prior independent bridge or runtime does not match")
    return runs, pins, bridge, blocked


def _allocation(code,budget,registrations):
    return {"policy":allocation.POLICY,"commit":code["commit"],"source_hash":code["source_hash"],
            "budget":budget,"registrations":registrations,"routes":allocation.routes(budget,registrations)}


def register(design_pin, support_audit_pin, code, *, issue):
    require_frozen_code(code)
    runs, pins, bridge, blocked = _prior_inputs(design_pin)
    prior_prefix = _support_audit(runs,support_audit_pin)
    admission.require(
        prior_prefix["design_sha256"]==DESIGN and prior_prefix["model_forwards"]==prior_prefix["optimizer_updates"]==0,
        "Independent support-only audit must belong to the frozen design")
    admission.require(type(issue) is int and issue>0,"An actual source Issue is required")
    out = external_directory(runs/ROOT)
    out.mkdir(exist_ok=False)
    (out/"inputs").mkdir()
    scene,record,files = dev.load_development(bridge["development"])
    config,upstream,_,_,_,_ = wp02.accepted_inputs(bridge["upstream"]["registration"],bridge["upstream"]["acceptance"])
    admission.require(record["wp02"]==upstream and bridge["upstream"]==upstream and scene.dataset==wp02.SCENE
        and len(record["train_support"]["indices"])==len(record["validation_support"]["indices"])==45
        and [sum(r["role"]==role for r in record["roles"]) for role in ("devtrain","devval")]==[73,5],
        "Original Xuzhou population, shape or WP02 release differs")
    rows,plan = budget_prefixes(scene,record)
    budget = None if plan is None else plan.updates
    admission.require(rows==prior_prefix["rows"] and budget==prior_prefix["chosen_budget"],
                      "Official exposure and independent support-only prefix results differ")
    prefix = dev._json_new(out/"inputs"/"prefixes.json",{"rows":rows,"chosen_budget":budget,"design":design_pin,
        "independent":support_audit_pin,"optimizer_updates":0,"model_forwards":0})
    if plan is None:
        return dev._json_new(out/"campaign.json",{"policy":POLICY,"status":"blocked_exposure","prefixes":prefix})
    model = dev._load_source(bridge["source"],bridge["source_sidecar"])
    reconstructed = _source(config)
    admission.require(route_state_hash(model)==route_state_hash(reconstructed)==bridge["source_state"]
        and model.export_config()==reconstructed.export_config() and model.training_config==reconstructed.training_config,
        "Original fresh GroupNorm initialization or architecture differs")
    registration = dev.register_route(bridge["source"]["path"],bridge["development"],plan,
        out/"inputs"/"donor.json",lr=bridge["donor_contract"]["lr"],torch_seed=MODEL_SEED,expected_commit=code["commit"])
    train_time,validation_time = [row["seconds"] for row in blocked["timing"]]
    pure_forward = 2*budget*(train_time+5*validation_time)
    steps_with_reverse_estimate = 2*budget*(3*train_time+5*validation_time)
    all_pins = {x["path"]:x for x in [*pins.values(),support_audit_pin,prefix,registration,bridge["source"],
                                     bridge["source_sidecar"],*upstream.values(),*files]}
    admission.check_pins(all_pins.values())
    receipt = dev._json_new(out/"campaign.json",{"campaign_policy":POLICY,"status":"registered","issue":issue,
        "root":str(out),"inputs":all_pins,"prior":pins,"prefixes":prefix,"runtime":admission.runtime_identity(),
        "source":bridge["source"],"source_metadata":bridge["source_sidecar"],"development":bridge["development"],
        "source_state":bridge["source_state"],"exposure_hash":plan.plan_hash,"controls":deepcopy(matched.SETTINGS),
        "authorized_updates":{"donor":2*budget,"six_arms_conditional":12*budget,"total_maximum":14*budget,
                              "completed_reload":0},
        "future_arm_routes":allocation.routes(budget,allocation.ARM_NAMES),
        "estimates":{"donor_training_and_validation_forwards":12*budget,
                     "donor_forward_only_seconds":pure_forward,
                     "donor_forward_and_three_times_train_estimate_seconds":steps_with_reverse_estimate,
                     "prior_forward_measurement_reused":True,"checkpoint_and_audit_io_measured":False,
                     "matched_planning_forward_bound":317,"frozen_review_forward_bound":35,
                     "solver_calls_bound":312,"solver_seconds_bound":9360},
        **_allocation(code,budget,{"donor":registration})})
    dev._json_new(out/"campaign-seal.json",{"campaign":receipt,"budget":budget,"maximum_updates":14*budget})
    return receipt


def _support_audit(runs,pin):
    admission.require(Path(pin["path"])==runs/CONTROL/"published-support-prefix-audit.json"
                      and pin["sha256"]==PREFIX_AUDIT,"Expected the frozen independent prefix audit")
    return read(pin)


def load_campaign(campaign_pin,code):
    require_frozen_code(code)
    campaign = read(campaign_pin)
    root = Path(campaign["root"])
    runs, pins, bridge, _ = _prior_inputs(campaign["prior"]["design"])
    budget = campaign["budget"]
    expected = _allocation(code,budget,{"donor":campaign["registrations"]["donor"]})
    seal = read(dev._pin(root/"campaign-seal.json"))
    admission.require(root==runs/ROOT and Path(campaign_pin["path"])==root/"campaign.json"
        and seal=={"campaign":campaign_pin,"budget":budget,"maximum_updates":14*budget}
        and root.resolve()==root and campaign["status"]=="registered" and budget in BUDGETS
        and campaign["campaign_policy"]==POLICY and campaign["prior"]==pins
        and campaign["runtime"]==admission.runtime_identity()
        and campaign["source"]==bridge["source"] and campaign["source_metadata"]==bridge["source_sidecar"]
        and campaign["development"]==bridge["development"] and campaign["source_state"]==bridge["source_state"]
        and campaign["authorized_updates"]=={"donor":2*budget,"six_arms_conditional":12*budget,
                                            "total_maximum":14*budget,"completed_reload":0}
        and all(campaign[k]==v for k,v in expected.items())
        and campaign["future_arm_routes"]==allocation.routes(budget,allocation.ARM_NAMES)
        and campaign["controls"]==matched.SETTINGS,"Real campaign identity, runtime or authorized budget differs")
    prefix = read(campaign["prefixes"])
    independent = _support_audit(runs,prefix["independent"])
    admission.require(prefix["design"]==pins["design"] and prefix["chosen_budget"]==budget==independent["chosen_budget"]
        and prefix["rows"]==independent["rows"]
        and next((r["budget"] for r in prefix["rows"] if r["eligible"]),None)==budget,
        "Campaign budget differs from the accepted exposure prefix")
    registration_pin = campaign["registrations"]["donor"]
    registration = read(registration_pin)
    admission.require(Path(registration_pin["path"])==root/"inputs"/"donor.json"
        and registration["source"]==campaign["source"] and registration["metadata"]==campaign["source_metadata"]
        and registration["development"]==campaign["development"] and registration["bank"] is None
        and registration["expected_commit"]==code["commit"] and registration["torch_seed"]==MODEL_SEED
        and registration["lr"]==bridge["donor_contract"]["lr"]
        and registration["spec"]["source_state"]==campaign["source_state"] and registration["plan"]["updates"]==budget
        and registration["plan"]["seed"]==EXPOSURE_SEED and registration["plan"]["minimum_support_visits"]==1
        and registration["plan"]["minimum_class_blocks"]=={str(c):1 for c in wp02.CLASSES}
        and digest(registration["plan"]["events"])==next(r["prefix_hash"] for r in prefix["rows"] if r["budget"]==budget)
        and registration["plan"]["plan_hash"]==campaign["exposure_hash"],"Wrong real donor registration")
    return root,campaign


def verify_execution_claim(claim_pin, registration, out, target, resume, measure_at, code):
    admission.require(claim_pin is not None,"Real WP02 route execution requires an authorized campaign claim")
    claim = read(claim_pin)
    allocation_pin = claim["allocation"]
    arm,segment = claim["arm"],claim["segment"]
    if arm=="donor":
        root,config = load_campaign(allocation_pin,code)
    else:
        arms = read(allocation_pin)
        root,config = load_arms(arms["campaign"],allocation_pin,code)
    folder,token = allocation.location(root,config,arm,segment)
    interval = config["routes"][arm][segment]
    admission.require(claim_pin["path"]==str(token) and not token.with_suffix(".result.json").exists()
        and external_directory(out)==folder and claim["output"]==str(folder)
        and claim["registration"]==registration==config["registrations"][arm] and claim["interval"]==interval
        and target==interval["stop"] and measure_at==(2 if arm=="donor" and segment=="full" else None),
        "Real claim does not match the frozen execution interval")
    previous = interval["previous"]
    if previous is None:
        admission.require(claim["prior"] is resume is None,"Fresh route cannot consume a checkpoint")
    else:
        status,result = allocation.read_attempt(root,allocation_pin,config,arm,previous)
        admission.require(claim["prior"]==result["attempt"] and resume==status["checkpoint"],
                          "Real route must resume its own successful prefix")


@torch.no_grad()
def selected_metrics(run):
    model = _selected(run)
    classes = run.record["classes"]
    truth = sparse_labels(run.record,run.scene.plan,"validation_support")
    matrices = np.zeros((3,len(classes),len(classes)),dtype=np.int64)
    losses = [[],[],[]]
    tiles = [tile for tile in run.record["roles"] if tile["role"]=="devval"]
    for tile in tiles:
        labels = truth[tile_slice(tile)].ravel()
        outputs = run._predict(model,run.scene.plan["tiles"][tile["id"]],0)[:3]
        for head,probabilities in enumerate(outputs):
            for i in np.flatnonzero(labels):
                actual = int(labels[i])-1
                probabilities_at_pixel = probabilities[i]
                matrices[head,actual,int(probabilities_at_pixel.argmax())]+=1
                losses[head].append(-math.log(max(float(probabilities_at_pixel[actual]),1e-8)))
    stored = run.history[run.selected_step-1]["validation"]
    admission.require(matrices[0].tolist()==stored["confusion"] and
        abs(math.fsum(losses[0])/len(losses[0])-stored["nll"])<=2e-6, "Independent selected validation differs")
    result = {}
    for i,name in enumerate(("main","graph","cnn")):
        row = metrics_from_confusion(matrices[i],classes)
        admission.require([c["support"] for c in row["per_class"]]==[5]*len(classes),
                          "Incomplete released validation population")
        result[name] = {**row,"nll":math.fsum(losses[i])/len(losses[i]),
                        "zero_recall_classes":[c["class_id"] for c in row["per_class"] if c["recall"]==0]}
    return result


def _audit_arm(root, pin, config, name):
    reg,base,pins = dev._load_registration(config["registrations"][name])
    budget = config["budget"]
    fingerprints, status_files, loaded = {}, {}, {}
    for key in allocation.segments(budget):
        status,result = allocation.read_attempt(root,pin,config,name,key)
        status_files[key] = result["attempt"]
        checkpoint = status["checkpoint"]
        pins.extend([result["attempt"], checkpoint])
        restored = GraphRouteRun.load(base.source,base.scene,base.record,checkpoint["path"],plan=base.plan,
                                      expected_sha256=checkpoint["sha256"],expected_contract_hash=base.contract_hash,
                                      bank=base.bank,expected_bank_hash=reg["bank_hash"])
        if key!="split-first":
            admission.require(canonical_json(restored.finish())==canonical_json(status["acceptance"]),
                              "Independent reload differs from the completed route result")
            fingerprints[key] = digest(matched.snapshot(restored))
            loaded[key] = restored
    admission.require(len(set(fingerprints.values()))==1,"Full, split and completed-reload scientific states differ")
    run = loaded["full"]
    history = run.history
    selected = run.selected_step
    expected = min(range(1,budget+1),key=lambda step:history[step-1]["validation"]["nll"])
    exposure = run.plan.exposure()
    selected_exposure = run.plan.exposure(selected)
    groups = {name:{"gradient_l1":sum(row["gradient_l1"][name] for row in history),
                    "changed_parameters":sum(row["changed_parameters"][name] for row in history)}
              for name in ("adapter","cnn","graph","heads")}
    admission.require(exposure["meets_minimum"] and selected==expected and
        all(g["gradient_l1"]>0 and g["changed_parameters"]>0 for g in groups.values()),
        "Completed exposure, first minimum or active group evidence differs")
    heads = selected_metrics(run)
    admission.check_pins(pins)
    return {"attempts":status_files,"full_split_reload_fingerprint":fingerprints["full"],
            "selected_step":selected,"selection_exposure_complete":selected_exposure["meets_minimum"],
            "selection_missing_support":sum(n>0 for n in selected_exposure["support_deficits"].values()),
            "selection_missing_classes":sum(n>0 for n in selected_exposure["class_deficits"].values()),
            "completed_exposure":{"class_blocks":exposure["class_blocks"],"class_visits":exposure["class_label_visits"],
                "minimum_visits":min(exposure["support_visits"].values()),"visited_tiles":sum(n>0 for n in exposure["tile_counts"].values())},
            "groups":groups,"first_loss":history[0]["loss"],"last_loss":history[-1]["loss"],
            "heads":heads,"updates":{"full":budget,"split_first":7,"split_last":budget-7,"completed":0}}


def audit_donor(campaign_pin,code):
    root,campaign = load_campaign(campaign_pin,code)
    started = time.monotonic()
    result = _audit_arm(root,campaign_pin,campaign,"donor")
    admission.check_pins(campaign["inputs"].values())
    require_frozen_code(code)
    audit = dev._json_new(root/"inputs"/"donor-audit.json",{
        "policy":POLICY,"status":"passed","campaign":campaign_pin,"producer":code["commit"],
        "result":result,"optimizer_updates":2*campaign["budget"],"audit_updates":0,"query":0,
        "wall_seconds":time.monotonic()-started})
    candidate = {"name":"xuzhou-real-dynamic","registration":campaign["registrations"]["donor"],
                 "attempt":result["attempts"]["full"],"source_issue":campaign["issue"],"provenance":None}
    summary,*_ = admission.metadata(candidate)
    proof = dev._json_new(root/"inputs"/"provenance.json",{**summary,"policy":admission.ORIGIN,"status":"accepted",
        "synthetic":False,"runtime":campaign["runtime"],"registration":candidate["registration"],
        "attempt":candidate["attempt"],"independent_bridge":campaign["prior"]["bridge_audit"],"donor_audit":audit,
        "campaign":campaign_pin,"raw_gt_accessed":False,"query_accessed":False})
    candidate["provenance"] = proof
    return dev._json_new(root/"inputs"/"catalog.json",{"policy":admission.CATALOG,"candidates":[candidate]})


def _bind_review(root,campaign_pin,campaign,scope_pin,review_pin,code):
    scope,preflight = matched.load_scope(scope_pin,code)
    review = read(review_pin)
    catalog_pin = dev._pin(root/"inputs"/"catalog.json")
    candidate = matched.choose(catalog_pin,"xuzhou-real-dynamic")
    proof = read(candidate["provenance"])
    accepted = read(proof["donor_audit"])
    admission.require(Path(scope_pin["path"])==root/"analysis"/"matched"/"scope.json"
        and Path(review_pin["path"])==root/"analysis"/"review"/"review.json"
        and accepted["campaign"]==campaign_pin and accepted["status"]=="passed"
        and accepted["optimizer_updates"]==2*campaign["budget"] and accepted["audit_updates"]==0
        and proof["independent_bridge"]==campaign["prior"]["bridge_audit"]
        and accepted["result"]["attempts"]["full"]==candidate["attempt"]==proof["attempt"]
        and candidate["registration"]==campaign["registrations"]["donor"]
        and scope["catalog"]==catalog_pin and scope["synthetic"] is False and scope["budget"]==campaign["budget"]
        and preflight["settings"]==matched.SETTINGS and review["scope"]==scope_pin
        and review["donor"]==preflight["donor"]
        and review["optimizer_updates"]==review["query"]==0,
        "Matched review or independent donor acceptance is outside the campaign")
    return scope,review


def match_donor(campaign_pin,code):
    root,campaign = load_campaign(campaign_pin,code)
    catalog = dev._pin(root/"inputs"/"catalog.json")
    candidate = matched.choose(catalog,"xuzhou-real-dynamic")
    source = read(candidate["provenance"])
    accepted = read(source["donor_audit"])
    admission.require(source["campaign"]==accepted["campaign"]==campaign_pin and accepted["status"]=="passed"
        and accepted["optimizer_updates"]==2*campaign["budget"] and accepted["result"]["attempts"]["full"]==candidate["attempt"],
        "The real donor must pass independent acceptance before matching")
    scope = matched.prepare(catalog,"xuzhou-real-dynamic",root/"analysis"/"matched",code,settings=matched.SETTINGS)
    review = matched.frozen_review(scope,root/"analysis"/"review",code)
    return register_arms(campaign_pin,scope,review,code)


def register_arms(campaign_pin,scope_pin,review_pin,code):
    root,campaign = load_campaign(campaign_pin,code)
    scope,review = _bind_review(root,campaign_pin,campaign,scope_pin,review_pin,code)
    if scope["status"]!="registered" or review["status"]!="passed":
        return dev._json_new(root/"inputs"/"matched-blocked.json",{
            "policy":POLICY,"status":"blocked_whole_six_arm_campaign","campaign":campaign_pin,
            "scope":scope_pin,"review":review_pin,"optimizer_updates":0,
            "unavailable":{name:row["reasons"] for name,row in scope["arms"].items() if row["status"]=="unavailable"}})
    registrations = {name:scope["arms"][name]["registration"] for name in allocation.ARM_NAMES}
    return dev._json_new(root/"inputs"/"matched.json",{"status":"registered","campaign":campaign_pin,
        "scope":scope_pin,"review":review_pin,**_allocation(code,campaign["budget"],registrations)})


def load_arms(campaign_pin,arms_pin,code):
    root,campaign = load_campaign(campaign_pin,code)
    arms = read(arms_pin)
    admission.require(Path(arms_pin["path"])==root/"inputs"/"matched.json" and arms["status"]=="registered"
                      and arms["campaign"]==campaign_pin,"Explicit real six-arm allocation is required")
    scope,review = _bind_review(root,campaign_pin,campaign,arms["scope"],arms["review"],code)
    admission.require(scope["status"]=="registered" and review["status"]=="passed",
                      "Every matching arm and context must be feasible")
    expected = _allocation(code,campaign["budget"],{name:scope["arms"][name]["registration"] for name in allocation.ARM_NAMES})
    admission.require(all(arms[k]==v for k,v in expected.items()),"Real arm identity or total allocation differs")
    return root,arms


def audit_arms(campaign_pin,arms_pin,code):
    root,arms = load_arms(campaign_pin,arms_pin,code)
    results = {name:_audit_arm(root,arms_pin,arms,name) for name in allocation.ARM_NAMES}
    baseline = results["none"]["heads"]
    for row in results.values():
        for name,head in row["heads"].items():
            recalls = [c["recall"] for c in head["per_class"]]
            baseline_recalls = [c["recall"] for c in baseline[name]["per_class"]]
            head["protection"] = [{"class_id":i,"delta_pp":100*(a-b),"passes_2pp":a-b>=-.02}
                                  for i,(a,b) in enumerate(zip(recalls,baseline_recalls,strict=True),1)]
    require_frozen_code(code)
    return dev._json_new(root/"inputs"/"matched-audit.json",{"policy":POLICY,"status":"passed","allocation":arms_pin,
        "arms":results,"optimizer_updates":12*arms["budget"],"audit_updates":0,"query":0})
