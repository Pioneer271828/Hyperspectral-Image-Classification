"""Post-freeze confirmation and attribution gates for synthetic spectral controls."""

from collections import defaultdict
from copy import deepcopy
from itertools import product
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, paired, require, scores
from evaluation.native_positive_oracle import oracle
from evaluation.wp06_observation_audit import V1_FREEZE, V1_REVIEW
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_scale_training as native
from transfer.native_coverage_queue import case_lock
from transfer.native_long_run import checkpoints
from transfer.native_positive_fixture import (ARMS, CLOCK, CLASSES, ROLE_SEEDS, SCIENTIFIC, SCHEMA, STEPS,
                                               TRAIN_VIEWS, VIEWS, profiles, publish_bank)
from transfer.native_positive_training import PositiveProvider, admitted_forward, load_plan

STAGES = ("initial","middle","selected","last")


def frozen(root,place,producer,pin):
    plan,plan_pin = load_plan(root,place,producer)
    receipt = json_pin(root,pin)
    require(pin == file_pin(place/"freeze.json") and receipt["plan"] == plan_pin
            and receipt["schema"] == SCHEMA and receipt["scope"] == "synthetic_only"
            and set(receipt["cases"]) == set(plan["cases"])
            and receipt["updates"] == STEPS*len(plan["cases"])
            and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False,
            "Confirmation needs the full frozen synthetic roster")
    return plan,receipt


def selected_step(spec,history):
    require(spec["selection"] == SCIENTIFIC["selection"] and spec["steps"] == STEPS
            and all(row["band"] == "full" for row in spec["validation"])
            and [row["step"] for row in history] == list(CLOCK),
            "Selection may use only the original full-band calibration clock")
    return min(history,key=lambda row:row["fused_nll"])["step"]


def positive_gate(profile, spectrum, pixel_aa, native_aa, center_aa, rare_delta):
    """Keep measurement success separate from the scientific gates."""
    local = bool(profile["local"])
    context = bool(profile["context"])
    controls_ok = all(spectrum.values())
    local_ready = controls_ok and local and pixel_aa >= SCIENTIFIC["spectral_reference_valid_aa"]
    null = not local and not context
    return dict(data=controls_ok,local_reference_valid=local_ready,
                context_reference_valid=controls_ok and context,
                native_positive_valid=controls_ok and (local or context) and native_aa >= SCIENTIFIC["native_positive_aa"],
                null_detected_false_positive=null and max(pixel_aa,native_aa,center_aa) > SCIENTIFIC["negative_upper_aa"],
                center_supported=local_ready and center_aa >= SCIENTIFIC["native_positive_aa"]
                                 and center_aa-native_aa >= SCIENTIFIC["candidate_delta_aa"]
                                 and rare_delta >= -SCIENTIFIC["maximum_rare_loss"])


def point_capture(model,samples,producer,execution,*,shuffle=False):
    captured, traces = [],[]
    for sample in samples:
        actual = sample["observation"]
        length = actual.receipt["pixels"]
        order = None
        if shuffle:
            rng = np.random.default_rng([ROLE_SEEDS["confirmation"], actual.receipt["source"]["tile"],919])
            order = rng.permutation(length)
            require(not np.array_equal(order,np.arange(length)),"Position control did not change position order")
        outputs,trace,_ = admitted_forward(model,actual,producer,actual.receipt["role"],execution,order=order)
        probs = {key:value.detach().cpu().numpy()[sample["cells"]] for key,value in zip(HEADS,outputs[:3],strict=True)}
        traces.append(dict(graph=trace,point_roster=digest(sample["ids"]),
                           original_positions=actual.receipt["source"]["positions"],
                           coordinate_permutation=None if order is None else digest(order.tolist())))
        for i, (identity, truth) in enumerate(sample["ids"]):
            captured.append(dict(id=[actual.receipt["source"]["tile"],identity],truth=truth,
                                 probabilities={head:probs[head][i].tolist() for head in HEADS}))
    return captured,traces


@torch.no_grad()
def stage_confirmation(model,rows,producer,execution,*,check_cpu=False):
    started = perf_counter()
    device = next(model.parameters()).device
    traces,samples = {},{}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for view in VIEWS:
        basic = "full" if view == "position_shuffled_full" else view
        samples[view], traces[view] = point_capture(model,rows[("confirmation",basic)],producer,execution,
                                                   shuffle=view == "position_shuffled_full")
    result = {}
    for view in VIEWS:
        choice = samples[view]
        out = scores(choice,CLASSES)
        transitions = paired(samples["full"],choice,CLASSES,{1:2,2:1,3:4,4:3})
        for head,head_scores in out.items():
            head_scores["aa"] = float(np.mean(head_scores["recall"]))
            changes = transitions[head]["transitions"]
            transitions[head]["correct_to_error"] = sum(n for truth,a,b,n in changes if a == truth and b != truth)
            transitions[head]["error_to_correct"] = sum(n for truth,a,b,n in changes if a != truth and b == truth)
        result[view] = dict(heads=out,comparison=transitions,graph=digest(traces[view]),
                            class_points=digest([(row["id"],row["truth"]) for row in choice]))
    comparison = None
    if check_cpu:
        cpu = deepcopy(model).cpu().eval()
        comparison = {}
        for view in TRAIN_VIEWS:
            replay,_ = point_capture(cpu,rows[("confirmation",view)],producer,None)
            reference = samples[view]
            require([(r["id"],r["truth"]) for r in replay] == [(r["id"],r["truth"]) for r in reference],
                    "Cross-device replay changed the confirmation support")
            stats = {}
            for head in HEADS:
                a,b = (np.asarray([r["probabilities"][head] for r in group]) for group in (replay,reference))
                stats[head] = dict(max_abs=float(np.max(np.abs(a-b))),
                                   changed_argmax=int(np.count_nonzero(a.argmax(1)!=b.argmax(1))))
            comparison[view] = stats
    return dict(views=result,wall_seconds=perf_counter()-started,device=str(device),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                cpu_representative_comparison=comparison,
                position_invariant=all(samples["full"][i]["probabilities"]==q["probabilities"]
                                       for i,q in enumerate(samples["position_shuffled_full"])))


def audit_case(root,place,review,out,code,producer,freeze_pin,name,execution):
    plan,roster = frozen(root,place,producer,freeze_pin)
    require(name in roster["cases"],"Unknown frozen verification case")
    origin = roster["cases"][name]
    if out.is_file():
        registered = json_pin(root,file_pin(out))
        require(registered["freeze"] == freeze_pin and registered["origin"] == origin
                and registered["consumer"] == code["commit"],"Existing audit has a foreign identity")
        return file_pin(out)
    started = perf_counter()
    fit = json_pin(root,origin)
    spec = json_pin(root,fit["spec"])
    require(spec["numeric"] == execution.numeric,"Audit must replay the original optimizer arithmetic")
    journal = checkpoints(root,place/"cases"/name,fit["registration"],read_all=True)
    require([row["pin"] for row in journal] == fit["checkpoints"] and journal[-1]["pin"] == fit["last"]
            and fit["registration"] == digest(spec),"Frozen checkpoint ledger changed")
    states = {row["step"]:row["item"]["training"] for row in journal}
    last = states[STEPS]
    selection = selected_step(spec,last["validation"])
    require(selection == last["selected"]["step"] == fit["selected"]
            and native._exact(last["selected"]["state"],states[selection]["model"]),
            "Selected checkpoint or saved selector differs")
    provider = PositiveProvider(place,plan,name)
    run = native.Run(spec,producer["commit"],execution=execution,released=provider)
    verified_events = {}
    for epoch in CLOCK:
        run.restore(states[epoch],epoch)
        validation = run.evaluate(epoch)
        original = next(p for p in last["validation"] if p["step"] == epoch)
        require(validation["rows"] == original["rows"] and validation["fused_nll"] == original["fused_nll"]
                and native.state_hash(run.model) == original["state_sha256"],
                "Ordinary frozen full-band inference differs from original calibration")
        verified_events[str(epoch)] = validation["fused_nll"]
    profile = plan["cases"][name]
    family = profile["family"]
    condition = profiles()[family]
    dataset = review/"confirmation"
    dataset.mkdir(exist_ok=True)
    locks = review/"confirmation-locks"
    locks.mkdir(exist_ok=True)
    with case_lock(locks/family):
        bank,roles,spatial_ratio = publish_bank(dataset/family,condition,confirm=True)
        fit_stats,final_stats = provider.bank.release.statistics,bank.release.statistics
        require(all(fit_stats[key] == final_stats[key] for key in ("minimum","maximum"))
                and not set(provider.roles["calibration"]) & set(roles["confirmation"]),
                "Confirmation changed training statistics or reused a calibration tile")
    rows,report = oracle(bank,roles,producer["commit"],"confirmation",condition)
    decisions = {}
    for stage,step in zip(STAGES,(0,120,selection,STEPS),strict=True):
        run.model.load_state_dict(states[step]["model"],strict=True)
        run.model.eval()
        representative = (profile["arm"] == "native" and run.device.type == "cuda" and stage == "selected")
        owner = native.state_hash(run.model)
        decisions[stage] = dict(step=step,state=owner,**stage_confirmation(run.model,rows,producer["commit"],execution,
                                                                           check_cpu=representative))
        require(native.state_hash(run.model) == owner,"Confirmation modified the sealed model")
        if profile["arm"] == "pixel":
            require(decisions[stage]["position_invariant"],"Pure spectral readout depends on coordinates")
    require_frozen_code(code)
    timing = {key:sum(item["timing"][key] for item in journal) for key in journal[0]["timing"]}
    return _write(out,dict(status="passed",schema=SCHEMA,consumer=code["commit"],freeze=freeze_pin,origin=origin,
                     case=name,profile=profile,device=spec["numeric"],initial=spec["initialization"]["state"],
                     parameters=spec["parameters"],training=digest(spec["training"]),inputs=digest(spec["inputs"]),
                     confirmed_bank=bank.pin,confirmation_spatial_ratio=spatial_ratio,oracle=report,
                     full_calibration_history=verified_events,stages=decisions,training_timing=timing,
                     training_wall_seconds=sum(row["wall_seconds"] for row in journal),audit_wall_seconds=perf_counter()-started,
                     optimizer_updates=0,raw_gt_loaded=False,query_accessed=False))


def publish(root,place,review,out,code,producer,freeze_pin,v1_pin):
    plan,roster = frozen(root,place,producer,freeze_pin)
    outputs,linked = {},{}
    for name,profile in plan["cases"].items():
        path = review/"cases"/f"{name}.json"
        pin = file_pin(path)
        value = json_pin(root,pin)
        require(value["case"] == name and value["status"] == "passed" and value["profile"] == profile
                and value["consumer"] == code["commit"] and value["origin"] == roster["cases"][name]
                and value["freeze"] == freeze_pin and value["optimizer_updates"] == 0
                and value["raw_gt_loaded"] is value["query_accessed"] is False,"Independent case lineage differs")
        outputs[name], linked[name] = value, pin["sha256"]
    factors,matched,aggregates = {},{},defaultdict(list)
    families = profiles()
    for family,seed in product(families,(64,512,2048)):
        trio = {arm:outputs[f"{family}-m{seed}-{arm}"] for arm in ARMS}
        base,center,pixel = (trio[key] for key in ("native","center","pixel"))
        same = {digest(item["device"]) for item in trio.values()}
        require(len(same) == 1 and base["parameters"] == center["parameters"]
                and abs(base["parameters"]-pixel["parameters"])/base["parameters"] < .005
                and base["initial"] == center["initial"]
                and len({(i["training"],i["inputs"],i["confirmed_bank"]["sha256"]) for i in trio.values()}) == 1,
                "Triplet is not matched on arithmetic, capacity, exposure and confirmation geometry")
        full = {arm:trio[arm]["stages"]["selected"]["views"]["full"]["heads"]["fused"] for arm in ARMS}
        rare = min(after-before for before,after in zip(full["native"]["recall"][-2:],full["center"]["recall"][-2:],strict=True))
        gate = positive_gate(families[family],base["oracle"]["gates"],*(full[arm]["aa"] for arm in ARMS),rare)
        key = f"{family}/m{seed}"
        factors[key] = gate
        matched[key] = dict(full_aa={arm:full[arm]["aa"] for arm in ARMS},rare_delta=rare,
                            center_minus_native_full_aa=full["center"]["aa"]-full["native"]["aa"],
                            arithmetic=base["device"]["protocol"])
        for arm,stage,view,head in product(ARMS,STAGES,VIEWS,HEADS):
            line = trio[arm]["stages"][stage]["views"][view]["heads"][head]
            aggregates[(family,arm,stage,view,head)].append((line["oa"],line["aa"],line["nll"]))
    summary = {}
    for key,values in aggregates.items():
        vals = np.asarray(values,dtype=np.float64)
        require(vals.shape == (3,3),"Three distinct model seeds required")
        summary["/".join(key)] = dict(mean=dict(zip(("oa","aa","nll"),vals.mean(0).tolist(),strict=True)),
                                      sample_sd=dict(zip(("oa","aa","nll"),vals.std(0,ddof=1).tolist(),strict=True)))
    retro = None
    if v1_pin is not None:
        earlier = json_pin(root,v1_pin)
        require(earlier["scope"] == "retrospective_v1" and earlier["freeze"]["sha256"] == V1_FREEZE
                and earlier["audit"]["sha256"] == V1_REVIEW and len(earlier["cases"]) == 12
                and earlier["new_optimizer_updates"] == 0 and earlier["query_accessed"] is False,
                "Foreign retrospective receipt cannot support this candidate")
        retro = dict(sha256=v1_pin["sha256"],cases=len(earlier["cases"]),mode="read_only")
    require_frozen_code(code)
    return _write(out,dict(status="passed",scope="synthetic_confirmation",producer=producer["commit"],
                       consumer=code["commit"],frozen_source=freeze_pin,policy=SCHEMA,preregistered=SCIENTIFIC,
                       reports=linked,cases=outputs,preregistered_gate_results=factors,matched=matched,cohort_summary=summary,
                       retrofit_v1=retro,updates=roster["updates"],new_audit_updates=0,query_accessed=False,
                       raw_gt_loaded=False,stage_heads=len(outputs)*len(STAGES)*len(VIEWS)*len(HEADS)))
