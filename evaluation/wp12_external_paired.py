"""Publish complete selected pairs and independently consume their authorized evidence."""

from collections import Counter
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS, hard
from evaluation.frozen_training_stages import file_pin, json_pin, require
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_qualification import classification_status
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts.run_frozen_band_region import _write
from transfer import native_external_paired as paired
from transfer.native_coverage_queue import case_lock
from transfer.native_paired_head_fixture import VIEWS
from transfer.xuzhou_multiscale_audit import _exact

CAPTURE = "wp12_external_paired_capture_v1"
REVIEW = "wp12_external_paired_independent_review_v1"


def add_json(root, path, value):
    if path.exists():
        link = file_pin(path)
        require(json_pin(root,link)==value,"An existing publication marker has a different identity")
        return link
    return _write(path,value)


def selected_record(root,out,code,release,seed,arm,*,test,reference=False):
    p = paired.ExternalPairedProvider(root,out,code,release,seed,arm,test=test)
    place = paired.arm_dir(out if not reference else out/"reference",release,seed,arm)
    receipt = file_pin(place/"receipt.json")
    row = json_pin(root,receipt)
    spec = json_pin(root,row["contract"])
    require(spec==p.contract(p.classes,seed,"native",1,p.producer,numeric=spec["numeric"]),
            "Completed arm differs from its admitted workload")
    lineage = paired.journal.checkpoints(root,place,digest(spec),read_all=True,steps=p.budget)
    require(lineage and paired.finish(root,place,code,p,spec,lineage)==receipt,
            "Completed paired arm or selection differs")
    training = lineage[-1]["item"]["training"]
    body = dict(schema=paired.PUBLISH, release=release, seed=seed, arm=arm, scope=row["scope"],
                registration=p.registration, source=p.source, receipt=receipt, contract=row["contract"],
                checkpoint=row["checkpoint"], selected=row["selected"], state=training["selected"]["state"],
                history=[event["step"] for event in lineage], updates=row["updates"])
    return p,spec,training,body


def publish(root,out,code,*,test,fail_after=None):
    plan,registration = paired.load(root,out,code,test=test)
    require(fail_after is None or test and fail_after==1,"Injected publication failures are fixture-only")
    place = out/"selected"
    work = []
    for release,record in sorted(plan["releases"].items()):
        req = record["request"]
        for seed in req["model_seeds"]:
            for arm in paired.ARMS[req["kind"]]:
                name = f"{release}-s{seed}-{arm}"
                finished = paired.arm_dir(out,release,seed,arm)/"receipt.json"
                require(finished.is_file(),"Every registered arm must finish before selected publication")
                work.append((name,release,seed,arm))
    with case_lock(place):
        members = {}
        for name,release,seed,arm in work:
            _,_,_,payload = selected_record(root,out,code,release,seed,arm,test=test)
            path = place/(name+".pt")
            if path.exists():
                link = file_pin(path)
                require(_exact(paired.journal.packet(root,link),payload),"Retry found a foreign staged selected member")
            else:
                link = paired.journal.write_checkpoint(path,payload)
            members[name] = link
            if fail_after==len(members):
                raise RuntimeError("External-format fixture interrupted the selected publication")
        require_frozen_code(code)
        return add_json(root,place/"publication.json",dict(schema=paired.PUBLISH,scope=plan["scope"],
                        registration=registration,producer=code["commit"],members=members,
                        raw_gt_loaded=False,query_accessed=False))


@torch.no_grad()
def capture(root,out,output,code,execution,*,test,release=None):
    plan,registration = paired.load(root,out,code,test=test)
    publication = file_pin(out/"selected"/"publication.json")
    marker = json_pin(root,publication)
    require(marker["registration"]==registration and marker["producer"]==code["commit"] and
            marker["scope"]==plan["scope"] and marker["schema"]==paired.PUBLISH,"Foreign selected publication")
    expected = {f"{key}-s{seed}-{arm}" for key,record in plan["releases"].items()
                for seed in record["request"]["model_seeds"] for arm in paired.ARMS[record["request"]["kind"]]}
    require(expected==set(marker["members"]) and (release is None or release in plan["releases"]),
            "The selected publication and capture must use the exact admitted roster")
    arms = {}
    for name,link in marker["members"].items():
        key,seed,arm = name.split("-")
        if release is not None and release!=key:
            continue
        seed = int(seed[1:])
        p,spec,training,body = selected_record(root,out,code,key,seed,arm,test=test)
        require(_exact(paired.journal.packet(root,link),body),"Selected consumer rejected foreign checkpoint contents")
        model = p.baseline(seed).to(execution.device)
        model.load_state_dict(body["state"],strict=True)
        rows = []
        for band in VIEWS:
            for tile in p.roles["evaluation"]:
                accuracy,cost = None,None
                if p.intake["uses"]["independent_accuracy"]["eligible"]:
                    actual,_,targets = p.consume(tile,band,use="independent_accuracy")
                    probs,graph,_ = p.registered_forward(model,actual,"devval",execution)
                    cells = targets.cells
                    accuracy = dict(truth=[label for _,label in targets.points], population=targets.summary(),
                                    probabilities={head:value.detach().cpu().numpy()[cells].tolist()
                                                   for head,value in zip(HEADS,probs[:3],strict=True)},
                                    observation=actual.receipt["observation"], graph=graph,
                                    bands=actual.receipt["source"]["band_ids"])
                if p.intake["uses"]["cost"]["eligible"]:
                    times = []
                    for rep in range(3):
                        execution.sync()
                        if execution.device.type=="cuda":
                            torch.cuda.reset_peak_memory_stats(execution.device)
                        start = perf_counter()
                        actual,labels,targets = p.consume(tile,band,use="cost")
                        require(labels is targets is None,"Cost reached the classification target interface")
                        materialized = perf_counter()
                        values,graph,details = p.registered_forward(model,actual,"devval",execution)
                        projection = [list(value.shape) for value in values[:3]]
                        execution.sync()
                        end = perf_counter()
                        if rep:
                            times.append(dict(total=end-start, materialize=materialized-start,
                                              forward=end-materialized, graph=sum(details["graph_seconds"]),
                                              cuda_peak_bytes=(torch.cuda.max_memory_allocated(execution.device)
                                                               if execution.device.type=="cuda" else None)))
                    cost = dict(times=times,projection=projection,graph=graph,label_blind=True,
                                observation=actual.receipt["observation"],bands=actual.receipt["source"]["band_ids"])
                rows.append(dict(tile=tile,band=band,accuracy=accuracy,cost=cost))
        require(all(tensor_hash(x)==tensor_hash(body["state"][key]) for key,x in model.state_dict().items()),
                "Independent consumption modified the selected state")
        freeze = {key:value for key,value in spec["initialization"]["inventory"].items() if key not in spec["trainable"]}
        head = {key:spec["initialization"]["inventory"][key] for key in spec["trainable"]}
        exposure = dict(Counter(item["band"] for item in spec["training"]))
        arms[name] = dict(key=key, seed=seed, arm=arm, selected=link,
                          checkpoint=body["checkpoint"], uses=p.intake["uses"], selected_step=body["selected"],
                          arm_receipt=body["receipt"], head=digest(head), frozen=digest(freeze), source=p.source,
                          calibration=[dict(step=item["step"],nll=item["fused_nll"]) for item in training["validation"]],
                          schedule=[event["tile"] for event in spec["training"]], exposure=exposure,
                          dimension=spec["recognition_dim"], trainable_parameters=spec["trainable_parameters"],
                          classes=p.classes, train_device=spec["numeric"], history=body["history"], rows=rows)
    require_frozen_code(code)
    output.parent.mkdir(parents=True,exist_ok=True)
    return _write(output,dict(schema=CAPTURE, scope=plan["scope"], producer=code["commit"], selected=publication,
                              numeric=execution.numeric, arms=arms, query_accessed=False, raw_gt_loaded=False))


def review(root,raw,output,code):
    capture_doc = json_pin(root,raw)
    marker = json_pin(root,capture_doc["selected"])
    plan = json_pin(root,marker["registration"])
    require(capture_doc["schema"]==CAPTURE and marker["producer"]==capture_doc["producer"]==plan["producer"]
            and capture_doc["scope"]==plan["scope"]==marker["scope"] and capture_doc["raw_gt_loaded"] is False
            and capture_doc["query_accessed"] is False and set(capture_doc["arms"])==set(marker["members"]),
            "Independent scoring requires a complete captured publication")
    results = {}
    for key, entry in plan["releases"].items():
        req = entry["request"]
        scene = paired.SCENES[req["kind"]]
        assessments = {}
        starts = set()
        for seed in req["model_seeds"]:
            names = [f"{key}-s{seed}-{arm}" for arm in paired.ARMS[req["kind"]]]
            old,new = (capture_doc["arms"][name] for name in names)
            require(all(capture_doc["arms"][name]["selected"]==marker["members"][name] for name in names)
                    and old["uses"]==entry["intake"]["uses"]
                    and old["source"]==req["source"] and old["key"]==new["key"]==key
                    and old["seed"]==new["seed"]==seed
                    and all(old[field]==new[field] for field in ("source","frozen","head","schedule","uses","classes",
                            "dimension","trainable_parameters","train_device")) and old["dimension"]==64
                    and old["trainable_parameters"]==65*old["classes"],"Independent paired identity differs")
            starts.add(old["head"])
            aggregate, costs = {}, {}
            for band in VIEWS:
                left = [row for row in old["rows"] if row["band"]==band]
                right = [row for row in new["rows"] if row["band"]==band]
                require(len(left)==len(right)==entry["intake"]["role_reports"]["evaluation"]["blocks"],
                        "A captured evaluation block was omitted")
                truth, a, b = [], {head:[] for head in HEADS}, {head:[] for head in HEADS}
                for x,y in zip(left,right,strict=True):
                    require(x["tile"]==y["tile"],"Paired probes differ in evaluation geometry")
                    ca,cb = x["accuracy"],y["accuracy"]
                    cost_a,cost_b = x["cost"],y["cost"]
                    require((ca is not None)==(cb is not None)==old["uses"]["independent_accuracy"]["eligible"]
                            and (cost_a is not None)==(cost_b is not None)==old["uses"]["cost"]["eligible"],
                            "A result crossed its independent use authorization")
                    if ca is not None:
                        require(all(ca[field]==cb[field] for field in ("truth","population","observation","bands","graph"))
                                and all(ca["probabilities"][head]==cb["probabilities"][head] for head in ("cnn","graph")),
                                "Evaluation geometry, deleted columns or diagnostics changed within the pair")
                        labels = ca["truth"]
                        truth.extend(labels)
                        for head in HEADS:
                            a[head].extend(hard(ca["probabilities"][head],len(labels),old["classes"]))
                            b[head].extend(hard(cb["probabilities"][head],len(labels),old["classes"]))
                    if cost_a is not None:
                        for candidate in (cost_a,cost_b):
                            require(candidate["label_blind"] and len(candidate["times"])==2 and
                                    all(row["total"]>=row["forward"]>=row["graph"]>0 and row["materialize"]>0
                                        for row in candidate["times"]),"Incomplete or label-dependent cost")
                        require(cost_a["observation"]==cost_b["observation"] and cost_a["bands"]==cost_b["bands"],
                                "Cost arms have different original observations")
                        costs.setdefault(band,[]).append(dict(tile=x["tile"],before=cost_a["times"],after=cost_b["times"]))
                if band in costs:
                    statistics = {}
                    for arm in ("before","after"):
                        samples = [timing["total"] for row in costs[band] for timing in row[arm]]
                        statistics[arm] = dict(p50_seconds=float(np.quantile(samples,.5)),
                                               p95_seconds=float(np.quantile(samples,.95)))
                    costs[band] = dict(blocks=costs[band],end_to_end=statistics)
                if truth:
                    threshold = 20 if plan["scope"]==paired.REAL else 5
                    require([truth.count(c) for c in range(1,old["classes"]+1)]==
                            entry["intake"]["role_reports"]["evaluation"]["per_class"],
                            "Independent labels differ from the approved population")
                    aggregate[band] = {head:pair_scores(truth,a[head],b[head],old["classes"],scene,head_kind=head,
                                                       guard_population=threshold) for head in HEADS}
            require((set(new["exposure"])==set(VIEWS) and set(old["exposure"])=={"full"} if req["kind"]=="xuzhou"
                    else old["exposure"]==new["exposure"] and set(old["exposure"])=={"full"}),
                    "Training observations do not implement the registered comparison")
            eligibility = {band:{head:classification_status(aggregate.get(band,{}).get(head),
                        accuracy_allowed=old["uses"]["independent_accuracy"]["eligible"],
                        cost_allowed=old["uses"]["cost"]["eligible"],real=plan["scope"]==paired.REAL)
                        for head in HEADS} for band in VIEWS}
            assessments[str(seed)] = dict(classification=aggregate or None,qualification=eligibility,cost=costs or None,
                                           arm_selected_steps=[old["selected_step"],new["selected_step"]],
                                           checkpoint_lineages=[old["history"],new["history"]])
        require(len(starts)==len(req["model_seeds"]),"Named model seeds reuse an identical initialized head")
        require(len({capture_doc["arms"][f"{key}-s{x}-{paired.ARMS[req['kind']][0]}"]["frozen"]
                     for x in req["model_seeds"]})==1,"A seed altered the frozen trunk")
        results[key] = dict(scene=scene,k=req["k"],support_seed=req["support_seed"],spatial_repeat=req["spatial_repeat"],
                            band_axis=entry["band_axis"],bank=entry["bank"],geometry=entry["geometry"],
                            uses=entry["intake"]["uses"],
                            effective_model_initializations=len(starts),assessment=assessments)
    require_frozen_code(code)
    output.parent.mkdir(parents=True,exist_ok=True)
    return _write(output,dict(schema=REVIEW, scope=plan["scope"], runtime=plan["producer"], consumer=code["commit"],
                              raw=raw, results=results, raw_gt_loaded=False, query_accessed=False,
                              external_real_updates=plan["approved_updates"] if plan["scope"]==paired.REAL else 0,
                              real_accuracy_available=plan["scope"]==paired.REAL and
                              any(r["uses"]["independent_accuracy"]["eligible"] for r in results.values()),
                              formal_independent_accuracy_qualified=any(
                                  all(item["qualification"]["central_gap" if row["scene"]=="xuzhou" else "full"]
                                      ["fused"]["qualified"] for item in row["assessment"].values())
                                  for row in results.values())))
