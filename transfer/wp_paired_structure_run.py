"""Durable single-case execution and all-arm checkpoint publication."""

from collections import Counter
import io
from time import perf_counter

import torch

from evaluation import wp12_cross_backend as comparison
from evaluation.frozen_training_stages import file_pin, json_pin, require
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts.run_frozen_band_region import _write
from transfer import native_long_run as journal, native_scale_training as native, wp_cross_backend as backend
from transfer.development_bank import _atomic_new
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.wp_paired_structure import REAL, SCHEMA, StructuralProvider, open_plan
from transfer.wp_support_match import exposure as count_exposure
from transfer import wp_region_graph as regional

PUBLICATION = "wp05_wp08_paired_structure_publication_v1"


def arm_dir(out, group, arm, *, reference=False):
    return out/("reference" if reference else "cases")/group/arm


def memory(execution, *, reset=False):
    if execution.device.type!="cuda":
        return None
    execution.sync()
    if reset:
        torch.cuda.reset_peak_memory_stats(execution.device)
        return None
    return dict(allocated=torch.cuda.max_memory_allocated(execution.device),
                reserved=torch.cuda.max_memory_reserved(execution.device))


def receipt(root, place, group, arm, plan_pin, spec, rows):
    done = place/"receipt.json"
    last = rows[-1]
    saved = journal.packet(root,last["pin"])["training"]
    require(saved["cursor"]==spec["steps"]==sum(r["updates"] for r in rows)
            and [v["step"] for v in saved["validation"]]==spec["events"]
            and saved["selected"]["step"]==min(saved["validation"],key=lambda v:v["fused_nll"])["step"],
            "Durable checkpoints did not complete the authorized selection schedule")
    changed = [key for key,x in spec["initialization"]["inventory"].items()
               if tensor_hash(saved["model"][key])!=x["sha256"]]
    exposed = Counter(x["band"] for x in saved["history"])
    grouped = {}
    for name,params in spec["initialization"]["groups"].items():
        grouped[name] = dict(tensors=len(params),updated=sum(key in changed for key in params),
            gradient_steps=sum(any((event["fresh"][key]["gradient_l1"] or 0)>0 for key in params)
                               for event in saved["history"]) if params else 0)
    body = dict(schema=SCHEMA,status="complete",group=group,arm=arm,campaign=plan_pin,
        registration=file_pin(place/"contract.json"),checkpoint=last["pin"],budget=spec["steps"],
        selected=saved["selected"]["step"],calibration=[dict(step=v["step"],nll=v["fused_nll"]) for v in saved["validation"]],
        query_accessed=False,raw_gt_loaded=False,real_updates=spec["real_updates"],
        device=spec["numeric"],exposure=dict(exposed),
        class_exposure=count_exposure(saved["history"],spec["classes"]),
        weights_changed=len(changed),groups=grouped,parameters=spec["initialization"]["parameters"],
        timings=[{k:v for k,v in row.items() if k not in ("registration","model_state","serialization_seconds")} for row in rows])
    if done.exists():
        require(json_pin(root,file_pin(done))==body,"Completion receipt disagrees with the durable trajectory")
        return file_pin(done)
    return _write(done,body)


def execute(root, out, code, group, arm, execution, *, test, until=None, reference=False):
    plan, link = open_plan(root,out,code,test=test)
    require(group in plan["groups"] and arm in plan["groups"][group]["arms"],
            "Unknown registered training arm")
    expected = plan["groups"][group]
    budget = expected["steps"]
    until = budget if until is None else until
    require(type(until) is int and 0 < until <= budget and (execution.numeric["physical_uuid"] or "cpu")==expected["device"],
            "Drain cursor or physical assignment differs")
    require(not reference or test and expected["purpose"]=="recovery" and until==budget,
            "Only an engineering recovery arm may run a continuous reference")
    place = arm_dir(out,group,arm,reference=reference)
    place.parent.mkdir(parents=True,exist_ok=True)
    with case_lock(place), journal.drain_on_signal() as drain:
        provider = StructuralProvider(root,out,plan,group,arm,code,test=test)
        spec = provider.contract(provider.classes,provider.seed,arm,expected["factor"],code["commit"],
                                 numeric=execution.numeric)
        contract = digest(spec)
        spec_path = place/"contract.json"
        if spec_path.exists():
            require(json_pin(root,file_pin(spec_path))==spec,"Cannot rebind a started trajectory")
        else:
            _write(spec_path,spec)
        rows = journal.checkpoints(root,place,contract,steps=budget)
        if rows and rows[-1]["step"]==budget:
            return dict(status="complete",updates=0,receipt=receipt(root,place,group,arm,link,spec,rows))
        require((not rows or not reference) and not (place/"receipt.json").exists(),"Foreign terminal marker or interrupted reference")
        cursor = first = rows[-1]["step"] if rows else 0
        require(until>=cursor,"Cannot resume backwards")
        memory(execution,reset=True)
        started = perf_counter()
        run = native.Run(spec,code["commit"],execution=execution,released=provider)
        if rows:
            run.restore(journal.packet(root,rows[-1]["pin"])["training"],cursor)
        else:
            prior = dict(run.timing)
            run.start()
            rows.append(journal.append(place,run,None,0,0,contract,prior,"initial",started,cuda_peak=memory(execution)))
        stride = max(1,min(32,budget//3))
        while cursor<until and drain["requested"] is None:
            memory(execution,reset=True)
            start,started,prior = cursor,perf_counter(),dict(run.timing)
            limit = min(until,(cursor//stride+1)*stride)
            for event in spec["training"][cursor:limit]:
                if drain["requested"]:
                    break
                run.update(event)
                cursor = event["step"]
            if cursor==start:
                break
            provider.bank.verify_authority(provider.bank.pin)
            reason = drain["requested"] or ("complete" if cursor==budget else "drain" if cursor==until else "interval")
            rows.append(journal.append(place,run,rows[-1]["pin"]["sha256"],start,cursor,contract,
                                        prior,reason,started,cuda_peak=memory(execution)))
        require_frozen_code(code)
        if cursor==budget:
            return dict(status="complete",updates=cursor-first,receipt=receipt(root,place,group,arm,link,spec,rows))
        return dict(status="drained",updates=cursor-first,cursor=cursor,signal=drain["requested"],checkpoint=rows[-1]["pin"])


def completed(root,out,group,arm,plan_pin,g,*,reference=False):
    place = arm_dir(out,group,arm,reference=reference)
    require((place/"receipt.json").is_file(),"A registered arm has not finished")
    done = file_pin(place/"receipt.json")
    record = json_pin(root,done)
    spec = json_pin(root,record["registration"])
    durable = journal.checkpoints(root,place,digest(spec),steps=g["steps"])
    require(durable and record["checkpoint"]==durable[-1]["pin"]
            and record["campaign"]==plan_pin and record["group"]==group and record["arm"]==arm
            and record["budget"]==spec["steps"]==g["steps"]==durable[-1]["step"]
            and record["query_accessed"] is record["raw_gt_loaded"] is False
            and (spec["numeric"]["physical_uuid"] or "cpu")==g["device"] and spec["events"]==g["clock"]
            and spec["factor"]==g["factor"] and spec["seed"]==g["seed"] and spec["architecture"]==g["arms"][arm],
            "Frozen arm does not match its durable budget, grid, architecture and device")
    require(receipt(root,place,group,arm,plan_pin,spec,durable)==done,"Invalid completion receipt")
    return done,record,spec,journal.packet(root,record["checkpoint"])["training"]


def write_once(root,path,payload):
    if path.exists():
        link = file_pin(path)
        require(json_pin(root,link)==payload,"Frozen publication payload changed")
        return link
    return _write(path,payload)


def export_selected(root, out, plan, link, name, group, arm, data):
    _,record,spec,packet = data
    backend_name = group["arms"][arm]["backend"]
    bank = plan["inputs"][group["release"]]
    from transfer.wp_paired_structure import admit, base_model, open_source

    entry = plan["releases"][group["release"]]
    release,_,_ = admit(root,out,entry,test=plan["scope"]!=REAL)
    source = open_source(root,release,entry["source"])
    base,_ = base_model(release,group["seed"],source)
    model = backend.train_variant(base,group["arms"][arm],spec["module_seed"])
    require(state_hash(model)==spec["initialization"]["state"],"Cannot reconstruct the registered model")
    architecture = {**model.export_config(),"training_config":{}}
    selected = packet["selected"]
    anchor = None
    if backend_name==regional.BACKEND:
        anchor = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    model.load_state_dict(selected["state"],strict=True)
    identity = state_hash(model)
    require(identity==next(v["state_sha256"] for v in packet["validation"] if v["step"]==selected["step"]),
            "Selected weights differ from the saved calibration event")
    path = out/"frozen"/name/(arm+".pt")
    path.parent.mkdir(parents=True,exist_ok=True)
    meta = write_once(root,path.with_suffix(".pt.json"),architecture)
    if not path.exists():
        with io.BytesIO() as stream:
            torch.save(selected["state"],stream)
            _atomic_new(path,stream.getvalue())
    export = dict(backend=backend_name,weights=file_pin(path),metadata=meta,state=identity)
    if anchor is not None:
        anchor_path = path.parent/"stage-two-initial.pt"
        if not anchor_path.exists():
            with io.BytesIO() as stream:
                torch.save(anchor,stream)
                _atomic_new(anchor_path,stream.getvalue())
        export["graph_program"] = dict(weights=file_pin(anchor_path),state=spec["initialization"]["state"],
            policy=group["arms"][arm]["policy"],seed=group["seed"]+19000,budget=group["steps"])
    backend.load_export(root,export)
    context = {k:spec[k] for k in ("seed","factor","steps","numeric","roles","bank","graph_scale","band_axis",
                                   "learning_rate","optimizer_group","objective","training","validation","events","inputs","selection")}
    context["initialization"] = spec["initialization"]["base"]
    protocol = dict(budget=group["steps"],factor=group["factor"],data=digest([bank["bank"],spec["roles"],spec["inputs"]]),
              exposure=digest(spec["training"]),initialization=context["initialization"],
              class_axis=list(range(1,spec["classes"]+1)),band_axis=spec["band_axis"],
              arithmetic=digest(spec["numeric"]),selection=comparison.SELECTION)
    if "delivered_columns" in bank["authority"]:
        from evaluation.wp12_scientific_claim import COLUMNS

        protocol["observation_protocol"] = COLUMNS
    endpoint = write_once(root,path.with_suffix(".json"),dict(schema=comparison.ENDPOINT,protocol=protocol,
        seed=group["seed"],phase="selected",step=selected["step"],calibration=record["calibration"],checkpoint=export))
    return protocol,endpoint,context


def publication(root,out,code,*,test):
    plan,link = open_plan(root,out,code,test=test)
    with case_lock(out/"frozen"):
        # Complete and authenticate the entire roster before exposing any comparison request.
        all_arms = {name:{arm:completed(root,out,name,arm,link,g)[:2] for arm in g["arms"]}
                    for name,g in plan["groups"].items()}
        requests, all_cases = {},{}
        for name,group in plan["groups"].items():
            roi = requests.setdefault(group["release"],{})
            saved,proofs,reference = {},{},None
            for arm in group["arms"]:
                data = completed(root,out,name,arm,link,group)
                require(data[:2]==all_arms[name][arm],"Completed checkpoint changed during publication")
                protocol,pin,context = export_selected(root,out,plan,link,name,group,arm,data)
                if reference is None:
                    reference = context
                require(context==reference,"Paired models did not share geometry, support, clock, budget or initialization")
                saved[arm],proofs[arm] = pin,data[0]
            roi[name] = dict(seed=group["seed"],protocol=protocol,arms=saved,
                    pairs=[[p["baseline"],p["candidate"]] for p in group["pairs"]])
            all_cases[name] = proofs
        releases = {}
        for name, groups in requests.items():
            entry = plan["releases"][name]
            releases[name] = write_once(root,out/"frozen"/(name+".json"),dict(schema=comparison.DESIGN,
                                         intake=entry["manifest"],authority=entry["authority"],groups=groups))
        require_frozen_code(code)
        return write_once(root,out/"publication.json",dict(schema=PUBLICATION,producer=code["commit"],campaign=link,
                          scope=plan["scope"],releases=releases,cases=all_cases,optimizer_updates=plan["optimizer_updates"],
                          query_accessed=False,raw_gt_loaded=False))
