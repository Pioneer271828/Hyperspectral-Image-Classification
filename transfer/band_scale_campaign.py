"""Disjoint new real freeze and synthetic scale cells in one registered spatial entry."""

from collections import Counter, defaultdict
from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import resource

import torch

from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _read, _write, check, pin, source_documents
from scripts.run_frozen_mechanisms import case_lock
from transfer import band_scale_real as real, native_scale_training as native
from transfer.band_scale import BANDS, FACTORS, NAME, condition, evidence, measure
from transfer.band_scale_fixture import ARMS, EVENTS, LAYOUTS, PREFIX, SEEDS, STEPS, TASKS, open_fixture
from transfer.spatial_coverage_training import load

PHASES = ("full","prefix","resume","completed")
SYNTHETIC = {f"{task}-{layout}-{seed}":(task,layout,seed) for task,layout,seed in product(TASKS,LAYOUTS,SEEDS)}


def specification(root,code):
    return dict(schema=NAME, commit=code["commit"], source_hash=code["source_hash"],
                former=real.source_pins(root),synthetic=list(SYNTHETIC),real=list(real.CASES),
                focus=real.FOCUS,bands=list(BANDS),factors=list(FACTORS),arms=list(ARMS),steps=STEPS,
                prefix=PREFIX,validation_steps=list(EVENTS),recall_guard=-.02,
                hypothesis=dict(minimum_delta=.10,pairs_required=4,
                        spectral="fused off full minus gap at factor 1",
                        region="graph_content fused minus cnn_content on factor 1 full",
                        negative="same region contrast below required count"),
                real_updates=0,query_accessed=False,raw_gt_loaded=False)


def register(root,out,code):
    sources = specification(root,code)
    for item in sources["former"].values():
        _read(item)
    out.mkdir()
    return _write(out/"registration.json",sources)


def registered(root,out,code,sha=None):
    item = pin(out/"registration.json",sha)
    check(_read(item)==specification(root,code),"Registered band-scale campaign differs")
    return item


def branch_updates(model,packet):
    groups = defaultdict(lambda:Counter(parameter_tensors=0,changed_tensors=0,adam_slots=0,adam_steps=0))
    opt = packet["optimizer"]
    check(len(opt["param_groups"])==1,"Unexpected registered Adam group roster")
    for (name,value),pid in zip(model.named_parameters(),opt["param_groups"][0]["params"],strict=True):
        key = ("adapter" if ".adapter." in name else "cnn" if ".CNN_Branch." in name or name.startswith("cnn_heads.")
               else "graph" if "learnable_graph." in name or ".HGCN_Branch." in name or ".HGNN_Branch." in name
               or ".range_" in name or name.startswith("hgnn_heads.") else "fusion_and_other")
        row = groups[key]
        row["parameter_tensors"]+=1
        row["changed_tensors"]+=not torch.equal(value,packet["model"][name])
        if pid in opt["state"]:
            row["adam_slots"]+=1
            row["adam_steps"]+=int(opt["state"][pid]["step"])
    return {name:dict(value) for name,value in groups.items()}


def finish_arm(path, fixture, arm, scope):
    model_seed, code = fixture.seed,fixture.producer
    spec = fixture.contract(9,model_seed,arm,1,code)
    if path.exists():
        item = pin(path/"receipt.json")
        content = _read(item)
        check(content["scope"]==scope and content["arm"]==arm,"Foreign completed band-scale arm")
        return item["sha256"]
    with TemporaryDirectory(dir=path.parent,prefix=f".{arm}-") as directory:
        stage = Path(directory)/"sealed"
        stage.mkdir()
        phases, snapshots = {}, {}
        parents = {"completed":"resume","resume":"prefix"}
        for phase in PHASES:
            previous = None if phase not in parents else load(stage/f"{parents[phase]}.pt")
            artifact = stage/f"{phase}.pt"
            phases[phase] = native.execute(spec,phase,artifact,code,previous,released=fixture)
            snapshots[phase] = pin(artifact)["sha256"]
        full,resumed,complete = (load(stage/f"{name}.pt") for name in ("full","resume","completed"))
        check(native._exact(full,resumed) and native._exact(full,complete),"Exact scale recovery differs")
        check(full["selected"]["step"]==min(full["validation"],key=lambda x:x["fused_nll"])["step"],
              "Selected checkpoint differs")
        model = fixture.model_factory(fixture.baseline(model_seed),arm,spec["module_seed"]).eval()
        updates = branch_updates(model,full)
        model.load_state_dict(full["selected"]["state"],strict=True)
        items = []
        for event in spec["validation"]:
            actual,_,targets = fixture.materialize("devval",event,code)
            reading = measure(model,actual,targets,code,selector=condition(actual,"matched"))
            items.append(dict(event=event,observed=evidence(actual,targets),
                              labels=[label for _,label in targets.points],**reading))
        negative = None
        if arm=="joint":
            event = spec["validation"][-1]
            actual,_,targets = fixture.materialize("devval",event,code)
            base = items[-1]
            variants = {}
            content = {}
            handles = []
            def capture(key):
                def record(_,__,values):
                    content[key] = values.detach().softmax(dim=-1).tolist()
                return record
            first = model.backbone.CNN_Branch[0]
            handles.append(first.content_selector.register_forward_hook(capture("cnn")))
            handles.append(model.backbone.range_content.register_forward_hook(capture("graph")))
            try:
                for candidate in ("permuted","missing"):
                    variants[candidate] = measure(model,actual,targets,code,selector=condition(actual,candidate))
            finally:
                for handle in handles:
                    handle.remove()
            check(all(v["graph"]["q"]==base["graph"]["q"] and v["graph"]["a0"]==base["graph"]["a0"] for v in variants.values())
                  and variants["permuted"]["graph"]["weights"]!=base["graph"]["weights"]
                  and all(variants["missing"]["graph"]["weights"][key]["weights"]==weight for key,weight in content.items()),
                  "Metadata case changed observed graph or fallback rule")
            negative = dict(event=event,controls=variants,fallback_matches_content=True)
        result = _write(stage/"receipt.json",dict(arm=arm,scope=scope, phases=phases,snapshots=snapshots,
                         chosen=full["selected"]["step"], state=phases["full"]["state"],
                         initial=spec["initialization"]["source"],training=spec["training"],
                         trajectory=full["history"],updates_by_branch=updates, parameters=spec["parameters"],
                         validation=full["validation"],endpoints=items,physical_controls=negative))
        os.rename(stage,path)
    return result["sha256"]


def execute(root,out,code,selector):
    scope = registered(root,out,code)["sha256"]
    if selector=="history":
        target = out/"history.json"
        if target.exists():
            check(_read(pin(target))["scope"]==scope,"Wrong historical scope")
            print("reused history",flush=True)
            return
        _,docs = source_documents(root)
        value = real.history(root,real.source_pins(root),docs)
        _write(target,dict(scope=scope,**value))
        print("completed history",flush=True)
        return
    if selector in real.CASES:
        target = out/f"{selector}.json"
        with case_lock(out/f".{selector}.lock"):
            if target.exists():
                check(_read(pin(target))["scope"]==scope,"Wrong frozen scope")
                print(f"reused {selector}",flush=True)
                return
            record = real.run(root,code,selector)
            require_frozen_code(code)
            _write(target,dict(scope=scope,**record))
            print(f"completed {selector}",flush=True)
        return
    enabled = [name for name in SYNTHETIC if selector=="all-synthetic" or name==selector
               or name.rsplit("-",1)[0]==selector]
    check(enabled,"Unknown registered scale campaign selector")
    for name in enabled:
        started = perf_counter()
        task,layout,seed = SYNTHETIC[name]
        path = out/name
        path.mkdir(exist_ok=True)
        with case_lock(path/".lock"):
            result = path/"receipt.json"
            if result.exists():
                check(_read(pin(result))["scope"]==scope,"Wrong synthetic bank scope")
                print(f"reused {name}",flush=True)
                continue
            fixture,witness = open_fixture(path,task,layout,seed,code["commit"])
            arms = {arm:finish_arm(path/arm,fixture,arm,scope) for arm in ARMS}
            require_frozen_code(code)
            _write(result,dict(case=name,scope=scope,arms=arms,plant=witness,seconds=perf_counter()-started,
                               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                               real_updates=0,raw_gt_loaded=False,query_accessed=False))
            print(f"completed {name}",flush=True)
