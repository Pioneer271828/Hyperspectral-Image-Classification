"""Registered paired use of the native optimizer, checkpoint and publication primitives."""

from itertools import product

from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_paired_head as head, native_paired_head_fixture as fixture, native_scale_training as native
from transfer.native_coverage_queue import case_lock
from transfer.native_long_run import packet
from transfer.xuzhou_multiscale_audit import _exact

SCHEMA=head.SCHEMA
REFERENCE={"xuzhou":"xuzhou-s64-k211","honghu":"honghu-s64-k211","fallback":"fallback-s64-k211"}
SOURCE_HISTORY=("wp05-wp06-production-20261119-qa/accepted/production.json",
                "f432c0e4862ac6ae3def6c639697e688ef2c6c74a7487836753c55fb8575cc52")
SOURCE_B=("wp05-wp06-production-20261119-qa/program-v2/cases/source-b/receipt.json",
          "e3c0b58cf02aef470fe3a184362bff6a84e61cb662b22d8814098e1e2a637ff65")
GATES=dict(aggregate_delta=-.005,ordinary_delta_recall=-.02,honghu_rare_added_errors=0,
           targets={"xuzhou":[5,6],"whu-honghu":[15,21]},target_correct_gain_min=1,
           every_target_nonnegative=True,minimum_supported_confirmation_per_class=5,
           minimum_real_confirmation_per_class=20,synthetic_erased_gap_max_aa=.25)


def design():
    return {f"{kind}-s{seed}-k{k}":dict(kind=kind,seed=seed,support=k,budget=head.BUDGET[kind],
                      policies=list(head.ARMS[kind])) for kind,seed,k in product(fixture.FIT,fixture.SEEDS,fixture.SUPPORT)}


def source(root):
    prior=json_pin(root,dict(path=str(root/SOURCE_HISTORY[0]),sha256=SOURCE_HISTORY[1]))
    b=json_pin(root,dict(path=str(root/SOURCE_B[0]),sha256=SOURCE_B[1]))
    chain=prior["source_lineage"]
    require(chain["source-b"]["parent"]==chain["source-a"]["state"]
            and chain["source-b"]["state"]==b["selected_export"]["state"]
            and prior["independent_real_accuracy"] is None and b["real_updates"]==0,"Frozen multisource ancestry differs")
    for key in ("weights","metadata"):
        pinned(root,b["selected_export"][key])
    return b["selected_export"],{name:dict(state=row["state"],detector_bands=row["detector_bands"],
                         updates=row["updates"],parent=row["parent"]) for name,row in chain.items()}


def register(root,out,code):
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"registration.json"
    if manifest.exists(): return load(root,out,code["commit"])[1]
    weight, ancestry=source(root)
    inputs={}
    for kind,k in product((*fixture.FIT,*fixture.FINAL),fixture.SUPPORT):
        name=f"{kind}-k{k}"
        _,proof=fixture.publish(out/"inputs"/name,kind,k)
        inputs[name]=dict(kind=kind,support=k,**proof)
    disjoint={}
    for kind in fixture.FIT:
        rows=[fixture.definition(kind,k)[0].rows for k in fixture.SUPPORT]
        supports=[{p for row in q if row["role"]=="devtrain" for p,_ in row["samples"]} for q in rows]
        require(supports[0].isdisjoint(supports[1]),"Paired support draws share actual released points")
        disjoint[kind]=list(map(len,supports))
    k1=fixture.definition("honghu",fixture.SUPPORT[0],override_k=1)[-1]
    value=dict(schema=SCHEMA,producer=code["commit"],scope="synthetic_only",cases=design(),gates=GATES,
               inputs=inputs,source=weight,multisource_ancestry=ancestry,support_draw_sizes=disjoint,
               role_k1_feasibility=dict(engineering_only=True,updates=0,**k1),
               references=REFERENCE,center_feature_dim=64,
               real_updates=0,optimizer_updates_at_registration=0,query_accessed=False,raw_gt_loaded=False)
    require_frozen_code(code)
    return _write(manifest,value)


def load(root,out,producer):
    link=file_pin(out/"registration.json")
    doc=json_pin(root,link)
    require(doc["schema"]==SCHEMA and doc["producer"]==producer and doc["scope"]=="synthetic_only"
            and doc["cases"]==design() and doc["gates"]==GATES and doc["references"]==REFERENCE
            and doc["center_feature_dim"]==64 and doc["real_updates"]==doc["optimizer_updates_at_registration"]==0
            and doc["query_accessed"] is doc["raw_gt_loaded"] is False,"Paired case registration differs")
    weight,ancestry=source(root)
    require((doc["source"],doc["multisource_ancestry"])==(weight,ancestry),"Source checkpoint no longer admitted")
    return doc,link


def bank_for(out,plan,kind,k):
    key=f"{kind}-k{k}"
    require((out/"inputs"/key/"bank.json").is_file(),"Paired Bank is absent")
    bank,proof=fixture.publish(out/"inputs"/key,kind,k)
    require(plan["inputs"][key]==dict(kind=kind,support=k,**proof),"Cleanly restored Bank or role inventory differs")
    return bank,proof


def provider(root,out,plan,case,arm):
    spec=plan["cases"][case]
    bank,proof=bank_for(out,plan,spec["kind"],spec["support"])
    return head.PairedHeadProvider(root,bank,proof["roles"],kind=spec["kind"],policy=arm,seed=spec["seed"],
              producer=plan["producer"],source=plan["source"]),proof


def case_location(out,name,arm):
    return out/"cases"/name/arm


def fit(root,out,code,name,execution):
    plan,link=load(root,out,code["commit"])
    require(name in plan["cases"],"Unknown registered scene and support pair")
    proof=plan["cases"][name]
    output=[]
    for arm in proof["policies"]:
        path=case_location(out,name,arm)
        path.parent.mkdir(parents=True,exist_ok=True)
        with case_lock(path):
            receipt=path/"receipt.json"
            if receipt.exists():
                stored=json_pin(root,file_pin(receipt))
                require(stored["registration"]==link["sha256"] and stored["case"]==name and stored["condition"]==arm,
                        "Existing paired receipt belongs to another case")
                output.append(dict(condition=arm,updates=0,receipt=file_pin(receipt)))
                continue
            p,_=provider(root,out,plan,name,arm)
            spec=p.contract(p.classes,p.seed,p.config.arm,1,p.producer,numeric=execution.numeric)
            contract=path/"contract.json"
            if contract.exists():
                require(json_pin(root,file_pin(contract))==spec,"Resume changes registered optimizer/device contract")
            else:
                _write(contract,spec)
            exact_reference=(name==REFERENCE[proof["kind"]] and arm==proof["policies"][0])
            phases=("prefix","resume","completed","full") if exact_reference else ("full",)
            old=None
            records={}
            updates=0
            for phase in phases:
                checkpoint=path/(phase+".pt")
                if not checkpoint.exists():
                    result=native.execute(spec,phase,checkpoint,p.producer,
                        prior=old if phase in ("resume","completed") else None,execution=execution,released=p)
                    _write(path/(phase+".json"),result)
                    updates+=result["updates"]
                link_to_state=file_pin(checkpoint)
                state=packet(root,link_to_state)
                require(state["registration"]==digest(spec),"Paired checkpoint was produced by another contract")
                if phase in ("prefix","resume"):
                    old=state
                records[phase]=dict(checkpoint=link_to_state,
                    timing=json_pin(root,file_pin(path/(phase+".json")))["timing"])
            if exact_reference:
                require(all(_exact(state,packet(root,records[phase]["checkpoint"])) for phase in ("resume","completed")),
                        "Separate restored head trajectory differs from continuous execution")
            require(state["cursor"]==len(state["history"])==p.budget and
                    all(item["supervised_area"]>0 for item in state["history"]),"Incomplete effective head budget")
            p.validate_history(spec,state)
            best=min(state["validation"],key=lambda row:row["fused_nll"])
            require(best["step"]==state["selected"]["step"] and best["fused_nll"]==state["selected"]["fused_nll"],
                    "Head selection did not use the first minimum held calibration NLL")
            require_frozen_code(code)
            value=dict(schema=SCHEMA,scope="synthetic_fit_and_calibration",producer=code["commit"],
                registration=link["sha256"],case=name,condition=arm,contract=file_pin(contract),expected_updates=p.budget,
                phases=records,exact_reference=exact_reference,selected=best["step"],device=execution.numeric,
                real_updates=0,query_accessed=False,raw_gt_loaded=False)
            output.append(dict(condition=arm,updates=updates,receipt=_write(receipt,value)))
    return output


def restored(root,out,plan,link,name,arm,execution):
    p,proof=provider(root,out,plan,name,arm)
    receipt=json_pin(root,file_pin(case_location(out,name,arm)/"receipt.json"))
    require(receipt["registration"]==link["sha256"] and receipt["case"]==name and receipt["condition"]==arm
            and receipt["expected_updates"]==p.budget and receipt["real_updates"]==0
            and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False,"Held checkpoint scope differs")
    spec=json_pin(root,receipt["contract"])
    require(spec==p.contract(p.classes,p.seed,p.config.arm,1,p.producer,numeric=receipt["device"]),
            "Independent fit roster or frozen parameters differ")
    state=packet(root,receipt["phases"]["full"]["checkpoint"])
    require(state["registration"]==digest(spec) and state["cursor"]==p.budget
            and receipt["selected"]==min(state["validation"],key=lambda row:row["fused_nll"])["step"]==state["selected"]["step"],
            "Independent first minimum selection differs")
    p.validate_history(spec,state)
    if receipt["exact_reference"]:
        require(all(_exact(state,packet(root,receipt["phases"][key]["checkpoint"])) for key in ("resume","completed")),
                "Independent optimizer/RNG/cursor verification failed")
    require(not {row["tile"] for row in (*spec["training"],*spec["validation"])}&set(p.roles["evaluation"]),
            "An independent point entered support or model selection")
    model=p.model_factory(p.baseline(p.seed),p.config.arm,p.seed+7321)
    model.load_state_dict(state["selected"]["state"],strict=True)
    return p,proof,receipt,spec,state,model.to(execution.device).eval()
