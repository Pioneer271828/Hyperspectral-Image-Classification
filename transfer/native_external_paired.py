"""Externally released paired heads through the native optimizer and append-only worker."""

from time import perf_counter

import torch

from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require, stage_plan, load_stage
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts import run_frozen_band_region as archive
from scripts.run_frozen_band_region import _write
from scripts.run_wp12_retained_evidence import RELEASE, pin
from transfer import native_independent_release as independent, native_long_run as journal
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.native_paired_head import ARMS, BUDGET, PairedHeadProvider, freeze_for_head
from transfer.native_production_model import Switches
from transfer.native_production_provider import ProductionProvider
from transfer.p4_runtime_device import UUID

SCHEMA="wp05_wp06_external_paired_campaign_v1"
REAL_AUTHORITY="wp05_wp06_external_paired_execution_authority_v1"
TEST_AUTHORITY="wp05_wp06_external_paired_test_only_authority_v1"
PUBLISH="wp05_wp06_external_paired_selected_v1"
REAL="external_real_execution"
TEST="test_external_format_only"
SCENES={"xuzhou":"xuzhou","honghu":"whu-honghu"}
SEEDS=(64,512,2048)
DEVICE_CPU="cpu"
TEST_STEPS=12
CLOCK_INTERVAL=32


def source_model(root, kind, source, *, test):
    require(set(source)==({"checkpoint","state","weights","sidecar"} if test else {"checkpoint","state"}),
            "Explicit frozen source identity is required")
    if test:
        path=pinned(root,source["weights"])
        metadata=pinned(root,source["sidecar"])
        require(metadata==path.with_suffix(".pt.json") and source["checkpoint"]==source["weights"]["sha256"],
                "Foreign external test source sidecar")
        model=load_model_checkpoint(path,"cpu")
        checkpoint=source["checkpoint"]
    elif kind=="xuzhou":
        plan=stage_plan(root,pin(root,RELEASE))
        rel=json_pin(root,plan["release"])
        spec=plan["stages"]["selected"]
        checkpoint=spec["state"]["sha256"]
        require(checkpoint==source["checkpoint"],"Unexpected frozen real Xuzhou source")
        model=load_stage(root,rel,spec,"selected")
    else:
        from transfer.native_coverage_replay import frozen_model

        parents,docs=archive.source_documents(root)
        bank,intake=archive._freeze_bank(root,"whu-honghu",{"sources":parents})
        model,_,_,_,checkpoint=frozen_model(root,"whu-honghu",bank,intake,docs)
    require(source["checkpoint"]==checkpoint and source["state"]==state_hash(model)
            and SCENES[kind] in model.heads and model.backbone.graph_mode=="learned",
            "Model source, architecture or checkpoint differs")
    model.backbone.adapter.cache_attention=False
    model.backbone.adapter.clear_attention_cache()
    return model.eval()


def expected_approval(request, request_pin, out, code, *, test):
    updates=sum(len(row["model_seeds"])*len(ARMS[row["kind"]])*(TEST_STEPS if test else BUDGET[row["kind"]])
                for row in request["releases"].values())
    return dict(schema=TEST_AUTHORITY if test else REAL_AUTHORITY,decision="approved",
        scope=TEST if test else REAL,request=request_pin,producer=code["commit"],source_hash=code["source_hash"],
        destination=str(out),updates=updates,raw_gt_authorized=False,query_authorized=False)


def read_admission(root,out,code,request_pin,approval_pin,*,test):
    require(request_pin is not None and approval_pin is not None,"Independent release and execution approval are required")
    request,approval=(json_pin(root,item) for item in (request_pin,approval_pin))
    require(set(request)=={"schema","scope","releases"} and request["schema"]==SCHEMA
            and request["scope"]==(TEST if test else REAL) and bool(request["releases"]),
            "Foreign campaign request or missing external releases")
    for row in request["releases"].values():
        eligible(row,test=test)
    require(approval==expected_approval(request,request_pin,out,code,test=test),
            "Separate trusted execution approval does not bind the requested scope, source, budget or destination")
    return request,approval


def open_request_bank(root,out,row,*,test):
    bank,result,roles=independent.open_independent(root,row["manifest"],row["authority"],
                                                   test_history=[] if test else None)
    location=bank.root
    require(location not in (out,*out.parents) and out not in location.parents,
            "Optimizer output overlaps an external release")
    supports=result["role_reports"]["support"]
    calibration=result["role_reports"]["calibration"]
    require((bank.release.profile=="synthetic")==test and bank.release.scene==SCENES[row["kind"]]
            and bank.release.custody["support_seed"]==row["support_seed"]
            and bank.release.custody.get("spatial_repeat")==row["spatial_repeat"]
            and supports["per_class"]==[row["k"]]*bank.release.classes
            and all(n>0 for n in calibration["per_class"])
            and "support/calibration" not in result["role_buffer_conflicts"],
            "External K coverage, spatial repeat, support/calibration population or buffer differs")
    return bank,result,roles


def eligible(row,*,test):
    require(set(row)=={"kind","k","support_seed","spatial_repeat","manifest","authority","source","model_seeds",
                       "initialization","device"}
            and row["kind"] in SCENES and row["k"] in (1,5,10) and row["support_seed"] in SEEDS
            and isinstance(row["spatial_repeat"],str) and row["spatial_repeat"]
            and row["model_seeds"]==sorted(set(row["model_seeds"])) and row["model_seeds"]
            and all(seed in SEEDS for seed in row["model_seeds"])
            and (bool(UUID.fullmatch(row["device"])) or test and row["device"]==DEVICE_CPU),
            "An actual released K/support/space/device identity is required")
    require(row["initialization"]=="seeded_head" or
            row["initialization"]=="inherit_head" and len(row["model_seeds"])==1,
            "An inherited identical head has one effective model initialization")


def register(root,out,code,request_pin,approval_pin,*,test):
    req,approved=read_admission(root,out,code,request_pin,approval_pin,test=test)
    releases={}
    for key,row in req["releases"].items():
        require(key.isascii() and key.isalnum(),"Use stable alphanumeric publication keys")
        eligible(row,test=test)
        bank,report,roles=open_request_bank(root,out,row,test=test)
        source=source_model(root,row["kind"],row["source"],test=test)
        require(source.heads[bank.release.scene].in_features==64
                and source.heads[bank.release.scene].out_features==bank.release.classes,
                "Source head and external class order differ")
        releases[key]=dict(request=row,bank=bank.pin,roles=digest(roles),band_axis=bank.release.bands.fingerprint,
                           intake=report,metadata_only_reads=bank.reads)
        require(bank.reads==0,"Registration opened retained perception data")
    require(len({row["bank"]["sha256"] for row in releases.values()})==len(releases)
            and len({(r["kind"],r["k"],r["support_seed"],r["spatial_repeat"])
                  for r in req["releases"].values()})==len(releases),
            "Duplicated Bank or spatial protocol is not independent support evidence")
    doc=dict(schema=SCHEMA,scope=req["scope"],producer=code["commit"],source_hash=code["source_hash"],
        request=request_pin,approval=approval_pin,approved_updates=approved["updates"],releases=releases,
        query_accessed=False,raw_gt_loaded=False)
    require_frozen_code(code)
    out.mkdir(parents=True,exist_ok=True)
    output=out/"registration.json"
    if output.exists():
        value=json_pin(root,file_pin(output))
        require(value==doc,"Existing external campaign has a different admission")
        return file_pin(output)
    return _write(output,doc)


def load(root,out,code,*,test):
    link=file_pin(out/"registration.json")
    plan=json_pin(root,link)
    require(plan["schema"]==SCHEMA and plan["producer"]==code["commit"] and plan["source_hash"]==code["source_hash"]
            and plan["scope"]==(TEST if test else REAL) and plan["query_accessed"] is plan["raw_gt_loaded"] is False,
            "External worker scope changed")
    request,approval=read_admission(root,out,code,plan["request"],plan["approval"],test=test)
    require({key:record["request"] for key,record in plan["releases"].items()}==request["releases"]
            and approval["updates"]==plan["approved_updates"],"External approval no longer matches the registration")
    return plan,link


class ExternalPairedProvider(PairedHeadProvider):
    """Exact external admission wrapper using the existing paired-head optimizer."""

    def __init__(self,root,out,code,release_key,seed,arm,*,test):
        plan,link=load(root,out,code,test=test)
        record=plan["releases"][release_key]
        request=record["request"]
        eligible(request,test=test)
        kind=request["kind"]
        require(seed in request["model_seeds"] and arm in ARMS[kind],"Unknown paired external seed or arm")
        bank,report,roles=open_request_bank(root,out,request,test=test)
        require(bank.pin==record["bank"] and report==record["intake"] and digest(roles)==record["roles"]
                and bank.release.bands.fingerprint==record["band_axis"],"External role, order or Bank receipt differs")
        ProductionProvider.__init__(self,root,bank,roles,axis=bank.release.bands.band_ids,config=Switches(),
                     source=request["source"],seed=seed,budget=TEST_STEPS if test else BUDGET[kind],
                     producer=code["commit"],approval=None if test else plan["approval"]["sha256"])
        self.kind,self.policy=kind,arm
        self.registration,self.intake,self.release_key,self.request=link,report,release_key,request
        self.test=test

    def baseline(self,seed):
        model=source_model(self.root,self.kind,self.source,test=self.test)
        if self.request["initialization"]=="seeded_head":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model.heads[self.scene].reset_parameters()
        return freeze_for_head(model,self.scene)

    def model_factory(self,source,arm,seed):
        require(arm=="native","External paired model uses the unchanged original backbone")
        return freeze_for_head(source,self.scene)

    def contract(self,classes,seed,arm,factor,producer,*,numeric=None):
        require(numeric is not None and (numeric["physical_uuid"] or DEVICE_CPU)==self.request["device"],
                "Paired training cannot change its approved physical device")
        body=super().contract(classes,seed,arm,factor,producer,numeric=numeric)
        body.update(prefix=self.budget//3,external_registration=self.registration["sha256"],
                    independent_role_intake=digest(self.intake),initialization_mode=self.request["initialization"],
                    real_updates=0 if self.test else self.budget)
        return body

    def validate_history(self,spec,packet):
        super().validate_history(spec,packet)
        expected=spec["initialization"]["inventory"]
        for states in (packet["model"],packet["selected"]["state"]):
            require(all(tensor_hash(value)==expected[name] for name,value in states.items()
                        if name not in spec["trainable"]),"External worker modified a frozen source tensor")

    def consume(self,tile,band,*,use):
        return independent.materialize_reviewed(self.bank,self.intake,self.roles,"evaluation",tile,
                                                self.producer,band,use=use)


def arm_dir(out,release,seed,arm):
    return out/"cases"/release/f"s{seed}"/arm


def execute(root,out,code,release,seed,arm,execution,*,test,until=None,reference=False):
    from transfer.native_scale_training import Run

    p=ExternalPairedProvider(root,out,code,release,seed,arm,test=test)
    require(not reference or test,"A reference duplicate is only authorized for external-format fixtures")
    require((execution.numeric["physical_uuid"] or DEVICE_CPU)==p.request["device"],
            "Alternate physical device cannot update a registered case")
    until=p.budget if until is None else until
    require(type(until) is int and 0<until<=p.budget,"Invalid external drain cursor")
    place=arm_dir(out if not reference else out/"reference",release,seed,arm)
    place.parent.mkdir(parents=True,exist_ok=True)
    with case_lock(place), journal.drain_on_signal() as drain:
        spec=p.contract(p.classes,p.seed,"native",1,p.producer,numeric=execution.numeric)
        reg=place/"contract.json"
        if not reg.exists():
            _write(reg,spec)
        require(json_pin(root,file_pin(reg))==spec,"Existing paired worker contract differs")
        key=digest(spec)
        rows=journal.checkpoints(root,place,key,steps=p.budget)
        done=place/"receipt.json"
        if rows and rows[-1]["step"]==p.budget:
            receipt=finish(root,place,code,p,spec,rows)
            return dict(status="complete",case=release,seed=seed,arm=arm,updates=0,receipt=receipt)
        require(not done.exists() and (not reference or not rows),"Incomplete trajectory has a foreign marker")
        with_run=Run(spec,p.producer,execution=execution,released=p)
        cursor=rows[-1]["step"] if rows else 0
        first=cursor
        require(until>=cursor,"Cannot move an external head worker backwards")
        if rows:
            with_run.restore(journal.packet(root,rows[-1]["pin"])["training"],cursor)
        else:
            start=perf_counter()
            before=dict(with_run.timing)
            with_run.start()
            rows.append(journal.append(place,with_run,None,0,0,key,before,"initial",start))
        while cursor<until and not drain["requested"]:
            start=perf_counter()
            prior=dict(with_run.timing)
            previous=cursor
            stop=min(until,(cursor//CLOCK_INTERVAL+1)*CLOCK_INTERVAL)
            for item in spec["training"][cursor:stop]:
                if drain["requested"]:
                    break
                with_run.update(item)
                cursor=item["step"]
            if cursor==previous:
                break
            p.bank.verify_authority(p.bank.pin)
            rows.append(journal.append(place,with_run,rows[-1]["pin"]["sha256"],previous,cursor,key,prior,
                                           drain["requested"] or ("complete" if cursor==p.budget else "drain"),start))
        if cursor==p.budget:
            receipt=finish(root,place,code,p,spec,rows)
            return dict(status="complete",case=release,seed=seed,arm=arm,updates=cursor-first,receipt=receipt)
        return dict(status="drained",case=release,seed=seed,arm=arm,updates=cursor-first,cursor=cursor,
                    signal=drain["requested"],checkpoint=rows[-1]["pin"])


def finish(root,place,code,p,spec,rows):
    last=rows[-1]
    saved=journal.packet(root,last["pin"])["training"]
    require(saved["cursor"]==p.budget and sum(x["updates"] for x in rows)==p.budget,
            "External paired schedule is incomplete")
    p.validate_history(spec,saved)
    best=min(saved["validation"],key=lambda row:row["fused_nll"])
    require(best["step"]==saved["selected"]["step"],"Unregistered external selection")
    receipt=place/"receipt.json"
    body=dict(schema=SCHEMA,status="complete",registration=p.registration["sha256"],release=p.release_key,
             arm=p.policy,seed=p.seed,contract=file_pin(place/"contract.json"),source=p.source,
             checkpoint=last["pin"],selected=best["step"],updates=p.budget,scope=TEST if p.test else REAL,
             real_updates=0 if p.test else p.budget,query_accessed=False,raw_gt_loaded=False)
    require_frozen_code(code)
    if not receipt.exists():
        return _write(receipt,body)
    require(json_pin(root,file_pin(receipt))==body,"Existing external final state differs")
    return file_pin(receipt)
