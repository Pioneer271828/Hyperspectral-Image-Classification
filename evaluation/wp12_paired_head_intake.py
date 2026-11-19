"""Read-only release audit and runnable role preflight for future paired heads."""

from pathlib import Path
import json

import torch

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require, stage_plan
from evaluation.wp12_mechanism_real import LOCKED_PINS
from scripts import run_frozen_band_region as archive
from scripts.run_wp12_retained_evidence import RELEASE, pin
from transfer import native_independent_release as independent
from transfer.g3_training import state_hash
from transfer.native_paired_head import ARMS, BUDGET, freeze_for_head
from transfer.native_paired_head_campaign import GATES
from transfer.native_paired_head_fixture import SEEDS
from transfer.native_coverage_replay import frozen_model

SCHEMA="wp12_paired_real_custody_v1"
CASE="s512-k5-shared_transfer-none"
SPLIT=("wp05-coverage-real-execution01/"+CASE+"/split/state.pt",
       "88539529de4c77e952b5ffbc8723e55af16ac663e3b819cf8daec0821e93ca8f")
HISTORY=("wp05-wp06-frozen-stage-20261119-qa/verified/wp12-stages.json",
         "e4b1d9732cd89cbac4276f3ba51bdcc3889dd2dbb6c57e768b4e7c5724bd1f70")
WP11=("wp11-confirmation-binding-cli01/release.json","26ea62ffc1f2bd8ddebe4092be526af73c664afa2e8df04f1df56ff7cdda0d7c")
REQUEST="wp05_wp06_paired_head_read_only_request_v1"
AUTHORITY="wp05_wp06_paired_head_read_only_authority_v1"


def candidates(root):
    matches=[]
    for design in ("*/approval.json","*/*/approval.json","*/*/*/approval.json",
                   "*/authority.json","*/*/authority.json","*/*/*/authority.json"):
        for path in sorted(root.glob(design)):
            if not path.is_file() or path.stat().st_size>256*1024:
                continue
            try:
                row=json.loads(path.read_bytes())
            except (UnicodeError,ValueError):
                continue
            if type(row) is dict and row.get("schema")==independent.AUTHORITY:
                matches.append(dict(sha256=file_pin(path)["sha256"],mode=row.get("source_mode"),
                        manifest=row.get("approved_manifest_sha256"),uses=row.get("uses")))
    return matches


def hung_historical(root):
    parents,docs=archive.source_documents(root)
    exact=next(row for row in docs["hh_scope"]["cases"] if row["case"]["id"]==CASE)
    proof=exact["state"]
    forward_pin=dict(path=proof["path"],sha256=proof["state_sha256"])
    left,right=(torch.load(pinned(root,entry),map_location="cpu",weights_only=True)
                for entry in (forward_pin,pin(root,SPLIT)))
    require(left["identity"]==right["identity"]==proof["identity"]
            and left["cursor"]==right["cursor"]==24 and left["best"]["step"]==right["best"]["step"]==24
            and left["history"]==right["history"],"Historical same-protocol short states differ")
    loads=sorted(str(path.relative_to(root)) for path in Path(proof["path"]).parent.parent.glob("*/*.pt"))
    require(loads==sorted([str(Path(proof["path"]).relative_to(root)),SPLIT[0]]),"Re-audit newly found saved state files")
    known=json_pin(root,pin(root,("wp05-wp06-production-20261119-qa/locked-v2/whu-honghu.json",
                                 LOCKED_PINS["whu-honghu"])))
    stages={}
    for event in left["history"]:
        heads={}
        for name,score in event["metrics"].items():
            matrix=score["confusion"]
            metrics=metrics_from_confusion(matrix,range(1,23))
            require(all(item["support"]==5 for item in metrics["per_class"]),"Historical denominator differs")
            heads[name]=dict(oa=metrics["oa"],aa=metrics["aa"],total=metrics["count"],confusion=matrix,
                              focus=[dict(class_id=i,correct=matrix[i-1][i-1],support=sum(matrix[i-1])) for i in (15,21)])
            if event["step"]==24:
                observed=known["result"]["views"]["full"]["native_scores"][name]["confusion"]
                require(matrix==observed,"Existing same-point full-band replay differs from old checkpoint")
        stages[str(event["step"])]=heads
    prior=json_pin(root,pin(root,HISTORY))["honghu"]
    return dict(protocol=CASE,guarded_support_seed=512,model_seed=64,k=5,physical_saved_cursors=[24,24],
        same_geometry_and_role=True,history_scope=parents["hh_scope"]["sha256"],saved_weights_24=proof["state_sha256"],
        candidate_scope_differences=dict(older_or_longer_trials_same_protocol=False,
                other_120_update_accepted=prior["accepted"],related_bank_with_no_matching_120_files=sorted(prior["missing"]),
                reason="other budget cohorts differ in support seed, model seed, K or initialization"),
        observations=stages,history_22_weight_materialized=False,independent_model_receipt_at_24=True,
        never_used_alternative_whole_scene_K10_to_replace_K5=True,query_accessed=False,raw_gt_loaded=False)


def inspect(root):
    old=json_pin(root,pin(root,WP11))
    handed=json_pin(root,old["handoff"])
    design=json_pin(root,handed["confirmation_design"])
    require("/fixture/" in handed["confirmation_design"]["path"],
            "External release candidate requires individual authority intake")
    prior=stage_plan(root,pin(root,RELEASE))
    history={}
    for scene,bank in independent.historical_banks(root).items():
        if scene not in GATES["targets"]:
            continue
        roles=dict(support=[x["id"] for x in bank.role_rows("devtrain")],calibration=[],
                   evaluation=[x["id"] for x in bank.role_rows("devval")])
        report=independent.assess(bank,[bank],roles,gap=max(1,bank.release.custody["minimum_buffer"]),
                                  permitted=dict.fromkeys(independent.USES,True))
        require(not report["uses"]["independent_accuracy"]["eligible"],"Historical dev labels cannot be promoted")
        history[scene]=dict(band_quality=bank.release.bands.wavelength_quality,bank=bank.pin["sha256"],
                            report=report)
    recognized=candidates(root)
    return dict(schema=SCHEMA,scope="read_only_inventory",scenes=history,authority_candidates=recognized,
        external_candidates_found=sum(row["mode"]=="external_real" for row in recognized),
        previous_confirmation_artifact=dict(sha256=WP11[1],scope="synthetic_fixture",
                                            design_sha256=handed["confirmation_design"]["sha256"],
                                            only_outer_metadata_read=True),
        previous_xuzhou=dict(release=prior["release"]["sha256"],development=prior["development"]["sha256"],
            original_support_k=10,active_seed_or_roles_for_new_trials=None,original_steps=prior["budget"]),
        honghu=hung_historical(root),new_independent_role_or_paired_optimizer_authorization=False,
        raw_gt_loaded=False,query_accessed=False,real_updates=0,independent_real_accuracy=None)


def preflight(root, request_pin, approval_pin, producer):
    request,approval=(json_pin(root,item) for item in (request_pin,approval_pin))
    fields={"schema","kind","independent_manifest","independent_authority","model_seeds","support_seed",
            "scope_checkpoint","comparison"}
    require(set(request)==fields and request["schema"]==REQUEST and request["kind"] in ("xuzhou","honghu")
            and request["model_seeds"]==list(SEEDS) and type(request["support_seed"]) is int,
            "Explicit paired scene, roles and model/label seeds are required")
    kind=request["kind"]
    require(request["comparison"]==dict(arms=list(ARMS[kind]),steps=BUDGET[kind],dimension=64),
            "Paired request changes the locked model budgets or heads")
    require(approval==dict(schema=AUTHORITY,approved_request_sha256=request_pin["sha256"],producer=producer,
                            allowed_use="read_only_preflight",train_authorized=False),
            "Separate trusted read-only authority required")
    bank,result,_=independent.open_independent(root,request["independent_manifest"],request["independent_authority"])
    require(bank.release.profile!="synthetic" and bank.release.scene==("xuzhou" if kind=="xuzhou" else "whu-honghu")
            and result["uses"]["independent_accuracy"]["eligible"] and result["uses"]["cost"]["eligible"],
            "For real use all spatial, classification and cost roles must first be admitted")
    roles=result["role_reports"]
    k=10 if kind=="xuzhou" else 5
    require(roles["support"]["per_class"]==[k]*bank.release.classes and
            min(roles["evaluation"]["per_class"])>=GATES["minimum_real_confirmation_per_class"],
            "Independent real support K or confirmation class denominator is insufficient")
    if kind=="xuzhou":
        from evaluation import frozen_training_stages as stages

        plan=stage_plan(root,pin(root,RELEASE)); release=json_pin(root,plan["release"])
        spec=plan["stages"]["selected"]; model=stages.load_stage(root,release,spec,"selected")
        checkpoint=spec["state"]["sha256"]
    else:
        sources,docs=archive.source_documents(root)
        historical_bank,intake=archive._freeze_bank(root,"whu-honghu",{"sources":sources})
        model,_,_,_,checkpoint=frozen_model(root,"whu-honghu",historical_bank,intake,docs)
    require(request["scope_checkpoint"]==checkpoint,"Future paired scope uses an unexpected frozen state")
    require(bank.release.custody.get("support_seed") == request["support_seed"],
            "Published support draw differs from the paired request")
    model=freeze_for_head(model,bank.release.scene)
    return dict(schema=SCHEMA,scope="authorized_read_only_preflight",scene=bank.release.scene,comparison=request["comparison"],
                bank=bank.pin["sha256"],checkpoint=checkpoint,model_state=state_hash(model),role_population=roles,
                trainable=sum(param.numel() for param in model.parameters() if param.requires_grad),
                query_accessed=False,raw_gt_loaded=False,real_updates=0,independent_real_accuracy=None)
