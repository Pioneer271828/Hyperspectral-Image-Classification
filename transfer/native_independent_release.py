"""Validate externally authorized development roles against published Bank history."""

from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts import run_frozen_band_region as archive
from transfer import development_bank as custody
from transfer.development_bank_training import _baseline
from transfer.development_blocks import read_pin
from transfer.native_coverage_queue import generated
from transfer.native_released_points import scores
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_observation, bank_view, counts, release_statistics, retained_positions

MANIFEST = "wp02_independent_development_intake_v1"
AUTHORITY = "wp02_independent_development_approval_v1"
ROLES = {"support": "devtrain", "calibration": "devval", "evaluation": "devval"}
USES = ("independent_accuracy", "cost", "frozen_diagnostics")
COLUMN_VIEWS = ("full", "central_gap", "matched_dispersed", "metadata_missing")
PHYSICAL_COVERAGE_VIEWS = (*COLUMN_VIEWS, "held_range", "misaligned_gap")
FIT_COVERAGE_VIEWS = COLUMN_VIEWS[:3]


def validate_exposure(exposure, views, *, columns):
    allowed = COLUMN_VIEWS if columns else PHYSICAL_COVERAGE_VIEWS
    check(isinstance(exposure,dict) and set(exposure)==ROLES.keys()
          and all(isinstance(row,list) and row and len(row)==len(set(row))
                  and all(v in allowed for v in row) for row in exposure.values())
          and exposure["evaluation"]==views
          and set(exposure["calibration"])<=set(exposure["support"])<=set(FIT_COVERAGE_VIEWS)
          and set(exposure["support"])<set(views) and "full" in exposure["calibration"],
          "Exposure must register optimization, calibration and independently approved held-out observations")
    return deepcopy(exposure)



def column_inventory(bands):
    retained = {view: [bands.band_ids[i] for i in retained_positions(
        bands, "full" if view == "metadata_missing" else view)] for view in COLUMN_VIEWS}
    return dict(axis=list(bands.band_ids), retained=retained)


def observation_view(band, intake):
    if "delivered_columns" not in intake:
        if "observation_exposure" in intake:
            check(band in intake["observation_views"], "Physical coverage view was not approved")
            if band == "metadata_missing":
                return "full", "missing"
            if band == "misaligned_gap":
                return "central_gap", "rejected_axis"
        return band, None
    check(band in intake["observation_views"], "Delivered-column view was not approved")
    return ("full", "missing") if band == "metadata_missing" else (band, "ordinal")


def observation_bands(bands, band, intake):
    original, _ = observation_view(band, intake)
    return [bands.band_ids[i] for i in retained_positions(bands, original)]


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def role_rows(bank, assigned):
    check(set(assigned)==ROLES.keys(),"Explicit support, calibration and evaluation assignment required")
    rows = {}
    used = set()
    for role, internal in ROLES.items():
        ids = assigned[role]
        check(isinstance(ids,list) and all(type(i) is int for i in ids) and ids==sorted(set(ids))
              and not used.intersection(ids),"Independent roles reuse a block or have foreign IDs")
        rows[role] = [bank.role_row(internal,i) for i in ids]
        used.update(ids)
    allowed = {row["id"] for internal in set(ROLES.values()) for row in bank.role_rows(internal)
               if row["blocked"] is None}
    check(used==allowed,"Independence review must account for every published block")
    return rows


def scope_identity(bank):
    return dict(bank=bank.pin["sha256"],registration=bank.pin["registration"],role_geometry=bank.release.plan_hash)


def assess(bank, history, assigned, *, gap, permitted, delivered_columns=False):
    """Return separate qualifications without promoting historical selection labels."""
    check(isinstance(bank,custody.Bank) and history and all(isinstance(p,custody.Bank) for p in history),
          "Published candidate and historical Banks are required")
    check(type(gap) is int and gap>=1 and set(permitted)==set(USES) and
          all(type(v) is bool for v in permitted.values()),"Explicit positive use and buffer contract required")
    release=bank.release
    check(gap>=release.custody["minimum_buffer"] and
          all(gap>=old.release.custody["minimum_buffer"] for old in history),
          "Independent buffer cannot be narrower than a published buffer")
    seen=set()
    historical=[]
    for old in history:
        old.verify_authority(old.pin)
        prior=old.release
        same_axis = (prior.bands.band_ids, prior.bands.valid) == (release.bands.band_ids, release.bands.valid)
        check(prior.scene==release.scene and prior.shape==release.shape
              and (same_axis if delivered_columns else prior.bands==release.bands),
              "Historical comparison requires the same scene coordinates, original band axis and applicable metadata")
        key=old.pin["sha256"]
        check(key not in seen,"Repeated historical Bank")
        seen.add(key)
        historical.extend(prior.rows)

    bank.verify_authority(bank.pin)
    rows=role_rows(bank,assigned)
    occupied={role:{sample for row in vals for sample,_ in row["samples"]} for role,vals in rows.items()}
    parent_ids={role:{r["parent_tile"] for r in vals} for role,vals in rows.items()}
    short_buffers=[]
    for first, second in combinations(ROLES,2):
        if (occupied[first]&occupied[second] or parent_ids[first]&parent_ids[second] or
            any(custody._gap(a["parent_box"],b["parent_box"])<gap for a in rows[first] for b in rows[second])):
            short_buffers.append(f"{first}/{second}")
    historical_samples={i for row in historical for i,_ in row["samples"]}
    historical_parents={row["parent_tile"] for row in historical}
    checks={}
    reasons=[]
    for role, vals in rows.items():
        tally=counts(vals,release.classes)
        past_parents=parent_ids[role]&historical_parents
        past_samples=occupied[role]&historical_samples
        proximity=sum(any(custody._gap(row["parent_box"],old["parent_box"])<gap for old in historical) for row in vals)
        checks[role]=dict(blocks=len(vals),per_class=tally,shared_historical_parents=len(past_parents),
                         shared_historical_samples=len(past_samples),historical_buffer_conflicts=proximity)
        if not vals:
            reasons.append(f"{role}_absent")
        if vals and min(tally)==0:
            reasons.append(f"{role}_classes_missing")
        if past_parents or past_samples or proximity:
            reasons.append(f"{role}_historical_overlap")
    if short_buffers:
        reasons.append("roles_share_spatial_buffer")
    candidates={"independent_accuracy":not reasons,
                "cost":bool(rows["evaluation"]),
                "frozen_diagnostics":bool(occupied["evaluation"])}
    capability={}
    for use, ready in candidates.items():
        enabled=permitted[use] and ready
        reason=None if enabled else ("not_authorized" if not permitted[use] else
                                     reasons if use=="independent_accuracy" else "no_eligible_released_evaluation_blocks")
        capability[use]=dict(eligible=enabled,reason=reason)
    return dict(scene=release.scene, source_mode="synthetic" if release.profile=="synthetic" else "external_real",
                bank=scope_identity(bank), history=[scope_identity(item) for item in history],
                role_reports=checks, role_buffer_conflicts=short_buffers, independent_checks=reasons,
                minimum_buffer=gap, band_axis=release.bands.fingerprint, role_assignment=digest(assigned),
                physical_wavelengths_verified=release.bands.wavelength_quality=="verified",
                uses=capability, optimizer_updates=0, query_accessed=False)


def authorized_document(root, pin):
    return json.loads(read_pin(pin,root))


def bank_from_entry(root, entry):
    check(set(entry)=={"pin","release"} and set(entry["pin"])=={"root","sha256","registration"},
          "Explicit existing Bank pin and release required")
    definition = deepcopy(entry["release"])
    definition["bands"] = BandMetadata(**definition["bands"])
    definition["shape"] = tuple(definition["shape"])
    definition["rows"] = tuple(definition["rows"])
    release = custody.BankRelease(**definition)
    path = Path(entry["pin"]["root"]).resolve(strict=True)
    check(root in path.parents,"Published Bank is outside the authorized runs")
    return custody.Bank(entry["pin"],path.parent,release)


def historical_banks(root, scene=None):
    source, _ = archive.source_documents(root)
    return {name:archive._freeze_bank(root,name,{"sources":source})[0]
            for name in ((scene,) if scene else archive.SCENES)}


def open_independent(root, manifest_pin, authority_pin, *, test_history=None):
    """Open the pinned intake. The release owner provides the separate authority receipt."""
    manifest = authorized_document(root,manifest_pin)
    authority = authorized_document(root,authority_pin)
    base_fields={"schema","candidate","roles","minimum_buffer","history"}
    check(base_fields <= manifest.keys() <= base_fields | {"spatial_annotations", "observation_views", "observation_factors", "delivered_columns", "observation_exposure"},
          "Exact independent manifest fields required")
    views = manifest.get("observation_views")
    columns = "delivered_columns" in manifest
    exposure = None
    if "observation_exposure" in manifest:
        exposure = validate_exposure(manifest["observation_exposure"],views,columns=columns)
    if "observation_views" in manifest:
        check(isinstance(views, list) and views and all(
            view in (COLUMN_VIEWS if columns else PHYSICAL_COVERAGE_VIEWS if exposure else COLUMN_VIEWS[:3]) for view in views)
            and len(set(views)) == len(views), "Explicit unique approved retained-band views required")
    factors = manifest.get("observation_factors")
    if "observation_factors" in manifest:
        check(isinstance(factors,list) and factors and all(type(f) is int and f in (1,2,4) for f in factors)
              and factors==sorted(set(factors)), "Explicit unique approved native lattice factors required")
    annotations=manifest.get("spatial_annotations")
    approved_fields={"schema","approved_manifest_sha256","historical_bank_sha256","uses","source_mode"}
    if "spatial_annotations" in manifest:
        approved_fields.add("spatial_annotations_sha256")
        check(isinstance(annotations,dict) and set(annotations)=={"path","sha256"} and
              authority.get("spatial_annotations_sha256")==annotations["sha256"],
              "Spatial annotations require the same release owner approval")
    check(manifest["schema"]==MANIFEST and set(authority)==approved_fields and authority["schema"]==AUTHORITY
          and authority["approved_manifest_sha256"]==manifest_pin["sha256"],"Independent manifest is not authorized")
    bank=bank_from_entry(root,manifest["candidate"])
    if columns:
        inventory = column_inventory(bank.release.bands)
        check(views == list(COLUMN_VIEWS)
              and digest(manifest["delivered_columns"]) == digest(inventory)
              and len(inventory["retained"]["full"]) > len(inventory["retained"]["central_gap"])
                  == len(inventory["retained"]["matched_dispersed"]),
              "Delivered-column protocol requires the approved actual column order and exact matched deletions")
        if authority["uses"].get("cost") and not authority["uses"].get("independent_accuracy"):
            check(set(authority["uses"]) == set(USES) and not authority["uses"]["frozen_diagnostics"]
                  and "spatial_annotations" not in manifest
                  and all(not row["samples"] for row in bank.release.rows),
                  "Cost-only delivered columns require an observation-only Bank without labels or annotations")
    if exposure and not columns:
        check(bank.release.bands.wavelength_quality=="verified",
              "Physical coverage requires the owner-bound verified original axis")
        if "held_range" in views:
            held = retained_positions(bank.release.bands,"held_range")
            check(all(held!=retained_positions(bank.release.bands,view) for view in exposure["support"]),
                  "Held-out range duplicates an optimization observation")
    supplemental = [bank_from_entry(root,row) for row in manifest["history"]]
    if bank.release.profile=="synthetic":
        check(authority["source_mode"]=="synthetic" and test_history is not None and
              all(old.release.profile=="synthetic" for old in [*test_history,*supplemental]),
              "Synthetic intake requires its isolated engineering history")
        historical = [*test_history,*supplemental]
    else:
        check(authority["source_mode"]=="external_real" and test_history is None
              and all(old.release.profile!="synthetic" for old in supplemental),"Wrong publication authority")
        archived=historical_banks(root,bank.release.scene)
        historical=[*archived.values(),*supplemental]
    check(authority["historical_bank_sha256"]==sorted(old.pin["sha256"] for old in historical),
          "The release owner must bind the history used to establish independence")
    result=assess(bank,historical,manifest["roles"],gap=manifest["minimum_buffer"],permitted=authority["uses"],
                  delivered_columns=columns or exposure is not None)
    if annotations and result["uses"]["independent_accuracy"]["eligible"]:
        from evaluation.wp12_spatial_release import read_annotations

        read_annotations(root,bank,manifest["roles"],annotations)
        result["spatial_annotations"]=annotations
    if views is not None:
        result["observation_views"] = views
    if factors is not None:
        result["observation_factors"] = factors
    if columns:
        result["delivered_columns"] = inventory
    if exposure is not None:
        result["observation_exposure"] = exposure
    return bank,result,manifest["roles"]


def materialize_reviewed(bank, report, assigned, role, tile, producer, band, *, use, factor=1):
    """Only an admitted purpose may consume the exact manifest role and original band view."""
    if (report["bank"]!=scope_identity(bank) or report["role_assignment"]!=digest(assigned)
            or role not in ROLES or use not in USES or type(factor) is not int
            or factor not in report.get("observation_factors",[1])
            or not report["uses"][use]["eligible"] or tile not in assigned[role]
            or (use!="independent_accuracy" and role!="evaluation")
            or ("observation_exposure" in report and band not in report["observation_exposure"][role])):
        raise ValueError("Independent consumption crosses role, publication or approved use")
    observed, metadata = observation_view(band, report)
    if use=="cost" and "delivered_columns" in report:
        row = bank.observation_row(ROLES[role],tile)
        return bank_observation(bank,row,producer,observed,factor=factor,metadata=metadata,
                                observation_only=True),None,None
    row = bank.role_row(ROLES[role],tile)
    if use=="cost":
        return bank_observation(bank,row,producer,observed,factor=factor,metadata=metadata),None,None
    return bank_view(bank,row,producer,observed,factor=factor,metadata=metadata)


def audit_existing(root, out, code):
    output={}
    for scene,bank in historical_banks(root).items():
        existing=dict(support=sorted(r["id"] for r in bank.role_rows("devtrain") if r["blocked"] is None),calibration=[],
                      evaluation=sorted(r["id"] for r in bank.role_rows("devval") if r["blocked"] is None))
        gap=max(1,bank.release.custody["minimum_buffer"])
        output[scene]=assess(bank,[bank],existing,gap=gap,permitted=dict.fromkeys(USES,True))
        check(not output[scene]["uses"]["independent_accuracy"]["eligible"],"Archived split was promoted")
    require_frozen_code(code)
    out.mkdir()
    return archive._write(out/"history-admission.json",dict(producer=code["commit"],scenes=output,
                          query_accessed=False,new_real_labels=False,optimizer_updates=0))


def synthetic_bank(root, name, shift, *, scene="xuzhou", support_seed=512):
    release, arrays=generated(scene,support_seed,5,"a")
    rows=deepcopy(release.rows)
    shape=(80,*release.shape[1:])
    for row in rows:
        key=row["id"]
        low,high,start,stop=row["box"]
        row["id"]+=shift*100
        row["parent_tile"]=row["id"]
        row["box"]=[low+shift,high+shift,start,stop]
        row["parent_box"]=row["box"]
        row["samples"]=[[i+shift*shape[1],label] for i,label in row["samples"]]
        old=arrays.pop(key)
        values=old+np.random.default_rng([key,shift]).normal(size=old.shape).astype(np.float32)
        arrays[row["id"]]=values
        row["origin"]["array_sha256"]=custody.array_hash(values)
    plan=digest([release.plan_hash,shift,shape])
    new=replace(release,shape=shape,plan_hash=plan,statistics=release_statistics(arrays,rows,plan),rows=rows)
    registered=custody.publish(root/name,root,new,lambda row:arrays[row["id"]])
    pin={k:registered[k] for k in ("root","sha256","registration")}
    return custody.Bank(pin,root,new),dict(pin=pin,release=asdict(new))


@torch.no_grad()
def synthetic_demo(root, out, code):
    out.mkdir()
    history,_=synthetic_bank(out,"history",0)
    current,entry=synthetic_bank(out,"independent",40)
    val=sorted(row["id"] for row in current.role_rows("devval"))
    roles=dict(support=sorted(row["id"] for row in current.role_rows("devtrain")),
               calibration=val[::2],evaluation=val[1::2])
    manifest=archive._write(out/"manifest.json",dict(schema=MANIFEST,candidate=entry,roles=roles,minimum_buffer=3,history=[]))
    approval=archive._write(out/"approval.json",dict(schema=AUTHORITY,approved_manifest_sha256=manifest["sha256"],
                            source_mode="synthetic",historical_bank_sha256=[history.pin["sha256"]],uses=dict.fromkeys(USES,True)))
    bank,result,assigned=open_independent(root,manifest,approval,test_history=[history])
    check(all(r["eligible"] for r in result["uses"].values()),"Legitimate independent fixture did not qualify")
    model=_baseline(bank.release,64)
    model.eval()
    role_evidence={}
    for role in ROLES:
        events=[]
        for tile in assigned[role]:
            actual,_,targets=materialize_reviewed(bank,result,assigned,role,tile,code["commit"],"central_gap",
                                                  use="independent_accuracy")
            timing={}
            started=perf_counter()
            output,graph=forward(model,actual,code["commit"],ROLES[role],timed=timing)
            events.append(dict(seconds=perf_counter()-started,graph_seconds=sum(timing["graph_seconds"]),
                               counts=targets.summary()["per_class"],graph_nnz=[graph["q_nnz"], graph["a0_nnz"]],
                               scores=scores(output,targets,detailed=True) if role=="evaluation" else None))
        role_evidence[role]=events
    actual, labels, targets=materialize_reviewed(bank,result,assigned,"evaluation",assigned["evaluation"][0],
                                                  code["commit"],"full",use="cost")
    check(labels is targets is None,"Cost view must withhold released labels")
    diag, _, targets=materialize_reviewed(bank,result,assigned,"evaluation",assigned["evaluation"][0],
                                           code["commit"],"full",use="frozen_diagnostics")
    check(diag.receipt==actual.receipt and targets.summary()["points"]>0,"Reviewed frozen diagnostic differs")
    require_frozen_code(code)
    return archive._write(out/"synthetic-admission.json",dict(producer=code["commit"],report=result,
                          label_blind_cost_view=True, frozen_diagnostic_view=True,
                          manifest=manifest["sha256"],authority=approval["sha256"],roles=role_evidence))
