"""Exercise three released roles and holder-provided geometry through the native Bank."""

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, require
from evaluation.linear_probe import POLICY, fingerprint, prepare
from evaluation import wp12_block_readouts as scoring, wp12_native_pathways as pathways, wp12_spatial_release as spatial
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent, native_paired_head_fixture as fixture
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.native_external_paired_fixture import release_bank
from transfer.native_scale_observation import forward


def rejected(call,match):
    try:
        call()
    except ValueError as error:
        require(match in str(error),"Rejected for an unrelated reason")
        return str(error)
    raise ValueError("Foreign annotation or readout was accepted")


def annotation_fixture(out,entry,roles):
    kind = "honghu"
    _,classes,channels,_,seed,shape,_ = fixture.SPECS[kind]
    names = fixture.master(kind,channels)
    result = {}
    for tile in roles["evaluation"]:
        _,truth = fixture.field(kind,names,classes,*shape,"erased" if tile==7 else "retained",
                                tile,seed,tile==8)
        geometry = truth>0
        narrow = np.isin(truth,[15,21])
        result[str(tile)] = dict(labels=truth.tolist(),geometry=geometry.tolist(),narrow=narrow.tolist())
    return _write(out/"trusted-test-geometry.json",dict(schema=spatial.SCHEMA,bank=entry["pin"]["sha256"],
                roles=digest(roles),scope="synthetic",source=digest(["planted_roles",str(out)]),tiles=result))


@torch.no_grad()
def observe_roles(root,out,code,execution):
    out.mkdir(parents=True)
    kind = "honghu"
    old,_,_ = release_bank(out/"historical",kind,1,64,"h")
    now,roles,release = release_bank(out/"test-candidate",kind,5,512,"a")
    annotation = annotation_fixture(out,now,roles)
    manifest = _write(out/"intake.json",dict(schema=independent.MANIFEST,candidate=now,roles=roles,
                        history=[old],minimum_buffer=8,spatial_annotations=annotation))
    owner = dict(schema=independent.AUTHORITY,approved_manifest_sha256=manifest["sha256"],
                  historical_bank_sha256=[old["pin"]["sha256"]],source_mode="synthetic",
                  uses=dict(independent_accuracy=True,cost=True,frozen_diagnostics=False))
    wrong = _write(out/"unbound-test-authority.json",owner)
    refusal = {"unbound_annotations":rejected(lambda:independent.open_independent(root,manifest,wrong,test_history=[]),
                                               "same release owner approval")}
    owner["spatial_annotations_sha256"] = annotation["sha256"]
    approval = _write(out/"test-authority.json",owner)
    bank,report,assigned = independent.open_independent(root,manifest,approval,test_history=[])
    certified = spatial.read_annotations(root,bank,assigned,report["spatial_annotations"])
    blocked = {**owner,"uses":{**owner["uses"],"independent_accuracy":False}}
    blocked_approval = _write(out/"cost-only-authority.json",blocked)
    bb,rr,aa = independent.open_independent(root,manifest,blocked_approval,test_history=[])
    _,labels,targets = independent.materialize_reviewed(bb,rr,aa,"evaluation",aa["evaluation"][0],
                                     code["commit"],"full",use="cost")
    require("spatial_annotations" not in rr and labels is targets is None,"Cost-only scope revealed class geometry")
    model = _baseline(release,64).to(execution.device).eval().requires_grad_(False)
    identity = state_hash(model)
    views = {}
    for band in fixture.VIEWS:
        views[band] = {}
        for role,tiles in assigned.items():
            groups = {name:[] for name in scoring.LAYERS}
            labels,ids,block_ids,ann = [],[],[],[]
            native = {head:[] for head in HEADS}
            for tile in tiles:
                actual,_,target = independent.materialize_reviewed(bank,report,assigned,role,tile,
                                         code["commit"],band,use="independent_accuracy")
                if not target.points:
                    continue
                clean,q,a,old_native = pathways.capture_call(model,
                       lambda:forward(model,actual,code["commit"],independent.ROLES[role],execution=execution)[0])
                _,features,_,_ = pathways.spatial_forward(model,release.scene,clean,q,a,features=True)
                labels.extend(y for _,y in target.points)
                ids.extend(i for i,_ in target.points)
                block_ids.extend([tile]*len(target.points))
                for layer in groups:
                    groups[layer].append(features[layer][target.cells].detach().cpu())
                for head,result in zip(HEADS,old_native,strict=True):
                    native[head].extend((result[target.cells].argmax(1)+1).tolist())
                if role=="evaluation":
                    ann.append(spatial.classify(bank,assigned,certified,role,tile,target.points))
            views[band][role] = dict(labels=labels,ids=ids,blocks=block_ids,annotations=ann,native=native,
                                      layers={name:torch.cat(parts) for name,parts in groups.items()})
    placebo = views["full"]["evaluation"]
    refusal["annotation_crossed_role"] = rejected(lambda:spatial.classify(bank,assigned,certified,"support",
                  assigned["support"][0],[]),"authenticated evaluation")
    roles_data = views["full"]
    pack = dict(identity={"query_labels_accessed":False},features={layer:{
           label:roles_data[role]["layers"][layer] for label,role in (("train","support"),("val","calibration"))}
           for layer in scoring.LAYERS},labels={k:torch.tensor(roles_data[v]["labels"],dtype=torch.int64)-1
               for k,v in (("train","support"),("val","calibration"))},
           row_ids={k:torch.tensor(roles_data[v]["ids"],dtype=torch.int64)
               for k,v in (("train","support"),("val","calibration"))})
    native_strata = {name:[] for name in spatial.STRATA}
    offset = 0
    for row in placebo["annotations"]:
        for name,flags in row["strata"].items():
            native_strata[name].extend(offset+i for i,flag in enumerate(flags) if flag)
        offset += row["n"]
    require(offset==len(placebo["labels"]) and all(native_strata.values()),
            "Boundary, interior and independently supplied narrow regions must all occur")
    spatial_results = {name:spatial.score_stratum(placebo["labels"],placebo["native"],
         views["central_gap"]["evaluation"]["native"],indices,release.classes,release.scene,
         threshold=20,real=False) for name,indices in native_strata.items()}
    require(all(not row["qualification"]["fused"]["protection_evaluable"] for row in spatial_results.values()),
            "Small strata were promoted to formal protection")
    fitted = {}
    for seed in scoring.SEEDS:
        plan = dict(policy=POLICY,width=64,classes=release.classes,seed=seed,
                    steps=500,lr=.1,l2=.0001,scale_floor=1e-12)
        _,_,norm = prepare(pack,plan)
        # Explicit frozen engineering weights. No optimizer or selection pass is called.
        gen = torch.Generator().manual_seed(seed)
        frozen = {name:dict(weight=torch.randn(release.classes,64,generator=gen,dtype=torch.float64)*.01,
                             bias=torch.zeros(release.classes,dtype=torch.float64)) for name in scoring.LAYERS}
        fit_sha = fingerprint([norm,frozen])
        scored = {}
        for band,members in views.items():
            scored[band] = {}
            for role,data in members.items():
                receipt = dict(role=role,model=identity,fit=fit_sha,blocks=digest(data["blocks"]),
                        population=digest([data["ids"],data["labels"]]),
                        observation=digest([bank.pin,role,band]),approved_independent=role=="evaluation")
                record,_ = scoring.role_score(role,receipt,receipt,data["layers"],data["labels"],data["ids"],
                                               data["blocks"],frozen,norm,plan)
                require(record["independent"]==(role=="evaluation"),"Support or selection was treated as independent")
                scored[band][role] = record
        paired = {band:{role:scoring.paired(scored["full"][role],scored[band][role]) for role in scoring.ROLES}
                  for band in fixture.VIEWS[1:]}
        refusal["cross_role_readout_pair"] = rejected(lambda:scoring.paired(scored["full"]["support"],
                                          scored["full"]["calibration"]),"identical model, fit and ordered role")
        fitted[str(seed)] = dict(fit=fit_sha,plan=plan,scores=scored,paired=paired)
    require(state_hash(model)==identity,"Frozen readout handling modified original weights")
    execution.sync()
    return dict(schema="wp12_native_three_roles_test_v1",producer=code["commit"],model=identity,bank=bank.pin,
        manifest=manifest,approval=approval,spatial=spatial_results,probes=fitted,negative=refusal,
        cost_annotations_hidden=True,optimizer_updates=0,raw_gt_loaded=False,query_accessed=False,
        scope="synthetic_external_format_only")
