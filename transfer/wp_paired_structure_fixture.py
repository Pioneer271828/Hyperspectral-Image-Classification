"""Preregister three complementary engineering panels and explicit protection controls."""

from dataclasses import asdict
from itertools import product

import numpy as np
import torch

from evaluation.frozen_training_stages import file_pin, json_pin, require
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_qualification import classification_status
from evaluation.wp12_spatial_release import SCHEMA as GEOMETRY
from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import development_bank as custody, native_independent_release as independent
from transfer.development_blocks import SCENES
from transfer.g3_training import state_hash
from transfer.spatial_coverage_protocol import release_statistics, retained_positions
from transfer.wp_paired_structure import SCHEMA, SYNTHETIC, VIEWS, expected_approval

FIXTURE = "wp05_wp08_external_structure_engineering_v1"
SCENES_BY_KEY = {"xz":"xuzhou","pavia":"paviaU","hh":"whu-honghu"}
OFF = {"backend":"native"}
SCALES = {"backend":"multiscale","trial":{"cnn_multiscale":"content","cnn_dilation":1,
                                           "graph_multiscale":"content","graph_radius":1}}
FIXED = {"backend":"multiscale","trial":{"cnn_multiscale":"fixed","cnn_dilation":1,
                                          "graph_multiscale":"off","graph_radius":1}}


def message(mode,direction="both"):
    return dict(backend="region_message",mode=mode,direction=direction)


def pair(a,b,*,equal=False):
    return dict(baseline=a,candidate=b,comparison="equal_parameters" if equal else "architecture_with_capacity_change")


def designs(devices):
    slots = [*devices[:3]] or ["cpu"]
    science = {}
    xz = {"off":OFF,"fixed1":FIXED,"scale":SCALES,"aligned":{"backend":"aligned"}}
    xz_pairs = [pair("off","fixed1",equal=True),pair("off","scale"),pair("off","aligned")]
    for name,direction in (("both","both"),("cg","cnn_to_graph"),("gc","graph_to_cnn")):
        a,b = (f"{mode}_{name}" for mode in ("local","combined"))
        xz.update({a:message("local",direction),b:message("combined",direction)})
        xz_pairs.append(pair(a,b,equal=True))
    science["xz-s64-f1"] = dict(release="xz",seed=64,factor=1,arms=xz,pairs=xz_pairs,device=slots[0])
    for i,seed in enumerate((64,512,2048)):
        science[f"pavia-s{seed}-f2"] = dict(release="pavia",seed=seed,factor=2,device=slots[i%len(slots)],
                arms=dict(off=OFF,scale=SCALES,message=message("combined")),
                pairs=[pair("off","scale"),pair("off","message"),pair("scale","message")])
    science["hh-s512-f1"] = dict(release="hh",seed=512,factor=1,device=slots[-1],
                 arms=dict(off=OFF,scale=SCALES,message=message("combined")),
                 pairs=[pair("off","scale"),pair("off","message"),pair("scale","message")])
    for group in science.values():
        group.update(steps=24,clock=[0,4,8,12,16,20,24],purpose="science")
    science["xz-recovery"] = dict(release="xz",seed=2048,factor=1,device="cpu",
                arms=dict(local=message("local","graph_to_cnn"),combined=message("combined","graph_to_cnn")),
                pairs=[pair("local","combined",equal=True)],steps=12,clock=[0,4,8,12],purpose="recovery")
    return science


def bank_at(out, scene, layout, shared_state):
    classes = SCENES[scene][1]
    scene_index = tuple(SCENES_BY_KEY.values()).index(scene)
    width = 24
    shape = (224,126,width)
    bands = BandMetadata(f"{FIXTURE}:{scene}",tuple(11+7*i for i in range(width)),
             tuple(430.+13*i+2*scene_index for i in range(width)),"nm","verified",(True,)*width)
    retained = set(retained_positions(bands,"central_gap"))
    all_bands = set(range(width))
    rng = np.random.default_rng(5820+scene_index)
    template = rng.normal(0,.85,(classes,width)).astype(np.float32)
    offset = {"a":4,"b":78,"h":152}[layout]
    rows, arrays, truth_by_tile = [],{},{}
    roles = {role:[] for role in independent.ROLES}
    h,w = 14,18
    for i,role in enumerate(["support"]*4+["calibration"]*2+["evaluation"]*3):
        y,x = offset+(i//4)*23,4+(i%4)*29+(3 if layout=="b" else 0)
        key = offset*100+i
        rng = np.random.default_rng([2084,offset,i,scene_index])
        truth = rng.permutation((np.arange(h*w)%classes+1)).reshape(h,w)
        kind = "removed_only" if i in (1,5,7) else "retained"
        active = sorted(all_bands-retained if kind=="removed_only" else retained)
        data = np.full((h,w,width),.42,np.float32)
        data[:,:,active] += template[truth-1][:,:,active]
        if kind=="retained":
            data[:,:,active] += rng.normal(0,.025,(h,w,len(active))).astype(np.float32)
        samples = []
        for label in range(1,classes+1):
            spots = np.argwhere(truth==label)
            count = 1+int((label+i)%4==0) if role=="support" else (
                5 if role=="calibration" else 7 if scene=="whu-honghu" else 3)
            for a,b in spots[rng.permutation(len(spots))[:count]]:
                samples.append([(y+int(a))*shape[1]+x+int(b),label])
        box = [y,y+h,x,x+w]
        arrays[key],truth_by_tile[key] = data,truth
        roles[role].append(key)
        rows.append(dict(id=key,role=independent.ROLES[role],partition="train" if role=="support" else "val",
            box=box,parent_tile=key,parent_box=box,samples=sorted(samples),representation="raw_fp32",
            origin=dict(array_sha256=custody.array_hash(data),pattern=kind),blocked=None))
    rows.sort(key=custody._row_key)
    plan = digest([FIXTURE,scene,layout,roles,[r["box"] for r in rows]])
    values = release_statistics(arrays,rows,plan)
    release = custody.BankRelease(scene,"synthetic",shape,classes,plan,bands,values,
                dict(shared_state=shared_state,kind=FIXTURE),dict(k=5,minimum_buffer=4,layout=layout,
                roles=digest(roles),support_seed=512),tuple(rows))
    saved = custody.publish(out,out.parent,release,lambda row:arrays[row["id"]])
    return dict(pin={k:saved[k] for k in ("root","sha256","registration")},release=asdict(release)),roles,truth_by_tile


def source(out):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72051)
        model = MultiDatasetVCASAHGCN({"whu-honghu":22},graph_mode="static",norm_type="group_norm",
                                     latent_channels=128,adapter_embed_dim=64).eval()
    path = out/"source.pt"
    save_model_checkpoint(model,path)
    return dict(weights=file_pin(path),metadata=file_pin(path.with_suffix(".pt.json")),state=state_hash(model))


def generate(out,campaign,code,devices,*,qualification=False):
    out.mkdir(parents=True)
    src = source(out)
    releases, plan = {},designs(devices)
    if qualification:
        plan = {k:g for k,g in plan.items() if k in ("xz-recovery","pavia-s64-f2")}
        for g in plan.values():
            g.update(steps=12,clock=[0,4,8,12],purpose="recovery")
    for name in sorted({g["release"] for g in plan.values()}):
        scene = SCENES_BY_KEY[name]
        branch = out/name
        branch.mkdir()
        old,_,_ = bank_at(branch/"historical",scene,"h",src["state"])
        bank,roles,truth = bank_at(branch/"release",scene,"a" if name=="xz" else "b",src["state"])
        # Restore two sources through the exact four P4 families and fresh task graph.
        # HongHu scratch keeps a separate symmetric control for its own training cohort.
        identification = None if name=="hh" else src
        annotation = _write(branch/"geometry.json",dict(schema=GEOMETRY,bank=bank["pin"]["sha256"],
            roles=digest(roles),scope="synthetic",source=digest([FIXTURE,scene,"geometry"]),
            tiles={str(tile):dict(labels=truth[tile].tolist(),geometry=(truth[tile]>0).tolist(),
                                  narrow=np.isin(truth[tile],[15,21] if name=="hh" else [5,6]).tolist())
                   for tile in roles["evaluation"]}))
        manifest = _write(branch/"intake.json",dict(schema=independent.MANIFEST,candidate=bank,roles=roles,
            minimum_buffer=4,history=[old],observation_views=list(VIEWS),observation_factors=[1,2],
            spatial_annotations=annotation))
        owner = _write(branch/"owner-test-only.json",dict(schema=independent.AUTHORITY,
            approved_manifest_sha256=manifest["sha256"],historical_bank_sha256=[old["pin"]["sha256"]],
            uses=dict(independent_accuracy=True,cost=True,frozen_diagnostics=False),source_mode="synthetic",
            spatial_annotations_sha256=annotation["sha256"]))
        releases[name] = dict(manifest=manifest,authority=owner,source=identification,k=5)
    data = dict(schema=SCHEMA,scope=SYNTHETIC,releases=releases,groups=plan)
    request = _write(out/"request.json",data)
    approval = _write(out/"engineering-only.json",expected_approval(data,request,campaign,code,test=True))
    return dict(request=request,approval=approval)


def controls(root, entry, producer):
    from transfer.spatial_coverage_protocol import bank_observation

    bank,report,roles = independent.open_independent(root,entry["manifest"],entry["authority"],test_history=[])
    retained_checks = {}
    for tile,band in product(roles["evaluation"],VIEWS):
        row = bank.role_row("devval",tile)
        actual = bank_observation(bank,row,producer,band)
        raw = actual.view.observation.spectral
        varies = bool(np.ptp(raw,axis=(0,1)).max()>1e-7)
        kind = row["origin"]["pattern"]
        expected = not (kind=="removed_only" and band=="central_gap")
        require(varies==expected,"Planted signal was removed from the wrong physical view or leaked")
        retained_checks[f"{tile}/{band}"] = dict(pattern=kind,variable_spectra=varies)
    if bank.release.scene!="whu-honghu":
        return dict(signal=retained_checks,protection=None)
    truth = [label for tile in roles["evaluation"] for _,label in bank.role_row("devval",tile)["samples"]]
    passed = {}
    base,positive,negative = map(list,(truth,truth,truth))
    for label in range(1,23):
        points = [i for i,y in enumerate(truth) if y==label]
        for idx in points[:3]:
            base[idx] = label%22+1
        if label in (15,21):
            for idx in points[:4]:
                negative[idx] = label%22+1
    for name,guesses in (("target_gain",positive),("rare_sacrifice",negative)):
        scored = pair_scores(truth,base,guesses,22,bank.release.scene,head_kind="fused",guard_population=20)
        require(scored["adequate"] and scored["accepted"]==(name=="target_gain") and scored["delta_aa"]>0,
                "Engineering adoption gates did not reject the sacrificed target category")
        passed[name] = dict(scores=scored,qualification=classification_status(scored,accuracy_allowed=True,
                                                                              cost_allowed=True,real=False))
    return dict(signal=retained_checks,protection=passed)
