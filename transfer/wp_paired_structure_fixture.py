"""Preregister three complementary engineering panels and explicit protection controls."""

from dataclasses import asdict
from itertools import product

import numpy as np
import torch

from evaluation.frozen_training_stages import file_pin, json_pin, require
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_qualification import classification_status
from evaluation.wp12_spatial_release import SCHEMA as GEOMETRY
from evaluation.wp12_scientific_claim import COLUMNS, PAVIA
from transfer.wp_support_match import MODE as MATCHED_EXPOSURE
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
    if len(devices)==1:
        slots.append("cpu")
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


def bank_at(out, scene, layout, shared_state, *, k=5, support_seed=512, matched=False,
            columns=False, observation_only=False, regional_signals=False):
    classes = SCENES[scene][1]
    scene_index = tuple(SCENES_BY_KEY.values()).index(scene)
    width = 24
    shape = (224,126,width)
    bands = BandMetadata(f"{FIXTURE}:{scene}",tuple(11+7*i for i in range(width)),
             tuple(430.+13*i+2*scene_index for i in range(width)),"nm","verified",(True,)*width)
    if columns:
        bands = BandMetadata(bands.source_id,bands.band_ids,tuple(np.linspace(0.,1.,width)),
                             "normalized","index_only",(True,)*width)
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
        if regional_signals:
            kind = "retained"
            data,truth,signal = planted_region_tile(h,w,classes,i,width,retained,bands,template,rng)
        samples = []
        for label in range(1,classes+1):
            spots = np.argwhere(truth==label)
            count = 1+int((label+i)%4==0) if role=="support" else (
                5 if role=="calibration" else 7 if scene=="whu-honghu" else 3)
            selection = rng
            if matched:
                if role == "support":
                    count = k if (label-1) % 4 == i else 0
                    selection = np.random.default_rng([support_seed,offset,scene_index,label])
                elif role == "evaluation":
                    count = 7
            if observation_only:
                count = 0
            for a,b in spots[selection.permutation(len(spots))[:count]]:
                samples.append([(y+int(a))*shape[1]+x+int(b),label])
        box = [y,y+h,x,x+w]
        arrays[key],truth_by_tile[key] = data,truth
        roles[role].append(key)
        rows.append(dict(id=key,role=independent.ROLES[role],partition="train" if role=="support" else "val",
            box=box,parent_tile=key,parent_box=box,samples=sorted(samples),representation="raw_fp32",
            origin=dict(array_sha256=custody.array_hash(data),pattern=kind,
                        **({"regional_signal":signal} if regional_signals else {})),blocked=None))
    rows.sort(key=custody._row_key)
    plan = digest([FIXTURE,scene,layout,roles,[r["box"] for r in rows]])
    values = release_statistics(arrays,rows,plan)
    release = custody.BankRelease(scene,"synthetic",shape,classes,plan,bands,values,
                dict(shared_state=shared_state,kind=FIXTURE),dict(k=k,minimum_buffer=4,layout=layout,
                roles=digest(roles),support_seed=support_seed),tuple(rows))
    saved = custody.publish(out,out.parent,release,lambda row:arrays[row["id"]])
    return dict(pin={k:saved[k] for k in ("root","sha256","registration")},release=asdict(release)),roles,truth_by_tile


def planted_region_tile(h,w,classes,index,width,retained,bands,template,rng):
    """Insert the only distinctive class-specific feature into a known emitting area."""
    rare = (15,21) if classes==22 else (5,6)
    truth = np.zeros((h,w),dtype=np.int64)
    target_rows = [(0,5),(h-5,h)]
    if index%2:
        target_rows.reverse()
    for label,(start,end) in zip(rare,target_rows,strict=True):
        truth[start:end,:3] = label
    ordinary = [label for label in range(1,classes+1) if label not in rare]
    truth[truth==0] = rng.permutation(np.resize(ordinary,int((truth==0).sum())))
    reference = truth.copy()
    reference[np.isin(reference,rare)] = rare[0]
    data = np.full((h,w,width),.42,np.float32)
    base = sorted(retained)
    data[:,:,base] += template[reference-1][:,:,base]
    data[:,:,base] += rng.normal(0,.025,(h,w,len(base))).astype(np.float32)
    cue = sorted(set(range(width))-retained)
    regime = ("local","across_region","deleted")[index%3]
    sources = {}
    for sign,label,(start,end) in zip((-1,1),rare,target_rows,strict=True):
        visible = np.zeros((h,w),dtype=bool)
        if regime=="local":
            visible = truth==label
        elif regime=="across_region":
            visible[start:end,-3:] = True
        data[visible] += np.asarray([sign*1.8 if b in cue else 0. for b in range(width)],dtype=np.float32)
        sources[str(label)] = np.flatnonzero(visible).tolist()
    return data,truth,dict(case=regime,cue_bands=[bands.band_ids[b] for b in cue],sources=sources)


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


def generate_scientific(out, campaign, code, devices):
    """An anchor-star per scene that changes K, label draw or layout one at a time."""
    out.mkdir(parents=True)
    src = source(out)
    releases, groups, contrasts = {},{},{}
    variations = {"anchor":(5,512,"a",None),"k1":(1,512,"a","support_k"),
                  "k10":(10,512,"a","support_k"),"draw64":(5,64,"a","support_seed"),
                  "draw2048":(5,2048,"a","support_seed"),"layout-b":(5,512,"b","spatial_layout")}

    def release(branch, scene, history, *, layout, k=5, seed=512, columns=True,
                matched=True, cost=False, accuracy=True, prospective=False):
        branch.mkdir()
        bank,roles,truth = bank_at(branch/"bank",scene,layout,src["state"],k=k,support_seed=seed,
                                   matched=matched,columns=columns,observation_only=not accuracy)
        views = independent.COLUMN_VIEWS if columns else VIEWS
        manifest = dict(schema=independent.MANIFEST,candidate=bank,roles=roles,minimum_buffer=4,
                        history=[history],observation_views=list(views),observation_factors=[1,2])
        options, approval = {},{}
        if columns:
            manifest["delivered_columns"] = independent.column_inventory(
                BandMetadata(**bank["release"]["bands"]))
            options["observation_protocol"] = COLUMNS
        if accuracy:
            annotation = _write(branch/"geometry.json",dict(schema=GEOMETRY,bank=bank["pin"]["sha256"],
                roles=digest(roles),scope="synthetic",source=digest([FIXTURE,scene,layout,"geometry"]),
                tiles={str(tile):dict(labels=truth[tile].tolist(),geometry=(truth[tile]>0).tolist(),
                                      narrow=np.isin(truth[tile],[15,21] if scene=="whu-honghu" else [5,6]).tolist())
                       for tile in roles["evaluation"]}))
            manifest["spatial_annotations"] = annotation
            approval["spatial_annotations_sha256"] = annotation["sha256"]
        pin = _write(branch/"intake.json",manifest)
        authority = _write(branch/"owner-test-only.json",dict(schema=independent.AUTHORITY,
            approved_manifest_sha256=pin["sha256"],historical_bank_sha256=[history["pin"]["sha256"]],
            uses=dict(independent_accuracy=accuracy,cost=cost,frozen_diagnostics=False),source_mode="synthetic",
            **approval))
        if matched:
            options["matched_exposure"] = MATCHED_EXPOSURE
        if prospective:
            options["claim_policy"] = PAVIA
        return dict(manifest=pin,authority=authority,source=None if scene=="whu-honghu" else src,k=k,**options)

    for i,(alias,scene) in enumerate(SCENES_BY_KEY.items()):
        device = devices[i % len(devices)] if devices else "cpu"
        if len(devices)==1 and alias=="hh":
            device = "cpu"
        history,_,_ = bank_at(out/(alias+"-history-columns"),scene,"h",src["state"],columns=True)
        for variation,(k,draw,layout,factor) in variations.items():
            key = f"{alias}-{variation}"
            releases[key] = release(out/key,scene,history,layout=layout,k=k,seed=draw)
            groups[key] = dict(release=key,seed=64,factor=1,steps=40,clock=[0,10,20,30,40],
                arms=dict(off=OFF,scale=SCALES),pairs=[pair("off","scale")],device=device,purpose="science")
            if factor:
                contrasts[key] = dict(baseline=alias+"-anchor",comparison=key,factor=factor)
        for seed in (512,2048):
            groups[f"{alias}-model-{seed}"] = {**groups[alias+"-anchor"],"seed":seed}

    for alias in ("xz","pavia"):
        scene = SCENES_BY_KEY[alias]
        history,_,_ = bank_at(out/(alias+"-history-physical"),scene,"h",src["state"])
        key = alias+"-physical"
        releases[key] = release(out/key,scene,history,layout="a",columns=False,matched=False,cost=True,
                                prospective=alias=="pavia")
    groups["xz-recovery"] = dict(release="xz-physical",seed=2048,factor=1,device="cpu",
        arms=dict(local=message("local","graph_to_cnn"),combined=message("combined","graph_to_cnn")),
        pairs=[pair("local","combined",equal=True)],steps=12,clock=[0,4,8,12],purpose="recovery")
    groups["xz-column-recovery"] = {**groups["xz-recovery"],"release":"xz-anchor","steps":40,
                                    "clock":[0,10,20,30,40]}
    groups["xz-physical"] = dict(release="xz-physical",seed=64,factor=1,device=devices[0] if devices else "cpu",
        arms={"off":OFF,"aligned":{"backend":"aligned"}},pairs=[pair("off","aligned")],
        steps=12,clock=[0,4,8,12],purpose="science")
    for seed in (64,512,2048):
        groups[f"pavia-physical-s{seed}"] = dict(
            release="pavia-physical",seed=seed,factor=2,device=devices[0] if devices else "cpu",
            arms=dict(off=OFF,scale=SCALES),pairs=[pair("off","scale")],steps=12,clock=[0,4,8,12],purpose="science")
    history,_,_ = bank_at(out/"cost-history",SCENES_BY_KEY["xz"],"h",src["state"],columns=True)
    cost_only = release(out/"xz-cost-only",SCENES_BY_KEY["xz"],history,layout="b",cost=True,accuracy=False)
    _write(out/"cost-only-entry.json",cost_only)
    data = dict(schema=SCHEMA,scope=SYNTHETIC,releases=releases,groups=groups,support_contrasts=contrasts)
    request = _write(out/"request.json",data)
    approval = _write(out/"engineering-only.json",expected_approval(data,request,campaign,code,test=True))
    return dict(request=request,approval=approval)


def generate_regions(out, campaign, code, devices):
    from transfer import wp_region_graph as regional

    out.mkdir(parents=True)
    src = source(out)
    releases, groups = {},{}
    matrix = (("xz-k1","xuzhou",1,64,"a",64),("xz-k5","xuzhou",5,512,"a",512),
              ("hh-k10","whu-honghu",10,2048,"b",2048))
    for i,(name,scene,k,support_seed,layout,seed) in enumerate(matrix):
        branch = out/name
        branch.mkdir()
        old,_,_ = bank_at(branch/"history",scene,"h",src["state"],columns=True,regional_signals=True)
        bank,roles,truth = bank_at(branch/"release",scene,layout,src["state"],k=k,support_seed=support_seed,
                                   matched=True,columns=True,regional_signals=True)
        annotation = _write(branch/"geometry.json",dict(schema=GEOMETRY,bank=bank["pin"]["sha256"],
            roles=digest(roles),scope="synthetic",source=digest([FIXTURE,scene,layout,"regional_cues"]),
            tiles={str(tile):dict(labels=truth[tile].tolist(),geometry=(truth[tile]>0).tolist(),
                    narrow=np.isin(truth[tile],[15,21] if scene=="whu-honghu" else [5,6]).tolist())
                   for tile in roles["evaluation"]}))
        manifest = _write(branch/"intake.json",dict(schema=independent.MANIFEST,candidate=bank,roles=roles,
            minimum_buffer=4,history=[old],observation_views=list(independent.COLUMN_VIEWS),observation_factors=[1],
            delivered_columns=independent.column_inventory(BandMetadata(**bank["release"]["bands"])),
            spatial_annotations=annotation))
        owner = _write(branch/"owner-test-only.json",dict(schema=independent.AUTHORITY,
            approved_manifest_sha256=manifest["sha256"],historical_bank_sha256=[old["pin"]["sha256"]],
            uses=dict(independent_accuracy=True,cost=True,frozen_diagnostics=False),source_mode="synthetic",
            spatial_annotations_sha256=annotation["sha256"]))
        releases[name] = dict(manifest=manifest,authority=owner,source=None if scene=="whu-honghu" else src,
                              k=k,observation_protocol=COLUMNS,matched_exposure=MATCHED_EXPOSURE)
        policies = regional.POLICIES[:4]
        groups[f"region-{name}"] = dict(release=name,seed=seed,factor=1,steps=40,clock=[0,10,20,30,40],
            arms={policy:dict(backend=regional.BACKEND,policy=policy) for policy in policies},
            pairs=[pair("original",p,equal=True) for p in policies[1:]]+[
                pair("random_fixed","spectral_fixed",equal=True),pair("spectral_fixed","spectral_bounded",equal=True)],
            device=devices[i%len(devices)] if devices else "cpu",purpose="science")
    recovery = groups["region-xz-k1"]
    groups["region-recovery"] = {**recovery,"seed":2048,"steps":16,"clock":[0,4,8,12,16],
        "arms":{p:recovery["arms"][p] for p in ("original","spectral_bounded")},
        "pairs":[pair("original","spectral_bounded",equal=True)],"purpose":"recovery","device":"cpu"}
    request = dict(schema=SCHEMA,scope=SYNTHETIC,releases=releases,groups=groups)
    pin = _write(out/"request.json",request)
    approval = _write(out/"engineering-only.json",expected_approval(request,pin,campaign,code,test=True))
    return dict(request=pin,approval=approval)



def generate_coverage(out, campaign, code, devices, *, recovery_matrix=False):
    """Use the existing Bank and erased-signal fixture with separate observation roles."""
    from transfer import wp_coverage_release as coverage

    out.mkdir(parents=True)
    scene = "xuzhou"
    physical_history,_,_ = bank_at(out/"physical-history",scene,"h",None)
    column_history,_,_ = bank_at(out/"column-history",scene,"h",None,columns=True)
    variants = {"physical-a":("a",512,False), "physical-draw":("a",2048,False),
                "physical-layout":("b",512,False), "delivered-columns":("a",512,True)}
    if recovery_matrix:
        require(len(devices)<=2, "The coverage recovery matrix binds up to two devices")
        variants = {key:variants[key] for key in ("physical-a","delivered-columns")}
    arms = {name:dict(backend=coverage.BACKEND,condition=name) for name in coverage.MODES}
    links = [pair("none","continuous"), *(pair(name,"continuous",equal=True)
              for name in ("anchor_only","binary","shuffled"))]
    releases, groups = {}, {}
    for key,(layout,draw,columns) in variants.items():
        branch = out/key
        branch.mkdir()
        bank,roles,_ = bank_at(branch/"release",scene,layout,None,k=1,support_seed=draw,
                               matched=True,columns=columns)
        history = column_history if columns else physical_history
        views = independent.COLUMN_VIEWS if columns else independent.PHYSICAL_COVERAGE_VIEWS
        exposure = dict(support=list(VIEWS),calibration=["full"],evaluation=list(views))
        record = dict(schema=independent.MANIFEST,candidate=bank,roles=roles,minimum_buffer=4,
                      history=[history],observation_exposure=exposure,observation_views=list(views),
                      observation_factors=[1])
        if columns:
            record["delivered_columns"] = independent.column_inventory(BandMetadata(**bank["release"]["bands"]))
        pin = _write(branch/"intake.json",record)
        authority = _write(branch/"engineering-owner.json",dict(schema=independent.AUTHORITY,
            approved_manifest_sha256=pin["sha256"],historical_bank_sha256=[history["pin"]["sha256"]],
            uses=dict(independent_accuracy=True,cost=True,frozen_diagnostics=False),source_mode="synthetic"))
        releases[key] = dict(manifest=pin,authority=authority,source=None,k=1,coverage=coverage.GRAPH,
                             matched_exposure=MATCHED_EXPOSURE,**({"observation_protocol":COLUMNS} if columns else {}))
        device = devices[0] if devices and not columns else "cpu"
        groups[key] = dict(release=key,seed=64,factor=1,steps=12,clock=[0,4,8,12],
                           arms=arms,pairs=links,device=device,purpose="science")
    groups["coverage-recovery"] = {**groups["physical-a"],"seed":512,"purpose":"recovery",
                                  "arms":{name:arms[name] for name in ("none","continuous")},"pairs":links[:1]}
    contrasts = {key:dict(baseline="physical-a",comparison=key,factor=factor) for key,factor in
                 (("physical-draw","support_seed"),("physical-layout","spatial_layout"))}
    if recovery_matrix:
        physical = groups["physical-a"]
        column = groups["delivered-columns"]
        column["device"] = devices[-1] if len(devices)==2 else "cpu"

        def recovery(group,seed,references):
            return {**group,"seed":seed,"purpose":"recovery","steps":36,"clock":[0,6,12,18,24,30,36],
                    "arms":{name:arms[name] for name in ("none","continuous")},"pairs":links[:1],
                    "recovery_arms":references}

        groups = {"a-completed":physical,
                  "b-physical-recovery":recovery(physical,512,["continuous","none"]),
                  "c-column-recovery":recovery(column,2048,["continuous"]),
                  "z-column-followup":column}
        contrasts = {}
    data = dict(schema=SCHEMA,scope=SYNTHETIC,releases=releases,groups=groups)
    if contrasts:
        data["support_contrasts"] = contrasts
    request = _write(out/"request.json",data)
    approval = _write(out/"engineering-only.json",expected_approval(data,request,campaign,code,test=True))
    return dict(request=request,approval=approval)


def controls(root, entry, producer):
    from transfer.spatial_coverage_protocol import bank_observation

    bank,report,roles = independent.open_independent(root,entry["manifest"],entry["authority"],test_history=[])
    retained_checks = {}
    for tile,band in product(roles["evaluation"],report["observation_views"]):
        row = bank.role_row("devval",tile)
        observed, metadata = independent.observation_view(band, report)
        actual = bank_observation(bank,row,producer,observed,metadata=metadata)
        raw = actual.view.observation.spectral
        varies = bool(np.ptp(raw,axis=(0,1)).max()>1e-7)
        kind = row["origin"]["pattern"]
        expected = not (kind=="removed_only" and observed=="central_gap")
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
