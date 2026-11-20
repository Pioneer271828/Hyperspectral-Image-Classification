"""Independent synthetic Bank family for the directed two-layer mechanism."""

from dataclasses import asdict

import numpy as np

from evaluation.frozen_training_stages import require
from evaluation.wp12_scientific_claim import COLUMNS
from evaluation.wp12_spatial_release import SCHEMA as GEOMETRY
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import development_bank as custody, native_independent_release as independent
from transfer import wp_causal_graph_design as design, wp_region_graph as region
from transfer.spatial_coverage_protocol import release_statistics, retained_positions
from transfer.wp_paired_structure import SCHEMA, SYNTHETIC, expected_approval
from transfer.wp_paired_structure_fixture import pair, source
from transfer.wp_support_match import MODE as EXPOSURE


PARTS=("support",)*4+("calibration",)*2+("evaluation",)*10
CLOCK=[0,6,12,18,24]


def _tile(layout, classes, i, rng, spectra, focus, cue):
    size=design.SIZE
    members=design.memberships(layout)
    rare=focus if i%2==0 else focus[::-1]
    ordinary=[label for label in range(1,classes+1) if label not in focus]
    truth=rng.permutation(np.resize(ordinary,size*size)).reshape(size,size)
    for node,label in zip(design.TARGETS,rare,strict=True):
        truth[members==node]=label
    common=truth.copy()
    common[np.isin(common,focus)]=focus[0]
    data=spectra[common-1]+rng.normal(0,.025,(size,size,spectra.shape[1])).astype(np.float32)
    data[:,:,cue]=1.
    regime="new_path" if i<6 else design.REGIMES[(i-6)//2]
    sign=(i//2)%2
    for node,value in zip(design.UNRELATED,(0.,2.) if sign else (2.,0.),strict=True):
        y,x=np.where(members==node)
        data[y[:,None],x[:,None],cue]=value
    emit=design.emitters(regime)
    sources={str(label):[] for label in focus}
    for node,label in zip(emit,rare):
        y,x=np.where(members==node)
        data[y[:,None],x[:,None],cue]=2. if label==focus[1] else 0.
        sources[str(label)]=np.flatnonzero(members==node).tolist()
    return data,truth,dict(case=regime,sources=sources),members


def bank_at(out, scene, layout, shared_state, *, support_seed):
    size=design.SIZE
    classes=9 if scene=="xuzhou" else 22
    axis=tuple(11+7*j for j in range(24))
    bands=BandMetadata(f"{design.VERSION}:{scene}",axis,tuple(np.linspace(0.,1.,len(axis))),
                       "normalized","index_only",(True,)*len(axis))
    cue=sorted(set(range(len(axis)))-set(retained_positions(bands,"central_gap")))
    cue_bands=[axis[j] for j in cue]
    shape=(500,160,len(axis))
    base={"a":4,"b":171,"h":338}[layout]
    scene_no=int(classes==22)
    spectra=np.random.default_rng(9481+scene_no).normal(.6,.5,(classes,len(axis))).astype(np.float32)
    roles={role:[] for role in independent.ROLES}
    rows,arrays,annotations,tiles=[],{},{},{}
    for i,role in enumerate(PARTS):
        y,x=base+(i//4)*38,4+(i%4)*38
        key=base*100+i
        rng=np.random.default_rng([82051,base,i,scene_no])
        data,truth,signal,members=_tile(layout,classes,i,rng,spectra,design.FOCUS[scene],cue)
        signal["cue_bands"]=cue_bands
        roles[role].append(key)
        box=[y,y+size,x,x+size]
        samples=[]
        for label in range(1,classes+1):
            count=(5 if (label-1)%4==i else 0) if role=="support" else (
                6 if role=="calibration" else 11 if label in design.FOCUS[scene] else 3)
            points=np.flatnonzero(truth==label)
            require(len(points)>=count,"Insufficient synthetic support for a registered class")
            draw=np.random.default_rng([support_seed,base,i,scene_no,label]) if role=="support" else rng
            chosen=draw.permutation(points)[:count]
            samples.extend([(y+int(p)//size)*shape[1]+x+int(p)%size,label] for p in chosen)
        role_name=independent.ROLES[role]
        tiles[str(key)]=dict(role=role_name,regime=signal["case"])
        arrays[key]=data
        rows.append(dict(id=key,role=role_name,partition="train" if role=="support" else "val",
             box=box,parent_tile=key,parent_box=box,samples=sorted(samples),representation="raw_fp32",
             origin=dict(array_sha256=custody.array_hash(data),pattern="retained",regional_signal=signal),blocked=None))
        if role=="evaluation":
            annotations[str(key)]=dict(labels=truth.tolist(),geometry=(truth>0).tolist(),
                                       narrow=np.isin(members,design.TARGETS).tolist())
    rows.sort(key=custody._row_key)
    plan=digest([design.VERSION,scene,layout,roles,[r["box"] for r in rows]])
    statistics=release_statistics(arrays,rows,plan)
    release=custody.BankRelease(scene,"synthetic",shape,classes,plan,bands,statistics,
                dict(shared_state=shared_state,kind=design.VERSION),dict(k=5,minimum_buffer=4,layout=layout,
                    roles=digest(roles),support_seed=support_seed),tuple(rows))
    saved=custody.publish(out,out.parent,release,lambda row:arrays[row["id"]])
    entry=dict(pin={k:saved[k] for k in ("root","sha256","registration")},release=asdict(release))
    return entry,roles,annotations,design.contract(entry["pin"]["sha256"],scene,roles,layout,tiles,cue_bands)


def make_release(place, scene, layout, source_pin, history):
    place.mkdir()
    draw={"a":512,"b":2048}[layout]
    bank,roles,annotations,causal=bank_at(place/"bank",scene,layout,source_pin["state"],support_seed=draw)
    annotation=_write(place/"geometry.json",dict(schema=GEOMETRY,bank=bank["pin"]["sha256"],
        roles=digest(roles),scope="synthetic",source=digest([design.VERSION,scene,layout,"geometry"]),tiles=annotations))
    manifest=_write(place/"intake.json",dict(schema=independent.MANIFEST,candidate=bank,roles=roles,
        minimum_buffer=4,history=[history],observation_views=list(independent.COLUMN_VIEWS),observation_factors=[1],
        delivered_columns=independent.column_inventory(BandMetadata(**bank["release"]["bands"])),
        spatial_annotations=annotation))
    authority=_write(place/"engineering-owner.json",dict(schema=independent.AUTHORITY,
        approved_manifest_sha256=manifest["sha256"],historical_bank_sha256=[history["pin"]["sha256"]],
        uses=dict(independent_accuracy=True,cost=True,frozen_diagnostics=False),source_mode="synthetic",
        spatial_annotations_sha256=annotation["sha256"]))
    return dict(manifest=manifest,authority=authority,
                source=None if scene=="whu-honghu" else source_pin,k=5,observation_protocol=COLUMNS,
                matched_exposure=EXPOSURE,causal_graph=_write(place/"causal-design.json",causal))


def generate(out,campaign,code,devices):
    """A predeclared seed series and one second-layout repeat in each scene."""
    out.mkdir(parents=True)
    src=source(out)
    releases,groups={},{}
    arms={p.removeprefix("causal_"):dict(backend=region.BACKEND,policy=p) for p in design.POLICIES}
    pairs=[pair("original","matched",equal=True),pair("original","placebo",equal=True),
           pair("matched","evolving",equal=True),pair("placebo","matched",equal=True)]
    for short,scene in (("xz","xuzhou"),("hh","whu-honghu")):
        history,_,_,_=bank_at(out/(short+"-history"),scene,"h",src["state"],support_seed=512)
        for layout in ("a","b"):
            name=f"{short}-{layout}"
            releases[name]=make_release(out/name,scene,layout,src,history)
            for seed in ((64,512,2048) if layout=="a" else (64,)):
                device=devices[len(groups)%len(devices)] if devices else "cpu"
                groups[f"{name}-s{seed}"]=dict(release=name,seed=seed,factor=1,steps=24,clock=CLOCK,
                    arms=arms,pairs=pairs,device=device,purpose="science")
    groups["causal-recovery"]={**groups["xz-a-s64"],"seed":2048,"device":devices[0] if devices else "cpu",
        "arms":{k:arms[k] for k in ("matched","evolving")},"pairs":[pair("matched","evolving",equal=True)],
        "purpose":"recovery"}
    request=dict(schema=SCHEMA,scope=SYNTHETIC,releases=releases,groups=groups)
    pin=_write(out/"request.json",request)
    approval=_write(out/"engineering-only.json",expected_approval(request,pin,campaign,code,test=True))
    return dict(request=pin,approval=approval)
