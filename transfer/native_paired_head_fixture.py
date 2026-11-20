"""Deterministic spatial bank publications for two head protocols and ordinal fallback."""

from itertools import product

import numpy as np

from evaluation.frozen_training_stages import file_pin, require
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from transfer import development_bank as custody
from transfer.spatial_coverage_protocol import counts, release_statistics, retained_positions

SCHEMA="wp05_wp06_paired_head_spatial_fixture_v1"
SEEDS=(64,512,2048)
SUPPORT=(211,613)
VIEWS=("full","central_gap","matched_dispersed")
SPECS={
    "xuzhou":("xuzhou",9,436,10,63131,(28,36),5),
    "honghu":("whu-honghu",22,270,5,74869,(32,40),5),
    "fallback":("xuzhou",9,57,10,14171,(28,34),5),
    "fallback-a":("xuzhou",9,45,10,26627,(34,42),5),
    "fallback-b":("xuzhou",9,63,10,35407,(34,40),5),
}
FIT=("xuzhou","honghu","fallback")
FINAL=("fallback-a","fallback-b")
FOCUS={"xuzhou":(5,6),"whu-honghu":(15,21)}
ROLES=("support","calibration","evaluation")


def master(kind, width):
    low, high={"xuzhou":(414.,2450.),"honghu":(405.,995.),"fallback":(480.,960.),
                 "fallback-a":(565.,1090.),"fallback-b":(435.,835.)}[kind]
    ids=tuple(101+7*i+(i//5) for i in range(width))
    spacing=np.linspace(0.,1.,width)**(1.15 if kind=="fallback-a" else .92)
    return BandMetadata(f"{SCHEMA}:{kind}",ids,tuple(low+(high-low)*spacing),"nm","verified",(True,)*width,
                        (max(10.,(high-low)/width),)*width,"nm")


def signatures(metadata, classes, mode):
    width=len(metadata.band_ids)
    kept=np.asarray(retained_positions(metadata,"central_gap"))
    allowed=sorted(set(range(width))-set(kept)) if mode=="erased" else kept.tolist()
    x=np.arange(width)
    points=np.asarray(allowed)[np.linspace(0,len(allowed)-1,classes).round().astype(int)]
    sigma=max(1.15,width/(9.*classes))
    book=np.exp(-.5*((x[None,:]-points[:,None])/sigma)**2).astype(np.float32)
    mask=np.ones(width,dtype=bool)
    mask[allowed]=False
    book[:,mask]=0.
    return 2.5*book


def field(kind, metadata, classes, height, width, mode, tile, seed, shifted):
    layout=np.random.default_rng([seed,tile,7])
    noise=np.random.default_rng([seed,tile,23])
    nr=3 if classes==9 else 5
    nc=3 if classes==9 else 5
    ycut=np.linspace(0,height,nr+1,dtype=int)
    xcut=np.linspace(0,width,nc+1,dtype=int)
    if shifted:
        ycut[1:-1]+=np.where(np.arange(1,nr)%2,1,-1)
        xcut[1:-1]+=np.where(np.arange(1,nc)%2,-1,1)
    truth=np.zeros((height,width),dtype=np.int16)
    label_values=layout.permutation(classes)+1
    for (i,j),label in zip(product(range(nr),range(nc)),label_values,strict=False):
        y0,y1,x0,x1=ycut[i],ycut[i+1],xcut[j],xcut[j+1]
        if kind=="honghu" and label in (15,21):
            x0+=(x1-x0)//2
            x1=x0+2
        truth[y0:y1,x0:x1]=label
    book=signatures(metadata,classes,mode)
    book=np.vstack((np.zeros((1,len(metadata.band_ids)),np.float32),book))
    values=book[truth]
    if shifted:
        values=1.08*values + np.linspace(-.08,.08,len(metadata.band_ids),dtype=np.float32)
    values=values+noise.normal(0,.020,size=values.shape).astype(np.float32)
    if mode=="erased":
        kept=retained_positions(metadata,"central_gap")
        require(not book[:,kept].any(),"Planted removed signal can still be observed")
    return np.ascontiguousarray(values),truth


def definition(kind, support_seed, *, override_k=None):
    scene,classes,channels,k,seed,shape,confirmation_k=SPECS[kind]
    require(support_seed in SUPPORT and (override_k is None or scene=="whu-honghu" and override_k==1),
            "Unsupported paired cohort or K")
    if override_k is not None:
        k=override_k
    metadata=master(kind,channels)
    holds=("retained","erased","retained")
    design=[("support",mode,False) for mode in ("retained","erased","retained","retained")]
    design += [("calibration","retained",False),("calibration","erased",False)]
    design += [("evaluation",mode,domain) for mode,domain in zip(holds,(False,False,True),strict=True)]
    if kind in FINAL:
        design=[(role,mode,True if role=="evaluation" else shifted) for role,mode,shifted in design]
    roles={role:[] for role in ROLES}
    blocks,rows,arrays={},[],{}
    h,w=shape
    width=180
    extent=(180,width,channels)
    for index,(role,mode,shifted) in enumerate(design):
        y0,x0=(index//3)*60,(index%3)*60
        values,truth=field(kind,metadata,classes,h,w,mode,index,seed,shifted)
        samples=[]
        for label in range(1,classes+1):
            locations=np.argwhere(truth==label)
            if role=="support":
                rng=np.random.default_rng([seed,index,label,k])
                count=k//4+int((index-label)%4<k%4)
                start=SUPPORT.index(support_seed)*count
                # The independently registered support draws have no shared release points.
                take=rng.permutation(len(locations))[start:start+count]
            else:
                rng=np.random.default_rng([seed,index,label,5221])
                count=confirmation_k if scene!="whu-honghu" or label not in (15,21) else 3
                take=rng.permutation(len(locations))[:count]
            require(len(take)==count,"Insufficient independent class points in the field")
            samples.extend([[(y0+int(y))*width+x0+int(x),label] for y,x in locations[take]])
        box=[y0,y0+h,x0,x0+w]
        bank_role="devtrain" if role=="support" else "devval"
        rows.append(dict(id=index,role=bank_role,partition="train" if role=="support" else "val",
            box=box,parent_tile=index,parent_box=box,samples=sorted(samples),representation="raw_fp32",
            origin=dict(array_sha256=custody.array_hash(values),simulation=SCHEMA),blocked=None))
        roles[role].append(index);arrays[index]=values
        blocks[str(index)]=dict(role=role,mechanism=mode,layout="shift" if shifted else "base",
                                truth_sha256=custody.array_hash(truth))
    rows.sort(key=custody._row_key)
    plan=digest([SCHEMA,kind,extent,roles,[row["box"] for row in rows]])
    statistics=release_statistics(arrays,rows,plan)
    poisoned={row["id"]:arrays[row["id"]] if row["role"]=="devtrain"
              else np.full_like(arrays[row["id"]],np.nan) for row in rows}
    require(release_statistics(poisoned,rows,plan)==statistics,"Held array altered support statistics")
    by_id={r["id"]:r for r in rows}
    report={role:counts([by_id[i] for i in members],classes) for role,members in roles.items()}
    require(report["support"]==[k]*classes and all(min(tally)>0 for tally in report.values()),
            "Independent Bank misses a declared class or K")
    minimum=min(custody._gap(by_id[i]["box"],by_id[j]["box"])
                for a,b in (("support","calibration"),("support","evaluation"),("calibration","evaluation"))
                for i,j in product(roles[a],roles[b]))
    require(minimum>=8,"Synthetic spatial release lacks its buffer")
    release=custody.BankRelease(scene,"synthetic",extent,classes,plan,metadata,statistics,
        dict(shared_state=digest([SCHEMA,kind,seed]),kind="known_synthetic_spectra"),
        dict(minimum_buffer=8,roles=digest(roles),support_seed=support_seed,k=k),tuple(rows))
    return release,arrays,roles,blocks,dict(counts=report,minimum_buffer=minimum,training_statistics_only=True)


def publish(path,kind,support_seed):
    release,arrays,roles,blocks,eligibility=definition(kind,support_seed)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not (path/"bank.json").exists():
        custody.publish(path,path.parent,release,lambda row:arrays[row["id"]],resume=path.exists())
    pin=dict(root=str(path),sha256=file_pin(path/"bank.json")["sha256"],
             registration=file_pin(path/"registration.json")["sha256"])
    bank=custody.Bank(pin,path.parent,release)
    return bank,dict(bank=pin,roles=roles,blocks=blocks,eligibility=eligibility)
