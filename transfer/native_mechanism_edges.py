"""Independent synthetic boundary, narrow-target and matched-deletion native provider."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import file_pin, require
from evaluation.matched_edge_controls import freeze, verify
from loadData.observation_sampling import BandMetadata
from model.graph_state import GraphState
from model.research_config import digest
from monitoring.released_region import RegionAction, region_action
from research_runtime import tensor_hash
from transfer import development_bank as custody
from transfer.band_graph_control import FIXED_ARMS, ViewControl, fixed_graph, observation_id, row_coo
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_production_model import Switches
from transfer.native_production_provider import ProductionProvider
from transfer.native_released_points import LOSS, MODE
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import release_statistics

SCHEMA = "wp06_exact_native_edges_v1"
TASKS = ("local","remote","erased")
SEEDS = (64,512,2048)
ARMS = FIXED_ARMS
STEPS = 48
CLOCKS = (0,24,48)
PREFIX = 17
BAND = "central_gap"
VIEWS = ("full",BAND)
RANDOM = 7719
RARE = 3


def geometry(h,w):
    require((h,w) in ((12,32),(32,12)), "Unknown registered narrow field")
    yy,xx=np.mgrid[:h,:w]
    y,x=(xx,yy) if h==32 else (yy,xx)
    pickup=(y//6)*2+(x//2)
    groups=np.where(x<4,pickup,np.where(x>=28,4+y//3,8)).astype(np.int64)
    return y,x,groups


def fixture(task,seed):
    require(task in TASKS and seed in SEEDS,"Unknown synthetic boundary field")
    width,height=126,196
    raw, rows, known={"support":[],"calibration":[],"evaluation":[]},[],{}
    arrays={}
    bands=BandMetadata("synthetic-independent-edge-axis",tuple(113+9*i for i in range(8)),
                       tuple(460+46.*i for i in range(8)),"nm","verified",(True,)*8,(16.,)*8,"nm")
    for index in range(10):
        role="support" if index<4 else "calibration" if index<6 else "evaluation"
        shape=(12,32) if index%2==0 else (32,12)
        y,x,groups=geometry(*shape)
        plan_rng=np.random.default_rng([83027,seed,index,113])
        noise=np.random.default_rng([83027,seed,index,719])
        labels=plan_rng.permutation([1,1,2,3])
        values=noise.normal(0,.018,size=(*shape,8)).astype(np.float32)
        truth=np.zeros(shape,dtype=np.int16)
        points=[]
        for target,label in enumerate(labels):
            mask=(groups==target)&(y%6>=1)&(y%6<=4)
            require(int(mask.sum())==8,"Narrow region size changed")
            truth[mask]=label
            dy,dx=np.argwhere(mask&(y%6==1))[index%2]
            points.append((int(dy),int(dx),int(label),target))
            where=groups==4+target if task=="remote" else groups==target
            columns=(2,3,4) if task=="erased" else (0,1,7)
            values[where,columns[label-1]] += 2.2
        role_key="devtrain" if role=="support" else "devval"
        y0,x0=index//3*44,index%3*42
        box=[y0,y0+shape[0],x0,x0+shape[1]]
        row=dict(id=index,role=role_key,partition="train" if role_key=="devtrain" else "val",
                 box=box,parent_tile=index,parent_box=box,
                 samples=sorted([[(y0+a)*width+x0+b,c] for a,b,c,_ in points]),
                 origin=dict(array_sha256=array_hash(values)),representation="raw_fp32",blocked=None)
        rows.append(row);arrays[index]=values;raw[role].append(index)
        dy,dx=(0,1) if shape[0]==32 else (1,0)
        require(all(truth[a,b]==label and truth[a-dy,b-dx]==0 for a,b,label,_ in points),
                "Synthetic focus lacks a planted class-to-background boundary")
        known[str(index)]=dict(orientation="vertical" if shape[0]==32 else "horizontal",
            boundary_points=len(points),width=2,foreground=int((truth>0).sum()),truth_sha256=array_hash(truth),
            focus_region=[target for _,_,_,target in sorted(points)],classes=labels.tolist())
    plan=digest([SCHEMA,seed,task,[row["box"] for row in rows]])
    stats=release_statistics(arrays,rows,plan)
    source=dict(shared_state=digest([SCHEMA,seed,task,"independent_generator"]),origin="synthetic")
    release=custody.BankRelease("xuzhou","synthetic",(height,width,8),9,plan,bands,stats,source,
                                   dict(minimum_buffer=6,roles=digest(raw)),tuple(rows))
    for a,b in (("support","calibration"),("support","evaluation"),("calibration","evaluation")):
        require(min(custody._gap(rows[i]["box"],rows[j]["box"]) for i,j in product(raw[a],raw[b]))>=6,
                "Synthetic calibration/confirmation parent buffer differs")
    return release,arrays,raw,known


def exact_control(actual):
    began=perf_counter()
    h,w=actual.lattice.shape
    yy,xx=np.mgrid[:h,:w]
    _,_,groups=geometry(h,w)
    pixels=np.arange(h*w)
    q=torch.sparse_coo_tensor(torch.from_numpy(np.vstack((pixels,groups.ravel()))),
                             torch.ones(h*w),(h*w,9)).coalesce()
    coords=np.asarray([[yy[groups==i].mean(),xx[groups==i].mean()] for i in range(9)],dtype=np.float32)
    edges=[(i,i) for i in range(9)]
    weights=[.5]*9
    for a in range(4):
        edges.extend(((a,4+a),(a,4+(a^1))))
        weights.extend((2.,.25))
    a0=row_coo(edges,weights,9)
    graph=GraphState(q,a0,torch.from_numpy(coords/np.asarray((h-1,w-1),dtype=np.float32)),(h,w),(1,9)).validate()
    nonself=[(a,b) for a,b in edges if a!=b]
    lengths=np.unique([np.linalg.norm(coords[a].astype(np.float64)-coords[b]) for a,b in nonself])
    boundaries=[float((a+b)/2) for a,b in zip(lengths[:-1],lengths[1:],strict=True)] or [float(lengths[0]+1)]
    coords=torch.from_numpy(coords)
    plan=freeze(a0,coords,per_row=1,distance_bins=boundaries,seeds=[RANDOM],time_limit=2.)
    verify(plan,a0,coords,expected_hash=plan["plan_hash"])
    high=plan["baseline"]["removed"]
    require(high==[a*9+4+a for a in range(4)] and all(r["status"]=="optimal" for r in plan["controls"].values()),
            "Exact geometric fixture lacks matched high/low/random masks")

    def d2(edges):
        return sorted(float((coords[a//9]-coords[a%9]).square().sum()) for a in edges)

    exact=d2(high)
    require(all(d2(row["removed"])==exact for row in plan["controls"].values()),
            "Matched degrees and counts did not preserve exact squared distances")
    topo=dict(squared_distance_multiset=exact, degree_counts=plan["baseline"]["counts"],removed=len(high),
              masks_distinct_from_high=True,random_overlaps_low=plan["controls"][f"random:{RANDOM}"]["removed"]==
                                                           plan["controls"]["low_weight"]["removed"])
    return ViewControl(observation_id(actual),graph,coords,plan,perf_counter()-began,topo)


def publish(path,task,seed):
    release,arrays,roles,known=fixture(task,seed)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not (path/"bank.json").exists():
        custody.publish(path,path.parent,release,lambda row:arrays[row["id"]],resume=path.exists())
    pin=dict(root=str(path),sha256=file_pin(path/"bank.json")["sha256"],
             registration=file_pin(path/"registration.json")["sha256"])
    return custody.Bank(pin,path.parent,release),roles,known


class EdgeProvider(ProductionProvider):
    def __init__(self,root,bank,roles,seed,edge_arm,producer):
        require(bank.release.profile=="synthetic" and edge_arm in ARMS and seed in SEEDS,
                "Exact edge training uses only the sealed synthetic provider")
        super().__init__(root,bank,roles,axis=bank.release.bands.band_ids,config=Switches(),source=None,
                         seed=seed,budget=STEPS,producer=producer)
        self.edge_arm, self.controls = edge_arm, {}

    def control(self,actual):
        key=observation_id(actual)
        if key not in self.controls:
            self.controls[key]=exact_control(actual)
        return self.controls[key]

    def registered_forward(self,model,actual,role,execution,*,edge=None,region=None):
        from contextlib import nullcontext

        plan=self.control(actual)
        operation=fixed_graph(actual,plan,self.edge_arm if edge is None else edge)
        timing={}
        scope=nullcontext() if region is None else region_action(model.backbone,region)
        with scope:
            out,graph=forward(model,actual,self.producer,role,timed=timing,execution=execution,graph_override=operation)
        return out,graph,timing

    def sealed(self,actual,labels,targets):
        return self.evidence(actual,labels,targets)|dict(control=self.control(actual).fingerprint)

    def validate_observation(self,spec,actual,labels,targets,event):
        key=self.key(actual.receipt["role"],event)
        require(self.sealed(actual,labels,targets)==spec["inputs"][key],"Registered graph or retained exposure changed")

    def stage_evidence(self,actual):
        return super().stage_evidence(actual)|dict(control=self.control(actual).fingerprint)

    def validate_history(self,spec,packet):
        super().validate_history(spec,packet)
        for role,events in (("devtrain",packet["history"]),
                            ("devval",[r for row in packet["validation"] for r in row["rows"]])):
            for item in events:
                actual,_,_=self.materialize(role,item,self.producer)
                control=self.control(actual)
                expected=control.actual(actual,self.edge_arm).A0._nnz()
                require(item["control"]==control.fingerprint and item["graph"]["a0_nnz"]==expected,
                        "Recovered native sparse graph differs from registered controls")

    def contract(self,classes,seed,arm,factor,producer,*,numeric=None):
        require((classes,seed,arm,factor,producer)==(9,self.seed,"native",1,self.producer),"Edge candidate identity differs")
        rng=np.random.default_rng(seed)
        events=[int(tile) for _ in range(STEPS//len(self.roles["support"])) for tile in rng.permutation(self.roles["support"])]
        train=[dict(tile=tile,step=i,band=BAND,factor=1,condition="matched_exact_distance") for i,tile in enumerate(events,1)]
        validation=[dict(tile=tile,band=BAND,factor=1,condition="matched_exact_distance") for tile in self.roles["calibration"]]
        roster={}
        for role,items in (("devtrain",train),("devval",validation)):
            for item in items:
                key=self.key(role,item)
                if key not in roster:
                    roster[key]=self.sealed(*self.materialize(role,item,producer))
        model=self.model_factory(self.baseline(seed),arm,seed+7321)
        return dict(format=SCHEMA,producer=producer,classes=classes,seed=seed,arm=arm,factor=factor,
            edge_arm=self.edge_arm,supervision=MODE,objective=LOSS,training=train,validation=validation,inputs=deepcopy(roster),
            steps=STEPS,prefix=PREFIX,events=list(CLOCKS),learning_rate=.002,module_seed=seed+7321,rng_seed=seed+15000,
            selection="first_min_released_point_fused_nll",initialization=dict(state=state_hash(model),fresh={}),
            parameters=sum(p.numel() for p in model.parameters()),bank=self.bank.pin,numeric=numeric,
            roles=digest(self.roles),frozen_stage_protocol=SCHEMA,real_updates=0,query_accessed=False,raw_gt_loaded=False)
