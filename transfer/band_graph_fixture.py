"""Guarded synthetic fixtures for a band-by-matching campaign on the native runner."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from time import perf_counter

import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from scripts.run_frozen_band_region import check
from transfer import development_bank as bank
from transfer.band_graph_control import ARMS, ViewControl, fixed_graph, observation_id
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_released_points import LOSS, MODE, scores
from transfer.spatial_coverage_protocol import bank_view, release_statistics

SCHEMA="wp05_wp06_band_graph_engineering_v1"
TASKS=("spectral","remote","negative")
LAYOUTS=("horizontal","vertical")
SEEDS=(64,512,2048)
VIEWS=("full","continuous","dispersed")
STEPS=12
PREFIX=4


def generated(task, layout, seed):
    check(task in TASKS and layout in LAYOUTS and seed in SEEDS,"Expected registered planted task")
    bands=BandMetadata("guarded-interior-axis",tuple(101+7*i for i in range(8)),tuple(450.+50*i for i in range(8)),
                       "nm","verified",(True,)*8)
    wide=(12,32) if layout=="horizontal" else (32,12)
    shape=(104,108,8)
    arrays, rows, witness = {}, [], {}
    for tile in range(6):
        rng=np.random.default_rng(np.random.SeedSequence([811,seed,tile]))
        labels=np.asarray([1,1,2,2])[rng.permutation(4)]
        donor=labels.copy() if task=="remote" else np.asarray([1,1,2,2])[rng.permutation(4)]
        original=np.zeros((12,32,8),dtype=np.float32)
        original[:,16:17,2] = np.float32((tile+1)/64)
        for i in range(4):
            if task=="spectral":
                original[i*3:(i+1)*3,:4,3] = labels[i] - 1.5
            if task in ("remote","negative"):
                original[i*3:(i+1)*3,28:,0] = donor[i] - 1.5
        array=original if layout=="horizontal" else original.transpose(1,0,2).copy()
        y,x=2+(tile//3)*45,2+(tile%3)*36
        box=[y,y+wide[0],x,x+wide[1]]
        label_pos=[(i*3+1,1) if layout=="horizontal" else (1,i*3+1) for i in range(4)]
        released=sorted([[(y+j)*shape[1]+x+k,int(label)] for (j,k),label in zip(label_pos,labels,strict=True)])
        role="devtrain" if tile<4 else "devval"
        arrays[tile]=array
        rows.append(dict(id=tile,role=role,partition="train" if tile<4 else "val",box=box,
                         parent_tile=tile,parent_box=box,samples=released,
                         origin=dict(array_sha256=array_hash(array)),representation="raw_fp32",blocked=None))
        anchors=original[1::3,1]
        peers=original[1::3,30,0]
        if task=="spectral":
            check(np.all(anchors[:,3]+1.5 == labels) and not np.any(np.delete(anchors,3,axis=1)),
                  "Planted center spectrum differs")
        else:
            check(not anchors.any() and np.all(peers+1.5==donor),"Remote fixture leaks into the labeled center")
            if task=="remote":
                check(np.array_equal(labels,donor),"Registered owner differs from the planted remote signal")
        witness[str(tile)]=dict(label=labels.tolist(),partner=donor.tolist(),anchor_hash=array_hash(anchors))
    geometry=digest(dict(layout=layout,tiles=[r["box"] for r in rows],bands=bands.fingerprint))
    stats=release_statistics(arrays,rows,geometry)
    release=bank.BankRelease("xuzhou","synthetic",shape,9,geometry,bands,stats,
                             {"kind":"synthetic_shared_source","shared_state":digest(["seed",seed])},
                             {"minimum_buffer":2,"task":task,"layout":layout,"source_has_fresh_head":True},tuple(rows))
    return release, arrays, witness


def new_model(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=MultiDatasetVCASAHGCN({"xuzhou":9},graph_mode="learned",norm_type="group_norm",
                                    latent_channels=8,adapter_embed_dim=8)
    model.backbone.adapter.cache_attention=False
    return model


class BandGraphFixture:
    """A sealed engineering provider for native optimization, prediction and resume."""
    scores=staticmethod(scores)
    is_synthetic=True

    def __init__(self, admitted, seed, layout, band, arm, producer, catalog=None):
        if admitted.release.profile!="synthetic" or seed not in SEEDS or layout not in LAYOUTS or band not in VIEWS or arm not in ARMS:
            raise ValueError("Only a sealed synthetic band-graph fixture can update")
        self.bank,self.seed,self.layout,self.band,self.arm,self.producer=admitted,seed,layout,band,arm,producer
        self.views={} if catalog is None else catalog[0]
        self.controls={} if catalog is None else catalog[1]
        self.roster={}
        if not self.views:
            for role in ("devtrain","devval"):
                for row in admitted.role_rows(role):
                    view=bank_view(admitted,row,producer,band)
                    self.views[role,row["id"]]=view
                    control=ViewControl.create(view[0],layout)
                    self.controls[observation_id(view[0])]=control

    def baseline(self,seed):
        return new_model(seed)

    @staticmethod
    def model_factory(baseline,arm,seed):
        return baseline

    @staticmethod
    def selection_condition(actual,spec):
        return None

    def __getitem__(self,actual):
        actual.verify(self.producer,actual.receipt["role"])
        item=self.controls.get(observation_id(actual))
        if item is None:
            raise ValueError("No graph is registered for the exact current band and role")
        return item

    def graph_override(self,actual):
        return fixed_graph(actual,self[actual],self.arm)

    def variant(self,arm):
        return BandGraphFixture(self.bank,self.seed,self.layout,self.band,arm,self.producer,(self.views,self.controls))

    def materialize(self,role,event,producer):
        if producer!=self.producer or event["band"]!=self.band:
            raise ValueError("Bound band or producer differs")
        return self.views[role,event["tile"]]

    def observations(self, actual, targets):
        g=self[actual]
        return dict(observation=actual.receipt["observation"],actual_ids=actual.receipt["source"]["band_ids"],
                    released=targets.summary(),graph_binding=g.fingerprint,
                    fixed_nnz=None if self.arm=="native" else g.actual(actual,self.arm).A0._nnz())

    def stage_evidence(self,actual):
        g=self[actual]
        return dict(graph_binding=g.fingerprint,actual_ids=actual.receipt["source"]["band_ids"],
                    fixed_nnz=None if self.arm=="native" else g.actual(actual,self.arm).A0._nnz())

    def validate_observation(self,spec,actual,labels,targets,event):
        if self.observations(actual,targets)!=spec["inputs"][f'{actual.receipt["role"]}/{event["tile"]}']:
            raise ValueError("Observation, matching or point exposure changed")

    def validate_history(self,spec,packet):
        for role, records in (("devtrain",packet["history"]),
                              ("devval",[r for row in packet["validation"] for r in row["rows"]])):
            for row in records:
                item=spec["inputs"][f'{role}/{row["tile"]}']
                if any(row[k]!=item[k] for k in ("observation","released","graph_binding","actual_ids","fixed_nnz")):
                    raise ValueError("Recovered band graph identity differs")
                if item["fixed_nnz"] is not None and row["graph"]["a0_nnz"]!=item["fixed_nnz"]:
                    raise ValueError("Recovered actual graph density differs")

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        check((classes,seed,arm,factor,producer,numeric)==(9,self.seed,self.arm,1,self.producer,None),
              "Synthetic native configuration differs")
        self.bank.verify_authority(self.bank.pin)
        for (role,tile),(actual,_,target) in self.views.items():
            self.roster[f"{role}/{tile}"]=self.observations(actual,target)
        order=[int(tile) for _ in range(3) for tile in np.random.default_rng(seed).permutation(4)]
        def event(tile):
            return dict(tile=tile,band=self.band,factor=1,profile=None,condition="missing")
        train=[dict(step=i,**event(tile)) for i,tile in enumerate(order,1)]
        val=[event(tile) for tile in (4,5)]
        start=self.baseline(seed)
        return dict(format=SCHEMA,producer=producer,classes=classes,seed=seed,arm=arm,factor=1,
                    supervision=MODE,objective=LOSS,training=train,validation=val,inputs=deepcopy(self.roster),
                    steps=STEPS,prefix=PREFIX,events=[0,6,12],learning_rate=.001,module_seed=seed+7321,rng_seed=seed+15000,
                    initialization=dict(source=state_hash(start),state=state_hash(start),fresh={}),
                    registered_graph_scale=100,parameters=sum(p.numel() for p in start.parameters()),
                    selection="first_min_released_point_fused_nll",bank=self.bank.pin,numeric=None,
                    real_updates=0,query_accessed=False,raw_gt_loaded=False)

    def match_inventory(self):
        items={}
        for (role,tile),(actual,_,targets) in self.views.items():
            control=self[actual]
            row=dict(role=role,tile=tile,view=control.identity,assignment=control.state.Q._nnz(),
                     original_nnz=control.state.A0._nnz(),band_ids=actual.receipt["source"]["band_ids"],
                     graph_seconds=control.construction,match_seconds=control.plan["elapsed_seconds"],
                     topology=control.topology,baseline=control.plan["baseline"],controls=control.plan["controls"],
                     symmetric=control.plan.get("symmetric",False),points=targets.summary()["points"])
            items[f"{role}/{tile}"]=row
        return items
