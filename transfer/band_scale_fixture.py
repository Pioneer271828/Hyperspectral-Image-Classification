"""Three separated synthetic signals in the production released-point scale runner."""

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np

from model.research_config import digest
from scripts.run_frozen_band_region import check, pin
from transfer.band_graph_fixture import new_model
from transfer.band_scale import BANDS, FACTORS, NAME, PROFILES, condition, evidence, released_view
from transfer.development_bank import Bank, BankRelease, publish
from transfer.development_bank_training import _initial
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_released_points import LOSS, MODE, scores
from transfer.spatial_coverage_protocol import release_statistics
from transfer.spatial_multiscale import initialize

TASKS = ("spectral","context","negative")
LAYOUTS = ("row","column")
SEEDS = (64,512,2048)
ARMS = ("off","fixed","mean","joint","cnn_content","graph_content")
STEPS, PREFIX = 24, 7
EVENTS = (0,12,24)
TRAIN = tuple(product(FACTORS,BANDS))


def generated(task, layout, seed):
    check(task in TASKS and layout in LAYOUTS and seed in SEEDS,"Unknown signal registration")
    profile = PROFILES[{"row":"a","column":"b"}[layout]]
    bands = replace(profile.bands(8),band_ids=tuple(range(103,159,7)))
    scene = (91,109,8)
    raw, truth, roster, witnesses = {}, {}, [], {}
    for tile in range(6):
        role = "devtrain" if tile<4 else "devval"
        rng = np.random.default_rng(np.random.SeedSequence([seed,113 if layout=="row" else 227,
                                          8901 if role=="devtrain" else 9719,tile]))
        labels = rng.permutation([1,1,2,2])
        field = np.zeros((17,29,8),dtype=np.float32)
        dense = np.zeros(field.shape[:2],dtype=np.int64)
        positions = [(2+4*i,2) for i in range(4)]
        for (y,x),label in zip(positions,labels,strict=True):
            dense[y,x] = label
            signal = label-1.5
            if task=="spectral":
                field[y-1:y+2,x-1:x+2,3] = signal
            if task=="context":
                field[y-1:y+2,14:21,0] = signal
        field[16,:,2] = (tile+1)*.02
        centers = np.asarray([field[y,x] for y,x in positions])
        donor_values = np.asarray([field[y,17,0] for y,_ in positions])
        if task=="spectral":
            check(np.array_equal(centers[:,3]+1.5,labels) and not np.delete(centers,3,axis=1).any(),
                  "Unknown signal outside the preregistered center band")
        else:
            check(not centers.any(),"Center spectra unexpectedly predict synthetic labels")
        if task=="context":
            check(np.array_equal(donor_values+1.5,labels),"Context truth changed")
        if task=="negative":
            check(not donor_values.any(),"Negative case contains a remote carrier")
        if layout=="column":
            field, dense = field.transpose(1,0,2).copy(), dense.T.copy()
            positions = [(x,y) for y,x in positions]
        a,c = 1+(tile//3)*42, 1+(tile%3)*36
        box = [a,a+field.shape[0],c,c+field.shape[1]]
        points = sorted([[(a+y)*scene[1]+c+x,int(label)] for (y,x),label in zip(positions,labels,strict=True)])
        roster.append(dict(id=tile,role=role,partition="train" if tile<4 else "val",box=box,
                           parent_tile=tile,parent_box=box,samples=points,
                           origin=dict(array_sha256=array_hash(field)),representation="raw_fp32",blocked=None))
        raw[tile],truth[tile] = field,dense
        witnesses[str(tile)] = dict(truth=array_hash(dense),centers=array_hash(centers),
                                    donor=array_hash(donor_values),classes=labels.tolist())
    plan = digest(dict(task=task,layout=layout,seed=seed,bands=bands.fingerprint,roles=roster))
    source = dict(kind="native_band_scale_fixture",scope=NAME,seed=seed,shared_state=state_hash(new_model(seed)))
    release = BankRelease("xuzhou","synthetic",scene,9,plan,bands,release_statistics(raw,roster,plan),
                          source,dict(minimum_buffer=2,task=task,layout=layout, source_has_fresh_head=True),tuple(roster))
    return release,raw,truth,profile,witnesses


def open_fixture(root, task, layout, seed, code):
    release,raw,truth,profile,witnesses = generated(task,layout,seed)
    path = root/"bank"
    if (path/"bank.json").is_file():
        link = dict(root=str(path),sha256=pin(path/"bank.json")["sha256"],
                    registration=pin(path/"registration.json")["sha256"])
    else:
        link = {k:v for k,v in publish(path,root,release,lambda row:raw[row["id"]],resume=path.exists()).items()
                if k in ("root","sha256","registration")}
    return BandScaleFixture(Bank(link,root,release),seed,profile,truth,code),witnesses


class BandScaleFixture:
    is_synthetic = True
    scores = staticmethod(scores)
    model_factory = staticmethod(initialize)

    def __init__(self, bank, seed, profile, truth, code):
        check(bank.release.profile=="synthetic" and bank.release.source["scope"]==NAME
              and bank.release.source["seed"]==seed,"Unauthorized scale fixture")
        self.bank,self.seed,self.profile,self.truth,self.producer = bank,seed,profile,truth,code
        self._inputs = {}

    def baseline(self, seed):
        check(seed==self.seed,"A foreign initialization seed cannot use this bank")
        return new_model(seed)

    def materialize(self, role, event, code):
        check(code==self.producer and event["condition"]=="matched","Synthetic band-scale operation differs")
        key = role,event["tile"],event["band"],event["factor"]
        if key not in self._inputs:
            row = self.bank.role_row(role,event["tile"])
            self._inputs[key] = released_view(self.bank,row,code,event["band"],event["factor"],
                                               profile=self.profile,labels=self.truth[row["id"]])
        return self._inputs[key]

    @staticmethod
    def key(role, event):
        return f'{role}/{event["tile"]}/{event["band"]}/{event["factor"]}'

    def contract(self, classes, seed, arm, factor, code, *, numeric=None):
        check((classes,seed,factor,code,numeric)==(9,self.seed,1,self.producer,None) and arm in ARMS,
              "Unexpected native band-scale execution scope")
        self.bank.verify_authority(self.bank.pin)
        gen = np.random.default_rng(np.random.SeedSequence([seed,441]))
        ids = gen.permutation((0,1,2,3)).tolist()
        train = [dict(step=i+1,tile=ids[(i+i//4)%4],factor=view[0],band=view[1],
                      profile=self.profile.key,condition="matched")
                 for i,view in enumerate(TRAIN*(STEPS//len(TRAIN)))]
        val = [dict(tile=tile,factor=factor_value,band=band,profile=self.profile.key,condition="matched")
               for tile,(factor_value,band) in product((4,5),TRAIN)]
        inventory = {}
        for role,events in (("devtrain",train),("devval",val)):
            for event in events:
                actual,_,targets = self.materialize(role,event,code)
                inventory[self.key(role,event)] = evidence(actual,targets)
        source = self.baseline(seed)
        model = self.model_factory(source,arm,seed+7321)
        return dict(format=NAME, producer=code, classes=classes, seed=seed, arm=arm, factor=factor,
                    supervision=MODE,objective=LOSS,training=train,validation=val,inputs=inventory,
                    steps=STEPS,prefix=PREFIX,events=list(EVENTS),learning_rate=.001,module_seed=seed+7321,
                    rng_seed=seed+15901,registered_graph_scale=16,initialization=_initial(model,source),
                    numeric=None,schema=NAME,bank=self.bank.pin,physical_profiles=(self.profile.key,),
                    parameters=sum(p.numel() for p in model.parameters()),query_accessed=False,
                    raw_gt_loaded=False,real_updates=0)

    def selection_condition(self, actual, spec):
        return condition(actual,"matched")

    @staticmethod
    def stage_evidence(actual):
        source = actual.receipt["source"]
        return dict(parent_stats=source["statistics"]["parent_statistics_hash"],
                    retained_stats=source["retained_statistics"],band_ids=source["band_ids"])

    def validate_observation(self,spec,actual,labels,targets,event):
        check(evidence(actual,targets)==spec["inputs"][self.key(actual.receipt["role"],event)],
              "View, released points, retained statistics or native grid differ")

    def validate_history(self,spec,packet):
        for role,rows in (("devtrain",packet["history"]),
                          ("devval",[row for event in packet["validation"] for row in event["rows"]])):
            for row in rows:
                ref = spec["inputs"][self.key(role,row)]
                check(row["observation"]==ref["view"] and row["released"]==ref["released"]
                      and all(row[k]==ref[k] for k in ("parent_stats","retained_stats","band_ids")),
                      "Cross-grid, scale or split history rejected")
