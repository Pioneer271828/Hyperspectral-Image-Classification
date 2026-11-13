"""Preregistered native-grid training, exact continuation and endpoint replay."""

from copy import deepcopy
from itertools import product
import io
import random
from time import perf_counter

import numpy as np
import torch

from joint_training import _restore_rng, _rng_state
from model.research_config import digest
from model.task_losses import detached_loss_values
from research_runtime import tensor_hash
from transfer.development_bank import _atomic_new
from transfer.development_bank_training import _baseline, _initial, _state
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash
from transfer.native_scale_fixture import BANDS, PROFILES, Fixture, condition_for, registration
from transfer.native_scale_observation import EDGE_POLICY, FACTORS, forward, head_hashes
from transfer.native_scale_supervision import PURITY_DENOMINATOR, PURITY_NUMERATOR, loss, metrics
from transfer.spatial_multiscale import ARMS, initialize
from transfer.xuzhou_multiscale_audit import _exact


FORMAT = "wp07-native-scale-continuation-v1"
SEEDS = (64,512,2048)
CPU = torch.device("cpu")
MAIN = (96,7,12,"mixture")
HARD = (24,7,6,"high_purity")
EFFECT = dict(nll_margin=0.005, min_paired_seeds=3, min_matched_weight_change=1e-7,
              projected_oa_tolerance=0.01, rare_recall_tolerance=0.05,
              rare_definition="last four planted classes")
SLOTS = tuple((f,b,p) for f,b,p in product(FACTORS,BANDS,PROFILES) if not (f==4 and p in ("c","d")))
VAL_SLOTS = ((1,"full","a"),(1,BANDS[1],"c"),(2,"full","d"),(2,BANDS[1],"b"),
             (4,"full","a"),(4,BANDS[1],"b"))


def baseline(classes, seed):
    from types import SimpleNamespace
    return _baseline(SimpleNamespace(scene=registration(classes)["dataset"],classes=classes),seed)


def contract(classes, seed, arm, producer, *, mode="mixture"):
    if seed not in SEEDS or arm not in ARMS or not isinstance(producer,str) or not producer:
        raise ValueError("Invalid native campaign seed, arm or producer")
    if mode=="high_purity":
        if seed!=64 or arm!="joint":
            raise ValueError("Hard supervision has two fixed nine/twenty-two class recovery controls")
        steps,prefix,interval,_ = HARD
    elif mode=="mixture":
        steps,prefix,interval,_ = MAIN
    else:
        raise ValueError("Unknown native supervision mode")
    fixture = registration(classes)
    train = Fixture(classes,"devtrain")
    rng = np.random.default_rng(seed)
    orders = [int(i) for _ in range(steps//len(train.tiles)) for i in rng.permutation(train.tiles)]
    schedule = []
    for clock,tile in enumerate(orders,1):
        factor,band,profile = SLOTS[(clock-1)%len(SLOTS)]
        schedule.append(dict(step=clock,tile=tile,factor=factor,band=band,profile=profile,condition="matched"))
    val = [dict(tile=tile,factor=factor,band=band,profile=profile,condition="matched")
           for tile,(factor,band,profile) in product(Fixture(classes,"devval").tiles,VAL_SLOTS)]
    source = baseline(classes,seed)
    model = initialize(source,arm,seed+7321)
    return dict(format=FORMAT,producer=producer,classes=classes,seed=seed,arm=arm,supervision=mode,
                steps=steps,prefix=prefix,events=list(range(0,steps+1,interval)),training=schedule,validation=val,
                module_seed=seed+7321,rng_seed=seed+15000,learning_rate=0.0001,
                edge=EDGE_POLICY,graph_scale_native_pixels=16,fixture=fixture,
                area_weights="represented_original_pixels",
                purity=[PURITY_NUMERATOR,PURITY_DENOMINATOR],selection="first_min_mixture_fused_nll",
                effect=dict(EFFECT),initialization=_initial(model,source),
                parameters=sum(param.numel() for param in model.parameters()),real_updates=0)


class Run:
    def __init__(self, registered, producer):
        clean = deepcopy(registered)
        if clean != contract(clean["classes"],clean["seed"],clean["arm"],producer,mode=clean["supervision"]):
            raise ValueError("Native training registration, grid, supervision or producer differs")
        self._contract = clean
        self.producer = producer
        self.model = initialize(baseline(clean["classes"],clean["seed"]),clean["arm"],clean["module_seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=clean["learning_rate"])
        self._train = Fixture(clean["classes"],"devtrain")
        self._val = Fixture(clean["classes"],"devval")
        self._history, self._validation, self._selected = [], [], None
        self.timing = dict(train_update_seconds=0.,validation_seconds=0.,transform_seconds=0.,
                           forward_seconds=0.,graph_seconds=0.)
        self._fresh = tuple(clean["initialization"]["fresh"])

    @property
    def registered(self):
        return deepcopy(self._contract)

    def _materialize(self, fixture, event):
        start = perf_counter()
        result = fixture.materialize(event["tile"],event["factor"],event["band"],event["profile"],self.producer)
        self.timing["transform_seconds"] += perf_counter()-start
        return result

    def _forward(self, actual, role, intervention):
        condition = condition_for(actual,intervention)
        timed = {}
        out,graph = forward(self.model,actual,self.producer,role,condition=condition,timed=timed)
        self.timing["forward_seconds"] += timed["forward_seconds"]
        self.timing["graph_seconds"] += sum(timed["graph_seconds"])
        return out,graph,condition

    @torch.no_grad()
    def evaluate(self, step, *, detailed=False):
        started = perf_counter()
        self.model.eval()
        rows = []
        for event in self._contract["validation"]:
            actual,labels,targets = self._materialize(self._val,event)
            output,graph,condition = self._forward(actual,"devval",event["condition"])
            rows.append(dict(**event,observation=actual.receipt["observation"],
                             labels=array_hash(labels),condition_key=condition.source_key,
                             area=int(targets.area.sum()),heads=head_hashes(output,actual),graph=graph,
                             scores=metrics(output,actual,labels,targets,detailed=detailed)))
        numerator = sum(row["scores"]["fused"]["nll_sum"] for row in rows)
        denominator = sum(row["scores"]["fused"]["count"] for row in rows)
        self.timing["validation_seconds"] += perf_counter()-started
        return dict(step=step,fused_nll=numerator/denominator,rows=rows)

    def _record_validation(self, step):
        value = self.evaluate(step)
        self._validation.append(value)
        if self._selected is None or value["fused_nll"]<self._selected["fused_nll"]:
            self._selected = dict(step=step,fused_nll=value["fused_nll"],state=_state(self.model))

    def start(self):
        if self._validation:
            raise ValueError("Native run is already started")
        seed = self._contract["rng_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._record_validation(0)

    def update(self, requested):
        cursor = len(self._history)
        if (not self._validation or cursor>=self._contract["steps"]
                or requested!=self._contract["training"][cursor]):
            raise ValueError("Update differs from the next registered native exposure")
        event = self._contract["training"][cursor]
        started = perf_counter()
        actual,labels,targets = self._materialize(self._train,event)
        # Empty eligible supervision is rejected before changing optimizer
        # state, gradients, model mode or the exposure cursor.
        values, mask = targets.weights(self._contract["supervision"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        out,graph,condition = self._forward(actual,"devtrain",event["condition"])
        terms = loss(out,values,mask)
        if not torch.isfinite(terms["total"]):
            raise ValueError("Nonfinite native supervision")
        named = dict(self.model.named_parameters())
        before = {name:tensor_hash(named[name]) for name in self._fresh}
        terms["total"].backward()
        grads = {name:None if named[name].grad is None else float(named[name].grad.abs().sum())
                 for name in self._fresh}
        self.optimizer.step()
        self._history.append(dict(**event,observation=actual.receipt["observation"],labels=array_hash(labels),
            condition_key=condition.source_key,gsd_m=condition.gsd_m,area=int(targets.area.sum()),
            supervised_area=int(mask[:,0].sum()),histogram=array_hash(targets.histogram),
            heads=head_hashes(out,actual),graph=graph,loss=detached_loss_values(terms),
            fresh={name:dict(gradient_l1=grads[name],changed=tensor_hash(named[name])!=before[name])
                   for name in self._fresh}))
        self.timing["train_update_seconds"] += perf_counter()-started
        if event["step"] in self._contract["events"]:
            self._record_validation(event["step"])

    def pack(self):
        return dict(format=FORMAT,registration=digest(self._contract),cursor=len(self._history),
                    model=_state(self.model),optimizer=deepcopy(self.optimizer.state_dict()),rng=_rng_state(CPU),
                    history=deepcopy(self._history),validation=deepcopy(self._validation),
                    selected=deepcopy(self._selected))

    def restore(self, value, cursor):
        packet = deepcopy(value)
        expected = self._contract["training"][:cursor]
        if (packet["format"]!=FORMAT or packet["registration"]!=digest(self._contract)
                or packet["cursor"]!=cursor or len(packet["history"])!=cursor
                or [x["step"] for x in packet["validation"]] != [x for x in self._contract["events"] if x<=cursor]
                or any({k:a[k] for k in b}!=b for a,b in zip(packet["history"],expected,strict=True))):
            raise ValueError("Native checkpoint registration or cursor differs")
        selected = min(packet["validation"],key=lambda x:x["fused_nll"])
        if any(packet["selected"][key]!=selected[key] for key in ("step","fused_nll")):
            raise ValueError("Native checkpoint selection differs")
        self.model.load_state_dict(packet["model"],strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        if not self.optimizer.state or any(int(s["step"])!=cursor for s in self.optimizer.state.values()):
            raise ValueError("Native Adam cursor differs")
        self._history, self._validation, self._selected = packet["history"],packet["validation"],packet["selected"]
        _restore_rng(packet["rng"],CPU)


def execute(registered, phase, path, producer, prior=None):
    run = Run(registered,producer)
    spec = run.registered
    prefix,steps = spec["prefix"],spec["steps"]
    start,end = {"full":(0,steps),"prefix":(0,prefix),"resume":(prefix,steps),"completed":(steps,steps)}[phase]
    if (start>0)!=(prior is not None) or path.exists():
        raise ValueError("Native phase requires the pinned predecessor and a new destination")
    if start:
        run.restore(prior,start)
    else:
        run.start()
    for event in spec["training"][start:end]:
        run.update(event)
    packet = run.pack()
    if phase=="completed" and not _exact(prior,packet):
        raise ValueError("Completed native re-entry changed state")
    with io.BytesIO() as buffer:
        torch.save(packet,buffer)
        _atomic_new(path,buffer.getvalue())
    return dict(phase=phase,updates=end-start,state=state_hash(run.model),selected=packet["selected"]["step"],
                timing=run.timing)
