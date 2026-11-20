"""Registered bounded synthetic repair with exact replay and a withheld split."""

from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.nn import functional as F

from evaluation.classification_metrics import classification_metrics
from evaluation.representation_localization import Feature, HEADS, LayerIdentity, local_response
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, tensor_hash
from transfer.g3_training import state_hash
from transfer.registered_development import _json_new, _pin, _read
from transfer.representation_bypass import ARMS, BANDS, LAYOUTS, PREFIX, ROLE_COUNT, SEEDS, STEPS, TASKS, VALIDATE
from transfer.representation_bypass import ResidualBottleneckClassifier, observation

NAME = "wp05_wp06_frozen_representation_repair_v1"
CASES = {f"{task}-{layout}-{seed}":(task,layout,seed) for task,layout,seed in product(TASKS,LAYOUTS,SEEDS)}
PHASES = ("full","prefix","resume","completed")
RATE = .0125


def read(pin):
    return json.loads(_read(pin))


def spec(code):
    return dict(schema=NAME, producer=code["commit"], source_hash=code["source_hash"], cases=list(CASES),
                batch=2,arms=list(ARMS),steps=STEPS,prefix=PREFIX,validation=list(VALIDATE),roles=ROLE_COUNT,
                optimizer="Adam",rate=RATE,minimum_valid_cases=4,
                hypothesis={"recoverable_min_oa":.8,"recoverable_min_delta":.2,
                            "removed_and_noise_max_delta":.1,"context_min_oa":.8},
                query=False,raw_gt=False,real_updates=0)


def register(out,code):
    out.mkdir()
    return _json_new(out/"registration.json",spec(code))


def scope(out,code,expected=None):
    pin = _pin(out/"registration.json")
    if (expected is not None and pin["sha256"]!=expected) or read(pin)!=spec(code):
        raise ValueError("Registered representation campaign differs")
    return pin["sha256"]


def encode(value):
    if torch.is_tensor(value):
        return tensor_hash(value)
    if isinstance(value,dict):
        return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [encode(x) for x in value]
    return value


def state_fingerprint(state):
    return digest(encode({k:v for k,v in state.items() if k!="state"}))


class ReleasedFixture:
    def __init__(self,task,layout,seed):
        self.task,self.layout,self.seed = task,layout,seed
        self.train_val = {role:{band:[observation(task,layout,seed,role,i,band) for i in range(ROLE_COUNT[role])]
                                for band in BANDS} for role in ("train","validation")}
        self.signature = {role:{band:[w.identity for w in windows] for band,windows in x.items()}
                          for role,x in self.train_val.items()}

    def windows(self,role,band):
        if role=="confirmation":
            return [observation(self.task,self.layout,self.seed,role,i,band) for i in range(ROLE_COUNT[role])]
        return self.train_val[role][band]


class Run:
    def __init__(self,fixture,arm,registered):
        self.fixture,self.arm = fixture,arm
        self.model = ResidualBottleneckClassifier(fixture.seed,arm)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=RATE)
        self.initial = state_hash(self.model)
        self.contract = digest(dict(scope=registered,arm=arm,fixture=fixture.signature,initial=self.initial))
        self.history,self.validation,self.selected = [],[],None
        self.times = defaultdict(float)

    def update(self):
        step = len(self.history)+1
        if not self.validation or step>STEPS:
            raise ValueError("Update outside the fixed exposure contract")
        self.model.train()
        band = BANDS[(step-1)%2]
        source = self.fixture.windows("train",band)
        used = [source[(step-1+j*7)%len(source)] for j in (0,1)]
        start = perf_counter()
        predictions = [self.model(w) for w in used]
        self.times["train_forward_seconds"]+=perf_counter()-start
        loss = sum(F.cross_entropy(head[w.focus],w.label[w.focus])
                   for w,p in zip(used,predictions,strict=True) for head in p)/6.
        self.optimizer.zero_grad(set_to_none=True)
        start = perf_counter()
        loss.backward()
        gradients = {key:sum(float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None)
                     for key,layer in (("adapter",self.model.adapter),("graph",self.model.graph),
                                       ("local",self.model.local),("fusion",self.model.fused_head))}
        self.times["train_backward_seconds"]+=perf_counter()-start
        start = perf_counter()
        self.optimizer.step()
        self.times["optimizer_seconds"]+=perf_counter()-start
        self.history.append(dict(step=step,views=[w.identity for w in used],band=band,
                                 nnz=[w.a._nnz() for w in used],loss=float(loss.detach()),gradients=gradients))
        if step in VALIDATE:
            self.validate(step)

    @torch.no_grad()
    def score(self,role,details):
        self.model.eval()
        result = {}
        start = perf_counter()
        for band in BANDS:
            y, boundaries, views = [],[],[]
            heads = {name:[] for name in HEADS}
            for w in self.fixture.windows(role,band):
                score = self.model(w)
                y.extend((w.label[w.focus]+1).tolist())
                boundaries.extend(w.boundary[w.focus].tolist())
                views.append(w.identity)
                for name,p in zip(HEADS,score,strict=True):
                    heads[name].extend(p[w.focus].softmax(1).tolist())
            info = dict(views=views,population=len(y),boundary=sum(boundaries),metrics={})
            labels = torch.as_tensor(y)
            for name,probs in heads.items():
                matrix = torch.as_tensor(probs,dtype=torch.float64)
                guesses = matrix.argmax(1)+1
                same = guesses==labels
                n = len(labels)
                metrics = classification_metrics(y,guesses.tolist(),[1,2,3])
                metrics["nll"] = float(-matrix[torch.arange(n),labels-1].clamp(min=1e-20).log().mean())
                scores = {}
                for group,mask in (("boundary",torch.as_tensor(boundaries)),("interior",~torch.as_tensor(boundaries))):
                    count=int(mask.sum())
                    scores[group]=dict(denominator=count,correct=int(same[mask].sum()),
                                       oa=float(same[mask].float().mean()) if count else None)
                metrics["groups"]=scores
                info["metrics"][name]=metrics
            if details:
                info.update(labels=y,truth_boundary=boundaries,probabilities=heads)
            result[band]=info
        self.times[role+"_seconds"]+=perf_counter()-start
        return result

    def validate(self,step):
        proof = self.score("validation",False)
        row = dict(step=step,nll=sum(proof[x]["metrics"]["fused"]["nll"] for x in BANDS)/len(BANDS),metrics=proof)
        self.validation.append(row)
        if self.selected is None or row["nll"]<self.selected["nll"]:
            self.selected = dict(step=step,nll=row["nll"],model=deepcopy(self.model.state_dict()))

    def pack(self):
        value = deepcopy(dict(contract=self.contract,cursor=len(self.history),model=self.model.state_dict(),
                              optimizer=self.optimizer.state_dict(),history=self.history,
                              validation=self.validation,selected=self.selected))
        return dict(**value,state=state_fingerprint(value))

    def restore(self,packet,step):
        if (self.history or packet["contract"]!=self.contract or packet["cursor"]!=step or len(packet["history"])!=step
                or [v["step"] for v in packet["validation"]]!=[v for v in VALIDATE if v<=step]
                or packet["state"]!=state_fingerprint(packet)
                or any(int(x["step"])!=step for x in packet["optimizer"]["state"].values())):
            raise ValueError("Frozen residual restore identity, optimizer clock or route differs")
        self.model.load_state_dict(packet["model"],strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        self.history,self.validation,self.selected = (packet[k] for k in ("history","validation","selected"))


def load(path, expected):
    pin = _pin(path)
    if pin["sha256"]!=expected:
        raise ValueError("Residual checkpoint bytes changed")
    return torch.load(path,map_location="cpu",weights_only=True)


def execute_arm(place,fixture,arm,scope):
    if place.exists():
        row = read(_pin(place/"receipt.json"))
        if row["scope"]!=scope or row["arm"]!=arm:
            raise ValueError("Existing residual fit belongs to another registration")
        return _pin(place/"receipt.json")["sha256"]
    stages = {}
    with TemporaryDirectory(dir=place.parent,prefix=f".{arm}-") as temp:
        stage = Path(temp)/"sealed"
        stage.mkdir()
        for phase,start,end,parent in (("full",0,STEPS,None),("prefix",0,PREFIX,None),
                                       ("resume",PREFIX,STEPS,"prefix"),("completed",STEPS,STEPS,"resume")):
            run = Run(fixture,arm,scope)
            if parent:
                run.restore(load(stage/f"{parent}.pt",stages[parent]["checkpoint"]),start)
            else:
                run.validate(0)
            while len(run.history)<end:
                run.update()
            packet = run.pack()
            file = stage/f"{phase}.pt"
            torch.save(packet,file)
            stages[phase] = dict(steps=end-start,cursor=end,state=packet["state"],checkpoint=_pin(file)["sha256"],
                                 seconds=dict(run.times))
        if len({stages[p]["state"] for p in ("full","resume","completed")})!=1:
            raise ValueError("Recovered residual checkpoint differs from uninterrupted path")
        full = load(stage/"full.pt",stages["full"]["checkpoint"])
        result = _json_new(stage/"receipt.json",dict(arm=arm,scope=scope,initial=run.initial,
                          phases=stages,history=full["history"],validation=full["validation"],
                          chosen=full["selected"]["step"],parameters=sum(p.numel() for p in run.model.parameters())))
        os.rename(stage,place)
    return result["sha256"]


def bottleneck(run):
    observations = {}
    for band in BANDS:
        window = run.fixture.windows("confirmation",band)[0]
        with torch.no_grad():
            original = run.model.split(window)
        site = LayerIdentity(state_hash(run.model),"synthetic",window.identity,"residual","synthetic_confirmation")
        off = Feature(site,"zero_residual",torch.zeros_like(original.residual))
        own = Feature(site,"actual_observed_residual",original.residual)
        wanted = window.focus.tolist()[:6]
        truth = (window.label[wanted]+1).tolist()
        other = [i%3+1 for i in truth]
        def predictor(value):
            return tuple(x.softmax(1) for x in run.model.predictor(window,original,value))
        observations[band] = dict(witness=window.witness,
                                 math=local_response(predictor,off,own,pixels=wanted,truth=truth,competitors=other))
    return observations


def execute(out,code,selector):
    sha = scope(out,code)
    match = [case for case in CASES if selector=="all" or selector==case or case.rsplit("-",1)[0]==selector]
    if not match:
        raise ValueError("Unregistered representation case")
    for case in match:
        start = perf_counter()
        target = out/case
        target.mkdir(exist_ok=True)
        claim = target/".lock"
        import fcntl
        with claim.open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            pin = target/"receipt.json"
            if pin.exists():
                if read(_pin(pin))["scope"]!=sha:
                    raise ValueError("Existing case belongs to another campaign")
                print("reused",case,flush=True)
                continue
            fixture = ReleasedFixture(*CASES[case])
            arms = {arm:execute_arm(target/arm,fixture,arm,sha) for arm in ARMS}
            # No confirmation sample is instantiated until every arm is frozen.
            result = {}
            for arm,arm_pin in arms.items():
                row = read({"path":str(target/arm/"receipt.json"),"sha256":arm_pin})
                full = load(target/arm/"full.pt",row["phases"]["full"]["checkpoint"])
                run = Run(fixture,arm,sha)
                run.model.load_state_dict(full["selected"]["model"],strict=True)
                result[arm] = dict(endpoint=run.score("confirmation",True),times=dict(run.times),
                                   diagnostics=bottleneck(run) if arm=="aligned" else None)
            require_frozen_code(code)
            _json_new(pin,dict(case=case,scope=sha,arms=arms,results=result,seconds=perf_counter()-start,
                               query=False,raw_gt=False,real_updates=0))
            print("completed",case,flush=True)
