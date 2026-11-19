"""Native three-head training on a controlled physical observation grid."""

from copy import deepcopy
from functools import lru_cache
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from evaluation.frozen_training_stages import file_pin, json_pin, require
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from transfer.development_bank import Bank
from transfer.development_bank_training import _initial
from transfer.native_long_run import append, checkpoints, packet
from transfer.native_coverage_queue import case_lock
from transfer import native_scale_training as native
from transfer.native_positive_fixture import (ARMS, CLOCK, CLASSES, ROLE_SEEDS, SCENE, SCIENTIFIC,
                                               SCHEMA, SHAPE, STEPS, TRAIN_VIEWS, cases, definition,
                                               profiles, publish_bank)
from transfer.native_released_points import LOSS, MODE
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_view
from transfer.spatial_coverage_training import SpatialReleasedFixture
from scripts.run_frozen_band_region import _write

GRAPH_SCALE = SCIENTIFIC["native_graph_scale"]


def aligned_spectra(spectral, axis):
    flat = spectral.reshape(-1,spectral.shape[-1])
    value = flat.new_zeros((len(flat),SHAPE[-1]))
    return value.index_copy(1,axis,flat-.5)


class CenterResidual(MultiDatasetVCASAHGCN):
    """A fixed center-spectral residual to the existing fused head only."""

    def forward(self, dataset_name,x,*args,wavelengths=None,**kwargs):
        require(wavelengths is not None,"Synthetic residual requires retained physical centers")
        pos = (wavelengths.reshape(-1)-440.)/40.
        indices = pos.round().long()
        require(torch.allclose(pos,indices.to(pos.dtype),rtol=0,atol=1e-5)
                and (indices >= 0).all() and (indices < SHAPE[-1]).all()
                and not (indices == 2).any() and len(torch.unique(indices)) == len(indices),
                "Synthetic center residual received a foreign physical axis")
        self._center = aligned_spectra(x[...,:-2],indices)
        try:
            return super().forward(dataset_name,x,*args,wavelengths=wavelengths,**kwargs)
        finally:
            del self._center

    def predict_features(self, dataset_name,fused,graph_features,cnn_features,return_logits=False):
        repeats = (fused.shape[-1]+self._center.shape[-1]-1)//self._center.shape[-1]
        residual = self._center.repeat(1,repeats)[:,:fused.shape[-1]]
        return super().predict_features(dataset_name,fused+2.*residual,graph_features,cnn_features,return_logits)


class PixelModel(nn.Module):
    """Capacity-matched MLP sharing no context or coordinate input across pixels."""

    def __init__(self, channel_pair):
        super().__init__()
        first,second = channel_pair
        self.layers = nn.Sequential(nn.Linear(SHAPE[-1],first),nn.GELU(),
                                    nn.Linear(first,second),nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(second,CLASSES) for _ in range(3))

    def forward(self, spectra, slots):
        hidden = self.layers(aligned_spectra(spectra,slots))
        return tuple(head(hidden).softmax(-1) for head in self.heads)


def new_model(seed, arm):
    model = CenterResidual if arm == "center" else MultiDatasetVCASAHGCN
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        built = model({SCENE:CLASSES}, graph_mode="learned",norm_type="group_norm",
                      latent_channels=128,adapter_embed_dim=64)
        built.backbone.adapter.cache_attention = False
    return built


def count(model):
    return sum(p.numel() for p in model.parameters())


@lru_cache(maxsize=1)
def pixel_capacity():
    target = count(new_model(64,"native"))
    center = int(target**.5)
    def difference(width):
        remaining = target-(SHAPE[-1]+1)*width-3*CLASSES
        second = round(remaining/(width+1+3*CLASSES))
        return abs((SHAPE[-1]+1)*width+(width+1)*second+(second+1)*3*CLASSES-target),width,second
    error,a,b = min(difference(w) for w in range(int(.9*center),int(1.1*center)+1))
    require(error/target < .005,"Pixel/native trainable capacity did not match")
    return a,b


def make_model(arm, seed):
    require(arm in ARMS,"Unknown positive control arm")
    if arm != "pixel":
        return new_model(seed,arm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PixelModel(pixel_capacity())


def admitted_forward(model, actual, producer, role, execution, *, order=None):
    if isinstance(model,PixelModel):
        actual.verify(producer,role)
        device = next(model.parameters()).device
        spectra = torch.from_numpy(actual.view.observation.spectral.copy()).to(device)
        indices = torch.tensor(actual.receipt["source"]["positions"],dtype=torch.long,device=device)
        start = perf_counter()
        output = model(spectra,indices)
        if execution is not None:
            execution.sync()
        return output,{"kind":"per_pixel_only","q_nnz":0,"a0_nnz":0},dict(forward_seconds=perf_counter()-start,graph_seconds=[])
    timed = {}
    backbone = model.backbone
    split = backbone._split_spectral_extra

    def permuted(x,wavelengths):
        values, coords = split(x,wavelengths)
        shuffle = torch.as_tensor(order,dtype=torch.long,device=x.device)
        return values,coords.reshape(-1,coords.shape[-1]).index_select(0,shuffle).reshape(coords.shape)

    if order is None:
        output,graph = forward(model,actual,producer,role,timed=timed,execution=execution,graph_scale=GRAPH_SCALE)
    else:
        with patch.object(backbone,"_split_spectral_extra",new=permuted):
            output,graph = forward(model,actual,producer,role,timed=timed,execution=execution,graph_scale=GRAPH_SCALE)
    return output,graph,timed


def register(root,out,code):
    require(root in out.parents, "Use a specific synthetic external run directory")
    with case_lock(out):
        folder = out/"fit"
        folder.mkdir(exist_ok=True)
        registered = {}
        for family,profile in profiles().items():
            bank, roles, correlation = publish_bank(folder/family,profile)
            registered[family] = dict(bank=bank.pin, roles=roles, same_neighbor_ratio=correlation)
        plan = dict(schema=SCHEMA,scope="synthetic_only",producer=code["commit"],source_hash=code["source_hash"],
                    fit=registered,cases=cases(),scientific=SCIENTIFIC,optimizer_updates=len(cases())*STEPS,
                    confirmation_created=False,query_accessed=False,real_updates=0,
                    estimate=dict(per_update_cpu_seconds=[.02,.10],workers=8,wall_minutes=[8,40],
                                  representative_gpu_policy="whole matched triples under physical lease"))
        path = out/"plan.json"
        if path.is_file():
            require(json_pin(root,file_pin(path)) == plan,"Frozen synthetic plan changed")
            return file_pin(path)
        return _write(path,plan)


def load_plan(root,out,code):
    pin = file_pin(out/"plan.json")
    plan = json_pin(root,pin)
    require(plan["scope"] == "synthetic_only" and plan["schema"] == SCHEMA
            and plan["producer"] == code["commit"] and plan["source_hash"] == code["source_hash"]
            and plan["cases"] == cases() and plan["scientific"] == SCIENTIFIC
            and plan["optimizer_updates"] == len(cases())*STEPS and plan["real_updates"] == 0
            and plan["confirmation_created"] is plan["query_accessed"] is False,
            "Synthetic-only plan identity or predeclared gates differ")
    return plan,pin


class PositiveProvider(SpatialReleasedFixture):
    """Keep the production optimizer, phase rules and sparse supervision."""

    def __init__(self,out,plan,name):
        item = plan["cases"][name]
        self.arm,self.seed = item["arm"],item["seed"]
        self.profile = profiles()[item["family"]]
        self.scene,self.classes,self.producer = SCENE,CLASSES,plan["producer"]
        entry = plan["fit"][item["family"]]
        release,_,roles,ratio = definition(self.profile)
        expected = out/"fit"/item["family"]
        require(entry["roles"] == roles and entry["same_neighbor_ratio"] == ratio
                and entry["bank"]["root"] == str(expected),"Paired fit-release identity differs")
        self.bank = Bank(entry["bank"],expected.parent,release)
        self.roles,self.cache,self.roster = roles,{},{}

    def baseline(self,seed):
        require(seed == self.seed,"Model seed differs from the frozen roster")
        return make_model(self.arm,seed)

    def model_factory(self,source,arm,seed):
        require(arm == self.arm and seed == self.seed,"Model factory received a foreign arm or seed")
        return source

    def materialize(self,role,event,producer):
        released = {"devtrain":"support","devval":"calibration"}.get(role)
        require(producer == self.producer and released is not None and event["tile"] in self.roles[released]
                and event["factor"] == 1 and (event["band"] in TRAIN_VIEWS if role == "devtrain" else event["band"] == "full"),
                "Unreleased role, observation or calibration view")
        key = self.key(role,event)
        if key not in self.cache:
            self.cache[key] = bank_view(self.bank,self.bank.role_row(role,event["tile"]),producer,event["band"])
        return self.cache[key]

    def registered_forward(self,model,actual,role,execution):
        return admitted_forward(model,actual,self.producer,role,execution)

    def contract(self,classes,seed,arm,factor,producer,*,numeric=None):
        require((classes,seed,arm,factor,producer)==(CLASSES,self.seed,self.arm,1,self.producer),
                "Synthetic model contract differs")
        self.bank.verify_authority(self.bank.pin)
        tiles = self.roles["support"]
        rng = np.random.default_rng([seed,483])
        order = [int(tile) for _ in range((STEPS+len(tiles)-1)//len(tiles)) for tile in rng.permutation(tiles)]
        train = [dict(step=i+1,tile=tile,band=TRAIN_VIEWS[(i//len(tiles)+tiles.index(tile))%3],
                      factor=1,profile=None,condition="missing") for i,tile in enumerate(order[:STEPS])]
        validation = [dict(tile=tile,band="full",factor=1,profile=None,condition="missing") for tile in self.roles["calibration"]]
        if not self.roster:
            for role,events in (("devtrain",train),("devval",validation)):
                for event in events:
                    key = self.key(role,event)
                    if key not in self.roster:
                        actual,_,targets = self.materialize(role,event,producer)
                        self.roster[key] = self.evidence(actual,targets)
        model = self.baseline(seed)
        return dict(format=SCHEMA,producer=producer,classes=classes,seed=seed,arm=arm,factor=1,
                    supervision=MODE,objective=LOSS,training=train,validation=validation,
                    inputs=deepcopy(self.roster),steps=STEPS,prefix=STEPS//2,events=list(CLOCK),
                    learning_rate=.001,module_seed=seed,rng_seed=seed+15000,registered_graph_scale=GRAPH_SCALE,
                    initialization=_initial(model,model),parameters=count(model),bank=self.bank.pin,numeric=numeric,
                    frozen_stage_protocol=SCHEMA,selection=SCIENTIFIC["selection"],real_updates=0,
                    raw_gt_loaded=False,query_accessed=False)


def run_case(root,out,code,name,execution):
    plan,plan_pin = load_plan(root,out,code)
    require(name in plan["cases"],"Unknown execution case")
    provider = PositiveProvider(out,plan,name)
    registration = provider.contract(CLASSES,provider.seed,provider.arm,1,code["commit"],numeric=execution.numeric)
    key = digest(registration)
    cases_dir = out/"cases"
    cases_dir.mkdir(exist_ok=True)
    folder = cases_dir/name
    with case_lock(folder):
        spec = folder/"registration.json"
        if spec.is_file():
            require(json_pin(root,file_pin(spec)) == registration,"Case arithmetic changed after first deposit")
        else:
            _write(spec,registration)
        final = folder/"receipt.json"
        once = checkpoints(root,folder,key)
        if final.is_file():
            retained = json_pin(root,file_pin(final))
            require(retained["last"] == once[-1]["pin"] and retained["registration"] == key
                    and retained["updates"] == STEPS,"Completed optimization identity differs")
            return dict(reused=True,new_updates=0,receipt=file_pin(final))
        model = native.Run(registration,code["commit"],execution=execution,released=provider)
        initial = bool(once)
        if initial:
            row = once[-1]
            model.restore(packet(root,row["pin"])["training"],row["step"])
        else:
            timing,start = dict(model.timing),perf_counter()
            model.start()
            row = append(folder,model,None,0,0,key,timing,"initial",start)
            once.append(row)
        cursor,parent = row["step"],row["pin"]["sha256"]
        start = cursor
        for stop in (step for step in CLOCK if step > cursor):
            before,now = dict(model.timing),perf_counter()
            for event in registration["training"][cursor:stop]:
                model.update(event)
            row = append(folder,model,parent,cursor,stop,key,before,"finish" if stop == STEPS else "interval",now)
            once.append(row)
            cursor,parent = stop,row["pin"]["sha256"]
        require(cursor == STEPS and sum(x["updates"] for x in once) == STEPS, "Incomplete fixed-budget trajectory")
        state = packet(root,once[-1]["pin"])["training"]
        retained = dict(schema=SCHEMA,case=name,plan=plan_pin,registration=key,spec=file_pin(spec),
                        checkpoints=[x["pin"] for x in once],last=once[-1]["pin"],selected=state["selected"]["step"],
                        updates=STEPS,query_accessed=False,real_updates=0)
        return dict(reused=False,new_updates=STEPS-start,receipt=_write(final,retained))


def freeze(root,out,code):
    plan,plan_pin = load_plan(root,out,code)
    published = {}
    for name in plan["cases"]:
        path = out/"cases"/name/"receipt.json"
        require(path.is_file(),"Every preregistered case must complete before any confirmation")
        fit = json_pin(root,file_pin(path))
        spec = json_pin(root,fit["spec"])
        stages = checkpoints(root,path.parent,fit["registration"])
        require(digest(spec) == fit["registration"] and fit["plan"] == plan_pin and fit["case"] == name
                and fit["updates"] == STEPS and fit["query_accessed"] is False and fit["real_updates"] == 0
                and [p["pin"] for p in stages] == fit["checkpoints"]
                and [p["step"] for p in stages] == list(CLOCK),"Foreign or incomplete training trajectory")
        published[name] = file_pin(path)
    path = out/"freeze.json"
    value = dict(schema=SCHEMA,scope=plan["scope"],plan=plan_pin,cases=published,updates=len(cases())*STEPS,
                 raw_gt_loaded=False,query_accessed=False)
    if path.is_file():
        require(json_pin(root,file_pin(path)) == value,"Freeze differs from archived training")
        return file_pin(path)
    return _write(path,value)
