"""Prepare native requests once and execute explicit graph or light routes."""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import torch

from inference.light_cnn import NonfiniteLightOutput
from inference.tile_cascade import gate_score
from model.coverage_config import coverage_forward_kwargs
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.g3_training import state_hash

STATIC = ("full", "light")
ROUTED = ("adaptive", "random")
COUNTS = ("light_calls","full_calls","graph_builds","graph_reuses","gate_calls","coverage_fallbacks","light_failures")


@dataclass(frozen=True)
class Request:
    key: str
    scene: str
    observation: str
    parent: str
    band: str
    x: torch.Tensor
    waves: torch.Tensor | None
    widths: torch.Tensor | None
    meta: object
    scale: int
    coverage: float
    known: bool

    @property
    def cohort(self):
        return self.band, tuple(self.x.shape[:2]), self.coverage, self.known

    @property
    def identity(self):
        return digest([self.key,self.scene,self.observation,self.parent,self.band,self.scale,self.coverage,self.known,
                       tensor_hash(self.x),None if self.waves is None else tensor_hash(self.waves),
                       None if self.widths is None else tensor_hash(self.widths)])


def prepare(actual, producer, *, key, device, original_band_count, expected=None):
    raw = actual.verify(producer,"devval")
    if (raw["synthetic_profile"] is not None or raw["factor"]!=1
            or (expected is not None and raw["observation"]!=expected)):
        raise ValueError("Deployment observation differs from its fixed native request")
    src = raw["source"]
    mode = src.get("spectral_metadata_mode","declared")
    if mode not in ("declared","missing","index","shuffled"):
        raise ValueError("Unregistered deployment band metadata")
    bands=actual.view.bands if mode!="missing" else None
    dtype=torch.float32
    to_device=lambda array: torch.as_tensor(np.array(array,copy=True),dtype=dtype,device=device)
    observation=actual.view.observation
    host=np.concatenate((observation.spectral,observation.coordinates),axis=-1)
    if type(original_band_count) is not int or original_band_count<len(src["band_ids"]):
        raise ValueError("Explicit published original band count required")
    coverage=float(len(src["band_ids"])/original_band_count)
    known=(mode=="declared" and bands.wavelength_unit=="nm" and bands.fwhm is None)
    check_coverage=0<coverage<=1
    if not check_coverage:
        raise ValueError("Invalid original-band coverage")
    wave=None if bands is None else to_device(bands.wavelengths)
    width=None if bands is None or bands.fwhm is None else to_device(bands.fwhm)
    return Request(key,actual.dataset,raw["observation"],digest([src["bank"],raw["box"],raw["footprints"]]),
                   src["band_view"],to_device(host),wave,width,bands,50 if actual.dataset=="paviaU" else 100,
                   coverage,known)


class RecoverableLightFailure(RuntimeError):
    """Data-dependent fast-path failure that permits a proven healthy full route."""


class RouteFailure(RuntimeError):
    def __init__(self,key,stats):
        super().__init__(f"Required full deployment path failed for {key}")
        self.stats=dict(stats)


def uninstrumented(model):
    for module in model.modules():
        if module.training or module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Deployment requires eval modules without diagnostic hooks or temperature overrides")
    owner=getattr(model,"backbone",None)
    if owner is not None:
        sites=("cnn_stage1","cnn_stage2","graph_stage1","graph_stage2","fuse_features")
        if (set(sites)&vars(owner).keys() or "build_graph" in vars(owner.learnable_graph)
                or owner.adapter.cache_attention):
            raise ValueError("Deployment refuses mutable diagnostics or spectral caches")


def clock_ms(start):
    return (perf_counter_ns()-start)/1e6


class Runtime:
    def __init__(self, *, device, full=None, light=None, identity=None, graph_policy="rebuild", full_loader=None):
        self.device=torch.device(device)
        if graph_policy not in ("rebuild","reuse_frozen") or all(x is None for x in (full,light,full_loader)):
            raise ValueError("Explicit model and graph policy required")
        if full_loader is not None:
            if (not callable(full_loader) or full is not None or graph_policy != "rebuild"
                    or identity is None or set(identity) != ({"full", "light"} if light is not None else {"full"})
                    or not isinstance(identity["full"], str) or len(identity["full"]) != 64):
                raise ValueError("Deferred full deployment requires a pinned state and rebuilt graphs")
        for model in (full,light):
            if model is not None:
                uninstrumented(model)
                if next(model.parameters()).device.type!="cpu":
                    raise ValueError("Authenticate frozen weights on CPU before deployment")
                model.to(self.device)
        self.full,self.light=full,light
        self.full_loader = full_loader
        self.identity={k:state_hash(m) for k,m in (("full",full),("light",light)) if m is not None}
        if full_loader is not None:
            self.identity["full"] = identity["full"]
        if identity is not None and self.identity!=identity:
            raise ValueError("Deployment model identity differs from the registered pair")
        self.routable=("full" in self.identity and light is not None and
                       getattr(light,"source_state",None)==self.identity["full"])
        self.graph_policy=graph_policy
        if graph_policy=="reuse_frozen" and (full is None or type(full.backbone.learnable_graph) is not LearnableSuperpixelGraph
                                              or isinstance(full,ConditionalMultiscaleVCASAHGCN)):
            raise ValueError("Only a plain native graph policy may reuse a frozen graph")
        self.cache={}
        self.versions=self.version()

    def version(self):
        return tuple((n,p.data_ptr(),p._version) for m in (self.full,self.light) if m is not None
                     for n,p in tuple(m.named_parameters())+tuple(m.named_buffers()))

    def sync(self):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)

    def _full(self, request, stats, graph_clock):
        if self.full is None and self.full_loader is not None:
            # Preserve tensor version counters for the runtime's mutation guard.
            with torch.inference_mode(False):
                model = self.full_loader()
                uninstrumented(model)
                if next(model.parameters()).device.type != "cpu" or state_hash(model) != self.identity["full"]:
                    raise ValueError("Deferred full model differs from its pinned CPU state")
                model.to(self.device)
            self.full = model
            self.versions = self.version()
        model=self.full
        stats["full_calls"]+=1
        if model is None:
            raise ValueError("Full model was not loaded")
        builder=model.backbone.learnable_graph
        load=builder.build_graph
        cache_key=(self.identity["full"],request.identity) if self.graph_policy=="reuse_frozen" else None
        cached=self.cache.get(cache_key)

        def build(features,*args,**kwargs):
            if cached is not None:
                saved, feature = cached
                if feature!=tensor_hash(features):
                    raise ValueError("Frozen graph cache has foreign model features")
                stats["graph_reuses"]+=1
                return saved.validate(features)
            stats["graph_builds"]+=1
            started=perf_counter_ns()
            event=None
            if self.device.type=="cuda":
                event=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                event[0].record()
            result=load(features,*args,**kwargs)
            if event is None:
                graph_clock.append(clock_ms(started))
            else:
                event[1].record()
                graph_clock.append(event)
            if cache_key is not None:
                self.cache[cache_key]=(result,tensor_hash(features))
            return result

        builder.build_graph=build
        try:
            with self._band_context(request):
                options=coverage_forward_kwargs(model,request.meta)
                if request.meta is not None:
                    options["wavelength_unit"]=request.meta.wavelength_unit
                    if request.widths is not None:
                        options.update(fwhm=request.widths,fwhm_unit=request.meta.fwhm_unit)
                return model(request.scene,request.x,wavelengths=request.waves,
                             superpixel_scale=request.scale,**options)[:3]
        finally:
            delattr(builder,"build_graph")

    def _band_context(self,req):
        if isinstance(self.full,ConditionalMultiscaleVCASAHGCN):
            return self.full.partition_bands(req.meta)
        return nullcontext()

    def _light(self, request, stats):
        if self.light is None or not request.known:
            raise ValueError("Light path requires its model and admitted native wavelengths")
        stats["light_calls"]+=1
        return self.light(request.scene,request.x,wavelengths=request.waves)

    @torch.inference_mode()
    def run(self, requests, mode, *, threshold=None, min_coverage=0., random_fast=None, faults=None):
        if mode not in (*STATIC,*ROUTED) or not requests or len({r.key for r in requests})!=len(requests):
            raise ValueError("Explicit deployment path and unique native requests required")
        if mode in ROUTED and not self.routable:
            raise ValueError("Native routing requires light and full from one frozen source")
        if mode=="adaptive" and (threshold is None or not 0<=threshold<=1):
            raise ValueError("Only the pinned margin threshold may route")
        if mode=="random" and (random_fast is None or not set(random_fast)<={r.key for r in requests}):
            raise ValueError("Explicit matched workload random routes required")
        if self.version()!=self.versions:
            self.cache.clear()
            raise ValueError("Frozen deployment model changed")
        if any(r.x.device!=self.device for r in requests):
            raise ValueError("Request is resident on a foreign device")
        stats=Counter(dict.fromkeys(COUNTS,0))
        routes, probability, graph_clock=[],[],[]
        faults={} if faults is None else faults
        self.sync()
        started=perf_counter_ns()
        outer=None
        if self.device.type=="cuda":
            outer=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            outer[0].record()
        for request in requests:
            route,score,reason="full",None,"fixed"
            p=None
            allowed=request.known and request.coverage>=min_coverage
            if mode=="light":
                p=(self._light(request,stats),)
                route="light"
            if mode in ROUTED:
                if not allowed:
                    stats["coverage_fallbacks"]+=1
                    reason="metadata_or_coverage"
                else:
                    try:
                        light=self._light(request,stats)
                        if faults.get(request.key)=="light":
                            raise RecoverableLightFailure("Registered light failure")
                        if mode=="adaptive":
                            score=gate_score(light,"margin")
                            stats["gate_calls"]+=1
                            accepted=score>=threshold
                        else:
                            accepted=request.key in random_fast
                        if accepted:
                            p=(light,)
                            route="light"
                        reason="accepted" if accepted else "confidence_or_control"
                    except (NonfiniteLightOutput, RecoverableLightFailure):
                        stats["light_failures"]+=1
                        reason="light_failure"
            if p is None:
                try:
                    if faults.get(request.key)=="full":
                        raise RuntimeError("Registered full failure")
                    p=self._full(request,stats,graph_clock)
                except Exception as error:
                    self.sync()
                    raise RouteFailure(request.key,dict(stats,failed_elapsed_ms=clock_ms(started))) from error
            routes.append(dict(key=request.key,route=route,reason=reason,score=score))
            probability.append(p)
        if outer is not None:
            outer[1].record()
        self.sync()
        elapsed=clock_ms(started)
        if any(any(not torch.isfinite(t).all() for t in row) for row in probability):
            raise ValueError("Nonfinite deployment probabilities")
        if stats["graph_builds"]+stats["graph_reuses"]!=stats["full_calls"]:
            raise ValueError("Deployment full route bypassed its graph")
        return dict(routes=routes,counts=dict(stats),predictions=probability,elapsed_ms=elapsed,
                    gpu_event_ms=None if outer is None else outer[0].elapsed_time(outer[1]),
                    graph_ms=sum(graph_clock) if self.device.type=="cpu" else sum(a.elapsed_time(b) for a,b in graph_clock))


def output_hash(result):
    return digest([[tensor_hash(t) for t in row] for row in result["predictions"]])


def matched_random(requests, adaptive, seed, min_coverage):
    """Match actual fast/full counts for every workload shape and band."""
    accepted={r["key"] for r in adaptive["routes"] if r["route"]=="light"}
    strata={}
    for request in requests:
        if request.known and request.coverage>=min_coverage:
            strata.setdefault(request.cohort,[]).append(request.key)
    rng=np.random.default_rng(seed)
    picks=[]
    for values in strata.values():
        n=len(accepted.intersection(values))
        picks.extend(str(x) for x in rng.permutation(values)[:n])
    return sorted(picks)
