"""WP11 fixed CNN and full forward controls from one accepted WP03 synthetic pair."""
import math
from pathlib import Path
import resource
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import torch

from evaluation import wp03_engineering_evidence as evidence
from model.checkpoint_io import load_model_checkpoint
from research_runtime import hash_file, require_frozen_code, tensor_hash, write_json
from transfer.registered_wp11_learning import read, verify

REQUEST = "wp11_wp03_static_request_v1"
MANIFEST = "wp11_wp03_static_manifest_v1"
RESULT = "wp11_wp03_fixed_comparison_v1"
MODE = "fixed_pair"


def pair(request):
    spec=read(request)
    if set(spec)!={"policy","wp12","registration_sha256","role","models","cost_scope","confirmation"} or spec["policy"]!=REQUEST:
        raise ValueError("Exact WP03 fixed-path engineering inputs required")
    if spec["cost_scope"]!=evidence.ADAPTATION:
        raise ValueError("WP12 adaptation costs do not qualify as inference latency")
    if spec["confirmation"] is not None:
        raise ValueError("WP03 synthetic engineering carries no independent calibration or confirmation")
    wp12=read(spec["wp12"])
    if (wp12.get("policy")!=evidence.POLICY or not evidence.is_synthetic(wp12)
            or wp12["production_accuracy_qualified"] is not False or wp12["cost_scope"]!=evidence.ADAPTATION):
        raise ValueError("Use WP12 synthetic engineering cost from the same batch")
    verified=evidence.build(wp12["proofs"])
    if any(wp12[k]!=v for k,v in verified.items()):
        raise ValueError("WP12 fixed-path source differs from the verified batch")
    runs=[r for r in wp12["runs"] if r["registration"]["sha256"]==spec["registration_sha256"]]
    if len(runs)!=1:
        raise ValueError("WP11 fixed path needs one exact WP03 run")
    scene,seed,mode=spec["role"]
    if mode!="none":
        raise ValueError("Graph-free fixed control requires the observed uncoupled none structure")
    by_method={r["method"]:r for r in runs[0]["rows"] if [r["scene"],r["seed"],r["mode"]]==[scene,seed,mode]}
    if set(by_method)!=set(evidence.METHODS):
        raise ValueError("Both distinct methods must share this scene/seed/structure")
    light,full = (by_method[method] for method in evidence.METHODS)
    if (light["observation"]!=full["observation"] or light["binding"]["input_sha256"]!=full["binding"]["input_sha256"]
            or light["binding"]["initial_sha256"]!=full["binding"]["initial_sha256"]
            or light["checkpoint"]==full["checkpoint"] or spec["models"]!={"light":light["checkpoint"],"full":full["checkpoint"]}):
        raise ValueError("Fixed light/full checkpoints or paired inputs differ")
    return spec,light,full


def register(request,code,out):
    spec,light,full=pair(request)
    verify([request,spec["wp12"],light["checkpoint"],full["checkpoint"],light["observation"]],code,out)
    write_json(out,{"policy":MANIFEST,"request":request,"producer":spec["wp12"],"scope":evidence.SCOPE,
                    "commit":code["commit"],"source_hash":code["source_hash"],"rows":[light,full],
                    "routing":None,"real_training_authorized":False})


def fixed_cnn(model,scene,item):
    backbone=model.backbone
    spectral,extra=backbone._split_spectral_extra(item["net_input"],item["wavelengths"])
    latent=backbone.adapter(spectral,wavelengths=item["wavelengths"],fwhm=item["fwhm"])
    noise=torch.cat((latent,extra),dim=2).permute(2,0,1).unsqueeze(0)
    for block in backbone.CNN_denoise:
        noise=block(noise,scene)
    clean=noise.squeeze(0).permute(1,2,0)
    pixels=backbone.cnn_stage1(clean,scene)
    features=backbone.cnn_stage2(pixels,clean.shape[:2],scene)
    return (model.cnn_heads[scene](features).softmax(-1),)


def full_forward(model,scene,item):
    return model(scene,item["net_input"],item["Q"],item["A"],wavelengths=item["wavelengths"],fwhm=item["fwhm"],
                 superpixel_scale=item["superpixel_scale"])[:3]


def run(manifest,code,mode,out,*,export):
    if mode!=MODE:
        raise ValueError("Adaptive or other routed modes require independent calibration and confirmation")
    if export:
        raise ValueError("The fixed WP03 comparison exports receipts, not individual predictions")
    data=read(manifest)
    if (data["policy"]!=MANIFEST or data["scope"]!=evidence.SCOPE or data["commit"]!=code["commit"]
            or data["source_hash"]!=code["source_hash"] or data["routing"] is not None):
        raise ValueError("Exact synthetic WP11 manifest required")
    spec,light,full=pair(data["request"])
    if data["rows"]!=[light,full] or data["producer"]!=spec["wp12"]:
        raise ValueError("Fixed-path evidence differs from its registration")
    pins=[manifest,data["request"],spec["wp12"],light["observation"],light["checkpoint"],light["metadata"],
          full["checkpoint"],full["metadata"]]
    verify(pins,code,out)
    if out.exists():
        raise ValueError("Fresh WP11 fixed-path result required")
    scene=light["scene"]
    models={key:load_model_checkpoint(row["checkpoint"]["path"]).requires_grad_(False)
           for key,row in (("light",light),("full",full))}
    for model in models.values():
        b=model.backbone
        if (b.graph_mode!="static" or any(getattr(b,k) is not None for k in
             ("interaction","region_residual","coverage_modulation","graph_router","edge_updater"))
             or any(m.training for m in model.modules())):
            raise ValueError("WP03 static control requires the two frozen uncoupled graphs")
    obs=light["observation"]
    item=torch.load(obs["path"],map_location="cpu",weights_only=True)
    if item["name"]!=scene or any("label" in k or "indices" in k for k in item):
        raise ValueError("Synthetic observation-only input required")
    comp={"light":fixed_cnn,"full":full_forward}
    states={key:{k:tensor_hash(x) for k,x in model.state_dict().items()} for key,model in models.items()}
    with torch.inference_mode():
        native=full_forward(models["light"],scene,item)[2]
        if not torch.equal(native,fixed_cnn(models["light"],scene,item)[0]):
            raise ValueError("Fixed CNN row changed the head-only checkpoint's output")
        rows={}
        for key,model in models.items():
            function=comp[key]
            for _ in range(2):
                function(model,scene,item)
            with patch.object(model.backbone,"graph_stage1",wraps=model.backbone.graph_stage1) as graph:
                times,digests=[],[]
                for _ in range(12):
                    begin=perf_counter_ns()
                    values=function(model,scene,item)
                    times.append((perf_counter_ns()-begin)/1e6)
                    if any(not torch.isfinite(v).all() or v.shape!=(item["height"]*item["width"],item["class_num"])
                           for v in values):
                        raise ValueError("Finite fixed-path probabilities required")
                    digests.append(tuple(tensor_hash(v) for v in values))
            if len(set(digests))!=1 or graph.call_count!=(0 if key=="light" else 12):
                raise ValueError("Actual fixed graph invocation or output differs")
            rows[key]={"graph_stage1_calls":graph.call_count,"resident_ms":times,"p50_ms":median(times),
                       "p95_ms":sorted(times)[math.ceil(len(times)*.95)-1],"probability_sha256":digests[0]}
    if any(state!={k:tensor_hash(x) for k,x in models[key].state_dict().items()} for key,state in states.items()):
        raise ValueError("Read-only fixed inference modified its model")
    require_frozen_code(code)
    verify(pins,code,out)
    write_json(out,{"policy":RESULT,"scope":evidence.SCOPE,"manifest":manifest,"wp12":spec["wp12"],
                    "role":spec["role"],"recorded_training_cost":{k:row["training_cost"] for k,row in (("light",light),("full",full))},
                    "training_cost_scope":evidence.ADAPTATION,"resident_cost_scope":"synthetic_static_cpu_prebound_graph",
                    "graph_construction_measured":False,"rows":rows,"process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "independent_confirmation":None,"model_updates":0,"query_access":False,"production_accuracy_qualified":False})
