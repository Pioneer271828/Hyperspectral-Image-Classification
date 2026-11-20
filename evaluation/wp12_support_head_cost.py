"""Label-free native deployment requests and WP12 resident block measurements."""

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import torch

from evaluation import frozen_training_stages as stages
from evaluation.wp12_cost_primitives import timing_summary
from inference import support_readout as readout
from inference.native_deployment import Request, Runtime, output_hash, prepare
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from research_runtime import write_json
from transfer.development_bank import _atomic_new
from transfer.development_blocks import BlockCube
from transfer.registered_development import _load_source
from transfer.spatial_coverage_protocol import bank_observation

SCHEMA = "wp12_support_head_observation_plan_v1"
COST = "wp12_support_head_cost_v1"
MODES = ("native","fixed_full","same_view")
REPEATS = 3
require = stages.require


def middle_blocks(rows):
    """Select two adjacent area ranks using only admitted block geometry."""
    def area(row):
        y0,y1,x0,x1 = row["box"]
        return ((y1-y0)*(x1-x0),row["id"])

    ordered = sorted(rows,key=area)
    require(len(ordered)>=2,"Two authorized development blocks are required")
    middle = (len(ordered)-1)//2
    return ordered[middle:middle+2]


def xuzhou(root, source, producer):
    from scripts.run_wp12_retained_evidence import RELEASE, pin

    release_pin = pin(root,RELEASE)
    plan = stages.stage_plan(root,release_pin)
    release = stages.json_pin(root,release_pin)
    checkpoint = plan["stages"]["selected"]
    require(checkpoint["state"]["sha256"]==source["checkpoint"] and checkpoint["step"]==source["cursor"]
            and release["full_state"]==source["state"] and plan["development"]["sha256"]==source["authority"]
            and plan["bands"]==source["bands"],"Released Xuzhou endpoint differs")
    model = stages.load_stage(root,release,checkpoint,"selected")
    manifest = stages.json_pin(root,plan["development"])
    blocks = {r["id"]:r for r in manifest["blocks"] if r["role"]=="devval"}
    bands = BandMetadata(**manifest["bands"])
    tile_plan = manifest["plan"]
    stats = manifest["statistics"]
    folder = Path(plan["development"]["path"]).parent
    for slot,role in enumerate(middle_blocks([r for r in manifest["record"]["roles"] if r["role"]=="devval"])):
        number = role["id"]
        tile = tile_plan["tiles"][number]
        member = blocks[number]
        require(member["name"]==f"tile-{number}.npy","Unexpected observation block member")
        values = np.load(stages.pinned(root,dict(path=str(folder/member["name"]),sha256=member["sha256"])),
                         allow_pickle=False)
        scene = SimpleNamespace(cube=BlockCube(tile_plan["shape"],tile,values),plan=tile_plan,stats=stats,bands=bands)
        for view in readout.VIEWS:
            observed = stages.checked_view(scene,tile,view,source["bands"][view])
            obs,meta = observed.observation,observed.bands
            x = torch.from_numpy(np.concatenate((obs.spectral,obs.coordinates),axis=-1))
            widths = None if meta.fwhm is None else torch.tensor(meta.fwhm,dtype=torch.float32)
            request = Request(f"{slot}:{view}","xuzhou",digest([producer,member["sha256"],view,meta.fingerprint]),
                               digest([number,tile["box"]]),view,x,torch.tensor(meta.wavelengths,dtype=torch.float32),
                               widths,meta,100,len(meta.band_ids)/len(bands.band_ids),True)
            yield model,request,obs.global_ids


def honghu(root, source, producer):
    from evaluation.wp11_deployment_cost import context

    bank,previous,model = context(root,"whu-honghu",load=True)
    require(bank.pin["sha256"]==source["authority"] and previous["checkpoint"]==source["checkpoint"]
            and previous["selected_step"]==source["cursor"] and previous["model"]==source["state"],
            "Released HongHu endpoint differs")
    rows = (r for r in bank.role_rows("devval") if r["blocked"] is None)
    for slot,row in enumerate(middle_blocks(rows)):
        observed_row = bank.observation_row("devval",row["id"])
        for view in readout.VIEWS:
            actual = bank_observation(bank,observed_row,producer,view,observation_only=True)
            request = prepare(actual,producer,key=f"{slot}:{view}",device="cpu",
                              original_band_count=len(source["bands"]["full"]))
            yield model,request,actual.view.observation.global_ids


def save_packet(path, value):
    with BytesIO() as stream:
        torch.save(value,stream)
        _atomic_new(path,stream.getvalue())
    return stages.file_pin(path)


def prepare_plan(root, source, artifact, out, producer):
    """Seal runtime tensors and geometry, with no fit examples or label arrays."""
    from transfer.g3_training import state_hash

    collect = xuzhou if source["scene"]=="xuzhou" else honghu
    packets,requests = [],[]
    for model,request,pixel_ids in collect(root,source,producer):
        require(list(request.meta.band_ids)==source["bands"][request.band],"Actual retained columns changed")
        entry = vars(request).copy()
        entry["meta"] = asdict(request.meta)
        packets.append(dict(request=entry,pixel_ids=torch.from_numpy(np.array(pixel_ids,copy=True))))
        requests.append(dict(key=request.key,view=request.band,identity=request.identity,
                             shape=list(request.x.shape[:2]),band_ids=list(request.meta.band_ids)))
    require(len(packets)==2*len(readout.VIEWS) and state_hash(model)==source["state"],
            "Incomplete block coverage or weight state differs")
    weight = save_packet(out/"frozen-state.pt",model.state_dict())
    architecture = out/"frozen-state.json"
    write_json(architecture,dict(**model.export_config(),training_config={}))
    inputs = save_packet(out/"observation-only.pt",packets)
    plan = dict(schema=SCHEMA,producer=producer,source=source,head=artifact,weights=weight,
                architecture=stages.file_pin(architecture),inputs=inputs,requests=requests,
                workload="two_registered_devval_blocks_three_retained_views",labels_in_requests=False)
    path = out/"plan.json"
    write_json(path,plan)
    return stages.file_pin(path)


def open_plan(root, plan_pin, producer):
    plan = stages.json_pin(root,plan_pin)
    require(plan["schema"]==SCHEMA and plan["producer"]==producer and plan["labels_in_requests"] is False,
            "Foreign deployment observation plan")
    return plan


def load_inputs(root, plan, device):
    file = stages.pinned(root,plan["inputs"])
    seen = []
    for packet,expected in zip(torch.load(file,map_location="cpu",weights_only=True),plan["requests"],strict=True):
        require(packet.keys()=={"request","pixel_ids"},"Deployment inputs must contain only observations and geometry")
        raw = packet["request"]
        request = Request(**{**raw,"meta":BandMetadata(**raw["meta"])})
        require(expected==dict(key=request.key,view=request.band,identity=request.identity,
                             shape=list(request.x.shape[:2]),band_ids=list(request.meta.band_ids))
                and packet["pixel_ids"].shape==request.x.shape[:2] and request.x.ndim==3
                and request.x.shape[-1]==len(request.meta.band_ids)+2,
                "Foreign deployment request data or geometry")
        args = {**vars(request),"x":request.x.to(device),"waves":request.waves.to(device)}
        if request.widths is not None:
            args["widths"] = request.widths.to(device)
        seen.append((Request(**args),packet["pixel_ids"].numpy().ravel()))
    require(len(seen)==2*len(readout.VIEWS) and len({r.key for r,_ in seen})==len(seen),
            "Deployment needs two distinct blocks in each retained view")
    return seen


def deploy(root, plan_pin, out, code, device):
    import os
    import resource

    started = perf_counter()
    plan = open_plan(root,plan_pin,code["commit"])
    artifact = readout.load(root,plan["head"],plan["source"])
    for pin in (plan["weights"],plan["architecture"]):
        stages.pinned(root,pin)
    model = _load_source(plan["weights"],plan["architecture"]).requires_grad_(False).eval()
    runtime = Runtime(device=device,full=model,identity={"full":plan["source"]["state"]})
    rt = readout.SupportRuntime(runtime,artifact,"all")
    runtime.sync()
    model_and_head_load = perf_counter()-started
    started = perf_counter()
    requests = load_inputs(root,plan,runtime.device)
    runtime.sync()
    input_load = perf_counter()-started
    if runtime.device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    metrics,scores = {},{}
    for index,(item,px) in enumerate(requests):
        saved = {}
        raw = {mode:[] for mode in MODES}
        graph = {mode:[] for mode in MODES}
        for rep in range(REPEATS+1):
            order = MODES[rep%len(MODES):]+MODES[:rep%len(MODES)]
            for mode in order:
                result = rt.run(item,mode)
                native = result["native"]
                counts = result["counts"]
                require(counts["graph_builds"]==counts["full_calls"]==1 and counts["graph_reuses"]==0
                        and not runtime.cache and not model.backbone.adapter.cache_attention
                        and not model.heads[item.scene]._forward_pre_hooks,
                        "Native prediction used foreign feature or graph state")
                witness = output_hash(dict(predictions=[native]))
                key = "r"+str(index)
                if rep==0:
                    saved.setdefault("native_sha256",witness)
                    scores[f"{key}_{mode}"] = result["fused_decision"]
                    if mode=="native":
                        for name,value in zip(stages.HEADS,native,strict=True):
                            scores[f"{key}_native_{name}"] = value.argmax(1).cpu().numpy()+1
                        scores[f"{key}_ids"] = px
                else:
                    raw[mode].append(1000*result["elapsed_seconds"])
                    graph[mode].append(result["graph_ms"])
                require(witness==saved["native_sha256"] and
                        np.array_equal(scores[f"{key}_{mode}"],result["fused_decision"]),
                        "Native heads or repeat block prediction changed")
        if item.band=="full":
            require(np.array_equal(scores[f"r{index}_fixed_full"],scores[f"r{index}_same_view"]),
                    "Full-band fixed and same-view heads must agree")
        metrics[item.key] = dict(**saved,view=item.band,pixels=len(px),
            items={mode:dict(timing=timing_summary(raw[mode],repeats=REPEATS,blocks=1),graph_ms=graph[mode])
                   for mode in MODES})
    for index,(item,_) in reversed(tuple(enumerate(requests))):
        for mode in MODES:
            result = rt.run(item,mode)
            require(np.array_equal(scores[f"r{index}_{mode}"],result["fused_decision"]),
                    "Different block order changed the deployed decision")
    output = out/"decisions.npz"
    with BytesIO() as stream:
        np.savez_compressed(stream,**scores)
        _atomic_new(output,stream.getvalue())
    peak = None if runtime.device.type=="cpu" else torch.cuda.max_memory_allocated(runtime.device)
    receipt = out/"cost.json"
    write_json(receipt,dict(schema=COST,plan=plan_pin,artifact=plan["head"],producer=code["commit"],pid=os.getpid(),
                model_and_head_load_seconds=model_and_head_load,input_load_seconds=input_load,
                device=str(runtime.device),rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_bytes=peak,source=plan["source"],resident=metrics,decisions=stages.file_pin(output),
                labels_loaded=False,optimizer_updates=0,alternate_block_order_equal=True,repeats=REPEATS))
    return stages.file_pin(receipt)


def evidence(root, pin, producer, source, artifact):
    cost = stages.json_pin(root,pin)
    require(cost["schema"]==COST and cost["producer"]==producer and cost["source"]==source
            and cost["artifact"]==artifact and cost["labels_loaded"] is False,"Unbound deployment evidence")
    plan = open_plan(root,cost["plan"],producer)
    require(plan["source"]==source and plan["head"]==artifact,"Deployment plan and fit differ")
    return plan,stages.pinned(root,cost["decisions"])


def verify_overlap(deployment, view, native, features, ids, heads):
    plan,path = deployment
    point_idx = {int(key):i for i,key in enumerate(ids)}
    count = 0
    with np.load(path,allow_pickle=False) as arrays:
        for index,row in enumerate(plan["requests"]):
            if row["view"]!=view:
                continue
            key = f"r{index}"
            mapped = [(j,point_idx[int(p)]) for j,p in enumerate(arrays[f"{key}_ids"]) if int(p) in point_idx]
            if not mapped:
                continue
            pixels,reference = zip(*mapped,strict=True)
            for h,value in native.items():
                require(np.array_equal(arrays[f"{key}_native_{h}"][list(pixels)],value[list(reference)]),
                        "Deployed native three-head decisions differ from the archived released points")
            for mode,trained in (("fixed_full","full"),("same_view",view)):
                want = readout.decisions(readout.scores(features[list(reference)],heads[trained]))
                require(np.array_equal(arrays[f"{key}_{mode}"][list(pixels)],want),
                        "Deployed fitted-head predictions differ from archived features")
            count += len(mapped)
    require(count>0,"No sealed development points overlap the two deployed blocks")
    return count
