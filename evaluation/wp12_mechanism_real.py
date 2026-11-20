"""Offline single-tensor exchanges on published, frozen development points."""

from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from evaluation import frozen_training_stages as frozen
from evaluation.matched_edge_controls import freeze as match_edges, verify as verify_edges
from evaluation.graph_edge_intervention import delete_support
from evaluation.wp12_release_results import BAND, BAND_PINS
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import tensor_hash
from scripts import run_frozen_band_region as archived
from scripts.run_wp12_retained_evidence import RELEASE, PREVIOUS, pin, compare_probabilities
from transfer.band_scale import evidence
from transfer.band_scale_real import FOCUS
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward, state_hash
from transfer.native_coverage_replay import frozen_model
from transfer.native_scale_observation import forward
from transfer.registered_development import load_development
from transfer.spatial_coverage_protocol import bank_view

CASES = ("initial","prefix","selected","last","whu-honghu","paviaU-64","paviaU-512","paviaU-2048")
HEADS, VIEWS = frozen.HEADS, frozen.VIEWS
SITES = ("vca","pre_pool","pooled","cnn","graph","fused")
SCHEMA = "wp12_frozen_one_site_development_v1"
RIVALS = {"xuzhou":{5:1,6:1},"whu-honghu":{15:1,21:18},"paviaU":{2:8,6:4}}
EDGE_CASES = ("selected","whu-honghu","paviaU-64")
LOCKED_PINS = {
    "initial":"8d5bf053ce8ffd261862a3106208042459148994194f778f67721050c73b2fb8",
    "last":"089a6c0d6f1d01596765e486aa22680dacd50afd34fa12305f1c214c43e2600f",
    "paviaU-64":"b4b7e67f28d672f8690b5217ab52dc2e5a7df2924bb83b438f823f8b697d21f2",
    "prefix":"198ff26b6ce71771a3f2af68e52dd3fb4db10c150e71b883333d8f016be0b8d0",
    "selected":"9ce4a018b6180cef2c07624bc6f28d46588de5a56d11dfff480d6cdbc22b601c",
    "whu-honghu":"e12db74156ac52a2a7dab3e3db8b3101153272823af9a3da238aeddea16eb60c",
}
REQUIRE = frozen.require


@dataclass(frozen=True)
class ReleasedBlock:
    truth: list
    cells: object
    callbacks: dict
    historical: dict
    tile: int
    parent: int
    views: dict
    point_order: str
    native_receipts: dict | None = None


@dataclass(frozen=True)
class ReleasedEndpoint:
    scene: str
    classes: int
    step: int
    checkpoint: str
    authority: str
    model: object
    blocks: list
    release: dict


def switch(base, donor):
    REQUIRE(base.layout==donor.layout==torch.strided and base.shape==donor.shape
            and base.dtype==donor.dtype and base.device==donor.device and torch.isfinite(donor).all(),
            "Incompatible frozen intervention dimensions, dtype or device")
    return donor


@torch.inference_mode()
def probe(model, invoke, *, site=None, donor=None, graph_action=None, collect=False, trace=None):
    REQUIRE(not model.training and ((site is None)==(donor is None)), "Single-site interventions require a frozen model")
    owner = model.backbone
    result, states, hits = {}, [], []
    stage, fuse, build = owner.forward_stages, owner.fuse_features, owner.learnable_graph.build_graph

    def at(key, value):
        if collect:
            REQUIRE(key not in result, "Repeated intermediate tensor")
            result[key] = value.detach()
        if site==key:
            hits.append(key)
            return switch(value,donor)
        return value

    def vca(_module,_args,value):
        return (at("vca",value[0]),*value[1:]) if isinstance(value,tuple) else at("vca",value)

    def pre_pool(_module,_args,value):
        return at("pre_pool",value)

    def stages(cnn_input, regions, q, adjacency, *args, **kwargs):
        if trace is not None:
            trace.update(cnn_input=cnn_input, pooled=regions, used_q=q, used_adjacency=adjacency)
        cnn,graph,info=stage(cnn_input,at("pooled",regions),q,adjacency,*args,**kwargs)
        return at("cnn",cnn),at("graph",graph),info

    def fusion(cnn,graph):
        return at("fused",fuse(cnn,graph))

    def builder(*args,**kwargs):
        state=build(*args,**kwargs)
        states.append(state)
        if trace is not None:
            trace["builder_input"] = args[0]
        if graph_action is not None:
            REQUIRE(tensor_hash(state.Q)==tensor_hash(graph_action.Q), "Matched edge arm changed assignment")
            return graph_action
        return state

    with ExitStack() as hooks:
        if collect or site=="vca":
            hooks.callback(owner.adapter.register_forward_hook(vca).remove)
        if collect or site=="pre_pool":
            hooks.callback(owner.CNN_denoise[-1].register_forward_hook(pre_pool).remove)
        if collect or trace is not None or site in ("pooled","cnn","graph"):
            hooks.enter_context(patch.object(owner,"forward_stages",new=stages))
        if collect or site=="fused":
            hooks.enter_context(patch.object(owner,"fuse_features",new=fusion))
        prediction=invoke(builder if graph_action is not None or collect or trace is not None else None)
    REQUIRE((not collect or set(result)==set(SITES)) and len(hits)==int(site is not None),
            "Native model did not traverse the declared frozen site")
    return prediction[:3],result,states[0] if states else None


def shuffle_donor(site,value,seed):
    rng=np.random.default_rng(seed)
    if site=="pre_pool":
        perm=rng.permutation(value.shape[2]*value.shape[3])
        return value.flatten(2).index_select(2,torch.as_tensor(perm,device=value.device)).reshape_as(value)
    flat=value.reshape(-1,value.shape[-1])
    perm=torch.as_tensor(rng.permutation(len(flat)),device=value.device)
    return flat.index_select(0,perm).reshape_as(value)


def invoke_bank(model,actual,producer,operation):
    return forward(model,actual,producer,"devval",graph_override=operation)[0]


def invoke_xuzhou(model,scene,tile,view,operation):
    from contextlib import nullcontext

    scope = (nullcontext() if operation is None else
             patch.object(model.backbone.learnable_graph,"build_graph",new=operation))
    with scope:
        return _forward(model,scene,tile,observation_view=view)


def load(root,case,producer,device):
    if case in CASES[:4]:
        plan=frozen.stage_plan(root,pin(root,RELEASE))
        release=frozen.json_pin(root,plan["release"])
        spec=plan["stages"][case]
        model=frozen.load_stage(root,release,spec,case)
        scene,record,_=load_development(plan["development"])
        target=sparse_labels(record,scene.plan,"validation_support")
        authority=plan["development"]["sha256"]
        record_train, record_val = record["train_support"], record["validation_support"]
        support_map = sparse_labels(record, scene.plan, "train_support")
        overlapping_parents = sum(bool(support_map[tile_slice(tile)].any() and target[tile_slice(tile)].any())
                                  for tile in scene.plan["tiles"])
        config = frozen.json_pin(root, release["full_metadata"])["training_config"]
        release_info = dict(support_seed=record_train["seed"], validation_seed=record_val["seed"],
            support_sha256=plan["identities"]["support"], validation_sha256=plan["identities"]["validation"],
            space_sha256=plan["identities"]["space"], statistics_sha256=plan["statistics"],
            support_per_class=np.bincount(record_train["labels"], minlength=scene.classes+1)[1:].tolist(),
            evaluation_per_class=np.bincount(record_val["labels"], minlength=scene.classes+1)[1:].tolist(),
            support_evaluation_overlap=len(set(record_train["indices"]) & set(record_val["indices"])),
            support_evaluation_parent_overlap=overlapping_parents, model_seed=config.get("model_seed"),
            original_sampler_seed=config.get("sampler_seed"), space_seed=scene.plan.get("seed"))
        result=[]
        for tile,binding in zip(scene.plan["tiles"],record["roles"],strict=True):
            if binding["role"]!="devval": continue
            labels=target[tile_slice(tile)].ravel()
            cells=np.flatnonzero(labels)
            if not len(cells): continue
            views={band:frozen.checked_view(scene,tile,band,plan["bands"][band]) for band in VIEWS}
            callbacks={band:(lambda operation,view=view,tile=tile:invoke_xuzhou(model,scene,tile,view,operation))
                       for band,view in views.items()}
            points = views["full"].observation.global_ids.reshape(-1)[cells].tolist()
            result.append(ReleasedBlock(labels[cells].tolist(), cells, callbacks, {}, tile["id"], tile["id"],
                                        views, digest(points)))
        name,classes,step,state=scene.dataset,scene.classes,spec["step"],spec["state"]["sha256"]
    else:
        archive=frozen.json_pin(root,pin(root,(f"{BAND}/{case}.json",BAND_PINS[case])))
        parents, documents=archived.source_documents(root)
        name=archive["scene"]
        bank,intake=archived._freeze_bank(root,name,{"sources":parents})
        model,seed,identity,step,checkpoint=frozen_model(root,case,bank,intake,documents)
        REQUIRE((identity,seed,step,checkpoint,bank.pin["sha256"])==(archive["model"],archive["model_seed"],
                archive["selected_step"],archive["checkpoint"],archive["preflight_population"])
                and archive["role"]=="devval" and archive["real_updates"]==0
                and archive["raw_gt_loaded"] is archive["query_accessed"] is False,"Archived role or endpoint differs")
        classes=model.heads[name].out_features
        authority,state=bank.pin["sha256"],checkpoint
        support = [row for row in bank.role_rows("devtrain") if row["samples"]]
        evaluation = [row for row in bank.role_rows("devval") if row["samples"]]
        support_points = [point for row in support for point in row["samples"]]
        evaluation_points = [point for row in evaluation for point in row["samples"]]
        release_info = dict(support_seed=bank.release.custody.get("release_seed"),
            validation_seed=bank.release.custody.get("validation_seed"),
            support_sha256=digest(support_points), validation_sha256=digest(evaluation_points),
            space_sha256=bank.release.plan_hash, statistics_sha256=bank.release.statistics["statistics_hash"],
            support_per_class=np.bincount([v for _, v in support_points], minlength=classes+1)[1:].tolist(),
            evaluation_per_class=np.bincount([v for _, v in evaluation_points], minlength=classes+1)[1:].tolist(),
            support_evaluation_overlap=len({i for i, _ in support_points} & {i for i, _ in evaluation_points}),
            support_evaluation_parent_overlap=len({r["parent_tile"] for r in support}
                                                 & {r["parent_tile"] for r in evaluation}),
            model_seed=seed)
        result=[]
        focus_ordinal=0
        for row in bank.role_rows("devval"):
            observations={band:bank_view(bank,row,producer,band) for band in VIEWS}
            selected=observations["full"][2]
            if not len(selected.points): continue
            truth=[y for _,y in selected.points]
            REQUIRE(all(item[2].points==selected.points for item in observations.values()),"Released labels changed by view")
            callbacks={band:(lambda operation,actual=item[0]:invoke_bank(model,actual,producer,operation))
                       for band,item in observations.items()}
            focus=[i for i,y in enumerate(truth) if y in FOCUS[name]]
            historical={}
            if focus:
                for band in VIEWS[:2]:
                    prior,=[x for x in archive["rows"] if x["tile"]==focus_ordinal
                            and x["event"]==dict(band=band,factor=1,policy="native")]
                    now=evidence(observations[band][0],observations[band][2])
                    REQUIRE(all(v==now[k] for k,v in prior["evidence"].items() if k!="view")
                            and prior["labels"]==[truth[i] for i in focus],"Historical retained source differs")
                    historical[band]=(focus,prior["probabilities"])
                focus_ordinal+=1
            result.append(ReleasedBlock(truth, selected.cells, callbacks, historical, row["id"], row["parent_tile"],
                                        {band: item[0].view for band, item in observations.items()},
                                        digest([i for i, _ in selected.points]),
                                        {band: item[0].receipt for band, item in observations.items()}))
    REQUIRE(result,"No released development point roster")
    model.to(device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention=False
    model.backbone.adapter.clear_attention_cache()
    return ReleasedEndpoint(name, classes, step, state, authority, model, result, release_info)


def output_rows(probabilities,truth,rivals):
    rows={}
    for name,p in probabilities.items():
        ordered=np.argsort(-p,axis=1,kind="stable")
        rows[name]=[]
        for y,row,order in zip(truth,p,ordered,strict=True):
            highest=next(int(i)+1 for i in order if i+1!=y)
            rival=rivals.get(y)
            rows[name].append(dict(predicted=int(order[0])+1,highest_other=highest,
                fixed_rival=rival,margin=None if rival is None else float(row[y-1]-row[rival-1]),
                highest_margin=float(row[y-1]-row[highest-1])))
    return rows


def get_points(out,cells):
    return {head:val.detach().cpu().numpy()[list(cells)] for head,val in zip(HEADS,out,strict=True)}


def graph_probe(model,invoke,base,truth,cells,rivals):
    from dataclasses import replace

    state=base[2]
    require_shape=state.region_positions.shape[0] == state.A0.shape[0]
    REQUIRE(require_shape,"Native graph region positions do not align")
    h,w=state.image_shape
    xy=state.region_positions.detach().cpu()*torch.tensor([max(h-1,1),max(w-1,1)])
    graph=state.A0.detach().cpu().coalesce()
    plan=match_edges(graph,xy,per_row=1,distance_bins=[3,9,18,36],seeds=[7719],time_limit=2.)
    verify_edges(plan,graph,xy,expected_hash=plan["plan_hash"])
    info={arm:row["status"] for arm,row in plan["controls"].items()}
    stages={}
    if all(value=="optimal" for value in info.values()):
        groups={"important":plan["baseline"]["removed"]} | {k:v["removed"] for k,v in plan["controls"].items()}
        for arm,edges in groups.items():
            changed=replace(state,A0=delete_support(graph,edges).to(state.A0.device))
            out,_,_=probe(model,invoke,graph_action=changed)
            stages[arm]=output_rows(get_points(out,cells),truth,rivals)
    return dict(shape=list(state.Q.shape),edge_nnz=graph._nnz(),removed=len(plan["baseline"]["removed"]),
                match_status=info,matched_count_bins=plan["baseline"]["counts"],mode="exact_degree_and_distance_bins",
                plan_sha256=plan["plan_hash"],scores=stages)


def locked_history(root, case, checkpoint):
    if case not in LOCKED_PINS:
        return None
    old=frozen.json_pin(root,pin(root,(f"wp05-wp06-production-20261119-qa/locked-v2/{case}.json",LOCKED_PINS[case])))
    REQUIRE(old["result"]["checkpoint"]==checkpoint and old["optimizer_updates"]==0,"Locked view history binding differs")
    return old


def verify_baselines(block, originals, old, offset, exact):
    if old is not None:
        for band, values in originals.items():
            previous = old["result"]["views"][band]
            sl = slice(offset, offset+len(block.truth))
            REQUIRE(previous["truth"][sl] == block.truth, "Published point identities differ")
            for head, value in values.items():
                compare_probabilities([row[head] for row in previous["native_probabilities"][sl]], value, exact)
    for band, (indices, previous) in block.historical.items():
        for head in HEADS:
            compare_probabilities(previous[head], originals[band][head][indices], exact)


def run_case(root,case,producer,device):
    start=perf_counter()
    endpoint=load(root,case,producer,device)
    scene, classes, model = endpoint.scene, endpoint.classes, endpoint.model
    before=state_hash(model)
    old=locked_history(root,case,endpoint.checkpoint)
    exact=device.type=="cpu"
    groups=[]
    offset=0
    first_focus=set()
    for ordinal,block in enumerate(endpoint.blocks,1):
        truth,cells,callbacks=block.truth,block.cells,block.callbacks
        base={band:probe(model,invoke,collect=True) for band,invoke in callbacks.items()}
        orig={band:get_points(result[0],cells) for band,result in base.items()}
        size=len(truth)
        verify_baselines(block,orig,old,offset,exact)
        variants={}
        for target, donor in (("full","full"),("central_gap","full"),("full","central_gap"),
                              ("matched_dispersed","full"),("full","matched_dispersed")):
            for site in SITES:
                key=f"{target}<-{donor}/{site}"
                target_graph,donor_graph=base[target][2],base[donor][2]
                # Region tokens lack a common coordinate system when Q ownership changes.
                if site=="pooled" and tensor_hash(target_graph.Q)!=tensor_hash(donor_graph.Q):
                    variants[key]=dict(status="incompatible_region_assignment")
                    continue
                values=base[donor][1][site]
                for control in (("aligned","random") if target!="full" else ("aligned",)):
                    supplied=values if control=="aligned" else shuffle_donor(site,values,[7719,ordinal])
                    out,_,_=probe(model,callbacks[target],site=site,donor=supplied)
                    read=get_points(out,cells)
                    if target==donor:
                        for head in HEADS:
                            compare_probabilities(orig[target][head],read[head],True)
                    if site=="fused":
                        for head in ("cnn","graph"):
                            compare_probabilities(orig[target][head],read[head],True)
                    variants[key+"/"+control]=dict(status="paired",scores=output_rows(read,truth,RIVALS[scene]))
        eligible=sorted(set(truth)&set(FOCUS[scene])-first_focus)
        edge={}
        if case in EDGE_CASES and eligible:
            first_focus.update(eligible)
            for band in VIEWS:
                edge[band]=graph_probe(model,callbacks[band],base[band],truth,cells,RIVALS[scene])
        groups.append(dict(block=ordinal,truth=truth,offset=offset,base={v:output_rows(p,truth,RIVALS[scene])
            for v,p in orig.items()},variants=variants,edge=edge))
        offset+=size
    REQUIRE(state_hash(model)==before and (old is None or offset==len(old["result"]["views"]["full"]["truth"])),
            "Frozen checkpoint or published point denominator changed")
    return dict(schema=SCHEMA,scope="released_development_offline_only",producer=producer,scene=scene,
                case=case,step=endpoint.step,checkpoint=endpoint.checkpoint,authority=endpoint.authority,state=before,roles=["devval"],
                classes=classes,focus=FOCUS[scene],locked_rivals=RIVALS[scene],groups=groups,
                seconds=perf_counter()-start,raw_gt_loaded=False,query_accessed=False,optimizer_updates=0,
                independent_accuracy=None,full_donor_available_to_training_or_deployment=False)
