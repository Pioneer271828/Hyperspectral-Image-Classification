"""Independently replay native observations and probabilities on the original runtime."""

from dataclasses import replace
import json
import time
from contextlib import ExitStack
from unittest.mock import patch

if __package__:
    from . import runtime as rt
else:
    import runtime as rt


def reference_tile(run, model, block, rr):
    import numpy as np
    import torch
    from loadData.spatial_protocol import forward_partition
    from loadData.spatial_tiles import observe_tile

    tile_id = block["tile_id"]
    record = run.record["roles"][tile_id]
    rt.require(record["role"]==block["role"] and record["role"] in ("devtrain","devval"),
               "Reference cannot read a protected context")
    rt.require(run.bank is None and model.backbone.graph_router.step==0 and not model.training
               and model.backbone.coverage_modulation is None,"Unexpected non-native reference settings")
    tile = run.scene.plan["tiles"][tile_id]
    obs = observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,tile)
    x = torch.from_numpy(obs.spectral.copy())
    ids = obs.global_ids.ravel().tolist()
    positions = [ids.index(i) for i in block["target_ids"]]
    classes = run.scene.classes
    captured = {}
    rt.require(positions==sorted(set(positions)), "Wrong target ordering in reference")
    backbone = model.backbone
    graph = backbone.learnable_graph
    router = backbone.graph_router
    build = graph.build_graph
    route = router.forward
    first,second = backbone.graph_stage1,backbone.graph_stage2
    wave = np.asarray(run.scene.bands.wavelengths,dtype=np.float32)
    def build_reference(*args,**kwargs):
        state = build(*args,**kwargs)
        captured["q_hash"] = rr.tensor_hash(state.Q)
        captured["a0_hash"] = rr.tensor_hash(state.A0)
        return state
    def route_reference(*args,**kwargs):
        a,info = route(*args,**kwargs)
        indices = a.coalesce().indices()
        captured.update(final_a_hash=rr.tensor_hash(a),nodes=a.shape[0],
                        edges=(indices[0]*a.shape[0]+indices[1]).tolist())
        return a,info
    def trace_first(regions,a,dataset_name=None):
        captured["stage1"] = rr.tensor_hash(a)
        return first(regions,a,dataset_name)
    def trace_second(regions,a,dataset_name=None):
        captured["stage2"] = rr.tensor_hash(a)
        return second(regions,a,dataset_name)
    def evaluate(value):
        captured.clear()
        results = forward_partition(model,run.scene.dataset,replace(obs,spectral=value.numpy()),
                                    wave,50 if run.scene.dataset=="paviaU" else 100)
        stage1,stage2 = captured.pop("stage1"),captured.pop("stage2")
        rt.require(stage1==captured["a0_hash"] and stage2==captured["final_a_hash"],
                   "The native stages did not consume their own graphs")
        return {"scores":[[float(head[positions,i].mean()) for i in range(classes)] for head in results[:3]],
                "probability_hashes":[rr.tensor_hash(head) for head in results[:3]],
                "support":captured.copy(),"stage_graphs_distinct":stage1!=stage2,
                "spectrum_hash":rr.tensor_hash(value)}
    result = {}
    with ExitStack() as stack, torch.no_grad():
        for owner,name,replacement in ((graph,"build_graph",build_reference),(router,"forward",route_reference),
                (backbone,"graph_stage1",trace_first),(backbone,"graph_stage2",trace_second)):
            stack.enter_context(patch.object(owner,name,replacement))
        baseline = evaluate(x)
        for epsilon in sorted({r["epsilon"] for r in block["requests"]}):
            rows = []
            for global_id,band in block["sources"]:
                y,col = divmod(ids.index(global_id),x.shape[1])
                signs = []
                for direction in (1,-1):
                    changed = x.clone()
                    changed[y,col,band].add_(direction*epsilon)
                    row = evaluate(changed)
                    row["delta"] = float(changed[y,col,band]-x[y,col,band])
                    signs.append(row)
                rows.append(signs)
            result[epsilon] = rows
    return baseline,result


def compare_block(block, production, baseline, signals):
    rt.require(production["tile_id"]==block["tile_id"] and production["role"]==block["role"]
        and [r["request"] for r in production["rows"]]==block["requests"],"Incomplete or reordered requests")
    count = {"requests":0,"baseline_probability_hashes":0,"baseline_scores":0,
             "directional_scores":0,"directional_input_deltas":0,"directional_supports":0,
             "gradient_entries":0,"finite_difference_values":0}
    for item in production["rows"]:
        result = item["result"]
        req = item["request"]
        i = req["class_id"]-1
        eps = req["epsilon"]
        native = [head[i] for head in baseline["scores"]]
        rt.require(result["baseline_probability_hashes"]==baseline["probability_hashes"]
            and result["baseline_scores"]==native and result["baseline_support"]==baseline["support"]
            and item["plan"]["base"]["normalized_spectrum_hash"]==baseline["spectrum_hash"]
            and item["plan"]["base"]["target_ids"]==block["target_ids"]
            and item["plan"]["sources"]==block["sources"] and item["plan"]["base"]["class_id"]==req["class_id"]
            and item["plan"]["epsilon"]==eps,"Independent baseline or request identity differs")
        rt.require([r["source"] for r in result["rows"]]==block["sources"],"Incomplete source population")
        count["requests"]+=1
        count["baseline_probability_hashes"]+=3
        count["baseline_scores"]+=3
        for row,signs in zip(result["rows"],signals[eps],strict=True):
            values = [[head[i] for head in x["scores"]] for x in signs]
            deltas = [x["delta"] for x in signs]
            rt.require(deltas[0]>0>deltas[1] and row["input_deltas"]==deltas and row["probability_scores"]==values
                and row["supports"]==[x["support"] for x in signs]
                and row["spectrum_hashes"]==[x["spectrum_hash"] for x in signs],
                "Strict independent positive/negative probability, actual FP32 delta or Q/graph differs")
            derived = [(plus-minus)/(deltas[0]-deltas[1]) for plus,minus in zip(*values,strict=True)]
            rt.require(row["finite_difference"]==derived and len(row["gradient"])==3
                and row["absolute_difference"]==[abs(a-b) for a,b in zip(row["gradient"],derived,strict=True)]
                and row["score_deltas"]==[[v-base for v,base in zip(value,native,strict=True)] for value in values],
                "Finite-difference record differs from independent values")
            for field,name in (("topology_changed","edges"),("q_changed","q_hash"),("graph_values_changed","final_a_hash")):
                rt.require(row[field]==[s["support"][name]!=baseline["support"][name] for s in signs],
                           "Derived native graph transition differs")
            count["directional_scores"]+=6
            count["directional_input_deltas"]+=2
            count["directional_supports"]+=2
            count["gradient_entries"]+=3
            count["finite_difference_values"]+=3
    return count


def audit(scope_pin, producer):
    import torch
    from collections import Counter
    rr,admission,code = rt.bootstrap(producer)
    scope,_ = rt.load_scope(scope_pin,"audit.py",rr,admission,code)
    from model.VCASAHGCN import MultiDatasetVCASAHGCN
    from evaluation.graph_edge_intervention import _selected
    root = rt.checked(scope_pin).parent
    prod_pin = rt.pin(root/"production"/"receipt.json")
    production = rt.read(prod_pin)
    rt.require(production["scope"]==scope_pin and production["status"]=="passed"
        and production["requests"]==scope["requests"] and len(production["tiles"])==len(scope["rows"])
        and production["diagnostic_forwards"]==scope["production_forward_budget"], "Wrong production receipt")
    out = root/"reference"
    out.mkdir(exist_ok=False)
    forwards = 0
    def count(module,args):
        nonlocal forwards
        if type(module) is MultiDatasetVCASAHGCN:
            forwards+=1
    def forbidden(*args,**kwargs):
        raise AssertionError("The native reference performs neither optimization nor backward calls")
    handle = torch.nn.modules.module.register_module_forward_pre_hook(count)
    started = time.monotonic()
    verified = []
    counts = Counter()
    distinct = 0
    try:
        with patch.object(torch.optim.Adam,"step",forbidden),patch.object(torch.autograd,"grad",forbidden):
            summary,run,_,_ = admission.restore(scope["candidate"],code)
            rt.require(summary["selected_state"]==scope["selected_state_hash"] and forwards==10,
                       "Wrong reference authentication calls or checkpoint")
            before = rt.state_snapshot(run,rr)
            model = _selected(run)
            from transfer.graph_route_context import route_state_hash
            state = route_state_hash(model)
            rt.require(state==scope["selected_state_hash"] and before["batch_norm_modules"]==0,
                       "Reference selected weights or normalization differ")
            for block,tile_pin in zip(scope["rows"],production["tiles"],strict=True):
                previous = forwards
                actual = rt.read(tile_pin)
                rt.require(actual["scope"]==scope_pin,"Mismatched tile scope")
                baseline,signals = reference_tile(run,model,block,rr)
                count_row = compare_block(block,actual,baseline,signals)
                n_distinct = int(baseline["stage_graphs_distinct"])+sum(s["stage_graphs_distinct"]
                    for rows in signals.values() for pair in rows for s in pair)
                distinct+=n_distinct
                rt.require(forwards-previous==25,"Reference changed registered forward reuse")
                counts.update(count_row)
                verified.append(rt.write(out/f'tile-{block["tile_id"]:03}.json',{
                    "tile_id":block["tile_id"],"role":block["role"],"production":tile_pin,
                    "reference_forwards":25,"distinct_native_stage_graphs":n_distinct,"checks":count_row}))
                if len(verified)==1 or len(verified)%8==0 or len(verified)==len(scope["rows"]):
                    print(f"reference tiles {len(verified)}/{len(scope['rows'])} elapsed={time.monotonic()-started:.3f}",
                          flush=True)
            rt.require(forwards==scope["reference_forwards"]+10 and counts["requests"]==scope["requests"]
                and counts["directional_scores"]==scope["requests"]*36 and counts["gradient_entries"]==scope["requests"]*18
                and route_state_hash(model)==state and rt.state_snapshot(run,rr)==before,
                "Independent population, state or budget differs")
            admission.check_pins(scope["input_pins"])
            rr.require_frozen_code(code)
            rt.verify_deployment(scope["deployment"],"audit.py")
            return rt.write(out/"receipt.json",{"status":"passed","scope":scope_pin,"production":prod_pin,
                "tiles":verified,"checks":dict(counts),"whole_model_forwards":forwards,
                "restore_validation_forwards":10,"reference_forwards":forwards-10,
                "reuse":"same native forward serves all nine output indices; baseline also serves both epsilons",
                "distinct_native_stage_graphs":distinct,"optimizer_updates":0,"gradient_calls":0,"bn_updates":0,
                "state_before_equals_after":True,"raw_gt_accessed":False,"target_labels_used":False,"query":0,
                "wall_seconds":time.monotonic()-started})
    except BaseException as error:
        rt.write(out/"failure.json",{"scope":scope_pin,"error":repr(error),"completed_tiles":len(verified),
                                     "forwards":forwards,"optimizer_updates":0,"query":0})
        raise
    finally:
        handle.remove()


if __name__ == "__main__":
    parser = rt.common_parser(__doc__)
    args = parser.parse_args()
    if not args.scope:
        parser.error("The existing frozen scope is required")
    print(json.dumps(audit(rt.cli_pin(args.scope),args.producer)),flush=True)
