"""Register and measure the full Xuzhou selected-model request population."""

from pathlib import Path
import json
import time
from unittest.mock import patch

if __package__:
    from . import runtime as rt
else:
    import runtime as rt


def produce(scope_pin, producer):
    import torch
    rr,admission,code = rt.bootstrap(producer)
    scope,paths = rt.load_scope(scope_pin,"produce.py",rr,admission,code)
    api = rt.engine(paths)
    from model.VCASAHGCN import MultiDatasetVCASAHGCN
    root = rt.checked(scope_pin).parent/"production"
    root.mkdir(exist_ok=False)
    counters = {"forward":0,"backward":0}
    def on_forward(module,args):
        if type(module) is MultiDatasetVCASAHGCN:
            counters["forward"]+=1
    native_grad = torch.autograd.grad
    def on_backward(*args,**kwargs):
        counters["backward"]+=1
        return native_grad(*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError("Optimizer calls are outside this diagnostic")
    handle = torch.nn.modules.module.register_module_forward_pre_hook(on_forward)
    started = time.monotonic()
    try:
        with patch.object(torch.optim.Adam,"step",forbidden),patch.object(torch.autograd,"grad",on_backward):
            summary,run,_,_ = admission.restore(scope["candidate"],code)
            rt.require(summary["selected_state"]==scope["selected_state_hash"] and
                       counters=={"forward":10,"backward":0},"Unexpected authentication forwards")
            before = rt.state_snapshot(run,rr)
            rt.require(before["batch_norm_modules"]==0,"Expected the accepted GroupNorm source")
            files = []
            for block in scope["rows"]:
                t0 = time.monotonic()
                old = counters.copy()
                rows = []
                for request in block["requests"]:
                    args = dict(tile_id=block["tile_id"],target_ids=block["target_ids"],
                                sources=block["sources"],spectral_cap=scope["spectral_cap"],**request)
                    binding_start = counters.copy()
                    plan = api.SelectedSignalPlan.freeze(run,**args)
                    rt.require(counters==binding_start,"Identity binding unexpectedly called the model")
                    plan = api.SelectedSignalPlan(plan.to_dict(),expected_hash=plan.plan_hash)
                    result = api.review_selected_signals(run,plan,expected_plan_hash=plan.plan_hash)
                    rt.require(result["selected_state_hash"]==scope["selected_state_hash"]
                        and result["context"]["tile_id"]==block["tile_id"]
                        and result["context"]["role"]==block["role"]
                        and result["whole_model_forwards"]==14 and not result["target_labels_used"]
                        and result["optimizer_updates"]==0 and not result["query_labels_accessed"]
                        and counters["forward"]-binding_start["forward"]==14
                        and counters["backward"]-binding_start["backward"]==3,
                        "The existing engine's result or measured call count differs")
                    rows.append({"request":request,"plan":plan.to_dict(),"result":result})
                seconds = time.monotonic()-t0
                files.append(rt.write(root/f'tile-{block["tile_id"]:03}.json',{
                    "scope":scope_pin,"tile_id":block["tile_id"],"role":block["role"],"rows":rows,
                    "forwards":counters["forward"]-old["forward"],"backwards":counters["backward"]-old["backward"],
                    "seconds":seconds}))
                if len(files)==1 or len(files)%8==0 or len(files)==len(scope["rows"]):
                    print(f"production tiles {len(files)}/{len(scope['rows'])} tile_seconds={seconds:.3f} "
                          f"total_seconds={time.monotonic()-started:.3f}",flush=True)
            rt.require(rt.state_snapshot(run,rr)==before,"Model, Adam, existing gradients, modes or RNG changed")
            run._check()
            admission.check_pins(scope["input_pins"])
            rr.require_frozen_code(code)
            rt.verify_deployment(scope["deployment"],"produce.py")
            rt.require(counters=={"forward":scope["production_forward_budget"]+10,
                                  "backward":scope["production_gradient_calls"]},"Wrong complete producer call budget")
            return rt.write(root/"receipt.json",{"status":"passed","scope":scope_pin,"tiles":files,
                "requests":scope["requests"],"whole_model_forwards":counters["forward"],
                "restore_validation_forwards":10,"diagnostic_forwards":counters["forward"]-10,
                "gradient_calls":counters["backward"],"state_before_equals_after":True,
                "optimizer_updates":0,"bn_updates":0,"raw_gt_accessed":False,"target_labels_used":False,"query":0,
                "wall_seconds":time.monotonic()-started})
    except BaseException as error:
        rt.write(root/"failure.json",{"scope":scope_pin,"error":repr(error),"completed_tiles":len(list(root.glob("tile-*.json"))),
            "calls":counters,"optimizer_updates":0,"query":0})
        raise
    finally:
        handle.remove()


def main():
    parser = rt.common_parser(__doc__)
    parser.add_argument("--register",type=Path)
    parser.add_argument("--donor",type=rt.Path)
    parser.add_argument("--deployment",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args()
    if args.register is not None:
        if args.scope or not (args.donor and args.deployment):
            parser.error("Registration requires donor and deployment")
        result = rt.freeze_scope(args.register,args.donor,rt.cli_pin(args.deployment),"produce.py",
                                 *rt.bootstrap(args.producer))
    else:
        if not args.scope or args.donor or args.deployment:
            parser.error("Measurement requires the previously pinned scope")
        result = produce(rt.cli_pin(args.scope),args.producer)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
