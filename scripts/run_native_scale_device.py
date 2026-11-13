"""Register and verify strict native-scale CPU/CUDA engineering campaigns in LXC."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_scale_device_campaign as campaign, native_scale_device_cost as cost
from transfer import native_scale_device_evidence as evidence
from transfer import p4_runtime_device as p4
from transfer.container_runtime import training_container_kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--protocol",choices=campaign.DEVICES,default=p4.CPU)
    parser.add_argument("--gpu")
    modes = parser.add_subparsers(dest="mode",required=True)
    for name in ("geometry","qualify","register","stage","audit","cost","compare","supplement","consume"):
        sub = modes.add_parser(name)
        if name!="stage":
            sub.add_argument("--out",required=True,type=Path)
        if name in ("qualify","cost"):
            sub.add_argument("--geometry",nargs=2,required=True)
        if name=="register":
            sub.add_argument("--qualification",nargs=2,required=True)
        if name in ("stage","audit"):
            sub.add_argument("--matrix",nargs=2,required=True)
        if name=="stage":
            sub.add_argument("--case",choices=campaign.cases(),required=True)
            sub.add_argument("--phase",choices=campaign.PHASES,required=True)
        if name in ("compare","supplement","consume"):
            sub.add_argument("--cpu",nargs=2,required=True)
            sub.add_argument("--cuda",nargs=2,required=name=="compare")
        if name in ("supplement","consume"):
            sub.add_argument("--cpu-audit",nargs=2,required=True)
            sub.add_argument("--cuda-audit",nargs=2)
            sub.add_argument("--comparison",nargs=2)
        if name=="consume":
            sub.add_argument("--producer",required=True)
            sub.add_argument("--partial-cuda",action="store_true")
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized LXC with assertions required")
    if (args.protocol==p4.CPU)!=(args.gpu is None):
        parser.error("CUDA requires one physical UUID and CPU has none")
    code = code_snapshot()
    producer = args.expected_commit
    if code["commit"]!=producer:
        parser.error("Exact committed producer required")
    require_frozen_code(code)
    root = external_directory(args.runs.resolve(strict=True))
    out = external_directory(args.out) if hasattr(args,"out") else None
    if root.name!="runs" or (out is not None and (out==root or not out.is_relative_to(root) or out.exists())):
        parser.error("A new bounded output under the authorized runs root is required")
    pin = lambda pair: None if pair is None else dict(zip(("path","sha256"),pair,strict=True))
    target = (producer,args.protocol,args.gpu)
    if args.mode=="geometry":
        result = campaign.geometry(root,out,producer)
    elif args.mode=="qualify":
        result = campaign.qualify(root,pin(args.geometry),out,*target)
    elif args.mode=="register":
        result = campaign.register(root,pin(args.qualification),out,*target)
    elif args.mode=="stage":
        result = campaign.stage(root,pin(args.matrix),args.case,args.phase,*target)
    elif args.mode=="audit":
        result = campaign.audit(root,pin(args.matrix),out,*target)
    elif args.mode=="cost":
        result = cost.cost(root,pin(args.geometry),out,*target)
    elif args.mode=="compare":
        result = cost.compare_cost(root,pin(args.cpu),pin(args.cuda),out,producer)
    elif args.mode=="consume":
        if args.protocol!=p4.CPU or args.comparison is not None or (args.partial_cuda and not args.cuda_audit):
            parser.error("Consume reads fixed artifacts on CPU and verifies any paired comparison directly")
        result = evidence.publish(root,out,args.producer,producer,pin(args.cpu_audit),pin(args.cpu),
                                  gpu_audit_pin=pin(args.cuda_audit),gpu_cost_pin=pin(args.cuda),
                                  partial=args.partial_cuda)
    elif args.mode=="supplement":
        if args.comparison is not None:
            parser.error("Supplement verifies the CPU/CUDA comparison directly from the paired cost inputs")
        result = evidence.publish(root,out,producer,producer,pin(args.cpu_audit),pin(args.cpu),
                                  gpu_audit_pin=pin(args.cuda_audit),gpu_cost_pin=pin(args.cuda))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
