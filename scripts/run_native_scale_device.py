"""Register and verify strict native-scale CPU/CUDA engineering campaigns in LXC."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_scale_device_campaign as campaign, native_scale_device_cost as cost
from transfer import p4_runtime_device as p4, registered_development as transport
from transfer.container_runtime import training_container_kind
from transfer.development_block_custody import historical, _protect
from transfer.native_scale_campaign import read


def supplement(root,cpu_pin,audit_pin,out,producer,*,gpu_pin=None,comparison_pin=None,gpu_audit_pin=None):
    prior = read(historical(root,"wp12"),root)
    tables = {**prior["preserved"],**prior["tables"]}
    _protect(root,tables.values())
    audits = [read(pin,root) for pin in (audit_pin,gpu_audit_pin) if pin is not None]
    evidence = [read(pin,root) for pin in (cpu_pin,gpu_pin) if pin is not None]
    if len(tables)!=10 or any(item["producer"]!=producer or item["real_updates"] for item in (*audits,*evidence)):
        raise ValueError("Native numerical WP12 evidence differs")
    if comparison_pin is not None:
        compared = read(comparison_pin,root)
        if compared["cpu"]!=cpu_pin or compared["gpu"]!=gpu_pin or compared["producer"]!=producer:
            raise ValueError("WP12 numerical comparison pin differs")
    out.mkdir()
    with (out/"native-numerical-cost.csv").open("x",newline="",encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(("protocol","uuid","scene","tile","band","factor","arm","phase","p50_s","p95_s","pixels_s","peak_bytes"))
        for result in evidence:
            n = result["numeric"]
            for row in result["rows"]:
                for name,stat in row["timings"].items():
                    w.writerow((n["protocol"],n["physical_uuid"],row["scene"],row["tile"],row["band"],row["factor"],
                        row["arm"],name,stat["p50_seconds"],stat["p95_seconds"],stat["native_pixels_per_second"],
                        None if row["peaks"] is None else row["peaks"]["peak_allocated_bytes"]))
    return transport._json_new(out/"supplement.json",dict(producer=producer,old_tables=tables,old_models=65,old_heads=193,
        cpu=cpu_pin,cpu_audit=audit_pin,gpu=gpu_pin,gpu_audit=gpu_audit_pin,comparison=comparison_pin,
        real_updates=0,scope=transport._pin(out/"native-numerical-cost.csv")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--protocol",choices=campaign.DEVICES,default=p4.CPU)
    parser.add_argument("--gpu")
    modes = parser.add_subparsers(dest="mode",required=True)
    for name in ("geometry","qualify","register","stage","audit","cost","compare","supplement"):
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
        if name in ("compare","supplement"):
            sub.add_argument("--cpu",nargs=2,required=True)
            sub.add_argument("--cuda",nargs=2,required=name=="compare")
        if name=="supplement":
            sub.add_argument("--cpu-audit",nargs=2,required=True)
            sub.add_argument("--cuda-audit",nargs=2)
            sub.add_argument("--comparison",nargs=2)
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
    else:
        result = supplement(root,pin(args.cpu),pin(args.cpu_audit),out,producer,
                            gpu_pin=pin(args.cuda),gpu_audit_pin=pin(args.cuda_audit),comparison_pin=pin(args.comparison))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
