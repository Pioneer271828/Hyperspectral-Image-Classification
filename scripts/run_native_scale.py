"""Authorized LXC CLI for WP07 native scale and frozen held-out confirmation."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer import native_scale_campaign as campaign, native_scale_real as real, registered_development as transport
from transfer.container_runtime import training_container_kind
from transfer.development_blocks import scene_name
from transfer.development_block_custody import historical, _protect


def supplement(root, scans, confirm_pin, out, producer):
    prior = campaign.read(historical(root,"wp12"),root)
    tables = {**prior["preserved"],**prior["tables"]}
    _protect(root,tables.values())
    real_rows = [campaign.read(pin,root) for pin in scans]
    synthetic = campaign.read(confirm_pin,root)
    if (len(tables)!=10 or len(real_rows)!=3 or {row["scene"] for row in real_rows}!=set(real.REAL)
            or any(row["producer"]!=producer or any(row[k] for k in ("real_updates","raw_gt","query","new_labels"))
                   for row in real_rows)
            or synthetic["producer"]!=producer or set(synthetic["cases"])!=set(campaign.case_names())):
        raise ValueError("Native supplement scope differs")
    out.mkdir()
    with (out/"native-real-scope.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene","factor","blocks","observations","forward_calls","real_updates","purity"))
        for result in real_rows:
            for factor in (1,2,4):
                rows = [row for row in result["rows"] if row["observation"]["factor"]==factor]
                writer.writerow((result["scene"],factor,result["blocks"],len(rows),sum(len(r["heads"]) for r in rows),
                                 0,"unavailable_sparse_labels"))
    with (out/"native-synthetic-confirmation.csv").open("x",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case","endpoint","profile","band","condition","head","nll","original_pixels"))
        for case,endpoints in synthetic["cases"].items():
            for endpoint,rows in endpoints.items():
                for r in rows:
                    for condition,data in r["predictions"].items():
                        for head,metric in data["scores"].items():
                            writer.writerow((case,endpoint,r["profile"],r["band"],condition,head,
                                             metric["nll_sum"]/metric["count"],metric["count"]))
    return transport._json_new(out/"supplement.json",dict(producer=producer,old_tables=tables,
        old_models=65,old_heads=193,scans=scans,confirmation=confirm_pin,real_updates=0,
        tables=[transport._pin(out/name) for name in ("native-real-scope.csv","native-synthetic-confirmation.csv")]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",required=True,type=Path)
    commands = parser.add_subparsers(dest="command",required=True)
    for name in ("register","stage","audit","freeze","confirm","scan","supplement"):
        sub = commands.add_parser(name)
        if name!="stage":
            sub.add_argument("--out",required=True,type=Path)
        if name in ("stage","audit","freeze"):
            sub.add_argument("--matrix",nargs=2,required=True)
        if name=="stage":
            sub.add_argument("--case",required=True)
            sub.add_argument("--phase",choices=campaign.PHASES,required=True)
        if name=="freeze":
            sub.add_argument("--audit",nargs=2,required=True)
        if name=="confirm":
            sub.add_argument("--freeze",nargs=2,required=True)
        if name=="scan":
            sub.add_argument("--scene",required=True)
        if name=="supplement":
            sub.add_argument("--scan",nargs=2,required=True,action="append")
            sub.add_argument("--confirmation",nargs=2,required=True)
    args = parser.parse_args()
    if not training_container_kind() or not __debug__:
        parser.error("Authorized LXC with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    producer = args.expected_commit
    if code["commit"]!=producer:
        parser.error("Exact committed producer required")
    require_frozen_code(code)
    root = external_directory(args.runs.resolve(strict=True))
    out = external_directory(args.out) if hasattr(args,"out") else None
    if root.name!="runs" or (out is not None and (out==root or not out.is_relative_to(root) or out.exists())):
        parser.error("A new bounded output under the authorized runs directory is required")
    pin = lambda pair: dict(zip(("path","sha256"),pair,strict=True))
    if args.command=="register":
        result = campaign.register(root,out,producer)
    elif args.command=="stage":
        result = campaign.stage(root,pin(args.matrix),args.case,args.phase,producer)
    elif args.command=="audit":
        result = campaign.audit(root,pin(args.matrix),out,producer)
    elif args.command=="freeze":
        result = campaign.freeze(root,pin(args.matrix),pin(args.audit),out,producer)
    elif args.command=="confirm":
        result = campaign.confirm(root,pin(args.freeze),out,producer)
    elif args.command=="scan":
        result = transport._json_new(out,real.scan(root,scene_name(args.scene),producer))
    else:
        result = supplement(root,[pin(p) for p in args.scan],pin(args.confirmation),out,producer)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
