"""Produce bounded frozen development attribution shards for WP12."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import wp12_real_attribution as study
from evaluation.frozen_training_stages import file_pin, json_pin, require
from evaluation.wp12_release_results import proof
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts import run_frozen_band_region as archive
from scripts.run_native_released_points import arithmetic


def roster(root):
    path, sha = archive.PINS["hh_scope"]
    short = json_pin(root,dict(path=str(root/path),sha256=sha))
    long = json_pin(root,proof(root,"p4"))
    return dict(honghu_short=[r["case"]["id"] for r in short["cases"]],
                honghu_120=[r["case"]["id"] for r in long["cases"]],
                xuzhou=["wp11-full-k10"])


def run(root, out, code, group, keys, gpu):
    results = {}
    out.mkdir(parents=True,exist_ok=True)
    missing = []
    for key in keys:
        candidate = out/(key+".json")
        if candidate.exists():
            existing = json_pin(root,file_pin(candidate))
            require(existing["producer"]==code["commit"] and existing["evidence"]["case"]==key
                    and existing["evidence"]["group"]==group,"Foreign existing result")
            results[key] = file_pin(candidate)
        else:
            missing.append(key)
    if missing:
        with arithmetic(root,gpu) as execution:
            if group=="xuzhou":
                records = [study.xz_cases(root,execution.device)]
            else:
                function = study.short_cases if group=="honghu_short" else study.long_cases
                records = function(root,set(missing),execution.device)
            for row in records:
                key = row["case"]
                require(key in missing and key not in results,"A frozen shard returned foreign cases")
                require_frozen_code(code)
                results[key] = archive._write(out/(key+".json"),dict(schema=study.SCHEMA,producer=code["commit"],
                    device=execution.numeric,evidence=row,query_accessed=False,raw_gt_loaded=False))
                print(json.dumps(dict(case=key,receipt=results[key]),ensure_ascii=False),flush=True)
    require(set(results)==set(keys),"A registered case was omitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--group",choices=study.SHARD_GROUPS,required=True)
    parser.add_argument("--case",action="append")
    parser.add_argument("--shard",nargs=2,type=int,metavar=("INDEX","COUNT"),default=(0,1))
    parser.add_argument("--gpu")
    args = parser.parse_args()
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and code["commit"]==args.expected_commit,"Unexpected source or output scope")
    require_frozen_code(code)
    torch.set_num_threads(1)
    keys = roster(root)[args.group]
    index,count = args.shard
    require(0<=index<count and (args.case is None or len(set(args.case))==len(args.case)
            and set(args.case)<=set(keys)),"Unknown authorized case or shard")
    selected = keys if args.case is None else [item for item in keys if item in args.case]
    run(root,out,code,args.group,selected[index::count],args.gpu)


if __name__=="__main__":
    main()
