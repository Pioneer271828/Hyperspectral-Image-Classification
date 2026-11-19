"""Run the read-only long coverage consumer from an independent checkout."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch

from evaluation.frozen_training_stages import json_pin, require
from evaluation.wp12_long_coverage import independent_case, publish
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.native_scale_device import arithmetic
from transfer.p4_runtime_device import CPU, CUDA_FP64_WEIGHT
from transfer.shared_gpu_lease import physical_lease


def production(source, commit, hash_value):
    require(re.fullmatch("[a-f0-9]{40}", commit) is not None, "Exact frozen producer commit required")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, source["commit"]], check=True)
    paths = subprocess.check_output(["git", "diff", "--name-only", "-z", commit, source["commit"]], text=True)
    allowed = {"evaluation/wp12_long_coverage.py", "evaluation/wp12_long_cli.py",
               "scripts/ingest_wp12_evidence.py", "scripts/run_native_long_coverage.py"}
    require(all(path in allowed or path.startswith(("tests/", "docs/", "research/reports/", "research/当前搭建"))
                for path in paths.split("\0") if path), "Independent consumer changed the production runtime")
    tree = subprocess.check_output(["git","ls-tree","-r","--name-only","-z", commit],text=True).split("\0")
    changed = set(paths.split("\0"))
    files = {}
    for path in tree:
        if path.endswith(".py") or (path.startswith("configs/") and path.endswith(".json")):
            files[path] = (hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{path}"])
                                          .replace(b"\r\n",b"\n")).hexdigest()
                           if path in changed else source["files"][path])
    require(digest(files) == hash_value, "Producer source fingerprint differs from the frozen campaign")
    return {**source, "commit":commit, "source_hash":hash_value, "files":files}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--long-stage-case", action="store_true")
    modes.add_argument("--long-stage-publish", action="store_true")
    for name in ("runs","campaign","out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--freeze",nargs=2,required=True,metavar=("PATH","SHA256"))
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--case")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--historical-snapshot",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    root, place, out = args.runs.resolve(strict=True), external_directory(args.campaign), external_directory(args.out)
    require(root in place.parents and root in out.parents and out != place and place not in out.parents,
            "Independent results must stay outside the source production directory")
    code = code_snapshot()
    require_frozen_code(code)
    require(code["commit"] == args.expected_commit, "Independent consumer SHA differs")
    torch.set_num_threads(1)
    pin = dict(zip(("path","sha256"),args.freeze,strict=True))
    frozen_receipt = json_pin(root,pin)
    plan = json_pin(root,frozen_receipt["campaign"])
    producer = production(code,plan["producer"],plan["source_hash"])
    if args.long_stage_case:
        require(args.case and args.receipts is None and args.historical_snapshot is None, "One frozen case and original device required")
        require(out.parent.is_dir(), "Create the independent output directory first")
        lease = nullcontext() if args.gpu is None else physical_lease(args.gpu)
        with lease, arithmetic(CPU if args.gpu is None else CUDA_FP64_WEIGHT,args.gpu) as execution:
            result = independent_case(root,place,out,code,producer,pin,args.case,execution)
    else:
        require(args.receipts and not args.case and not args.gpu, "Frozen publish requires all independent receipts")
        history = dict(zip(("path","sha256"),args.historical_snapshot,strict=True)) if args.historical_snapshot else None
        result = publish(root,place,out,code,producer,pin,args.receipts.resolve(strict=True),history)
    print(json.dumps(result),flush=True)
