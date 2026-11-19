"""Run original native graph attribution or its bounded physical-engineering controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.frozen_training_stages import require
from research_runtime import code_snapshot, require_frozen_code, external_directory
from scripts.run_frozen_band_region import _write
from scripts.run_native_released_points import arithmetic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("controls","real"))
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--case")
    parser.add_argument("--drain-after",type=int)
    args = parser.parse_args(argv)
    code = code_snapshot()
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    require(code["commit"]==args.expected_commit and root in out.parents,
            "Use the frozen checkout and a result under the authorized runs")
    require_frozen_code(code)
    torch.set_num_threads(1)
    if args.action=="controls":
        from evaluation.wp12_pathway_fixture import exercise

        require(args.case is args.drain_after is None and not out.exists(),"Use a fresh fixed control receipt")
        with arithmetic(root,args.gpu) as execution:
            result = exercise(execution)
        out.parent.mkdir(parents=True,exist_ok=True)
        require_frozen_code(code)
        value = _write(out,{**result,"producer":code["commit"]})
    else:
        from evaluation.wp12_native_pathways_real import run

        require(args.case,"Select one already authorized frozen checkpoint")
        with arithmetic(root,args.gpu) as execution:
            value = run(root,out,code,args.case,execution,drain_after=args.drain_after)
    print(json.dumps(value),flush=True)


if __name__=="__main__":
    main()
