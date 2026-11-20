"""Audit frozen mechanism records and export their deidentified development table."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_mechanism_review as review
from evaluation.frozen_training_stages import require
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--program",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    args=parser.parse_args()
    root=args.runs.resolve(strict=True)
    program=args.program.resolve(strict=True)
    out=external_directory(args.out)
    require(root in program.parents and root in out.parents,"Mechanism receipts must use authorized runs paths")
    code=code_snapshot()
    require(code["commit"]==args.expected_commit,"Enter the exact clean consumer checkout")
    require_frozen_code(code)
    out.mkdir(parents=True,exist_ok=False)
    real=review.real_reduction(root,[program/"localization",program/"localization-v2"],
                               program/"localization-v2"/"cuda",out)
    matrix=review.factorial_reduction(root,program/"factorial")
    edges=review.edge_reduction(root,program/"edge-cpu",program/"edge-cuda",program/"edges")
    require(set(real["device"])=={"selected","whu-honghu","paviaU-64"}
            and set(matrix["device"])=={"s64-k211-native","s64-k211-pixel"}
            and set(matrix["coverage"])=={"cpu","cuda"} and set(edges["device"])=={"remote-64"},
            "Representative GPU acceptance is incomplete")
    require_frozen_code(code)
    result=dict(schema=review.SCHEMA,consumer=code["commit"],development=real,factorial=matrix,
                exact_edges=edges,independent_real_accuracy=None,real_updates=0,
                raw_gt_loaded=False,query_accessed=False,wp04_executed=False)
    write_json(out/"mechanism.json",result)
    print(json.dumps(dict(schema=result["schema"],status="passed")),flush=True)


if __name__=="__main__":
    main()
