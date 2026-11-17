"""Diagnose authenticated full54 v1 endpoints and check v2 without fitting."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from evaluation import probe_numerical_diagnostics as diagnosis
from research_runtime import code_snapshot, hash_file, require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope","execution","review","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or code["commit"] != args.expected_commit or not __debug__:
        raise ValueError("Exact Linux checkout required")
    require_frozen_code(code)
    torch.set_num_threads(1)
    path = diagnosis.diagnose(args.scope,args.execution,args.review,args.out,code)
    status = json.loads(path.read_text())["status"]
    print(json.dumps({"path":str(path),"sha256":hash_file(path),"status":status}))
    return int(status!="zero_update_checks_passed")


if __name__ == "__main__":
    raise SystemExit(main())
