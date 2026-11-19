"""Review pinned old graph results through the regional cross-backend consumer."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation.frozen_training_stages import require
from evaluation.wp12_regional_history import review_history
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import _write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    args = p.parse_args()
    root = args.runs.resolve(strict=True)
    out = external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and code["commit"]==args.expected_commit and not out.exists(),
            "Use the frozen committed consumer and a fresh authorized output path")
    require_frozen_code(code)
    review = review_history(root)
    require_frozen_code(code)
    print(json.dumps(_write(out,{**review,"consumer":code["commit"]})),flush=True)


if __name__=="__main__":
    main()
