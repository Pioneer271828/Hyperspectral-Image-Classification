"""Independently recompute frozen target-query metrics and paired guards."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.g4_target_verification import verify_saved
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','evaluation-dir','result-sha256','out-dir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected verifier SHA required')
    out=external_directory(a.out_dir);out.mkdir(exist_ok=False)
    receipt=verify_saved(a.evaluation_dir,a.result_sha256,source);require_frozen_code(source)
    write_json(out/'receipt.json',receipt)
    print('G4_INDEPENDENT_QUERY_PASSED',receipt['head_metric_cases'],receipt['class_recalls'],flush=True)


if __name__=='__main__':main()
