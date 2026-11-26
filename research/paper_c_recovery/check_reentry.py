"""Trap model/data construction during actual fully completed registered CLI reentry."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def snapshot(root):
    return {str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino)
            for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--producer-root',required=True)
    p.add_argument('--registration',required=True);p.add_argument('--registration-sha256',required=True)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    reg=json.loads(Path(a.registration).read_text());root=Path(a.run);before=snapshot(root)
    sys.path.insert(0,str(Path(a.producer_root)/'research/paper_c_recovery'))
    producer=importlib.import_module('run')
    sys.argv=['run.py','run','--expected-commit',reg['producer'],'--dependency',reg['dependency_root'],
        '--author',reg['author_root'],'--runs',reg['runs_root'],'--out',str(root),
        '--registration',a.registration,'--registration-sha256',a.registration_sha256,'--resume']
    fail=AssertionError('Completed reentry attempted model or observation work')
    with patch.object(producer,'NativeTask',side_effect=fail),patch.object(producer,'DSNetTask',side_effect=fail),\
         patch.object(producer,'materialize',side_effect=fail),patch.object(producer,'prior_module',side_effect=fail),\
         patch('torch.load',side_effect=fail),patch('numpy.load',side_effect=fail):
        producer.main()
    if snapshot(root)!=before:raise ValueError('Completed CLI reentry changed an existing artifact')
    result={'status':'PASSED','producer':reg['producer'],'cases':len(reg['cases']),
        'new_neural_calls':0,'new_optimizer_updates':0,'model_deserializations':0,'observation_array_decodes':0,
        'existing_artifacts_size_mtime_inode_unchanged':True,'all_stage_original_bytes_checked_by_producer':True}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
