"""Check exact repeat endpoints and trap completed registered routing CLI work."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


def snapshot(root):return {str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in root.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('routes','training-run','lineage-audit','producer-root','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();routes=Path(a.routes);lineage=json.loads(Path(a.lineage_audit).read_text());elements=0
    for r in lineage['cases']:
        folder=routes/r['case']
        for arm in ('light','full'):
            with np.load(folder/f'round-0-{arm}.npz',allow_pickle=False) as initial:
                reference={k:initial[k].copy() for k in initial.files}
            for repeat in range(1,6):
                with np.load(folder/f'round-{repeat}-{arm}.npz',allow_pickle=False) as actual:
                    if set(actual.files)!=set(reference) or any(not np.array_equal(actual[k],v) for k,v in reference.items()):
                        raise ValueError('Frozen repeated endpoint probabilities differ')
                    elements+=sum(v.size for v in reference.values())
    path=Path(a.producer_root)/'research/paper_c_recovery';sys.path.insert(0,str(path))
    spec=importlib.util.spec_from_file_location('paper_c_frozen_route_producer',path/'route_transfer.py');producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    fail=AssertionError('Completed routing reentry attempted model/data construction')
    cases=[]
    with patch.object(producer,'prior_module',side_effect=fail),patch('torch.load',side_effect=fail),patch('numpy.load',side_effect=fail):
        for r in lineage['cases']:
            folder=routes/r['case'];c=json.loads((folder/'contract.json').read_text());before=snapshot(folder)
            sys.argv=['route_transfer.py','--expected-commit',c['code'],'--training-run',a.training_run,'--lineage-audit',a.lineage_audit,
                      '--case',r['case'],'--out',str(folder),'--resume']
            producer.main()
            if snapshot(folder)!=before:raise ValueError('Completed case artifacts changed')
            cases.append(r['case'])
    result={'status':'PASSED','cases':cases,'repeated_endpoint_elements_checked':elements,
        'all_static_full_CNN_light_probabilities_exact_across_six_rounds':True,
        'completed_CLI_new_neural_calls':0,'completed_CLI_model_deserializations':0,'completed_CLI_observation_array_decodes':0,
        'new_optimizer_updates':0,'existing_case_artifacts_unchanged':True}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('status','repeated_endpoint_elements_checked','completed_CLI_new_neural_calls','new_optimizer_updates')}))


if __name__=='__main__':main()
