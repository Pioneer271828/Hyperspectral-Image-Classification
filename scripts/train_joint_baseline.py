"""Explicit BASE-97B profiles; full budget is never selected by default."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_training import import_preparation,load_plan,options,run_training
from research_runtime import code_snapshot,require_frozen_code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--plan',default='configs/experiments/base97_training.json')
    p.add_argument('--profile',required=True,choices=['engineering','full'])
    p.add_argument('--seed',type=int,required=True,choices=[64,512,2048])
    p.add_argument('--prepared-dir',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args(argv)
    source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote Linux, published expected SHA and clean checkout required')
    plan=load_plan(a.plan)
    baseline,items,provenance=import_preparation(a.prepared_dir,a.data_root,plan,source)
    settings=options(plan,a.profile,a.seed,baseline)
    outcome=run_training(settings,items,a.out_dir,provenance,source,resume=a.resume,
                         stop_after=a.stop_after,finish_guard=lambda:require_frozen_code(source))
    print(json.dumps({'operation':outcome['operation'],'updates_this_call':outcome['updates_this_call'],
                      'total_budget':settings['optimizer_steps'],
                      'selected_step':outcome.get('result',{}).get('selected_step'),
                      'restored_state_fingerprints':outcome.get('restored_state_fingerprints')}),flush=True)


if __name__=='__main__': main()
