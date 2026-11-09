"""Explicit independent branch/graph experiments; engineering/full budgets separate."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from branch_graph_training import AblationRuntime,ablation_settings
from model.branch_graph_ablation import VARIANTS,controlled_items
from joint_training import load_plan,import_preparation,options,run_training
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--profile',choices=['engineering','full'],required=True)
    p.add_argument('--seed',choices=[64,512,2048],type=int,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean checkout at expected published SHA required')
    plan=load_plan('configs/experiments/base97_training.json')
    base,items,provenance=import_preparation(a.prepared_dir,a.data_root,plan,source)
    items,binding=controlled_items(items,a.variant)
    provenance={**provenance,'ablation_graph_binding':binding}
    settings=ablation_settings(options(plan,a.profile,a.seed,base),a.variant)
    answer=run_training(settings,items,a.out_dir,provenance,source,resume=a.resume,stop_after=a.stop_after,
        runtime_adapter=AblationRuntime(a.variant),finish_guard=lambda:require_frozen_code(source))
    print(json.dumps({'operation':answer['operation'],'updates_this_call':answer['updates_this_call'],
        'variant':a.variant,'selected_step':answer.get('result',{}).get('selected_step'),'total_budget':settings['optimizer_steps']}),flush=True)


if __name__=='__main__':main()
