"""M4-S1 engineering entry; full campaigns are separately preregistered."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_supervision import MODES, SUPERVISION, SupervisionRuntime, supervision_settings
from joint_training import load_plan, options, import_preparation, run_training
from scripts.train_joint_m4 import graph_binding
from research_runtime import code_snapshot, require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'): p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--supervision',choices=SUPERVISION,required=True)
    p.add_argument('--seed',type=int,choices=[64,512,2048],required=True)
    p.add_argument('--resume',action='store_true'); p.add_argument('--stop-after',type=int)
    a=p.parse_args(); source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected SHA required')
    plan=load_plan('configs/experiments/base97_training.json')
    base,items,provenance=import_preparation(a.prepared_dir,a.data_root,plan,source)
    settings=supervision_settings(options(plan,'engineering',a.seed,base),a.mode,a.supervision)
    provenance={**provenance,'m4_graph_binding':graph_binding(items)}
    result=run_training(settings,items,a.out_dir,provenance,source,resume=a.resume,stop_after=a.stop_after,
                        finish_guard=lambda:require_frozen_code(source),runtime_adapter=SupervisionRuntime(a.mode,a.supervision))
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
                      'total_budget':settings['optimizer_steps'],'policy':settings['supervision_policy']}),flush=True)


if __name__=='__main__':main()
