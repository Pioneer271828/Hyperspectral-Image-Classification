"""M2-J1 bounded engineering only; full accuracy study is registered separately."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_adapter_modes import MODES,adapter_settings,AdapterModeRuntime
from joint_training import load_plan,options,import_preparation,run_training
from scripts.train_joint_m4 import graph_binding
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--seed',type=int,choices=[64,512,2048],required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected published source required')
    plan=load_plan('configs/experiments/base97_training.json')
    base,items,prov=import_preparation(a.prepared_dir,a.data_root,plan,source)
    settings=adapter_settings(options(plan,'engineering',a.seed,base),a.mode)
    prov={**prov,'m4_graph_binding':graph_binding(items)}
    result=run_training(settings,items,a.out_dir,prov,source,resume=a.resume,stop_after=a.stop_after,
        runtime_adapter=AdapterModeRuntime(a.mode),finish_guard=lambda:require_frozen_code(source))
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],'mode':a.mode}),flush=True)


if __name__=='__main__':main()
