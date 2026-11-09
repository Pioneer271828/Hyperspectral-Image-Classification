"""Audit and finalize an already fully trained M4 joint state; zero optimizer updates."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.joint_finalization import finalize
from joint_training import import_preparation,load_plan,options
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code
from scripts.train_joint_m4 import m4_settings,graph_binding,M4JointRuntime

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit','training-commit','run-dir','prepared-dir','data-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True,choices=(64,512,2048))
    p.add_argument('--mode',required=True,choices=('none','capacity_control','x_residual'))
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Clean remote expected verification SHA required')
    folder=Path(a.run_dir);progress=json.loads((folder/'progress.json').read_text())
    if Path(progress['state_file']).name!=progress['state_file'] or file_hash(folder/progress['state_file'])!=progress['state_sha256']:
        raise ValueError('Final training state changed')
    state=torch.load(folder/progress['state_file'],map_location='cpu',weights_only=True)
    producer=state['start_code']
    if producer['commit']!=a.training_commit:
        raise ValueError('Unexpected training SHA')
    del state
    plan=load_plan('configs/experiments/base97_training.json')
    baseline,items,prov=import_preparation(a.prepared_dir,a.data_root,plan,source)
    settings=m4_settings(options(plan,'full',a.seed,baseline),a.mode)
    prov={**prov,'m4_graph_binding':graph_binding(items)}
    result=finalize(folder,settings,items,prov,producer,source,M4JointRuntime(),device='cuda',
                    finish_guard=lambda:require_frozen_code(source))
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
                      'selected_step':result['result']['selected_step'],'policy':result['result'].get('finalization_policy')}),flush=True)

if __name__=='__main__':main()
