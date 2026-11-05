"""Short source-only train/resume; no target payload or test scoring."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from joint_training import load_plan,run_training
from source_data_context import load_source_context
from source_training import source_settings,SourceRuntime
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--held-out',choices=NAMES,required=True);p.add_argument('--sources',nargs='+',choices=NAMES)
    p.add_argument('--seed',type=int,choices=(64,512,2048),default=64)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected published commit required')
    plan=load_plan('configs/experiments/base97_training.json')
    context=load_source_context(a.prepared_dir,a.data_root,manifest_sha256=plan['prepared_manifest_sha256'],
        held_out=a.held_out,sources=a.sources,consumer=code)
    settings=source_settings(a.held_out,context.sources,a.seed)
    # Only source configuration fields may differ for the registered short budget.
    expected=context.config.to_dict();actual=settings['config']
    if any(expected[k]!=actual[k] for k in expected if k not in ('name','training')):
        raise ValueError('Prepared source/model/data configuration mismatch')
    def guard():
        require_frozen_code(code)
        if any(file_hash(p)!=h for p,h in context.record['protected_files'].items()):
            raise ValueError('Protected source artifact changed')
    result=run_training(settings,context.items,a.out_dir,context.record,code,resume=a.resume,
        stop_after=a.stop_after,finish_guard=guard,runtime_adapter=SourceRuntime())
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
        'total_budget':settings['optimizer_steps'],'source_tasks':list(context.sources),
        'held_out':a.held_out,'target_payload_accessed':False,'test_scoring':False}),flush=True)


if __name__=='__main__':main()
