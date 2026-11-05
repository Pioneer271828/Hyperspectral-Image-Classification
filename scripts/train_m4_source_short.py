"""Source-only short M4/control integration; no held-target payload access."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from joint_training import load_plan,run_training
from source_data_context import load_source_context
from m4_transfer import MODES,source_options,M4SourceRuntime,graph_binding
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--held-out',choices=NAMES,required=True);p.add_argument('--sources',nargs='+',choices=NAMES)
    p.add_argument('--seed',type=int,choices=(64,512,2048),default=64);p.add_argument('--mode',choices=MODES,default='none')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or code['dirty'] or code['commit']!=a.expected_commit:raise RuntimeError('Clean remote expected SHA required')
    plan=load_plan('configs/experiments/base97_training.json')
    context=load_source_context(a.prepared_dir,a.data_root,manifest_sha256=plan['prepared_manifest_sha256'],
        held_out=a.held_out,sources=a.sources,consumer=code)
    settings=source_options(a.held_out,a.mode,sources=context.sources,seed=a.seed)
    expected=context.config.to_dict();actual=copy.deepcopy(settings['config']);actual['model']['region_residual_mode']='none'
    if any(expected[k]!=actual[k] for k in expected if k not in ('name','training')):
        raise ValueError('Only declared M4 source model change allowed')
    provenance=copy.deepcopy(context.record)
    if a.mode!='none':provenance.update(m4_mode=a.mode,m4_graph_binding=graph_binding(context.items))
    def guard():
        require_frozen_code(code)
        if any(file_hash(p)!=h for p,h in context.record['protected_files'].items()):raise ValueError('Protected source input changed')
    result=run_training(settings,context.items,a.out_dir,provenance,code,resume=a.resume,stop_after=a.stop_after,
        finish_guard=guard,runtime_adapter=M4SourceRuntime())
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
        'total_budget':settings['optimizer_steps'],'mode':a.mode,'target_payload_accessed':False,'test_scoring':False}),flush=True)


if __name__=='__main__':main()
