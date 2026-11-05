"""Run exactly one source-only case from a frozen nine-case launch."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from joint_training import run_training,load_plan
from source_data_context import load_source_context
from source_pretrain import select_case,load_study,SourcePretrainRuntime
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','launch-plan','launch-sha256'):p.add_argument('--'+k,required=True)
    p.add_argument('--held-out',choices=NAMES,required=True)
    p.add_argument('--seed',type=int,choices=(64,512,2048),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();source=code_snapshot();load_study()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('Frozen launch changed')
    launch=json.loads(Path(a.launch_plan).read_text());case=select_case(launch,source,a.held_out,a.seed)
    context=load_source_context(launch['prepared'],launch['data_root'],held_out=a.held_out,consumer=source,
        manifest_sha256=load_plan('configs/experiments/base97_training.json')['prepared_manifest_sha256'])
    if context.record!=case['source_context']:raise ValueError('Source-only context changed')
    def guard():
        require_frozen_code(source)
        if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('Launch changed during training')
    result=run_training(case['settings'],context.items,case['out'],context.record,source,resume=a.resume,
        stop_after=a.stop_after,runtime_adapter=SourcePretrainRuntime(),finish_guard=guard)
    if any(file_hash(p)!=h for p,h in context.record['protected_files'].items()):raise ValueError('Source inputs changed')
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
        'held_out':a.held_out,'total_budget':1002,'test_scoring':False}),flush=True)


if __name__=='__main__':main()
