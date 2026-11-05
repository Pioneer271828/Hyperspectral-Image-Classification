"""Run one frozen full M4 source case; never construct a held-target payload."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from joint_training import run_training,load_plan
from source_data_context import load_source_context
from m4_source_study import MODES,select_case,load_study,M4SourceFullRuntime
from m4_transfer import graph_binding
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','launch-plan','launch-sha256'):p.add_argument('--'+k,required=True)
    p.add_argument('--held-out',choices=NAMES,required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--seed',type=int,choices=(64,512,2048),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--ordered-validation-v2',action='store_true')
    a=p.parse_args();code=code_snapshot()
    if a.ordered_validation_v2:
        from m4_ordered_source import load_study as study_loader, select_case as selector, M4SourceOrderedRuntime as Runtime
    else:
        study_loader,selector,Runtime=load_study,select_case,M4SourceFullRuntime
    study_loader()
    if sys.platform!='linux' or not __debug__ or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Clean expected remote commit and assertions required')
    if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('Frozen full M4 launch changed')
    launch=json.loads(Path(a.launch_plan).read_text());case=selector(launch,code,a.held_out,a.seed,a.mode)
    ctx=load_source_context(launch['prepared'],launch['data_root'],held_out=a.held_out,consumer=code,
        manifest_sha256=load_plan('configs/experiments/base97_training.json')['prepared_manifest_sha256'])
    if ctx.record!=case['source_context'] or graph_binding(ctx.items)!=case['graph_binding']:
        raise ValueError('Frozen M4 source context/graph changed')
    def guard():
        require_frozen_code(code)
        if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('M4 launch changed during training')
        if any(file_hash(p)!=h for p,h in ctx.record['protected_files'].items()):raise ValueError('Source inputs changed')
    guard()
    result=run_training(case['settings'],ctx.items,case['out'],case['provenance'],code,resume=a.resume,
        stop_after=a.stop_after,runtime_adapter=Runtime(case['initial_state_sha256']),finish_guard=guard)
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call'],
        'held_out':a.held_out,'mode':a.mode,'total_budget':1002,'test_scoring':False}),flush=True)


if __name__=='__main__':main()
