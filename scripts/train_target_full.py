"""Run one registered full target case; no query labels or test scoring."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from joint_training import run_training
from target_initialization import METHODS
from target_data_context import load_target_context
from target_study import select_case,validate_source_case,TargetStudyRuntime,load_study
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','launch-plan','launch-sha256'):p.add_argument('--'+k,required=True)
    p.add_argument('--target',choices=NAMES,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,choices=(64,512,2048),required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();source=code_snapshot();load_study()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected published commit required')
    if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('Frozen target launch changed')
    launch=json.loads(Path(a.launch_plan).read_text());case=select_case(launch,source,a.target,a.seed,a.method)
    validate_source_case(case);inputs=case['inputs']
    items,provenance,guard=load_target_context(a.target,inputs['observation'],inputs['record'],inputs['record_sha256'],
        inputs['family'],inputs['family_sha256'],per_class=10,support_seed=64)
    provenance['source_spec']=case['source_spec']
    if case['source_spec']:
        spec=case['source_spec'];binding=case['source_result']
        provenance['protected_files'].update({spec['source_path']:spec['source_weight_sha256'],
            spec['source_path']+'.json':spec['source_sidecar_sha256'],binding['path']:binding['sha256']})
    if provenance!=case['provenance']:raise ValueError('Frozen target provenance changed')
    def finish():
        guard();require_frozen_code(source)
        if file_hash(a.launch_plan)!=a.launch_sha256:raise ValueError('Frozen launch changed while training')
        if any(file_hash(p)!=h for p,h in provenance['protected_files'].items()):raise ValueError('Protected target/source input changed')
    finish()
    runtime=TargetStudyRuntime(case['settings'],items,case['out'],source_spec=case['source_spec'],
        initialization_state_sha256=case['initial_state_sha256'])
    result=run_training(case['settings'],items,case['out'],provenance,source,resume=a.resume,
        stop_after=a.stop_after,device='cuda',finish_guard=finish,runtime_adapter=runtime)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
