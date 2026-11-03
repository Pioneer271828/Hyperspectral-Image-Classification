"""One frozen M1-J2 case; requires an immutable complete six-case launch plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_training import import_preparation, load_plan, run_training
from joint_observation import JointObservationRuntime
from joint_observation_study import load_study, case_settings
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, require_frozen_code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--launch-plan',required=True)
    p.add_argument('--launch-sha256',required=True)
    p.add_argument('--preset',required=True,choices=['full','alternate75'])
    p.add_argument('--seed',required=True,type=int,choices=[64,512,2048])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--finalize-only',action='store_true')
    a=p.parse_args(argv)
    source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote Linux and clean expected published commit required')
    if file_hash(a.launch_plan)!=a.launch_sha256:
        raise ValueError('Frozen launch plan changed')
    launch=json.loads(Path(a.launch_plan).read_text())
    study=load_study()
    if launch['code']!=source or launch['study']!=study:
        raise ValueError('Wrong launch source or protocol')
    expected={(p,s) for p in study['presets'] for s in study['training_seeds']}
    if {(c['preset'],c['seed']) for c in launch['cases']}!=expected or len(launch['cases'])!=6:
        raise ValueError('Complete six-case launch required')
    case=next(c for c in launch['cases'] if (c['preset'],c['seed'])==(a.preset,a.seed))
    base_plan=load_plan('configs/experiments/base97_training.json')
    base,items,provenance=import_preparation(launch['prepared'],launch['data_root'],base_plan,source)
    settings=case_settings(study,base_plan,base,items,provenance,a.preset,a.seed)
    if settings!=case['settings'] or settings['optimizer_steps']!=1503:
        raise ValueError('Case settings differ from frozen plan')
    runtime=JointObservationRuntime(settings)
    def guard():
        require_frozen_code(source)
        if file_hash(a.launch_plan)!=a.launch_sha256: raise ValueError('Launch changed during execution')
    if a.finalize_only:
        from evaluation.joint_finalization import finalize
        result=finalize(case['out'],settings,items,provenance,source,source,runtime,device='cuda',finish_guard=guard)
    else:
        result=run_training(settings,items,case['out'],provenance,source,resume=a.resume,
                            runtime_adapter=runtime,finish_guard=guard)
    print(json.dumps({'operation':result['operation'],'updates_this_call':result['updates_this_call']}),flush=True)


if __name__=='__main__': main()
