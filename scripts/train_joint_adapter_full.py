"""One preregistered M2-J2 case with frozen nine-case launch identity."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_training import import_preparation, load_plan, run_training
from joint_adapter_modes import MODES, AdapterModeRuntime
from joint_adapter_study import load_study, case_settings, select_case
from scripts.train_joint_m4 import graph_binding
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, require_frozen_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit', 'launch-plan', 'launch-sha256'): p.add_argument('--'+key, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--seed', type=int, choices=[64, 512, 2048], required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--finalize-only', action='store_true')
    a = p.parse_args(); source = code_snapshot()
    if sys.platform != 'linux' or source['dirty'] or source['commit'] != a.expected_commit:
        raise RuntimeError('Remote clean expected published SHA required')
    if file_hash(a.launch_plan) != a.launch_sha256: raise ValueError('Frozen launch changed')
    launch = json.loads(Path(a.launch_plan).read_text()); study = load_study()
    case = select_case(launch, source, study, a.mode, a.seed)
    if not launch['inputs'] or any(file_hash(p) != h for p, h in launch['inputs'].items()):
        raise ValueError('Frozen inputs changed')
    plan = load_plan('configs/experiments/base97_training.json')
    base, items, provenance = import_preparation(launch['prepared'], launch['data_root'], plan, source)
    provenance = {**provenance, 'm4_graph_binding': graph_binding(items)}
    settings = case_settings(study, plan, base, a.mode, a.seed)
    if settings != case['settings'] or provenance != launch['provenance']:
        raise ValueError('Case settings/preparation differ from frozen launch')
    runtime = AdapterModeRuntime(a.mode)
    def guard():
        require_frozen_code(source)
        if file_hash(a.launch_plan) != a.launch_sha256: raise ValueError('Launch changed during execution')
    if a.finalize_only:
        from evaluation.joint_finalization import finalize
        result = finalize(case['out'], settings, items, provenance, source, source, runtime, device='cuda', finish_guard=guard)
    else:
        result = run_training(settings, items, case['out'], provenance, source, resume=a.resume,
                              runtime_adapter=runtime, finish_guard=guard)
    print(json.dumps({'operation': result['operation'], 'updates_this_call': result['updates_this_call']}), flush=True)


if __name__ == '__main__': main()
