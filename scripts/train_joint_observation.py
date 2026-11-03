"""M1-J1 bounded engineering entry; full accuracy study is separately registered."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_training import import_preparation, load_plan, options, run_training
from joint_observation import freeze_settings, JointObservationRuntime, PRESETS
from loadData.data_read import read_band_metadata
from research_runtime import code_snapshot, require_frozen_code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--plan', default='configs/experiments/base97_training.json')
    p.add_argument('--seed', type=int, choices=[64, 512, 2048], required=True)
    p.add_argument('--sampling-seed', type=int, required=True)
    p.add_argument('--preset', choices=list(PRESETS), required=True)
    p.add_argument('--prepared-dir', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args(argv)
    source = code_snapshot()
    if sys.platform != 'linux' or source['dirty'] or source['commit'] != a.expected_commit:
        raise RuntimeError('Remote Linux, expected published SHA and clean checkout required')
    plan = load_plan(a.plan)
    baseline, items, provenance = import_preparation(a.prepared_dir, a.data_root, plan, source)
    metadata = {n: read_band_metadata(n, i['bands']) for n, i in items.items()}
    settings = freeze_settings(options(plan, 'engineering', a.seed, baseline), items, metadata,
                               preset=a.preset, sampling_seed=a.sampling_seed,
                               plan_hash=provenance['prepared_manifest_sha256'])
    runtime = JointObservationRuntime(settings)
    result = run_training(settings, items, a.out_dir, provenance, source, resume=a.resume,
                          stop_after=a.stop_after, runtime_adapter=runtime,
                          finish_guard=lambda: require_frozen_code(source))
    print(json.dumps({'operation': result['operation'], 'updates_this_call': result['updates_this_call'],
                      'policy_hash': settings['joint_observation']['policy_hash'], 'test_scoring': False}), flush=True)


if __name__ == '__main__':
    main()
