"""Opt-in M4 modes for the calibrated BASE-97 joint trainer."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_calibrated_selection import calibrated_settings, enabled, load_selected
from joint_training import import_preparation, load_plan, options, run_training, train_step, validation_snapshot
from model.region_residual_decomposition import RegionResidualDecomposition
from research_runtime import code_snapshot, require_frozen_code, tensor_hash


def m4_settings(settings, mode):
    if mode not in RegionResidualDecomposition.MODES:
        raise ValueError('Unknown M4 mode')
    result = calibrated_settings(settings)
    result['config'] = copy.deepcopy(result['config'])
    if result['config']['model']['region_residual_mode'] != 'none':
        raise ValueError('Expected original default-off baseline')
    result['config']['model']['region_residual_mode'] = mode
    return result


def graph_binding(items):
    result = {}
    for name, item in items.items():
        if item.get('Q') is None or item.get('A') is None:
            raise ValueError('M4 joint training requires Q and A')
        result[name] = {'Q': tensor_hash(item['Q']), 'A': tensor_hash(item['A'])}
    return result


class M4JointRuntime:
    train_step = staticmethod(train_step)
    validation_snapshot = staticmethod(validation_snapshot)

    def validate(self, settings):
        if not enabled(settings):
            raise ValueError('Calibrated selection required')
        return True

    def build_model(self, config, counts):
        return config.build_model(counts)

    def load_selected(self, path, items, raw_source_path, *, device='cpu'):
        model = load_selected(path, items, raw_source_path, device=device)
        expected = model.training_config['provenance'].get('m4_graph_binding')
        if expected is None or expected != graph_binding(items):
            raise ValueError('M4 graph binding mismatch')
        return model

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--plan', default='configs/experiments/base97_training.json')
    p.add_argument('--profile', required=True, choices=['engineering', 'full'])
    p.add_argument('--seed', required=True, type=int, choices=[64, 512, 2048])
    p.add_argument('--mode', default='none', choices=RegionResidualDecomposition.MODES)
    p.add_argument('--prepared-dir', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args(argv)
    source = code_snapshot()
    if sys.platform != 'linux' or source['dirty'] or source['commit'] != a.expected_commit:
        raise RuntimeError('Remote Linux, published expected SHA and clean checkout required')
    plan = load_plan(a.plan)
    baseline, items, provenance = import_preparation(a.prepared_dir, a.data_root, plan, source)
    settings = m4_settings(options(plan, a.profile, a.seed, baseline), a.mode)
    provenance = {**provenance, 'm4_graph_binding': graph_binding(items)}
    result = run_training(settings, items, a.out_dir, provenance, source,
                          resume=a.resume, stop_after=a.stop_after,
                          finish_guard=lambda: require_frozen_code(source), runtime_adapter=M4JointRuntime())
    print(json.dumps({'operation': result['operation'], 'updates_this_call': result['updates_this_call'],
                      'total_budget': settings['optimizer_steps'], 'mode': a.mode,
                      'selected_step': result.get('result', {}).get('selected_step')}), flush=True)


if __name__ == '__main__':
    main()
