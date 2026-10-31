"""Validation-only coordinate sensitivity of frozen screening models (not retraining)."""
from contextlib import contextmanager
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_screening_branches import BRANCHES, CORE_PATHS, collect_partition
from scripts.intervene_screening_messages import paired_effect, check_equivalent

CONDITIONS = {'original': 'whole-scene normalized y,x', 'zero': 'both coordinates zero',
              'relative': 'tile-local y/(H-1),x/(W-1)'}


def transform_input(value, condition):
    """Clone only for interventions; preserve spectral channels and the input tensor."""
    import torch
    if condition not in CONDITIONS or value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError('Declared condition and H,W,B+2 input required')
    if condition == 'original':
        return value
    result = value.clone()
    result[..., -2:] = 0
    if condition == 'relative':
        h, w = value.shape[:2]
        result[..., -2] = torch.arange(h, device=value.device, dtype=value.dtype)[:, None] / max(h-1, 1)
        result[..., -1] = torch.arange(w, device=value.device, dtype=value.dtype)[None, :] / max(w-1, 1)
    return result


@contextmanager
def coordinate_condition(model, condition):
    if model.training or condition not in CONDITIONS or model.backbone.extra_channels != 2:
        raise ValueError('Require eval model with exactly two extra coordinate channels')
    def intervene(module, positional, keywords):
        if len(positional) != 2 or 'x' in keywords:
            raise ValueError('Expected model(dataset, x, ...)')
        return (positional[0], transform_input(positional[1], condition)), keywords
    handle = model.register_forward_pre_hook(intervene, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def run_validation(model, cube, labels, plan, stats, wave, scale, class_ids, out,
                   focus_class=6, confused_as=2):
    from research_runtime import tensor_hash, write_json
    before = {key: tensor_hash(value) for key, value in model.state_dict().items()}
    hooks_before = tuple(model._forward_pre_hooks)
    summaries, arrays = {}, {}
    for condition in CONDITIONS:
        with coordinate_condition(model, condition):
            summary, payload, tiles = collect_partition(model, cube, labels, plan, stats, 'val', 'paviaU',
                wave, scale, class_ids, focus_class, confused_as)
        if tuple(model._forward_pre_hooks) != hooks_before:
            raise AssertionError('Coordinate hook was not removed')
        summary.update(partition='val', coordinate_condition=CONDITIONS[condition],
            paired_vs_original=paired_effect(arrays.get('original', payload), payload, focus_class, confused_as))
        summary['fused_nll_delta_vs_original'] = summary['fused_nll'] - summaries.get('original', summary)['fused_nll']
        np.savez_compressed(out / (condition + '-predictions.npz'), **payload)
        write_json(out / (condition + '-report.json'), summary)
        write_json(out / (condition + '-tiles.json'), tiles)
        summaries[condition], arrays[condition] = summary, payload
    # Repeat the original after removing the intervention: catches persistent hooks/state effects.
    _, restored, _ = collect_partition(model, cube, labels, plan, stats, 'val', 'paviaU',
                                       wave, scale, class_ids, focus_class, confused_as)
    checks = {'original_after_restoration': check_equivalent(arrays['original'], restored)}
    if before != {key: tensor_hash(value) for key, value in model.state_dict().items()}:
        raise AssertionError('Coordinate diagnostic changed model state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError('Unexpected parameter gradients')
    return summaries, arrays['original'], checks

def audit(args, out, code):
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from train_spatial import SpatialConfig, restore_spatial_checkpoint
    from research_runtime import hash_file, require_frozen_code
    source = args.campaign_root.resolve(strict=True)
    accepted = args.accepted_diagnostics.resolve(strict=True)
    frozen = {}
    def read(path):
        frozen[str(path)] = hash_file(path)
        return json.loads(path.read_text())
    previous = read(accepted / 'manifest.json')
    if previous['status'] != 'passed' or previous.get('optimizer_steps') != 0:
        raise ValueError('Require successful frozen branch diagnostics')
    for upstream, paths in [(args.training_commit, CORE_PATHS),
                            (previous['commit'], ['scripts/diagnose_screening_branches.py'])]:
        subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', upstream, code['commit'], '--', *paths],
                       check=True, capture_output=True, text=True)
    campaign = read(source / 'campaign.json')
    if campaign['commit'] != args.training_commit or campaign['order'] != ['none', 'add', 'gated']:
        raise ValueError('Unexpected training campaign')
    if (source / '.campaign.lock').exists():
        raise ValueError('Training campaign is still locked')
    initial = source / 'shared_initial_state.pt'
    if hash_file(initial) != campaign['initialization_file_sha256']:
        raise ValueError('Shared initial weights changed')
    frozen[str(initial)] = hash_file(initial)
    data_root = args.data_root.resolve(strict=True)
    identity, labels = inspect_dataset(data_root, 'paviaU')
    _, image_key, classes, _ = DATASETS['paviaU']
    cube = loadmat(data_root / identity['image_path'], variable_names=[image_key])[image_key]
    wave, _ = readWavelengths('paviaU', cube.shape[2])
    variants = {}
    for variant in campaign['order']:
        require_frozen_code(code)
        completion = read(source / (variant + '-result.json'))
        if completion['status'] != 'completed':
            raise ValueError('Incomplete training variant')
        attempt = (source / 'training' / completion['run_id'] / 'attempts' / completion['attempt']).resolve(strict=True)
        if not attempt.is_relative_to(source):
            raise ValueError('Invalid attempt path')
        manifest = read(attempt / 'manifest.json')
        if (manifest['status'] != 'completed' or manifest['code']['commit'] != args.training_commit
                or manifest['code']['source_hash'] != campaign['source_hash'] or manifest['code']['dirty']
                or manifest['prepared_data']['raw'] != identity
                or manifest['initialization']['state_digest'] != campaign['shared_state_digest']):
            raise ValueError('Training identity mismatch')
        for filename, expected in manifest['artifacts'].items():
            path = (attempt / filename).resolve(strict=True)
            if not path.is_relative_to(attempt) or hash_file(path) != expected:
                raise ValueError('Frozen artifact changed: ' + filename)
            frozen[str(path)] = expected
        config = SpatialConfig(json.loads((attempt / 'resolved_config.json').read_text()))
        if (config.config_hash != campaign['configs'][variant] or config.research.interaction.graph_update_mode != 'none'
                or config.research.interaction.interaction_mode != variant
                or config.research.interaction.interaction_direction != 'both'):
            raise ValueError('Unexpected configuration or graph feedback')
        metrics = json.loads((attempt / 'metrics.json').read_text())
        if metrics['optimizer_steps'] != 10000 or metrics['selection'] != 'val_loss':
            raise ValueError('Unexpected training budget or selection')
        best = min(metrics['history'], key=lambda item: item['val']['nll'])
        accepted_report = read(accepted / (variant + '-val-report.json'))
        accepted_arrays = accepted / (variant + '-val-predictions.npz')
        frozen[str(accepted_arrays)] = hash_file(accepted_arrays)
        with np.load(accepted_arrays, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in ('truth', 'global_ids', 'tile_ids', 'probabilities')}
        if (accepted_report['checkpoint'] != str(attempt / 'model.pt')
                or accepted_report['selected_step'] != best['optimizer_steps']):
            raise ValueError('Accepted baseline refers to another checkpoint')
        model, plan, stats = restore_spatial_checkpoint(attempt / 'model.pt', attempt / 'protocol.json',
                                                       attempt / 'statistics.json', 'cuda')
        if (model.training_config['spatial_config'] != config.to_dict()
                or model.training_config['initialization'] != manifest['initialization']
                or model.training_config['run_id'] != completion['run_id']
                or plan['plan_hash'] != metrics['plan_hash'] or stats['statistics_hash'] != metrics['statistics_hash']):
            raise ValueError('Checkpoint metadata or spatial identity mismatch')
        folder = out / variant
        folder.mkdir()
        summaries, original, checks = run_validation(model, cube, labels, plan, stats, wave,
            config.research.data.pavia_superpixel_scale, list(range(1, classes+1)), folder,
            args.focus_class, args.confused_as)
        baseline_differences = check_equivalent(reference, original)
        current = summaries['original']
        if (current['classification']['fused']['confusion_matrix'] != best['val']['confusion']
                or abs(current['fused_nll'] - best['val']['nll']) > 1e-6):
            raise AssertionError('Original condition differs from training validation baseline')
        for branch in BRANCHES:
            if current['classification'][branch]['confusion_matrix'] != accepted_report['classification'][branch]['confusion_matrix']:
                raise AssertionError('Original auxiliary baseline changed')
        variants[variant] = {'conditions': summaries, 'equivalence_checks': checks,
                             'original_vs_accepted_max_abs_probability': baseline_differences,
                             'selected_step': best['optimizer_steps'], 'checkpoint': str(attempt / 'model.pt'),
                             'state_unchanged': True, 'coordinate_hooks_restored': True}
        for condition, summary in summaries.items():
            print(json.dumps({'variant': variant, 'condition': condition, 'partition': 'val',
                'oa': summary['classification']['fused']['oa'], 'nll': summary['fused_nll'],
                'focus_as_confused': summary['target_confusion']['fused_confusion_count'],
                'paired': summary['paired_vs_original']['fused']}), flush=True)
        del model
    if inspect_dataset(data_root, 'paviaU')[0] != identity:
        raise ValueError('Original data changed')
    if any(hash_file(Path(path)) != sha for path, sha in frozen.items()):
        raise ValueError('Frozen source inputs changed')
    require_frozen_code(code)
    return {'variants': variants, 'source_inputs': frozen, 'training_commit': args.training_commit,
            'accepted_diagnostics_commit': previous['commit'], 'partition': 'val', 'optimizer_steps': 0,
            'test_partition_evaluated': False, 'conditions': CONDITIONS,
            'frozen_inputs_unchanged': True, 'raw_dataset_unchanged': True,
            'interpretation': 'Frozen-weight coordinate sensitivity including downstream graph response; not proof of a position shortcut or a retrained comparison'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--training-commit', required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--accepted-diagnostics', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--focus-class', type=int, default=6)
    parser.add_argument('--confused-as', type=int, default=2)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists() or sys.flags.optimize:
        parser.error('Use the training container with assertions enabled')
    import torch
    from research_runtime import code_snapshot, external_directory, write_json
    code = code_snapshot()
    if (any(not re.fullmatch('[0-9a-f]{40}', value) for value in [args.expected_commit, args.training_commit])
            or args.expected_commit != code['commit'] or code['dirty']):
        parser.error('Require a clean checkout and matching full SHA')
    if not torch.cuda.is_available():
        parser.error('CUDA required; no fallback')
    torch.set_num_threads(1)
    out = external_directory(args.out_dir)
    for source in [args.campaign_root.resolve(strict=True), args.accepted_diagnostics.resolve(strict=True)]:
        if source == out or source in out.parents or out in source.parents:
            parser.error('Keep output separate from frozen input directories')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'branch': code['branch'], 'commit': code['commit'],
                'source_hash': code['source_hash'], 'command': [sys.executable, *sys.argv],
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(0),
                'partition': 'val', 'conditions': list(CONDITIONS)}
    write_json(out / 'manifest.json', manifest)
    try:
        report = audit(args, out, code)
        write_json(out / 'coordinate-intervention.json', report)
        manifest.update(status='passed', optimizer_steps=0, condition_count=9,
                        frozen_input_count=len(report['source_inputs']), test_partition_evaluated=False)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()

