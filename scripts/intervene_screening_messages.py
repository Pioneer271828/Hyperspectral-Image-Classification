"""Validation-only, reversible message interventions on frozen screening checkpoints."""
import argparse
from contextlib import contextmanager
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

# Retained messages, fixed before observing intervention outcomes.
CONDITIONS = {'original': ('graph_to_cnn', 'cnn_to_graph'),
              'cnn_to_graph': ('cnn_to_graph',), 'graph_to_cnn': ('graph_to_cnn',), 'none': ()}


def message_strengths(model):
    interaction = model.backbone.interaction
    return {name: getattr(interaction, name).strength if interaction is not None and
            getattr(interaction, name) is not None else None for name in CONDITIONS['original']}


@contextmanager
def retain_messages(model, condition):
    """Zero disabled residual strengths, keeping parameters/modules and restoring on errors."""
    if condition not in CONDITIONS or model.training:
        raise ValueError('Require a declared intervention and an eval model')
    before = message_strengths(model)
    interaction = model.backbone.interaction
    try:
        if interaction is not None:
            for name in CONDITIONS['original']:
                module = getattr(interaction, name)
                if module is not None and name not in CONDITIONS[condition]:
                    module.strength = 0.0
        yield {'condition': condition, 'retained_messages': list(CONDITIONS[condition]),
               'original_strengths': before, 'effective_strengths': message_strengths(model),
               'operator': 'zero residual strength; no parameter/module replacement'}
    finally:
        if interaction is not None:
            for name, strength in before.items():
                if strength is not None:
                    getattr(interaction, name).strength = strength


def paired_effect(reference, candidate, focus_class=6, confused_as=2):
    """Paired prediction changes, without treating spatial pixels as independent repeats."""
    for key in ('truth', 'global_ids', 'tile_ids'):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError('Paired evaluation support/order changed: ' + key)
    if reference['probabilities'].shape != candidate['probabilities'].shape:
        raise ValueError('Paired probability shapes differ')
    truth = reference['truth']
    result = {}
    for index, branch in enumerate(BRANCHES):
        first = reference['probabilities'][:, index].argmax(1) + 1
        second = candidate['probabilities'][:, index].argmax(1) + 1
        def counts(selected):
            n = int(selected.sum())
            corrected = int((selected & (first != truth) & (second == truth)).sum())
            harmed = int((selected & (first == truth) & (second != truth)).sum())
            return {'count': n, 'corrected': corrected, 'harmed': harmed, 'net_correct': corrected - harmed,
                    'oa_delta_pp': (corrected - harmed) / n * 100 if n else None,
                    'prediction_changed': int((selected & (first != second)).sum()),
                    'reference_focus_as_confused': int((selected & (truth == focus_class) & (first == confused_as)).sum()),
                    'candidate_focus_as_confused': int((selected & (truth == focus_class) & (second == confused_as)).sum())}
        result[branch] = {'all': counts(np.ones(len(truth), dtype=bool)), 'focus': counts(truth == focus_class)}
    return result


def check_equivalent(reference, candidate, branch_indices=(0, 1, 2)):
    """Same messages imply the same scored class IDs; bound CUDA probability roundoff."""
    for key in ('truth', 'global_ids', 'tile_ids'):
        if not np.array_equal(reference[key], candidate[key]):
            raise AssertionError('Equivalent condition changed evaluation support')
    differences = {}
    for index in branch_indices:
        left, right = reference['probabilities'][:, index], candidate['probabilities'][:, index]
        difference = float(np.max(np.abs(left-right)))
        if (not np.allclose(left, right, rtol=1e-6, atol=2e-5)
                or not np.array_equal(left.argmax(1), right.argmax(1))):
            raise AssertionError('Unchanged message path changed predictions: ' + BRANCHES[index] +
                                 '; max_abs=' + str(difference))
        differences[BRANCHES[index]] = difference
    return differences


def run_validation(model, cube, labels, plan, stats, wave, scale, class_ids, out,
                   focus_class=6, confused_as=2):
    """Hard-coded validation partition: no condition is evaluated on the test partition."""
    from research_runtime import tensor_hash, write_json
    if model.training:
        raise ValueError('Validation intervention requires eval mode')
    before = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    strengths = message_strengths(model)
    summaries, arrays = {}, {}
    for condition in CONDITIONS:
        with retain_messages(model, condition) as intervention:
            summary, payload, tiles = collect_partition(model, cube, labels, plan, stats, 'val', 'paviaU',
                wave, scale, class_ids, focus_class, confused_as)
            summary['message_intervention'] = intervention
        if message_strengths(model) != strengths:
            raise AssertionError('Message strengths were not restored')
        summary.update(partition='val', intervention_attributes_restored=True,
                       paired_vs_original=paired_effect(arrays.get('original', payload), payload, focus_class, confused_as))
        if condition != 'original':
            summary['fused_nll_delta_vs_original'] = summary['fused_nll'] - summaries['original']['fused_nll']
        else:
            summary['fused_nll_delta_vs_original'] = 0.0
        np.savez_compressed(out / (condition + '-predictions.npz'), **payload)
        write_json(out / (condition + '-report.json'), summary)
        write_json(out / (condition + '-tiles.json'), tiles)
        summaries[condition], arrays[condition] = summary, payload
    checks = {}
    if model.backbone.interaction is None:
        for condition in CONDITIONS:
            checks[condition] = check_equivalent(arrays['original'], arrays[condition])
    else:
        # Both exchanges use pre-interaction features: no cross-direction feedback in this one stage.
        for left, right, indices in [('original', 'graph_to_cnn', (2,)),
                                     ('original', 'cnn_to_graph', (1,)),
                                     ('none', 'cnn_to_graph', (2,)),
                                     ('none', 'graph_to_cnn', (1,))]:
            checks[left + '_vs_' + right] = check_equivalent(arrays[left], arrays[right], indices)
    if before != {name: tensor_hash(value) for name, value in model.state_dict().items()}:
        raise AssertionError('Intervention changed parameters or buffers')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError('Read-only intervention unexpectedly produced parameter gradients')
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
                             'state_unchanged': True, 'strengths_restored': True}
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
            'test_partition_evaluated': False, 'conditions': {key: list(value) for key, value in CONDITIONS.items()},
            'frozen_inputs_unchanged': True, 'raw_dataset_unchanged': True,
            'interpretation': 'Frozen-weight path intervention; not a retrained single-direction model or final method selection'}


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
        write_json(out / 'message-intervention.json', report)
        manifest.update(status='passed', optimizer_steps=0, condition_count=12,
                        frozen_input_count=len(report['source_inputs']), test_partition_evaluated=False)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
