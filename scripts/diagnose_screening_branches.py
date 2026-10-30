"""Read-only CNN/Graph/fusion audit of completed paired spatial screening runs."""
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
from evaluation.classification_metrics import classification_metrics, class_order, evaluation_mask
from evaluation.branch_diagnostics import branch_diagnostics

BRANCHES = ('fused', 'graph', 'cnn')
CORE_PATHS = ['model/', 'loadData/', 'createGraph/', 'evaluation/', 'train_spatial.py',
              'train_vca_multidataset.py', 'research_runtime.py', 'configs/']


def target_confusion(labels, predictions, class_ids, focus_class, confused_as, mask=None):
    """Attribute an observed fused confusion to auxiliary-head predictions, not causes."""
    ids = class_order(class_ids)
    if focus_class not in ids or confused_as not in ids or focus_class == confused_as:
        raise ValueError('Two distinct declared class IDs required')
    labels = np.asarray(labels)
    valid = evaluation_mask(labels, mask)
    # Also validate finite, declared predictions, including when the focus has no support.
    for name in BRANCHES:
        classification_metrics(labels, predictions[name], ids, valid)
    selected = valid & (labels == focus_class)
    truth_count = int(selected.sum())
    vectors = {name: np.asarray(value)[selected] for name, value in predictions.items() if name in BRANCHES}
    states = {name: np.where(value == focus_class, 0, np.where(value == confused_as, 1, 2))
              for name, value in vectors.items()}
    # Codes: 0=correct, 1=confused_as, 2=other wrong; preserve all 27 joint counts.
    joint = np.bincount(states['cnn'] * 9 + states['graph'] * 3 + states['fused'], minlength=27).reshape(3, 3, 3)
    wrong_as = vectors['fused'] == confused_as
    c, g = vectors['cnn'] == focus_class, vectors['graph'] == focus_class
    denominator = int(wrong_as.sum())
    strata = {'both_aux_correct': int((wrong_as & c & g).sum()),
              'cnn_correct_only': int((wrong_as & c & ~g).sum()),
              'graph_correct_only': int((wrong_as & ~c & g).sum()),
              'neither_aux_correct': int((wrong_as & ~c & ~g).sum())}
    return {'focus_class': focus_class, 'confused_as': confused_as, 'support': truth_count,
            'branch_counts': {name: {'correct': int((value == focus_class).sum()),
                'confused_as': int((value == confused_as).sum()),
                'other_wrong': int(((value != focus_class) & (value != confused_as)).sum())}
                for name, value in vectors.items()},
            'joint_state_order': ['correct', 'confused_as', 'other_wrong'],
            'joint_axes': ['cnn', 'graph', 'fused'], 'joint_counts': joint.tolist(),
            'fused_confusion_count': denominator,
            'fused_confusion_rate': denominator / truth_count if truth_count else None,
            'fused_confusion_aux_strata': strata,
            'fused_confusion_both_aux_same_wrong': int((wrong_as & (vectors['cnn'] == confused_as) &
                                                       (vectors['graph'] == confused_as)).sum()),
            'interpretation': 'Same forward-pass auxiliary heads; descriptive association, not an intervention'}


def summarize(labels, predictions, class_ids, focus_class=6, confused_as=2, mask=None):
    valid = evaluation_mask(labels, mask)
    classes = class_order(class_ids)
    classification = {name: classification_metrics(labels, predictions[name], classes, valid) for name in BRANCHES}
    overall = branch_diagnostics(labels, predictions['cnn'], predictions['graph'], predictions['fused'], classes, valid)
    per_class = {str(c): branch_diagnostics(labels, predictions['cnn'], predictions['graph'], predictions['fused'],
                                           classes, valid & (np.asarray(labels) == c)) for c in classes}
    if sum(s['count'] for s in overall['strata'].values()) != overall['count']:
        raise AssertionError('Branch strata do not partition the evaluation support')
    return {'classification': classification, 'branches': overall, 'branches_by_true_class': per_class,
            'target_confusion': target_confusion(labels, predictions, classes, focus_class, confused_as, valid)}


def describe(values):
    values = np.asarray(values)
    if not values.size:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    if not np.isfinite(values).all() or (values < -1e-6).any() or (values > 1 + 1e-6).any():
        raise ValueError('Gate must contain finite values in [0,1]')
    return {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
            'min': float(values.min()), 'max': float(values.max())}


def prediction_equivalence(actual, expected):
    """CUDA sparse reductions can differ slightly even between plain eval passes."""
    import torch
    differences = []
    for observed, reference in zip(actual, expected):
        if (not torch.allclose(observed, reference, rtol=1e-6, atol=1e-6)
                or not torch.equal(observed.argmax(-1), reference.argmax(-1))):
            raise AssertionError('Diagnostic probe changed probabilities beyond tolerance or class IDs')
        differences.append(float((observed - reference).abs().max()))
    return differences


def collect_partition(model, cube, labels, plan, stats, partition, dataset, wave, scale, class_ids,
                      focus_class=6, confused_as=2):
    """Use the strict forward entry; request diagnostic tensors without changing its math."""
    import torch
    from loadData.spatial_protocol import forward_partition
    from loadData.spatial_tiles import observe_tile, tile_slice
    from research_runtime import tensor_hash
    if partition not in ('val', 'test') or model.training:
        raise ValueError('Frozen eval model and val/test partition required')
    before = {key: tensor_hash(value) for key, value in model.state_dict().items()}
    predictions = {name: np.zeros(labels.shape, dtype=np.int16) for name in BRANCHES}
    selected = np.zeros(labels.shape, dtype=bool)
    probabilities, truths, global_ids, tile_ids = [], [], [], []
    gates = {key: [] for key in ('graph_to_cnn_gate', 'cnn_to_graph_gate_pixels')}
    tile_reports = []
    nll_sum = 0.0
    diagnostic_checked = False
    diagnostic_probe_differences = None
    def diagnostics_only(module, positional, keywords):
        return positional, {**keywords, 'return_diagnostics': True}
    hook = model.register_forward_pre_hook(diagnostics_only, with_kwargs=True)
    try:
        with torch.no_grad():
            for tile in plan['tiles']:
                if tile['partition'] != partition:
                    continue
                observation = observe_tile(cube, plan, stats, tile)
                outputs = forward_partition(model, dataset, observation, wave, scale)
                if not diagnostic_checked:
                    hook.remove()
                    plain = forward_partition(model, dataset, observation, wave, scale)
                    diagnostic_probe_differences = prediction_equivalence(outputs[:3], plain[:3])
                    hook = model.register_forward_pre_hook(diagnostics_only, with_kwargs=True)
                    diagnostic_checked = True
                ys, xs = tile_slice(tile)
                if selected[ys, xs].any():
                    raise AssertionError('Partition blocks overlap')
                selected[ys, xs] = True
                local = np.asarray(labels[ys, xs]).reshape(-1)
                valid = local > 0
                arrays = [output.detach().cpu().numpy() for output in outputs[:3]]
                if any(not np.isfinite(a).all() for a in arrays):
                    raise ValueError('Nonfinite branch probabilities')
                for name, array in zip(BRANCHES, arrays):
                    if not np.allclose(array.sum(1), 1, atol=1e-5) or (array < 0).any():
                        raise ValueError('Expected normalized per-head probabilities')
                    predictions[name][ys, xs] = array.argmax(-1).reshape(observation.spectral.shape[:2]) + 1
                truth = local[valid].astype(np.int64)
                if truth.size and ((truth < 1).any() or (truth > len(class_ids)).any()):
                    raise ValueError('Invalid evaluation label IDs')
                probs = np.stack([array[valid] for array in arrays], axis=1)
                nll_sum -= float(np.log(np.maximum(probs[np.arange(len(truth)), 0, truth - 1], 1e-8)).sum())
                probabilities.append(probs); truths.append(truth)
                global_ids.append(observation.global_ids.reshape(-1)[valid])
                tile_ids.append(np.full(truth.size, tile['id'], dtype=np.int32))
                info = outputs[3].get('interaction', {})
                for name in gates:
                    if name in info:
                        gate = info[name].detach().cpu().numpy().reshape(-1)
                        if gate.shape != local.shape:
                            raise ValueError('Gate pixels are not aligned with observations')
                        gates[name].append(gate[valid])
                local_predictions = {name: predictions[name][ys, xs].reshape(-1) for name in BRANCHES}
                summary = summarize(local, local_predictions, class_ids, focus_class, confused_as)
                tile_reports.append({'tile_id': tile['id'], 'box': tile['box'], 'count': int(valid.sum()),
                    'region_count': outputs[3].get('token_count'),
                    'focus_support': summary['target_confusion']['support'],
                    'focus_confusion_count': summary['target_confusion']['fused_confusion_count'],
                    'oa': {name: summary['classification'][name]['oa'] for name in BRANCHES},
                    'target_confusion': summary['target_confusion']})
    finally:
        hook.remove()
    if before != {key: tensor_hash(value) for key, value in model.state_dict().items()}:
        raise AssertionError('Frozen evaluation changed model state')
    if not truths:
        raise ValueError('Partition has no tiles')
    truth = np.concatenate(truths)
    if not truth.size:
        raise ValueError('Partition has no labeled pixels')
    score = np.concatenate(probabilities)
    payload = {'truth': truth, 'probabilities': score, 'global_ids': np.concatenate(global_ids),
               'tile_ids': np.concatenate(tile_ids)}
    summary = summarize(labels, predictions, class_ids, focus_class, confused_as, selected)
    summary['fused_nll'] = nll_sum / len(truth)
    summary['gate_descriptions'] = {}
    focus = truth == focus_class
    fused_wrong_as = focus & (score[:, 0].argmax(1) + 1 == confused_as)
    for name, pieces in gates.items():
        if not pieces:
            continue
        values = np.concatenate(pieces)
        if len(values) != len(truth):
            raise AssertionError('Missing gate values for some labeled pixels')
        payload[name] = values
        summary['gate_descriptions'][name] = {'all_labeled': describe(values), 'focus_class': describe(values[focus]),
            'focus_fused_as_confused': describe(values[fused_wrong_as]),
            'focus_fused_correct': describe(values[focus & (score[:, 0].argmax(1) + 1 == focus_class)])}
    summary.update(tile_count=len(tile_reports), diagnostic_flag_equivalent=True,
                   diagnostic_probe_max_abs_differences=diagnostic_probe_differences,
                   state_unchanged=True, gate_weighting='labeled pixels; cnn_to_graph region gates broadcast via Q',
                   gate_semantics='mode add yields constant ones; learned scalar gates are not calibrated reliability')
    return summary, payload, tile_reports


def audit_campaign(campaign_root, data_root, out, code, training_commit, focus_class, confused_as):
    import torch
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS, inspect_dataset
    from loadData.data_read import readWavelengths
    from research_runtime import hash_file, require_frozen_code, write_json
    from train_spatial import SpatialConfig, restore_spatial_checkpoint
    campaign_root = campaign_root.resolve(strict=True)
    if (campaign_root / '.campaign.lock').exists():
        raise ValueError('Training campaign still locked')
    campaign_file = campaign_root / 'campaign.json'
    campaign = json.loads(campaign_file.read_text())
    if campaign['commit'] != training_commit or campaign['order'] != ['none', 'add', 'gated']:
        raise ValueError('Unexpected training commit or campaign variants')
    # Only new diagnostic source may differ; freeze all model/data/training/evaluation implementations.
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', training_commit, code['commit'], '--', *CORE_PATHS],
                   check=True, capture_output=True, text=True)
    inputs = {str(campaign_file): hash_file(campaign_file)}
    initial = campaign_root / 'shared_initial_state.pt'
    if hash_file(initial) != campaign['initialization_file_sha256']:
        raise ValueError('Shared initialization file changed')
    inputs[str(initial)] = hash_file(initial)
    record, labels = inspect_dataset(data_root, 'paviaU')
    _, image_key, classes, _ = DATASETS['paviaU']
    cube = loadmat(data_root / record['image_path'], variable_names=[image_key])[image_key]
    wave, _ = readWavelengths('paviaU', cube.shape[2])
    result = {}
    for variant in campaign['order']:
        require_frozen_code(code)
        source = campaign_root / (variant + '-result.json')
        completion = json.loads(source.read_text())
        if completion['status'] != 'completed':
            raise ValueError('Variant not completed: ' + variant)
        inputs[str(source)] = hash_file(source)
        attempt = (campaign_root / 'training' / completion['run_id'] / 'attempts' / completion['attempt']).resolve(strict=True)
        if not attempt.is_relative_to(campaign_root):
            raise ValueError('Attempt path escapes campaign')
        manifest_file = attempt / 'manifest.json'
        manifest = json.loads(manifest_file.read_text())
        inputs[str(manifest_file)] = hash_file(manifest_file)
        if (manifest['status'] != 'completed' or manifest['code']['commit'] != training_commit
                or manifest['code']['source_hash'] != campaign['source_hash'] or manifest['code']['dirty']
                or manifest['prepared_data']['raw'] != record
                or manifest['initialization']['state_digest'] != campaign['shared_state_digest']):
            raise ValueError('Training provenance mismatch: ' + variant)
        for filename, expected in manifest['artifacts'].items():
            path = (attempt / filename).resolve(strict=True)
            if not path.is_relative_to(attempt) or hash_file(path) != expected:
                raise ValueError('Training artifact changed: ' + filename)
            inputs[str(path)] = expected
        config = SpatialConfig(json.loads((attempt / 'resolved_config.json').read_text()))
        if config.config_hash != campaign['configs'][variant]:
            raise ValueError('Resolved configuration mismatch')
        metrics = json.loads((attempt / 'metrics.json').read_text())
        if metrics['optimizer_steps'] != 10000 or metrics['selection'] != 'val_loss':
            raise ValueError('Unexpected completed budget or selection')
        selected_val = min(metrics['history'], key=lambda h: h['val']['nll'])
        model, plan, stats = restore_spatial_checkpoint(attempt / 'model.pt', attempt / 'protocol.json',
                                                       attempt / 'statistics.json', 'cuda')
        if (model.training_config['spatial_config'] != config.to_dict()
                or model.training_config['initialization'] != manifest['initialization']
                or model.training_config['run_id'] != completion['run_id']):
            raise ValueError('Checkpoint metadata mismatch')
        if plan['plan_hash'] != metrics['plan_hash'] or stats['statistics_hash'] != metrics['statistics_hash']:
            raise ValueError('Checkpoint protocol differs from recorded metrics')
        partitions = {}
        for partition, expected in [('val', selected_val['val']), ('test', metrics['test'])]:
            summary, payload, tiles = collect_partition(model, cube, labels, plan, stats, partition, 'paviaU',
                wave, config.research.data.pavia_superpixel_scale, list(range(1, classes + 1)), focus_class, confused_as)
            actual = summary['classification']['fused']
            if actual['confusion_matrix'] != expected['confusion'] or actual['count'] != expected['count']:
                raise AssertionError('Frozen fused confusion differs from completed run')
            for new, old in [('oa', 'oa'), ('aa', 'aa_supported'), ('kappa', 'kappa')]:
                if not np.isclose(actual[new], expected[old], rtol=0, atol=1e-12):
                    raise AssertionError('Frozen fused metric mismatch: ' + new)
            nll_difference = abs(summary['fused_nll'] - expected['nll'])
            if nll_difference > 1e-6:
                raise AssertionError('Frozen fused NLL mismatch')
            summary.update(fused_confusion_matches_training=True, fused_nll_abs_difference=nll_difference,
                           checkpoint=str(attempt / 'model.pt'), selected_step=selected_val['optimizer_steps'],
                           fusion={'type': 'fixed_feature_fusion_with_independent_heads',
                                   'cnn_weight': model.backbone.alph, 'graph_weight': 1-model.backbone.alph},
                           interaction_mode=variant)
            prefix = variant + '-' + partition
            np.savez_compressed(out / (prefix + '-predictions.npz'), **payload)
            write_json(out / (prefix + '-tiles.json'), tiles)
            write_json(out / (prefix + '-report.json'), summary)
            partitions[partition] = summary
            print(json.dumps({'variant': variant, 'partition': partition, 'status': 'passed',
                'oa': {name: summary['classification'][name]['oa'] for name in BRANCHES},
                'focus': summary['target_confusion']}), flush=True)
        result[variant] = partitions
        del model
    if inspect_dataset(data_root, 'paviaU')[0] != record:
        raise ValueError('Raw dataset changed during evaluation')
    if any(hash_file(Path(path)) != expected for path, expected in inputs.items()):
        raise ValueError('Frozen campaign input changed during evaluation')
    require_frozen_code(code)
    return {'variants': result, 'source_inputs': inputs, 'training_commit': training_commit,
            'training_source_hash': campaign['source_hash'], 'shared_initialization': campaign['shared_state_digest'],
            'frozen_inputs_unchanged': True, 'raw_dataset_unchanged': True, 'optimizer_steps': 0,
            'branch_order_in_npz': list(BRANCHES), 'scope': 'Post-hoc descriptive audit; no head, alpha or checkpoint selection'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--training-commit', required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--focus-class', type=int, default=6)
    parser.add_argument('--confused-as', type=int, default=2)
    args = parser.parse_args()
    if sys.platform != 'linux' or not Path('/.dockerenv').exists():
        parser.error('Use ssh rmvision in the training container')
    if sys.flags.optimize:
        parser.error('Assertions must be enabled')
    import torch
    from research_runtime import code_snapshot, external_directory, write_json
    code = code_snapshot()
    if (any(not re.fullmatch('[0-9a-f]{40}', sha) for sha in (args.expected_commit, args.training_commit))
            or code['commit'] != args.expected_commit or code['dirty']):
        parser.error('Require a clean checkout and matching full SHA')
    if not torch.cuda.is_available():
        parser.error('CUDA required for the configured real-data audit')
    torch.set_num_threads(1)
    out = external_directory(args.out_dir)
    source = args.campaign_root.resolve(strict=True)
    if out == source or source in out.parents or out in source.parents:
        parser.error('Keep diagnostics separate from the training campaign')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'commit': code['commit'], 'branch': code['branch'],
                'source_hash': code['source_hash'], 'command': [sys.executable, *sys.argv],
                'gpu': torch.cuda.get_device_name(0), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'manifest.json', manifest)
    try:
        result = audit_campaign(source, args.data_root.resolve(strict=True), out, code, args.training_commit,
                                args.focus_class, args.confused_as)
        write_json(out / 'branch-audit.json', result)
        manifest.update(status='passed', optimizer_steps=0, frozen_input_count=len(result['source_inputs']))
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
