"""Descriptive E0-B2 analysis of frozen M4-J2 packets. No training or inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.spatial_prediction import spatial_prediction_report, paired_difference
from loadData.dataset_registry import inspect_dataset, file_hash
from research_runtime import code_snapshot, require_frozen_code, external_directory, write_json
from scripts.audit_spatial_support import split_masks

SEEDS = (64, 512, 2048)
MODES = ('none', 'capacity_control', 'x_residual')
NAMES = ('paviaU', 'xuzhou', 'whu-honghu')
POLICIES = ('within_split_only', 'whole_annotation_posthoc_only')


def load(path):
    return json.loads(Path(path).read_text())


def packet_arrays(packet, labels, split_indices):
    indices = packet['pixel_indices'].cpu().numpy()
    expected = np.asarray(split_indices)
    if not np.array_equal(indices, expected):
        raise ValueError('Packet order differs from frozen original split')
    truth = packet['labels'].cpu().numpy()
    if not np.array_equal(truth, labels.ravel()[expected]):
        raise ValueError('Packet labels differ from raw labels')
    order = np.argsort(indices)
    heads = {h: packet['heads'][h]['prediction'].cpu().numpy()[order]
             for h in ('cnn', 'graph', 'fused')}
    return indices[order], truth[order], heads


def run(args):
    source = code_snapshot(ROOT)
    if source['dirty'] or source['commit'] != args.expected_commit:
        raise RuntimeError('Require expected clean source SHA')
    out = external_directory(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    original = load(args.analysis_dir / 'result.json')
    frozen = load(args.analysis_dir / 'frozen-before-test.json')
    support = load(args.support_json)
    if original['status'] != 'completed' or support['status'] != 'completed':
        raise ValueError('Historical evaluation/support incomplete')
    hashes = dict(frozen['input_files'])
    model_count = 0
    for case in frozen['models'].values():
        for name, sha in case['files'].items():
            hashes[str(Path(case['folder']) / name)] = sha
            model_count += 1
    for name, sha in original['artifacts'].items():
        hashes[str(args.analysis_dir / name)] = sha
    for path in (args.analysis_dir / 'result.json', args.analysis_dir / 'frozen-before-test.json', args.support_json):
        hashes[str(path)] = file_hash(path)

    def unchanged():
        for path, sha in hashes.items():
            if file_hash(path) != sha:
                raise RuntimeError('Frozen input/artifact changed: ' + path)
    unchanged()
    write_json(out / 'frozen.json', {'code': source, 'files': hashes, 'command': sys.argv})
    scenes = {}
    for name in NAMES:
        record, labels = inspect_dataset(args.data_root, name)
        path = args.prepared_dir / (name + '-split.json')
        raw = args.data_root / record['label_path']
        if str(path) not in hashes or hashes.get(str(raw)) != record['label_sha256']:
            raise ValueError('Data/split not bound by M4 frozen inputs')
        split = load(path)
        masks = split_masks(labels, split)
        classes = list(range(1, record['class_count'] + 1))
        counts = np.bincount(labels.ravel()[split['indices']['train']].astype(int),
                             minlength=len(classes) + 1)[1:].tolist()
        scenes[name] = (labels, split, masks, classes, counts)
    rows = {}
    full_checks = 0
    for mode in MODES:
        for seed in SEEDS:
            key = f'{mode}-{seed}'
            packet = torch.load(args.analysis_dir / (key + '-predictions.pt'), map_location='cpu', weights_only=True)
            old = load(args.analysis_dir / (key + '-analysis.json'))
            if old['variant'] != mode or old['seed'] != seed:
                raise ValueError('Historical case identity mismatch')
            rows[key] = {}
            for name, (labels, split, masks, classes, _) in scenes.items():
                indices, truth, heads = packet_arrays(packet[name]['test'], labels, split['indices']['test'])
                rows[key][name] = {}
                for policy in POLICIES:
                    geometry = masks['test'] if policy == 'within_split_only' else labels > 0
                    row = spatial_prediction_report(labels, classes, pixel_indices=indices, truth=truth,
                                                    heads=heads, geometry_mask=geometry)
                    if row['support'] != support['datasets'][name]['roles']['test'][policy]:
                        raise ValueError('Support differs from E0-B1 frozen audit')
                    for h in heads:
                        current, previous = row['whole_score']['heads'][h], old['datasets'][name]['test']['heads'][h]
                        if current['confusion_matrix'] != previous['confusion_matrix']:
                            raise ValueError('Whole-test confusion differs from frozen evaluation')
                        for metric in ('oa', 'aa', 'macro_f1', 'kappa'):
                            if abs(current[metric] - previous[metric]) > 1e-12:
                                raise ValueError('Historical aggregate mismatch')
                        full_checks += 1
                    rows[key][name][policy] = row
    contrasts = []
    for name in NAMES:
        for policy in POLICIES:
            for width in ('1', '2', '3'):
                for stratum in ('boundary', 'observed_nonboundary'):
                    for head in ('fused', 'cnn', 'graph'):
                        for metric in ('oa', 'aa', 'macro_f1'):
                            values = {mode: {s: rows[f'{mode}-{s}'][name][policy]['widths'][width][stratum]['heads'][head][metric]
                                             for s in SEEDS} for mode in MODES}
                            for ref in MODES[:2]:
                                contrasts.append({'dataset': name, 'policy': policy, 'width': int(width),
                                                  'stratum': stratum, 'head': head, 'metric': metric,
                                                  'reference': ref, 'candidate': 'x_residual',
                                                  **paired_difference(values['x_residual'], values[ref])})
    class_changes = []
    for name, (_, _, _, classes, counts) in scenes.items():
        for c, count in zip(classes, counts):
            for head in ('fused', 'cnn', 'graph'):
                values = {mode: {s: rows[f'{mode}-{s}'][name][POLICIES[0]]['whole_score']['heads'][head]['per_class'][c-1]['recall']
                                 for s in SEEDS} for mode in MODES}
                for ref in MODES[:2]:
                    class_changes.append({'dataset': name, 'class_id': c, 'train_support': count,
                                          'small_by_train_support_le25': count <= 25, 'head': head, 'reference': ref,
                                          **paired_difference(values['x_residual'], values[ref])})
    unchanged()
    require_frozen_code(source)
    result = {'status': 'completed', 'code': source, 'rows': rows, 'contrasts': contrasts,
              'class_changes': class_changes, 'historical_head_checks': full_checks,
              'training_files_unchanged': model_count, 'protected_files_unchanged': len(hashes),
              'input_files_unchanged': len(frozen['input_files']), 'new_training_updates': 0,
              'new_model_inference': 0, 'scope': 'posthoc descriptive, reused joint split, no selection or significance claim'}
    write_json(out / 'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'contrasts', 'class_changes', 'code')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis-dir', 'prepared-dir', 'data-root', 'support-json', 'out-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-commit', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
