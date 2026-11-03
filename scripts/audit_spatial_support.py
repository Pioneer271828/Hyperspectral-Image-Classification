"""Read-only support audit of a frozen joint-study launch; no model execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.classification_metrics import integer_values
from evaluation.spatial_diagnostics import spatial_support
from loadData.dataset_registry import inspect_dataset, file_hash
from loadData.fixed_splits import label_hash
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def split_masks(labels, split):
    if list(labels.shape) != split['shape']:
        raise ValueError('Split shape differs from labels')
    if label_hash(labels) != split['label_hash']:
        raise ValueError('Split label identity differs from raw labels')
    seen = np.zeros(labels.size, dtype=bool)
    masks = {}
    for role in ('train', 'val', 'test'):
        ids = integer_values(split['indices'][role], 'Split indices')
        if ids.ndim != 1 or (ids >= labels.size).any() or len(np.unique(ids)) != len(ids):
            raise ValueError('Invalid or duplicate split indices')
        if seen[ids].any() or (labels.ravel()[ids] == 0).any():
            raise ValueError('Split overlap or unlabeled supervision')
        seen[ids] = True
        mask = np.zeros(labels.size, dtype=bool)
        mask[ids] = True
        masks[role] = mask.reshape(labels.shape)
    if not np.array_equal(seen, labels.ravel() > 0):
        raise ValueError('Split must partition all labeled foreground')
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    source = code_snapshot(ROOT)
    if source['dirty'] or source['commit'] != args.expected_commit:
        raise RuntimeError('Expected clean exact commit before evaluation')
    out = external_directory(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    launch_hash = file_hash(args.launch)
    launch = json.loads(args.launch.read_text())
    frozen = dict(launch['inputs'])
    if not frozen or any(file_hash(p) != h for p, h in frozen.items()):
        raise RuntimeError('Frozen launch input mismatch')
    rows = {}
    for name in ('paviaU', 'xuzhou', 'whu-honghu'):
        path = Path(launch['prepared']) / (name + '-split.json')
        if str(path) not in frozen:
            raise ValueError('Split is not bound by the historical launch')
        record, labels = inspect_dataset(launch['data_root'], name)
        label_path = str(Path(launch['data_root']) / record['label_path'])
        # Some historical launches bind raw labels through prepared provenance,
        # so also record their actual bytes explicitly for this read-only audit.
        if label_path in frozen and frozen[label_path] != record['label_sha256']:
            raise RuntimeError('Raw labels changed after initial identity check')
        frozen.setdefault(label_path, record['label_sha256'])
        split = json.loads(path.read_text())
        masks = split_masks(labels, split)
        ids = list(range(1, record['class_count'] + 1))
        rows[name] = {'shape': list(labels.shape), 'labels_sha256': record['label_sha256'],
                      'split_sha256': file_hash(path), 'roles': {}}
        for role, score in masks.items():
            rows[name]['roles'][role] = {
                'within_split_only': spatial_support(labels, ids, geometry_mask=score, score_mask=score),
                'whole_annotation_posthoc_only': spatial_support(
                    labels, ids, geometry_mask=labels > 0, score_mask=score)}
    require_frozen_code(source)
    if file_hash(args.launch) != launch_hash or any(file_hash(p) != h for p, h in frozen.items()):
        raise RuntimeError('Inputs changed during read-only audit')
    result = {'status': 'completed', 'code': source, 'command': sys.argv,
              'launch_sha256': launch_hash, 'input_hashes': frozen, 'datasets': rows,
              'new_training_updates': 0, 'prediction_evaluations': 0,
              'use': 'Frozen descriptive support only; no model selection or training from test geometry.',
              'not_established': ['M4 boundary accuracy benefit', 'independent split generalization',
                                  'physical semantic boundaries through unlabeled areas']}
    write_json(out / 'support.json', result)
    print(json.dumps({'status': result['status'], 'output': str(out / 'support.json')}))


if __name__ == '__main__':
    main()
