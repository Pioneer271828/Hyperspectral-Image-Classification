"""Evaluate frozen prediction maps; no model construction or training-data loader."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from evaluation.classification_metrics import class_order, evaluation_mask, classification_metrics
from evaluation.spatial_metrics import spatial_metrics
from evaluation.branch_diagnostics import branch_diagnostics
from evaluation.region_diagnostics import graph_diagnostics

ROOT = Path(__file__).resolve().parent
PREDICTION_KEYS = ('prediction', 'cnn_prediction', 'graph_prediction')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_predictions(labels, evaluation_selection, arrays, class_ids, small_class_ids=None, graph_node_count=None):
    class_ids = class_order(class_ids)
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('Export requires a 2D label map')
    valid = evaluation_mask(labels, evaluation_selection)
    for key in PREDICTION_KEYS:
        if key not in arrays:
            raise ValueError('Missing frozen prediction map: ' + key)
    report = {'schema_version': 1, 'diagnostic_scope': 'frozen_predictions_only',
              'class_ids': class_ids, 'array_shape': list(labels.shape),
              'selected_labeled_pixels': int(valid.sum()),
              'classification': {key: classification_metrics(labels, arrays[key], class_ids, valid) for key in PREDICTION_KEYS},
              'spatial': {key: spatial_metrics(labels, arrays[key], class_ids, valid) for key in PREDICTION_KEYS},
              'branches': branch_diagnostics(labels, arrays['cnn_prediction'], arrays['graph_prediction'],
                                               arrays['prediction'], class_ids, valid)}
    if small_class_ids is None:
        report['small_classes'] = {'available': False, 'reason': 'No predeclared small-class IDs supplied'}
    else:
        selected_ids = class_order(small_class_ids)
        if not set(selected_ids) <= set(class_ids):
            raise ValueError('Small-class IDs must be a subset of declared classes')
        selected = valid & np.isin(labels, selected_ids)
        report['small_classes'] = {'available': True, 'class_ids': selected_ids,
                                   'metrics': {key: classification_metrics(labels, arrays[key], class_ids, selected) for key in PREDICTION_KEYS}}
    edge_keys = {'edge_indices', 'edge_weights_before', 'edge_weights_after'}
    supplied = edge_keys.intersection(arrays)
    if supplied:
        if not {'edge_indices', 'edge_weights_before'} <= supplied or graph_node_count is None:
            raise ValueError('Edge diagnostics need indices, before weights and explicit graph_node_count')
        report['graph'] = graph_diagnostics(graph_node_count, arrays['edge_indices'], arrays['edge_weights_before'],
                                           arrays.get('edge_weights_after'))
    else:
        if graph_node_count is not None:
            raise ValueError('graph_node_count supplied without sparse edge arrays')
        report['graph'] = None
    return report


def export_frozen_predictions(arrays_path, labels_path, mask_path, checkpoint_path, class_ids, out_dir, *,
                              prediction_commit, dataset, protocol_id, partition, exporter_commit,
                              small_class_ids=None, graph_node_count=None):
    if not re.fullmatch(r'[0-9a-f]{40}', prediction_commit) or not re.fullmatch(r'[0-9a-f]{40}', exporter_commit):
        raise ValueError('Prediction/exporter commits must be full lowercase SHAs')
    if partition not in ('val', 'test') or not all(isinstance(v, str) and v.strip() for v in (dataset, protocol_id)):
        raise ValueError('Dataset, protocol ID and val/test partition required')
    out = Path(out_dir).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Write diagnostics outside checkout')
    if out.exists():
        raise FileExistsError('Refusing to overwrite diagnostic output directory')
    paths = {'prediction_arrays': Path(arrays_path).resolve(strict=True), 'labels': Path(labels_path).resolve(strict=True),
             'evaluation_mask': Path(mask_path).resolve(strict=True), 'checkpoint': Path(checkpoint_path).resolve(strict=True)}
    before = {key: file_hash(path) for key, path in paths.items()}
    with np.load(paths['prediction_arrays'], allow_pickle=False) as archive:
        wanted = {*PREDICTION_KEYS, 'edge_indices', 'edge_weights_before', 'edge_weights_after'}
        arrays = {key: archive[key] for key in wanted if key in archive.files}
    labels = np.load(paths['labels'], allow_pickle=False)
    selected = np.load(paths['evaluation_mask'], allow_pickle=False)
    report = evaluate_predictions(labels, selected, arrays, class_ids, small_class_ids, graph_node_count)
    after = {key: file_hash(path) for key, path in paths.items()}
    if before != after:
        raise RuntimeError('Frozen input artifact changed during evaluation')
    report['provenance'] = {'dataset': dataset, 'protocol_id': protocol_id, 'partition': partition,
                            'prediction_commit': prediction_commit, 'exporter_commit': exporter_commit,
                            'inputs': {key: {'path': str(path), 'sha256': before[key]} for key, path in paths.items()},
                            'selection_use': 'diagnostic report; no automatic model selection'}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'research_diagnostics.json').write_text(encoded, encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', required=True, help='Frozen NPZ maps with 1-based declared class IDs')
    parser.add_argument('--labels', required=True, help='2D NPY labels; 0 means unlabeled')
    parser.add_argument('--mask', required=True, help='2D boolean NPY evaluation selection')
    parser.add_argument('--checkpoint', required=True, help='Actual frozen checkpoint, hashed without loading')
    parser.add_argument('--class-ids', required=True, help='Comma-separated IDs in fixed reporting order')
    parser.add_argument('--small-class-ids', help='Predeclared class IDs; never chosen by test performance')
    parser.add_argument('--graph-node-count', type=int)
    parser.add_argument('--prediction-commit', required=True)
    parser.add_argument('--expected-commit', required=True, help='Frozen current exporter checkout SHA')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--protocol-id', required=True)
    parser.add_argument('--partition', required=True, choices=('val', 'test'))
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    def git(*options):
        return subprocess.check_output(['git', '-C', str(ROOT), *options], text=True).strip()
    if git('rev-parse', 'HEAD') != args.expected_commit or git('status', '--porcelain', '--untracked-files=all'):
        parser.error('Exporter checkout SHA must match and worktree must be clean')
    try:
        ids = [int(value.strip()) for value in args.class_ids.split(',')]
        small = [int(value.strip()) for value in args.small_class_ids.split(',')] if args.small_class_ids else None
        report = export_frozen_predictions(args.arrays, args.labels, args.mask, args.checkpoint, ids, args.out_dir,
            prediction_commit=args.prediction_commit, dataset=args.dataset, protocol_id=args.protocol_id,
            partition=args.partition, exporter_commit=args.expected_commit,
            small_class_ids=small, graph_node_count=args.graph_node_count)
        if git('rev-parse', 'HEAD') != args.expected_commit or git('status', '--porcelain', '--untracked-files=all'):
            raise RuntimeError('Exporter source changed during evaluation')
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({'count': report['selected_labeled_pixels'], 'report': str(Path(args.out_dir) / 'research_diagnostics.json')}))


if __name__ == '__main__': main()
