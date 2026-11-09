"""Independent raw-packet, scipy-geometry, confusion and paired arithmetic audit."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from loadData.dataset_registry import inspect_dataset, file_hash


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10), (a, b)


def check_metrics(gt, pred, c, report):
    matrix = np.zeros((c, c), dtype=np.int64)
    np.add.at(matrix, (gt.astype(int)-1, pred.astype(int)-1), 1)
    assert matrix.tolist() == report['confusion_matrix']
    support, predicted = matrix.sum(1), matrix.sum(0)
    diagonal = matrix.diagonal()
    recalls, f1s = [], []
    for i, row in enumerate(report['per_class']):
        assert row['class_id'] == i+1 and row['support'] == support[i] and row['predicted_count'] == predicted[i]
        recall = diagonal[i]/support[i] if support[i] else None
        precision = diagonal[i]/predicted[i] if predicted[i] else None
        f1 = 2*diagonal[i]/(support[i]+predicted[i]) if support[i] else None
        close(recall, row['recall']); close(precision, row['precision']); close(f1, row['f1'])
        if support[i]: recalls.append(recall); f1s.append(f1)
    oa = float(np.mean(gt == pred)) if len(gt) else None
    expected = sum(float(a)*float(b) for a, b in zip(support, predicted))/len(gt)**2 if len(gt) else None
    kappa = (oa-expected)/(1-expected) if len(gt) and abs(1-expected) > 1e-15 else None
    close(oa, report['oa']); close(kappa, report['kappa'])
    close(statistics.mean(recalls) if recalls else None, report['aa'])
    close(statistics.mean(f1s) if f1s else None, report['macro_f1'])
    assert report['count'] == len(gt)
    assert report['macro_class_ids'] == (np.flatnonzero(support)+1).tolist()


def check_branches(gt, predictions, row):
    a, b, f = [predictions[h] == gt for h in ('cnn', 'graph', 'fused')]
    assert row['count'] == len(gt)
    for name, selected in [('both_correct', a & b), ('cnn_only_correct', a & ~b),
                           ('graph_only_correct', ~a & b), ('both_wrong', ~a & ~b)]:
        count, correct = int(selected.sum()), int(f[selected].sum())
        assert row['strata'][name]['count'] == count and row['strata'][name]['fused_correct'] == correct
        close(correct/count if count else None, row['strata'][name]['fused_accuracy'])
    for key, select, success in [('correction', a ^ b, f), ('harm', a & b, ~f)]:
        n, d = int((select & success).sum()), int(select.sum())
        assert (row[key]['numerator'], row[key]['denominator']) == (n, d)
        close(n/d if d else None, row[key]['rate'])
    for key, v in [('cnn_accuracy', a), ('graph_accuracy', b), ('fused_accuracy', f), ('branch_oracle_accuracy', a | b)]:
        close(float(v.mean()) if len(gt) else None, row[key])
    assert row['prediction_disagreement_count'] == int((predictions['cnn'] != predictions['graph']).sum())


def main():
    parser = argparse.ArgumentParser()
    for key in ('results', 'frozen', 'analysis-dir', 'prepared-dir', 'data-root', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    frozen = json.loads(args.frozen.read_text())
    assert report['status'] == 'completed' and report['code'] == frozen['code']
    assert all(file_hash(p) == h for p, h in frozen['files'].items())
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    metrics, branches = 0, 0
    for name in ('paviaU', 'xuzhou', 'whu-honghu'):
        record, labels = inspect_dataset(args.data_root, name)
        c = record['class_count']
        split = json.loads((args.prepared_dir/(name+'-split.json')).read_text())['indices']
        indices = np.array(split['test']); truth = labels.ravel()[indices]
        selected = np.isin(np.arange(labels.size), indices).reshape(labels.shape)
        masks = {}
        for policy in ('within_split_only', 'whole_annotation_posthoc_only'):
            valid = selected if policy == 'within_split_only' else labels > 0
            low = ndimage.minimum_filter(np.where(valid, labels, c+1), footprint=footprint, mode='constant', cval=c+1)
            high = ndimage.maximum_filter(np.where(valid, labels, 0), footprint=footprint, mode='constant', cval=0)
            band = valid & ((low < labels) | (high > labels))
            masks[policy] = {}
            for width in ('1', '2', '3'):
                masks[policy][width] = band.ravel()[indices].copy()
                band = ndimage.binary_dilation(band, structure=footprint, mask=valid)
        for key, case in report['rows'].items():
            packet = torch.load(args.analysis_dir/(key+'-predictions.pt'), map_location='cpu', weights_only=True)[name]['test']
            assert np.array_equal(packet['pixel_indices'].numpy(), indices)
            assert np.array_equal(packet['labels'].numpy(), truth)
            predictions = {h: packet['heads'][h]['prediction'].numpy() for h in ('cnn', 'graph', 'fused')}
            for policy, row in case[name].items():
                selections = [(np.ones(len(indices), bool), row['whole_score'])]
                for width, band in masks[policy].items():
                    selections.extend([(band, row['widths'][width]['boundary']),
                                       (~band, row['widths'][width]['observed_nonboundary'])])
                for mask, scores in selections:
                    for h, pred in predictions.items():
                        check_metrics(truth[mask], pred[mask], c, scores['heads'][h]); metrics += 1
                    check_branches(truth[mask], {h: p[mask] for h, p in predictions.items()}, scores['branches'])
                    branches += 1
    for row in report['contrasts'] + report['class_changes']:
        values = []
        for seed in (64, 512, 2048):
            pair = []
            for mode in ('x_residual', row['reference']):
                case = report['rows'][f'{mode}-{seed}'][row['dataset']]
                if 'class_id' in row:
                    score = case['within_split_only']['whole_score']['heads'][row['head']]['per_class'][row['class_id']-1]['recall']
                else:
                    score = case[row['policy']]['widths'][str(row['width'])][row['stratum']]['heads'][row['head']][row['metric']]
                pair.append(score)
            values.append((pair[0]-pair[1])*100 if pair[0] is not None and pair[1] is not None else None)
        for a,b in zip(values,row['values_pp']): close(a,b)
        close(statistics.mean(values) if values[0] is not None else None, row['mean_pp'])
        close(statistics.stdev(values) if values[0] is not None else None, row['sample_sd_pp'])
    assert all(file_hash(p) == h for p, h in frozen['files'].items())
    receipt = {'status':'passed', 'audited_commit':report['code']['commit'], 'metric_cases':metrics,
               'branch_cases':branches, 'contrast_cases':len(report['contrasts']),
               'class_contrasts':len(report['class_changes']), 'protected_files_unchanged':len(frozen['files']),
               'results_sha256':file_hash(args.results), 'verifier_sha256':file_hash(__file__), 'new_updates':0}
    with args.output.open('x') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
