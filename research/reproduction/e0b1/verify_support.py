"""Independent scipy morphology/count audit; execute only in the training container."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from loadData.dataset_registry import inspect_dataset, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--support', type=Path, required=True)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.support.read_text())
    launch = json.loads(args.launch.read_text())
    assert report['status'] == 'completed'
    assert file_hash(args.launch) == report['launch_sha256']
    assert all(file_hash(p) == h for p, h in report['input_hashes'].items())
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    comparisons = 0
    summary = {}
    for name, scene in report['datasets'].items():
        record, labels = inspect_dataset(launch['data_root'], name)
        assert record['label_sha256'] == scene['labels_sha256']
        split_path = Path(launch['prepared']) / (name + '-split.json')
        assert file_hash(split_path) == scene['split_sha256']
        split = json.loads(split_path.read_text())['indices']
        classes = np.arange(1, record['class_count'] + 1)
        summary[name] = {}
        for role, protocols in scene['roles'].items():
            score = np.isin(np.arange(labels.size), split[role]).reshape(labels.shape)
            summary[name][role] = {}
            for protocol, observed in protocols.items():
                valid = ((labels > 0) & score if protocol == 'within_split_only' else labels > 0)
                low = ndimage.minimum_filter(np.where(valid, labels, classes[-1] + 1),
                                             footprint=footprint, mode='constant', cval=classes[-1] + 1)
                high = ndimage.maximum_filter(np.where(valid, labels, 0),
                                              footprint=footprint, mode='constant', cval=0)
                current = valid & ((low < labels) | (high > labels))
                summary[name][role][protocol] = {}
                for width in (1, 2, 3):
                    row = observed['widths'][str(width)]
                    for key, select in [('boundary', current & score),
                                        ('observed_nonboundary', valid & score & ~current)]:
                        counts = [int(np.count_nonzero(select & (labels == c))) for c in classes]
                        assert counts == row[key]['per_class_count']
                        assert sum(counts) == row[key]['count']
                        assert row[key]['present_class_ids'] == classes[np.array(counts) > 0].tolist()
                        comparisons += 1
                    assert row['boundary']['count'] + row['observed_nonboundary']['count'] == len(split[role])
                    summary[name][role][protocol][str(width)] = {
                        'boundary_count': row['boundary']['count'],
                        'class_count': len(row['boundary']['present_class_ids']),
                        'missing_classes': row['boundary']['missing_class_ids'],
                        'score_count': len(split[role])}
                    current = ndimage.binary_dilation(current, structure=footprint, mask=valid)
    assert all(file_hash(p) == h for p, h in report['input_hashes'].items())
    result = {'status': 'passed', 'support_sha256': file_hash(args.support),
              'verifier_sha256': file_hash(__file__), 'audited_commit': report['code']['commit'],
              'histogram_comparisons': comparisons, 'inputs_unchanged': len(report['input_hashes']),
              'new_training_updates': 0, 'datasets': summary}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
