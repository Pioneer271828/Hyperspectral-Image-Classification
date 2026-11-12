"""Score complete, immutable WP09 predictions against a separate development label release."""
import argparse
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np

from evaluate_tta import HEADS
from evaluation.classification_metrics import classification_metrics, class_indices, integer_values
from research_runtime import external_directory, hash_file
from transfer.online_tta_run import committed_history, read_predictions, validate_predictions, _json
from transfer.online_tta_stream import ORDERS, StreamRegistration, fields_are, parse_json, read_pin, require_hash


def _packet(reg, frame, path, hashes):
    probs = read_predictions(path)
    inputs = reg.observation(frame['observation'])[0]
    validate_predictions(probs, inputs.shape[0] * inputs.shape[1], len(reg.value['class_ids']))
    if reg.fingerprints(probs) != hashes:
        raise ValueError('Three-head prediction identity differs')
    return probs


def _frozen_run(item):
    fields_are(item, 'registration directory replay')
    reg = StreamRegistration(item['registration'])
    root = external_directory(item['directory'])
    events, tip = committed_history(reg, root, complete=True)
    predictions = {
        row['frame']['frame_id']: _packet(reg, row['frame'], root / f'frame-{index:06d}' / 'prediction.npz',
                                          row['predictions']['heads'])
        for index, row in enumerate(events, start=1)
    }
    replay_pin = item['replay']
    manifest = parse_json(read_pin(replay_pin))
    fields_are(manifest, 'version kind registration tip checkpoint rows predictions optimizer_updates statistics_updates')
    if (manifest['version'] != 1 or manifest['kind'] != 'initial_final_anchor_replay'
            or manifest['registration'] != reg.pin['sha256'] or manifest['tip'] != tip
            or manifest['checkpoint'] != events[-1]['checkpoint']
            or manifest['optimizer_updates'] != 0 or manifest['statistics_updates'] != 0):
        raise ValueError('Replay does not belong to completed run or included updates')
    anchors = {}
    for frame in reg.value['frames']:
        anchors.setdefault(frame['observation'], frame)
    if (len(manifest['rows']) != len(anchors) or manifest['predictions'] != 2 * len(anchors)):
        raise ValueError('Incomplete diagnostic replay')
    root = Path(replay_pin['path']).parent
    replay = dict(initial={}, final={})
    for index, (row, frame) in enumerate(zip(manifest['rows'], anchors.values(), strict=True), start=1):
        fields_are(row, 'frame observation_sha256 initial final')
        if row['frame'] != frame or row['observation_sha256'] != reg.observation_identity(frame):
            raise ValueError('Replay anchor identity differs')
        for stage in replay:
            packet = row[stage]
            fields_are(packet, 'name sha256 heads')
            path = root / f'{stage}-{index:06d}.npz'
            if path.name != packet['name'] or path.is_symlink() or hash_file(path) != packet['sha256']:
                raise ValueError('Replay packet identity differs')
            replay[stage][frame['frame_id']] = _packet(reg, frame, path, packet['heads'])
    return reg, predictions, replay


def _labels(pin, reg):
    release = parse_json(read_pin(pin))
    fields_are(release, 'kind population class_ids release_receipt_sha256 training_class_counts frames')
    require_hash(release['release_receipt_sha256'])
    expected = 'synthetic' if reg.provenance['kind'] == 'synthetic' else 'registered_development'
    if (release['kind'] != expected or release['population'] != reg.population
            or release['class_ids'] != reg.value['class_ids']):
        raise ValueError('Label release and prediction population differ')
    counts = release['training_class_counts']
    if (not isinstance(counts, list) or len(counts) != len(release['class_ids'])
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError('Small classes require preregistered training support counts')
    frames = {f['frame_id']: f for f in reg.frames}
    labels = {}
    for row in release['frames']:
        fields_are(row, 'frame_id observation_sha256 labels')
        frame_id = row['frame_id']
        if (frame_id not in frames or frame_id in labels
                or row['observation_sha256'] != reg.observation_identity(frames[frame_id])):
            raise ValueError('Label frame ID or observation pin differs')
        data = np.load(BytesIO(read_pin(row['labels'])), allow_pickle=False)
        shape = reg.observation(frames[frame_id]['observation'])[0].shape[:2]
        if not isinstance(data, np.ndarray) or data.shape != shape or data.dtype.kind not in 'iu':
            raise ValueError('Released labels must be a single integer image of registered geometry')
        truth = integer_values(data, 'Released labels').ravel()
        class_indices(truth[truth > 0], release['class_ids'], 'Released labels')
        labels[frame_id] = truth
    if labels.keys() != frames.keys():
        raise ValueError('Label release omits registered frame identities')
    small = [c for c, n in zip(release['class_ids'], counts, strict=True) if n <= 25]
    return labels, small


def _metrics(predictions, labels, classes):
    ordered = sorted(predictions)
    truth = np.concatenate([labels[key] for key in ordered])
    keep = truth > 0
    class_columns = class_indices(truth[keep], classes, 'Labels')
    scores = {}
    for head in HEADS:
        probability = np.concatenate([predictions[key][head] for key in ordered])
        metrics = classification_metrics(truth, np.asarray(classes)[probability.argmax(1)], classes)
        nll = -np.log(probability[keep][np.arange(len(class_columns)), class_columns].astype(float).clip(1e-8, 1))
        scores[head] = dict(metrics, nll=float(nll.mean()) if len(nll) else None)
    return scores


def _paired(current, baseline, small):
    result = {}
    for head in HEADS:
        now, old = current[head], baseline[head]
        deltas = {}
        for row, base in zip(now['per_class'], old['per_class'], strict=True):
            class_id = row['class_id']
            if row['support'] != base['support']:
                raise ValueError('Paired recall support differs')
            position = now['class_ids'].index(class_id)
            diff = now['confusion_matrix'][position][position] - old['confusion_matrix'][position][position]
            deltas[class_id] = Fraction(diff, row['support']) if row['support'] else None
        supported = [x for x in deltas.values() if x is not None]
        gates = {str(k): None if v is None else v >= Fraction(-2, 100) for k, v in deltas.items()}
        result[head] = dict(
            metrics={k: now[k] - old[k] if now[k] is not None and old[k] is not None else None
                     for k in ('oa', 'aa', 'kappa', 'macro_f1', 'nll')},
            recall={str(k): float(v) if v is not None else None for k, v in deltas.items()},
            worst_recall_drop=float(max(0, -min(supported))) if supported else None,
            recall_protected_at_2pp=gates, small_class_protected_at_2pp={str(k): gates[str(k)] for k in small})
    return result


def score(plan):
    """Open label inputs only after every registered run and anchor packet has been verified."""
    request = parse_json(read_pin(plan))
    fields_are(request, 'version label_release runs')
    if request['version'] != 1 or not isinstance(request['runs'], list) or not request['runs']:
        raise ValueError('Explicit offline scoring runs required')
    runs = {}
    reference = None
    for item in request['runs']:
        reg, predictions, replay = _frozen_run(item)
        order = reg.value['order']
        key = (reg.settings['mode'], reg.settings['reset'], order)
        identity = (reg.population, reg.value['source_hash'], reg.value['source'],
                    reg.value['provenance'], reg.value['class_ids'],
                    {k: v for k, v in reg.settings.items() if k not in ('mode', 'reset')})
        if key in runs or (reference is not None and identity != reference):
            raise ValueError('Duplicate run or incompatible paired population/source')
        reference = identity
        runs[key] = (reg, predictions, replay)
    for mode, reset, order in runs:
        if ('frozen', reset, order) not in runs or any((mode, reset, variant) not in runs for variant in ORDERS):
            raise ValueError('Each comparison needs all registered orders and matching frozen controls')
    labels, small = _labels(request['label_release'], next(iter(runs.values()))[0])
    online, forgetting = {}, {}
    for key, (reg, predictions, replay) in runs.items():
        classes = reg.value['class_ids']
        online[key] = _metrics(predictions, labels, classes)
        initial = _metrics(replay['initial'], labels, classes)
        final = _metrics(replay['final'], labels, classes)
        forgetting['/'.join(key)] = dict(initial=initial, final=final, final_minus_initial=_paired(final, initial, small))
    result = {}
    for (mode, reset, order), metrics in online.items():
        result[f'{mode}/{reset}/{order}'] = dict(metrics=metrics,
            versus_frozen=_paired(metrics, online[('frozen', reset, order)], small),
            versus_original_order=_paired(metrics, online[(mode, reset, 'original')], small))
    return dict(version=1, population=reference[0], label_scope='development', small_classes=small,
                online_pre_update=result, zero_update_anchor_replay=forgetting)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    _json(external_directory(args.out), score(dict(path=str(args.plan.resolve()), sha256=args.plan_sha256)))


if __name__ == '__main__':
    main()
