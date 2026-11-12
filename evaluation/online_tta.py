"""Score complete, immutable WP09 predictions against a separate development label release."""
import argparse
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np

from evaluate_tta import HEADS
from evaluation.classification_metrics import classification_metrics, class_indices, integer_values
from evaluation.known_development_authority import read_authority
from evaluation.known_development_release import (release_metadata, released_labels, reserve_label,
                                                  reserve_label_directory)
from evaluation.known_source_admission import verify_admission
from research_runtime import external_directory, hash_file
from transfer.online_tta_run import committed_history, read_predictions, validate_predictions, _json
from transfer.online_tta_stream import (ORDERS, StreamRegistration, fields_are, optional_pin, parse_json, read_pin,
                                        require_hash, validate_pin)


def _packet(reg, frame, path, hashes):
    probs = read_predictions(path)
    inputs = reg.observation(frame['observation'])[0]
    if reg.admission is not None:
        r0, r1, c0, c1 = reg.admission['windows'][frame['observation']]['box']
        if inputs.shape[:2] != (r1 - r0, c1 - c0):
            raise ValueError('Observation geometry differs from admitted development window')
    validate_predictions(probs, inputs.shape[0] * inputs.shape[1], len(reg.value['class_ids']))
    if reg.fingerprints(probs) != hashes:
        raise ValueError('Three-head prediction identity differs')
    return probs


def _frozen_run(item, admission, label_path):
    fields_are(item, 'registration directory replay')
    root = external_directory(item['directory'])
    if label_path is not None:
        reserve_label(item['registration'], label_path)
        reserve_label(item['replay'], label_path)
        reserve_label_directory(root, label_path)
        reserve_label_directory(Path(item['replay']['path']).parent, label_path)
    reg = StreamRegistration(item['registration'], admission=admission)
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
    return reg, predictions, replay, dict(registration_sha256=reg.pin['sha256'], tip=tip,
                                          replay_sha256=replay_pin['sha256'])


def _labels(pin, reg):
    if reg.provenance['kind'] == 'e4b_pavia_development':
        raise ValueError('The historical E4b observation release carries no scoring authorization')
    if reg.provenance['kind'] != 'synthetic':
        raise ValueError('Approved development source requires the authority and verified label release')
    release = parse_json(read_pin(pin))
    fields_are(release, 'kind population class_ids release_receipt_sha256 training_class_counts frames')
    require_hash(release['release_receipt_sha256'])
    if (release['kind'] != 'synthetic' or release['population'] != reg.population
            or release['class_ids'] != reg.value['class_ids']):
        raise ValueError('Label release and prediction population differ')
    counts = release['training_class_counts']
    if (not isinstance(counts, list) or len(counts) != len(release['class_ids'])
            or any(type(n) is not int or n < 0 for n in counts)):
        raise ValueError('Small classes require preregistered training support counts')
    frames = {f['frame_id']: f for f in reg.frames}
    labels, pins = {}, {}
    for row in release['frames']:
        fields_are(row, 'frame_id observation_sha256 labels')
        frame_id = row['frame_id']
        if (frame_id not in frames or frame_id in pins
                or row['observation_sha256'] != reg.observation_identity(frames[frame_id])):
            raise ValueError('Label frame ID or observation pin differs')
        validate_pin(row['labels'])
        pins[frame_id] = row['labels']
    if pins.keys() != frames.keys():
        raise ValueError('Label release omits registered frame identities')
    for frame_id, label_pin in pins.items():
        data = np.load(BytesIO(read_pin(label_pin)), allow_pickle=False)
        shape = reg.observation(frames[frame_id]['observation'])[0].shape[:2]
        if not isinstance(data, np.ndarray) or data.shape != shape or data.dtype.kind not in 'iu':
            raise ValueError('Released labels must be a single integer image of registered geometry')
        truth = integer_values(data, 'Released labels').ravel()
        class_indices(truth[truth > 0], release['class_ids'], 'Released labels')
        labels[frame_id] = truth
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


def _runs(request, admission, label_path):
    if not isinstance(request['runs'], list) or not request['runs']:
        raise ValueError('Explicit offline scoring runs required')
    runs, proofs = {}, []
    reference = None
    for item in request['runs']:
        reg, predictions, replay, proof = _frozen_run(item, admission, label_path)
        order = reg.value['order']
        key = (reg.settings['mode'], reg.settings['reset'], order)
        identity = (reg.population, reg.value['source_hash'], reg.value['source'],
                    reg.value['provenance'], reg.value['class_ids'],
                    {k: v for k, v in reg.settings.items() if k not in ('mode', 'reset')})
        if key in runs or (reference is not None and identity != reference):
            raise ValueError('Duplicate run or incompatible paired population/source')
        reference = identity
        runs[key] = (reg, predictions, replay)
        proofs.append(proof)
    for mode, reset, order in runs:
        if ('frozen', reset, order) not in runs or any((mode, reset, variant) not in runs for variant in ORDERS):
            raise ValueError('Each comparison needs all registered orders and matching frozen controls')
    return runs, sorted(proofs, key=lambda row: row['registration_sha256'])


def _request(plan, authority, admission, source_receipt, *, release):
    supplied = sum(pin is not None for pin in (authority, admission, source_receipt))
    if supplied not in (0, 3):
        raise ValueError('Supply the independently approved data authority, source receipt and source admission')
    root = context = label_path = None
    if authority is not None:
        root, context, _ = read_authority(authority)
        label_path = context['labels']['path']
        reserve_label(admission, label_path)
        reserve_label(source_receipt, label_path)
        reserve_label(plan, label_path)
    request = parse_json(read_pin(plan))
    fields_are(request, 'version runs' if release else 'version label_release runs')
    if request['version'] != (1 if root is None else 2):
        raise ValueError('Trusted known development scoring plans require version 2')
    runs, proofs = _runs(request, admission, label_path)
    reg = next(iter(runs.values()))[0]
    expected = None
    if root is not None:
        expected = release_metadata(reg, authority, source_receipt, root, context, proofs)
        verify_admission(reg, authority, source_receipt)
    return request, runs, reg, expected


def prepare_release(plan, *, authority, admission, source_receipt):
    """Seal exact completion/replay and sample identities without opening labels."""
    if any(pin is None for pin in (authority, admission, source_receipt)):
        raise ValueError('Development release needs the approved authority, source receipt and admission')
    return _request(plan, authority, admission, source_receipt, release=True)[-1]


def score(plan, *, authority=None, admission=None, source_receipt=None):
    """Open label inputs only after every registered run and anchor packet has been verified."""
    request, runs, reg, expected = _request(plan, authority, admission, source_receipt, release=False)
    if expected is None:
        labels, small = _labels(request['label_release'], reg)
        scope = 'synthetic'
    else:
        labels = released_labels(request['label_release'], expected, reg)
        small = [c for c, n in zip(expected['class_ids'], expected['training_class_counts'], strict=True) if n <= 25]
        scope = expected['scope']
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
    result = dict(version=request['version'], population=reg.population, label_scope=scope, small_classes=small,
                online_pre_update=result, zero_update_anchor_replay=forgetting)
    if expected is not None:
        result.update(label_release=request['label_release'], authority_sha256=authority['sha256'],
                      source_admission_sha256=admission['sha256'], label_role=expected['role'],
                      annotation_costs=reg.admission['annotation_costs'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--source-receipt-sha256')
    parser.add_argument('--freeze-release', action='store_true')
    args = parser.parse_args()
    plan = optional_pin(args.plan, args.plan_sha256)
    trust = dict(authority=optional_pin(args.authority, args.authority_sha256),
                 admission=optional_pin(args.admission, args.admission_sha256),
                 source_receipt=optional_pin(args.source_receipt, args.source_receipt_sha256))
    action = prepare_release if args.freeze_release else score
    _json(external_directory(args.out), action(plan, **trust))


if __name__ == '__main__':
    main()
