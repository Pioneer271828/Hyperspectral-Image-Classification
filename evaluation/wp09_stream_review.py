"""Offline WP09 review over verified recordings. No session is created by this entry point."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import sys

import numpy as np

from evaluate_tta import HEADS, MODES
from evaluation.classification_metrics import classification_metrics
from evaluation.online_tta import _frozen_run, score
from evaluation.open_set import _labels
from evaluation.registered_open_set import _load
from research_runtime import code_snapshot, external_directory, require_frozen_code
from transfer.online_tta_run import _json
from transfer.online_tta_stream import ORDERS, optional_pin, parse_json, read_pin


# These are the published receipts for two distinct, previously authorized studies.
HISTORY = {
    'campaign': '36b42c6377357f7b93057adcbc25dc6ce5335e938eebf8ace8da79df1c90688d',
    'execution': '0e8bf626f51293fef16551fc54390a4c3781177557555d60c1478bc1131a0441',
    'audit': '8c4f945fbc637fd97aa92cdb45875b02d0aeeef95e0ef57216699f912a0e8154',
    'tent_receipt': '98a392bbfb417e0d948b6c47bba53c56446846458e83ab0de4fae7ee7943effd',
    'tent_cost': '82dadf10038d76901c88b320c45518be476d37e6aa256c004990f2601d778326',
    'openood_audit': '7ae8effa5ddff2c78b6ef1091e552b19e14adb3603d3d9a2e248963924a72198',
}


def _drift(current, reference):
    result = {}
    for head in HEADS:
        a, b = current[head], reference[head]
        if a.shape != b.shape:
            raise ValueError('Paired predictions differ in geometry or class order')
        distance = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result[head] = dict(
            pixels=len(a), decision_changes=int(np.count_nonzero(a.argmax(1) != b.argmax(1))),
            max_abs_probability=float(distance.max()),
            mean_total_variation=float(distance.sum(axis=1).mean() / 2))
    return result


def _worst(records, field):
    result = {}
    for head in HEADS:
        rows = [(record['frame_id'], record[field][head]) for record in records]
        peak, prob = max(rows, key=lambda pair: pair[1]['max_abs_probability'])
        flips, decision = max(rows, key=lambda pair: pair[1]['decision_changes'] / pair[1]['pixels'])
        result[head] = dict(
            max_abs_probability=prob['max_abs_probability'], probability_frame_id=peak,
            max_decision_change_rate=decision['decision_changes'] / decision['pixels'],
            decision_frame_id=flips)
    return result


def _read_cases(campaign, execution, audit, root):
    combinations = {(order, mode, reset): f'{order}-{mode}-{reset}'
                    for order, mode, reset in product(ORDERS, MODES, ('episodic', 'continual'))}
    case_names = set(combinations.values())
    if not (case_names == campaign['registrations'].keys() == execution['combinations'].keys()
            == audit['cases'].keys()):
        raise ValueError('Historical stream cohort differs from the registered four arms and three orders')
    runs = {}
    population = None
    for (order, mode, reset), name in combinations.items():
        accepted = audit['cases'][name]
        stages = execution['combinations'][name]
        run = dict(
            registration=campaign['registrations'][name], directory=root / f'{name}-full',
            replay=dict(path=str(root / 'audit' / f'{name}-replay' / 'replay.json'),
                        sha256=accepted['replay_sha256']))
        reg, packets, anchors, proof = _frozen_run(run, None, None)
        if population is None:
            population = reg.population
        if (reg.provenance['kind'] != 'e4b_pavia_development' or reg.population != population
                or reg.value['commit'] != campaign['commit'] or reg.value['source_hash'] != campaign['source_hash']
                or reg.value['provenance'] != campaign['provenance']
                or (reg.value['order'], reg.settings['mode'], reg.settings['reset']) != (order, mode, reset)
                or reg.frames != campaign['orders'][order]
                or proof['tip'] != accepted['full']['tip'] or proof['tip'] != accepted['split']['tip']
                or proof['tip'] != stages['full']['receipt']['tip']
                or accepted['identical_events'] != len(packets) or not accepted['complete_zero']):
            raise ValueError(f'Recorded run differs from independent historical acceptance: {name}')
        runs[order, mode, reset] = (reg, packets, anchors, proof)
    return runs


def _sequences(runs, execution, audit):
    sequences = {}
    for (order, mode, reset), (reg, packets, anchors, proof) in runs.items():
        matched = {variant: runs[variant, mode, reset][1] for variant in ORDERS}
        frozen = runs[order, 'frozen', reset][1]
        trajectory = []
        for frame in reg.frames:
            key = frame['frame_id']
            trajectory.append(dict(
                frame_id=key, scene=frame['scene'], segment=frame['segment'], episode=frame['episode'],
                versus_frozen=_drift(packets[key], frozen[key]),
                versus_order={other: _drift(packets[key], matched[other][key])
                              for other in ORDERS if other != order}))
        replay = []
        for frame, initial in anchors['initial'].items():
            if any(not np.array_equal(initial[head], frozen[frame][head]) for head in HEADS):
                raise ValueError('Initial anchor differs from the paired frozen run')
            replay.append(dict(frame_id=frame, final_minus_initial=_drift(anchors['final'][frame], initial)))
        name = f'{order}-{mode}-{reset}'
        stages = execution['combinations'][name]
        sequences[f'{mode}/{reset}/{order}'] = dict(
            evidence=proof, trajectory=trajectory, first_frame_id=trajectory[0]['frame_id'],
            last_frame_id=trajectory[-1]['frame_id'], worst_versus_frozen=_worst(trajectory, 'versus_frozen'),
            worst_versus_order={variant: _worst(
                [dict(frame_id=row['frame_id'], drift=row['versus_order'][variant]) for row in trajectory], 'drift')
                for variant in ORDERS if variant != order},
            anchor_drift=replay, worst_anchor=_worst(replay, 'final_minus_initial'),
            cost=dict(
                device='cpu', committed_prediction_and_update_counts=stages['full']['receipt']['actual'],
                process_wall_seconds={stage: record['seconds'] for stage, record in stages.items()},
                committed_frame_wall_seconds=audit['cases'][name]['full']['counts']['committed_frame_seconds']))
    return sequences


def _open_set(audit):
    approval = parse_json(read_pin(audit['approvals']))
    export = parse_json(read_pin(audit['original_export']))
    freeze = parse_json(read_pin(approval['frozen']))
    producer = parse_json(read_pin(approval['producer']))
    if (approval['authority'] != audit['authority'] or export['freeze'] != approval['frozen']
            or approval['label_reads'] != 0 or freeze['evaluation_labels_read'] is not False
            or export['threshold_retuning'] is not False or export['model_selection'] is not False
            or export['label_role'] != 'development_evaluation'
            or producer['arrays_registration'] != freeze['registration']
            or producer['network_output_verified'] is not True or producer['evaluation_labels_read'] is not False):
        raise ValueError('Open-set source calibration and pre-release freeze differ')
    registration, _, _, _, target, pins = _load(freeze['registration'])
    ids = freeze['development_ids']
    label_pin = approval['released_labels']
    if (registration['labels'] != label_pin or freeze['deferred_labels'] != label_pin
            or target['sample_ids'].tolist() != ids or pins != freeze['inputs']):
        raise ValueError('The synthetic open-set label release does not match the frozen sample IDs')
    labels = parse_json(read_pin(label_pin))
    if labels['role'] != 'development_evaluation' or labels['sample_ids'] != ids:
        raise ValueError('The synthetic open-set label release role differs')
    truth = _labels(labels['labels'], len(ids))
    known = approval['columns']
    keep = np.isin(truth, known)
    closed = {head: classification_metrics(
        truth[keep], np.asarray(known)[target[f'logits_{head}'][keep].argmax(1)], known) for head in HEADS}
    methods = []
    for independent, frozen, receipt in zip(audit['methods'], freeze['rows'], export['reports'], strict=True):
        for name in ('head', 'method'):
            if independent[name] != frozen[name] or independent[name] != receipt[name]:
                raise ValueError('Open-set head or scoring method differs from the frozen order')
        calibration = frozen['calibration']
        if (calibration['role'] != 'source_validation'
                or set(calibration['sample_ids']) & set(freeze['development_ids'])):
            raise ValueError('Open-set calibration must use the separate known source role')
        baseline = closed[independent['head']]
        matrix = np.asarray(independent['metrics']['confusion'])
        if matrix[:-1].sum() != baseline['count']:
            raise ValueError('Frozen open-set scores and closed-set control have different known support')
        known_recall = []
        for index, row in enumerate(baseline['per_class']):
            support = row['support']
            if matrix[index].sum() != support:
                raise ValueError('Frozen open-set scores and closed-set control have different class support')
            delta = int(matrix[index, index]) - baseline['confusion_matrix'][index][index]
            known_recall.append(dict(class_id=row['class_id'], support=support,
                                     recall_delta=delta / support if support else None))
        methods.append(dict(independent, released_report_sha256=receipt['report']['sha256'],
                            calibration_sha256=frozen['calibration_sha256'],
                            source_known_coverage=calibration['known_coverage'],
                            reject_rule=calibration['comparator'], threshold=calibration['threshold'],
                            known_recall_versus_closed=known_recall))
    return dict(
        scope='synthetic_frozen_openood_oracle', methods=methods, excluded_unlabeled_id=0,
        closed_reference_known_only=closed,
        freeze_sha256=approval['frozen']['sha256'],
        calibration_role='source_validation', label_role='development_evaluation',
        thresholds_retuned=False, real=None,
        gap='Independent known-only source and calibration, unknown and known evaluation roles, '
            'and a frozen real label release are pending.')


def review(pins):
    if pins.keys() != HISTORY.keys() or any(pin['sha256'] != HISTORY[key] for key, pin in pins.items()):
        raise ValueError('Supply the exact published WP09 evidence pins')
    records = {key: parse_json(read_pin(pin)) for key, pin in pins.items()}
    campaign, execution, audit = (records[key] for key in ('campaign', 'execution', 'audit'))
    if (execution['campaign_sha256'] != HISTORY['campaign'] or audit['campaign_sha256'] != HISTORY['campaign']
            or audit['execution_sha256'] != HISTORY['execution']
            or audit['status'] != 'passed_known_scene_online_tta_custody'
            or execution['commit'] != campaign['commit'] or audit['commit'] != campaign['commit']
            or execution['actual'] != campaign['budget'] or audit['query'] or audit['raw_gt']
            or audit['independent_optimizer_updates'] or audit['independent_statistics_updates']):
        raise ValueError('Historical WP09 acceptance does not cover these completed predictions')
    root = Path(pins['execution']['path']).parent
    runs = _read_cases(campaign, execution, audit, root)
    reference = next(iter(runs.values()))[0]
    tent, cost, openood = (records[key] for key in ('tent_receipt', 'tent_cost', 'openood_audit'))
    if (tent['status'] != 'passed' or tent['actual'] != campaign['budget']
            or tent['raw_gt'] or tent['query'] or openood['raw_gt'] or openood['query']
            or openood['optimizer_updates'] or openood['status'] != 'passed'
            or not cost['synthetic'] or cost['real_optimizer_updates'] or cost['query']):
        raise ValueError('Optional-method evidence scopes differ from published acceptance')
    return dict(
        version=1, kind='wp09_stream_review', input_sha256=HISTORY.copy(),
        historical_closed_set=dict(
            scope='paviaU_seen_training_regions_label_free', population=reference.population,
            class_ids=reference.value['class_ids'], execution_commit=campaign['commit'], device='cpu',
            orders=campaign['orders'], receipt_counts=execution['actual'], wall_seconds=execution['seconds'],
            independent_review_wall_seconds=audit['seconds'], sequences=_sequences(runs, execution, audit),
            accuracy=None, recall_forgetting=None,
            gap='No independently released matching evaluation labels; anchors measure prediction drift.'),
        released_closed_set=None,
        tent=dict(scope='same_seen_paviaU_scope_label_free_entropy_comparator', execution_commit=tent['commit'],
                  counts=tent['actual'], wall_seconds=tent['seconds'],
                  synthetic_cost=dict(execution_commit=cost['commit'], units='ms',
                      scope='observation_load_predict_adapt_and_checkpoint_serialization',
                      summary=cost['full_online_observation_predict_adapt_and_serialize']['summary'],
                      peak_rss_kib=cost['worker_peak_rss_kib'])),
        open_set=_open_set(openood),
        new_execution=dict(predict=0, optimize=0, update_statistics=0, refit_threshold=0))


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    for name in HISTORY:
        parser.add_argument('--' + name.replace('_', '-'), nargs=2, metavar=('PATH', 'SHA256'), required=True)
    for name in ('closed-set-plan', 'authority', 'admission', 'source-receipt'):
        parser.add_argument('--' + name, nargs=2, metavar=('PATH', 'SHA256'))
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != 'linux' or not __debug__ or code['commit'] != args.expected_commit:
        raise ValueError('Use the expected Linux checkout with assertions enabled')
    require_frozen_code(code)
    out = external_directory(args.out)
    if out.exists():
        raise FileExistsError(out)
    pins = {key: optional_pin(*getattr(args, key)) for key in HISTORY}
    trust = {key: optional_pin(*(getattr(args, key) or (None, None)))
             for key in ('authority', 'admission', 'source_receipt')}
    if args.closed_set_plan is None and any(trust.values()):
        parser.error('Supply --closed-set-plan with its own evaluation authority')
    result = review(pins)
    if args.closed_set_plan is not None:
        closed_pin = optional_pin(*args.closed_set_plan)
        scored = score(closed_pin, **trust)
        if 'label_release' in scored:
            scored['label_release'] = {'sha256': scored['label_release']['sha256']}
        result['released_closed_set'] = dict(plan_sha256=closed_pin['sha256'], results=scored)
    result['consumer_commit'] = code['commit']
    require_frozen_code(code)
    _json(out, result)


if __name__ == '__main__':
    main()
