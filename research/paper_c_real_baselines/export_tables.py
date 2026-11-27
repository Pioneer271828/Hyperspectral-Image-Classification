"""Export explicit aggregate/opaque-point fields; never export fitted coefficients.

Every export first rechecks cached predictions and statistics without refitting.
Actual image provenance and optimizer replay remain separate acceptance gates.
"""
import argparse
import csv
from pathlib import Path
import statistics

from audit import validate_package
from c8_bridge import read, write, digest, fresh


def csv_file(path, rows):
    if not rows:
        raise ValueError('Nonempty declared export required')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(experiment, producer, out):
    root, out = Path(experiment), Path(out)
    verified = validate_package(root, producer, replay_fits=False)
    evaluation = read(root/'evaluation01/evaluation.json')
    kind = evaluation.get('evidence_kind')
    if kind not in ('real_development', 'synthetic_test'):
        raise ValueError('Explicit real-development versus synthetic evidence kind required')
    out.mkdir()
    rows = evaluation['rows']
    detailed, points = [], []
    for row in rows:
        m = row['metrics']
        detailed.append(dict(evidence_kind=kind, id=row['id'], scene=row['scene'], method=row['method'], representation=row['representation'], seed=row['seed'], fit_scene=row['fit_scene'], fitting_labels=row['fit_labeled_points'], N=m['denominator'], ground_N=m['class_denominators'][0], tree_N=m['class_denominators'][1], OA=m['OA'], AA=m['AA'], macroF1=m['macroF1'], NLL=m['NLL'], binary_Brier=m['binary_Brier'], ground_recall=m['recall'][0], tree_recall=m['recall'][1], true0_pred0=m['confusion'][0][0], true0_pred1=m['confusion'][0][1], true1_pred0=m['confusion'][1][0], true1_pred1=m['confusion'][1][1], original_parent_clusters=row['parent_sensitivity']['original_parent_clusters'], descriptive_OA_lower=row['parent_sensitivity']['OA']['range_2_5_97_5'][0], descriptive_OA_upper=row['parent_sensitivity']['OA']['range_2_5_97_5'][1], model_sha256=row['model_sha256']))
        for i, point in enumerate(row['point_aliases']):
            points.append(dict(evidence_kind=kind, evaluation=row['id'], scene=row['scene'], point_alias=point, parent_alias=row['parent_aliases'][i], truth=row['truth'][i], predicted=row['predictions'][i], probability_tree=row['probabilities'][i], correct=int(row['truth'][i] == row['predictions'][i])))
    groups = {}
    for row in rows:
        key = (tuple(row['direction']), row['scene'], row['method'], row['representation'])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (direction, scene, method, rep), group in groups.items():
        seeds = [r['seed'] for r in group]
        if (len(group) != 3 or set(seeds) != {64, 512, 2048}) and not (len(group) == 1 and seeds == [None]):
            raise ValueError('Complete seed group or single deterministic reference required')
        summary = dict(evidence_kind=kind, source_scene=direction[0], recorded_target_scene=direction[1], evaluated_scene=scene, method=method, representation=rep, initialization_repeats=len(group), unique_development_points=10)
        for name in ('OA', 'AA', 'macroF1', 'NLL', 'binary_Brier'):
            values = [r['metrics'][name] for r in group]
            summary[name+'_mean'] = statistics.mean(values)
            summary[name+'_sampleSD'] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    pairs = [dict(evidence_kind=kind, A=p['A'], B=p['B'], scene=p['scene'], paired_points=p['points'], A_correct_B_wrong=p['A_correct_B_wrong'], A_wrong_B_correct=p['A_wrong_B_correct'], OA_delta=p['OA_delta'], NLL_delta=p['NLL_delta'], ground_recall_delta=p['class_OA_delta'][0], tree_recall_delta=p['class_OA_delta'][1], descriptive_delta_lower=p['parent_resampling_OA_delta']['range_2_5_97_5'][0], descriptive_delta_upper=p['parent_resampling_OA_delta']['range_2_5_97_5'][1]) for p in evaluation['paired_comparisons']]
    directions = [dict(evidence_kind=kind, representation=r['representation'], method=r['method'], OA_direction_A_minus_B=r['OA_A_minus_B'], observed_parent_draws=r['observed_parent_cross_product_draws'], descriptive_lower=r['descriptive_range']['range_2_5_97_5'][0], descriptive_upper=r['descriptive_range']['range_2_5_97_5'][1], point_paired=False) for r in evaluation['direction_differences']]
    for filename, content in [('per-endpoint.csv', detailed), ('per-point.csv', points), ('seed-summary.csv', summaries), ('paired.csv', pairs), ('directional.csv', directions)]:
        csv_file(out/filename, content)
    escape = lambda s: str(s).replace('_', r'\_')
    lines = [r'\begin{tabular}{llllrr}', r'Target & Method & Representation & Repeats & OA & SD \\', r'\hline']
    for row in summaries:
        sd = '--' if row['OA_sampleSD'] is None else f"{100*row['OA_sampleSD']:.2f}"
        lines.append(' & '.join([escape(row['evaluated_scene']), escape(row['method']), escape(row['representation']), str(row['initialization_repeats']), f"{100*row['OA_mean']:.2f}", sd])+r' \\')
    lines += [r'\end{tabular}', '% Evidence: '+kind+'; fixed exposed development points, not independent sensor replication.']
    (out/'summary.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(out.iterdir())]
    result = dict(evidence_kind=kind, producer=producer, files=files, cached_metric_audit=verified, fitted_coefficients_exported=False, feature_matrices_exported=False, original_point_ids_exported=False, pseudonymization_key_exported=False, per_point_rows=len(points), independent_original_image_and_optimizer_acceptance_required=True)
    write(out/'export-manifest.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit', 'producer-commit', 'experiment', 'out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    out = fresh(args.expected_commit, args.out)
    result = export(args.experiment, args.producer_commit, out)
    print(__import__('json').dumps({'evidence_kind': result['evidence_kind'], 'files': len(result['files']), 'per_point_rows': result['per_point_rows']}))


if __name__ == '__main__':
    main()
