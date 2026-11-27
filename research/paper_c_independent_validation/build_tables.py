"""Project existing evidence to paper tables; no data loader, fitting or role release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ('paviaU', 'xuzhou', 'whu-honghu', 'whu-longkou', 'whu-hanchuan', 'laoyuhesidi')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tables():
    for item in read(HERE/'reuse-pins.json'):
        path = HERE/'reuse'/item['file']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Reused original evidence changed')
    datasets = read(HERE/'candidates.json')['datasets']
    comparisons = read(HERE/'comparisons.json')['comparisons']
    by_id = {d['id']: d for d in datasets}
    sources = {s['id'] for s in read(HERE/'sources.json')['sources']}
    if len(by_id) != 17 or len(comparisons) != 16 or len({c['id'] for c in comparisons}) != 16:
        raise ValueError('Frozen bounded roster differs')
    for row in datasets:
        if not set(row['sources']) <= sources or not row['sources'] or row['released_confirmation_units'] != 0 or row['verified_independent_parcels'] is not None:
            raise ValueError('No independent confirmation/parcel evidence has been admitted')
    for row in comparisons:
        if not set(row['datasets']) <= set(by_id) or row['second_device_pair_ready'] or row['independent_confirmation_units_current'] != 0:
            raise ValueError('Candidate metadata cannot release a new experiment')
        if row['relation'] == 'same_published_family' and len({by_id[d]['family'] for d in row['datasets']}) != 1:
            raise ValueError('Same-family control identity differs')
        if row['relation'] == 'different_published_families':
            families = {by_id[d]['family'] for d in row['datasets']}
            if None in families or len(families) < 2:
                raise ValueError('Different-family claim lacks source identity')
    with (HERE/'reuse/D-candidate-pairs30.csv').open(encoding='utf-8-sig', newline='') as stream:
        old = list(csv.DictReader(stream))
    if len(old) != 30 or {(r['source'], r['target']) for r in old} != {(s, t) for s in OLD for t in OLD if s != t}:
        raise ValueError('Original complete thirty-direction roster required')
    projected = []
    for row in old:
        a, b = by_id[row['source']], by_id[row['target']]
        relation = 'local_identity_unbound' if a['family'] is None or b['family'] is None else ('same_published_family' if a['family'] == b['family'] else 'different_published_families')
        common = {a['id'], b['id']} == {'xuzhou', 'whu-honghu'}
        projected.append(dict(**row, C9_relation=relation, C9_C7_common2_development_inputs_admitted=common, C9_common2_untouched_confirmation=False, C9_real_learning_paused=True, C9_current_coarse_scope='C7 corroborated inference, not pristine G2' if common else 'No new shared-head admission'))
    cfg = read(HERE/'reuse/D-source-semantics.json')['author_code']['config_crop_ids']
    intersections = {name: sorted((set(cfg['enmap_cdl']) & set(cfg[target])) - {0}) for name, target in [('EnMAP-DESIS', 'desis_cdl'), ('EnMAP-EO1', 'eo1_cdl')]}
    with (HERE/'reuse/D-candidate-class-map.csv').open(encoding='utf-8-sig', newline='') as stream:
        class_map = list(csv.DictReader(stream))
    for name, ids in intersections.items():
        if sorted(int(r['raw_CDL_id']) for r in class_map if r['candidate_pair'] == name) != ids:
            raise ValueError('Existing candidate ontology does not match original raw-ID intersection')
    exposure = read(ROOT/'research/paper_c_real_baselines/exposure.json')['scenes']
    spatial = []
    budgets = read(HERE/'reuse/C7-selected-budget-and-context.json')
    probe = read(HERE/'reuse/C7-input-probe-aggregates.json')
    for item in exposure:
        scene = item['scene']
        budget = next(r for r in budgets if r['scene'] == scene)
        phase_rows = [p for direction in probe['directions'] for p in direction['phases'] if p['scene'] == scene]
        if budget['selected_parent_overlap'] or budget['cross_role_observation_box_overlap_pixels'] or budget['minimum_role_box_gap_pixels'] != 4 or any(p['points'] != 10 for p in phase_rows):
            raise ValueError('Frozen C7 spatial/point accounting differs')
        for role, key in [('devtrain', 'selected_training_parents'), ('devval', 'selected_development_parents')]:
            if {p['blocks'] for p in phase_rows if p['original_role'] == role} != {item[key]}:
                raise ValueError('Reported C8 parents and C7 selected blocks disagree; do not infer independence')
            spatial.append(dict(scene=scene, original_role=role, released_points=10, per_common_class=5, selected_parent_count_reported=item[key], C7_selected_blocks=item[key], original_acquisition_images=1, validated_independent_parcels=None, original_role_parent_overlap=0, minimum_box_gap_pixels=4, nominal_minimum_axis_gap_m=4*by_id[scene]['GSD_m'], untouched_confirmation_units_released=0, role_status='previously_exposed_development', parent_independence_inference='not established', per_class_parent_count=None))
    hh = read(HERE/'honghu-parent-receipt.json')
    if hh['source'] != '7dc59390e38269b3ff36735932836e29ea78b7dc' or hh['scene'] != 'whu-honghu' or not hh['independent_label_box_parent_join_passed'] or hh['original_NPY_members_decoded'] or hh['untouched_confirmation_units_released']:
        raise ValueError('Original metadata-only parent audit identity/scope differs')
    for row in spatial:
        if row['scene'] == 'whu-honghu':
            role = next(r for r in hh['roles'] if r['original_role'] == row['original_role'])
            if role['total_points'] != 10 or role['selected_parents'] != row['selected_parent_count_reported'] or any(c['points'] != 5 or c['independently_verified_parcels'] is not None for c in role['classes']):
                raise ValueError('Audited per-class parent counts cannot imply parcel independence')
            row['per_class_parent_count'] = {str(c['common_class']): c['original_parents'] for c in role['classes']}
            row['per_class_parent_evidence_source'] = hh['source']
    return datasets, comparisons, projected, spatial, intersections


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def build(out):
    out = Path(out)
    datasets, comparisons, projected, spatial, intersections = tables()
    out.mkdir()
    for name, rows in [('candidate-evidence.csv', datasets), ('paper-experiment-matrix.csv', comparisons), ('original30-current-qualification.csv', projected), ('spatial-unit-evidence.csv', spatial), ('source-evidence.csv', read(HERE/'sources.json')['sources'])]:
        write_csv(out/name, rows)
    hh = read(HERE/'honghu-parent-receipt.json')
    write_csv(out/'honghu-class-parent-evidence.csv', [dict(scene='whu-honghu', role=r['original_role'], common_class=c['common_class'], points=c['points'], original_parents=c['original_parents'], independently_verified_parcels=None, original_acquisition_images=1, confirmation_units_released=0) for r in hh['roles'] for c in r['classes']])
    result = dict(candidate_records=len(datasets), paper_comparison_rows=len(comparisons), original_directions_preserved=len(projected), same_family_original_directions=sum(r['C9_relation'] == 'same_published_family' for r in projected), different_family_original_directions=sum(r['C9_relation'] == 'different_published_families' for r in projected), unbound_local_original_directions=sum(r['C9_relation'] == 'local_identity_unbound' for r in projected), C7_common2_directions=2, unique_C7_points=40, C7_selected_tiles=17, C7_original_acquisition_images=2, new_independent_confirmation_units=0, newly_ready_second_HSI_pairs=0, candidate_CDL_raw_intersections=intersections, new_image_GT_download_bytes=0, optimizer_updates=0, query_reads=0, null_means='Unknown, never zero or an independence certificate', existing_records_superseded=False, scope='Bounded existing candidate roster, not exhaustive proof that no other dataset exists')
    (out/'table-receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def clean(expected):
    if sys.platform != 'linux' or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != expected or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('Clean exact published Linux source required')
    if subprocess.check_output(['systemd-detect-virt'], text=True).strip() != 'lxc':
        raise ValueError('Current execution is authorized LXC only')
    if subprocess.check_output(['git', 'diff', 'c6526763b874df55fab4991e3076a235adcac174', expected, '--', 'research/paper_c_hsi_admission', 'research/paper_c_hsi_pair', 'research/paper_c_real_baselines'], cwd=ROOT):
        raise ValueError('Frozen C6–C8 dependencies changed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    clean(args.expected_commit)
    out = Path(args.out).resolve()
    if out.exists() or out.is_relative_to(ROOT):
        raise ValueError('Fresh external output required')
    print(json.dumps(build(out)))
