"""Validate/export a blocked metadata proposal and English paper inserts; never fit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = '11d7a8b398c03c421fb6fb50ca881e182ac1f360'
DOCUMENTS = ('D_Methods_established.md', 'D_Limitations_ready.md', 'D_Table_notes_ready.md', 'D_Prospective_protocol_insert.md', 'data_request_draft.md')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(proposal=None):
    p = read(HERE/'hypotheses.json') if proposal is None else proposal
    record = read(HERE/'original-hyrank-record.snapshot.json')
    if digest(HERE/'original-hyrank-record.snapshot.json') != '660f64ded5897348dbc6db20be1ee9b7fed391710959ef672090b8e808a87092' or record['metadata']['version'] != 'v001' or record['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Original C7 version/permission metadata differs')
    if p['status'] != 'SPECIFIED_NOT_RUN_INPUT_ROLES_BLOCKED' or p['authorized_real_updates'] or p['authorized_new_Hyperion_points'] or p['current_C7_ontology_reused']:
        raise ValueError('Metadata cannot release fitting, points or C7 ontology reuse')
    if [(c['common_id'], c['xuzhou_original_ids']) for c in p['classes']] != [(0,[2]),(1,[6])] or any(c['HyRANK_delivered_values'] is not None for c in p['classes']):
        raise ValueError('New water/woody mapping requires delivered-value binding; no ordinal guess')
    if p['classes'][0]['HyRANK_figure_ordinals'] != [13] or p['classes'][1]['HyRANK_figure_ordinals'] != [6,7,8]:
        raise ValueError('Frozen semantic hypothesis differs')
    scenes = {s['id']: s for s in p['scenes']}
    if set(scenes) != {'X','D','L'} or scenes['D']['family'] != scenes['L']['family'] or scenes['X']['family'] == scenes['D']['family']:
        raise ValueError('One different-family candidate plus same-family scene control required')
    cells = p['evaluation_cells']
    if len(cells) != 9 or {(c['source'],c['target']) for c in cells} != {(a,b) for a in scenes for b in scenes}:
        raise ValueError('Complete three-source reuse/evaluation registry required')
    primary = {(c['source'],c['target']) for c in cells if c['role'] == 'primary_different_family'}
    same = {(c['source'],c['target']) for c in cells if c['role'] == 'same_instrument_cross_scene_control'}
    if primary != {('X','D'),('D','X')} or same != {('D','L'),('L','D')}:
        raise ValueError('Primary direction or same-instrument control changed')
    fit = p['fitting']
    if fit['root_seeds'] != [64,512,2048] or fit['steps'] != 64 or fit['per_transfer_arm_target_optimizer_updates'] or fit['external_pretrained_weights_allowed'] or fit['source_checkpoints_shared_between_source_scenes'] or not fit['representation'].startswith('C8 raw_distribution only'):
        raise ValueError('Fresh isolated fixed-budget source fits required')
    count = len(scenes)*len(fit['root_seeds'])
    budget = p['budget_if_separately_authorized']
    expected = dict(source_linear_fits=count, producer_updates=count*64, independent_replay_updates=count*64, maximum_total_real_updates=count*128, unique_source_fitting_points=30, unique_evaluation_points=30, total_unique_points=60, new_Hyperion_role_points_required_if_feasible=40, linear_evaluation_rows=len(cells)*3, centroid_evaluation_rows=len(cells), prediction_presentations=len(cells)*4*10, actual_updates_now=0)
    if any(budget[k] != v for k,v in expected.items()):
        raise ValueError('Count unique source fits once; cells do not multiply fitting cost')
    if set(p['access_and_selection']['forbidden_scenes']) != {'Erato','Nefeli','Kirki','Kiriki'} or set(fit['source_scenes']) != set(scenes):
        raise ValueError('No hidden validation or extra source scene admitted')
    request = read(HERE/'request_fields.json')
    if request['status'] != 'DRAFT_NOT_SENT' or request['outbound_messages_sent'] or request['later_subset_only_if_project_authorized']['current_authorization']:
        raise ValueError('Request remains unsent and later data scope unauthorized')
    if len(request['fields']) != 7 or request['later_subset_only_if_project_authorized']['prospective_centers_total'] != 40 or request['later_subset_only_if_project_authorized']['hidden_validation_requested']:
        raise ValueError('Minimum original metadata/public-training scope differs')
    c8 = read(ROOT/'research/paper_c_real_baselines/engineering-receipt.json')
    c9 = read(ROOT/'research/paper_c_independent_validation/honghu-parent-receipt.json')
    assert c8['tests_run'] == 31 and c8['real_fitting_updates'] == 0
    assert [[c['original_parents'] for c in r['classes']] for r in c9['roles']] == [[2,3],[1,1]]
    for name in DOCUMENTS:
        text = (HERE/name).read_text().lower()
        if any(phrase in text for phrase in ('we trained', 'we fine-tuned', 'we achieved transfer accuracy')):
            raise ValueError('Paper inserts cannot invent completed learning')
    return dict(status='METADATA_AND_DRAFTS_COMPLETE_REAL_INGRESS_BLOCKED', original_version=record['metadata']['version'], candidate_count=1, source_scenes=3, evaluation_cells=9, primary_directions=2, same_instrument_directions=2, conditional_counts=expected, new_role_points_admitted=0, current_optimizer_updates=0, query_reads=0, outbound_requests_sent=0, measured_SRF_prerequisite_for_classification=False, remaining_inputs=['original member/class/processing dictionary','per-scene class/group support and permitted spatial roles','conversion and input identity audit','actual future training authority and cost notice'])


def export(out):
    result = review()
    p = read(HERE/'hypotheses.json')
    out = Path(out)
    out.mkdir()
    for name in DOCUMENTS:
        shutil.copyfile(HERE/name, out/name)
    for name in ('evidence.json','hypotheses.json','request_fields.json'):
        shutil.copyfile(HERE/name, out/name)
    for name, rows in [('prospective-cells.csv', p['evaluation_cells']), ('data-request-fields.csv', read(HERE/'request_fields.json')['fields'])]:
        with (out/name).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result['files'] = [dict(name=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(out.iterdir())]
    (out/'delivery-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def clean(expected):
    if sys.platform != 'linux' or subprocess.check_output(['systemd-detect-virt'],text=True).strip() != 'lxc' or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() != expected or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
        raise ValueError('Clean published authorized-LXC source required')
    frozen = ['research/paper_c_hsi_admission','research/paper_c_hsi_pair','research/paper_c_real_baselines','research/paper_c_independent_validation']
    if subprocess.check_output(['git','diff',BASE,expected,'--',*frozen],cwd=ROOT):
        raise ValueError('Frozen C6–C9 source changed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected-commit',required=True);ap.add_argument('--out',required=True)
    args = ap.parse_args();clean(args.expected_commit)
    out = Path(args.out).resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Fresh independent external delivery required')
    result = export(out)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}))
