"""E4c-2: all-nine development-array freeze before separate label release.

File hashes authenticate bytes, not the truth of the producer's model/semantic
declarations. Real query and automatic model/temperature selection are absent.
"""
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from research_runtime import code_snapshot, require_frozen_code, external_directory
from transfer.container_runtime import training_container_kind
from transfer.registered_development import _read, _json_new
from .open_set import OpenSetScorer, calibrate, evaluate, digest, _sha, _ids, _labels


HEADS = ['fused', 'graph', 'cnn']
METHODS = ['msp', 'energy', 'prototype_squared_distance']


def _code(expected):
    code = code_snapshot()
    if (not training_container_kind() or sys.flags.optimize or code['dirty'] or
            code['commit'] != expected):
        raise ValueError('Clean exact SHA in authorized Linux container required')
    return code


def _deferred(pin):
    # Intentionally no exists/stat/open: labels may not yet be released.
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('Deferred label path/hash required')
    _sha(pin['sha256'])
    p = Path(pin['path'])
    if not p.is_absolute() or str(external_directory(p)) != str(p):
        raise ValueError('Canonical external absolute label path required')


def _pack(pin, role, dimensions, classes):
    raw = _read(pin)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or sum(a.file_size for a in z.infolist()) > 256*1024*1024:
            raise ValueError('Duplicate or oversized archive entries')
    feature = {'features_'+h for h in HEADS}
    logits = {'logits_'+h for h in HEADS}
    fields = {'sample_ids'} | feature
    if role != 'source_train': fields |= logits
    if role in ('source_train', 'source_validation', 'independent_development'): fields.add('labels')
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        if set(f.files) != fields: raise ValueError('Extra/missing array payload; evaluation labels forbidden')
        data = {k: f[k].copy() for k in fields}
    ids = data['sample_ids']
    if ids.ndim != 1 or ids.dtype.kind != 'U' or not 0 < len(ids) <= 200000:
        raise ValueError('Bounded unicode global sample IDs required')
    _ids(ids.tolist(), len(ids))
    for hi, h in enumerate(HEADS):
        for key, width in [('features_'+h, dimensions[hi])] + ([] if role=='source_train' else [('logits_'+h,classes)]):
            a = data[key]
            if a.shape != (len(ids), width) or a.dtype.kind != 'f' or not np.isfinite(a).all():
                raise ValueError('Frozen finite FP feature/logit shape required')
    if 'labels' in data: _labels(data['labels'], len(ids))
    return data


def _load(registration):
    v = json.loads(_read(registration))
    fields = {'version','kind','known','unknown','heads','temperature','known_coverage',
              'model','model_evidence','source','calibration','evaluation','labels','calibration_role'}
    if (set(v) != fields or v['version'] != 1 or v['kind'] != 'open_set_development_arrays_v1' or
            v['heads'] != HEADS or v['calibration_role'] not in ('source_validation','independent_development') or
            not isinstance(v['temperature'], list) or len(v['temperature']) != 3):
        raise ValueError('Complete three-head development registration required')
    _deferred(v['labels'])
    _read(v['model'])
    evidence = json.loads(_read(v['model_evidence']))
    ef = {'version','kind','model_sha256','known','heads','feature_dimensions',
          'representations','source_training_class_ids','unknown_training_excluded','logits_kind'}
    if (set(evidence) != ef or evidence['version'] != 1 or
            evidence['kind'] != 'open_set_producer_declaration_v1' or
            evidence['model_sha256'] != v['model']['sha256'] or evidence['known'] != v['known'] or
            evidence['heads'] != HEADS or evidence['source_training_class_ids'] != [x['id'] for x in v['known']] or
            evidence['unknown_training_excluded'] is not True or evidence['logits_kind'] != 'raw_logits'):
        raise ValueError('Producer declaration/known class columns differ')
    dims = evidence['feature_dimensions']; names = evidence['representations']
    if (not isinstance(dims,list) or len(dims)!=3 or any(type(x) is not int or not 0<x<=512 for x in dims) or
            not isinstance(names,list) or len(names)!=3 or any(not isinstance(x,str) or not x for x in names) or len(set(names))!=3):
        raise ValueError('Explicit per-head representation identities and dimensions required')
    source = _pack(v['source'],'source_train',dims,len(v['known']))
    cal = _pack(v['calibration'],v['calibration_role'],dims,len(v['known']))
    target = _pack(v['evaluation'],'development_evaluation',dims,len(v['known']))
    groups = [set(d['sample_ids'].tolist()) for d in (source,cal,target)]
    if any(groups[i] & groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Source/calibration/evaluation sample overlap')
    pins = [registration] + [v[k] for k in ('model','model_evidence','source','calibration','evaluation')]
    paths = [str(Path(p['path']).resolve()) for p in pins]
    if len(set(paths))!=len(paths) or str(Path(v['labels']['path']).resolve()) in paths:
        raise ValueError('Input roles must have distinct files')
    return v,evidence,source,cal,target,pins


def _rows(v,evidence,source,cal):
    result=[]
    for hi,head in enumerate(HEADS):
        for method in METHODS:
            proto = method == METHODS[-1]
            extras = dict(source_features=source['features_'+head],source_labels=source['labels'],
                source_ids=source['sample_ids'].tolist(),source_role='source_train',
                source_provenance=v['source']['sha256']) if proto else {}
            scorer=OpenSetScorer.freeze(method,v['known'],v['unknown'],model_sha256=v['model']['sha256'],
                representation=evidence['representations'][hi],temperature=1 if proto else v['temperature'][hi],**extras)
            kind='features' if proto else 'raw_logits';key=('features_' if proto else 'logits_')+head
            c,ch=calibrate(scorer,cal[key],cal['labels'],cal['sample_ids'].tolist(),input_kind=kind,
                role=v['calibration_role'],provenance_sha256=v['calibration']['sha256'],known_coverage=v['known_coverage'])
            result.append(dict(head=head,method=method,input_kind=kind,array_key=key,
                scorer=scorer.as_dict(),scorer_sha256=scorer.sha256,calibration=c,calibration_sha256=ch))
    return result


def _output(out,pins):
    out=external_directory(out)
    for pin in pins:
        parent=Path(pin['path']).resolve().parent
        if out==parent or out in parent.parents or parent in out.parents:
            raise ValueError('Output overlaps input directory')
    if out.exists(): raise FileExistsError('Fresh output directory required')
    return out


def freeze_open_set(registration,*,out,expected_commit):
    code=_code(expected_commit);v,e,source,cal,target,pins=_load(registration)
    rows=_rows(v,e,source,cal);folder=_output(out,pins+[v['labels']])
    for pin in pins:_read(pin)
    require_frozen_code(code)
    value=dict(version=1,code=code,registration=registration,inputs=pins,rows=rows,
        development_ids=target['sample_ids'].tolist(),deferred_labels=v['labels'],
        evaluation_labels_read=False,query=0,optimizer_updates=0,producer_truth_verified=False)
    value['freeze_hash']=digest(value);folder.mkdir(parents=True,exist_ok=False)
    return _json_new(folder/'freeze.json',value)


def export_open_set(frozen,*,out,expected_commit):
    code=_code(expected_commit);f=json.loads(_read(frozen))
    fields={'version','code','registration','inputs','rows','development_ids','deferred_labels',
        'evaluation_labels_read','query','optimizer_updates','producer_truth_verified','freeze_hash'}
    if (set(f)!=fields or f['version']!=1 or f['code']!=code or f['evaluation_labels_read'] is not False or
            f['query']!=0 or f['optimizer_updates']!=0 or f['producer_truth_verified'] is not False or
            f['freeze_hash']!=digest({k:v for k,v in f.items() if k!='freeze_hash'})):
        raise ValueError('Frozen schema/code/hash differs')
    # Reconstruct all nine source-derived rows before opening any evaluation labels.
    v,e,source,cal,target,pins=_load(f['registration'])
    if (pins!=f['inputs'] or _rows(v,e,source,cal)!=f['rows'] or v['labels']!=f['deferred_labels'] or
            target['sample_ids'].tolist()!=f['development_ids']):
        raise ValueError('Frozen inputs, rows or IDs changed')
    folder=_output(out,pins+[v['labels'],frozen]);folder.mkdir(parents=True,exist_ok=False)
    _json_new(folder/'started.json',dict(status='running',freeze=frozen,code=code,query=0))
    start=time.monotonic()
    try:
        released=json.loads(_read(v['labels']))
        if (set(released)!={'version','role','sample_ids','labels'} or released['version']!=1 or
                released['role']!='development_evaluation' or released['sample_ids']!=f['development_ids']):
            raise ValueError('Separate development label release/ID order required')
        labels=_labels(released['labels'],len(f['development_ids']));reports=[]
        for row in f['rows']:
            scorer=OpenSetScorer(row['scorer'],row['scorer_sha256'])
            report=evaluate(scorer,row['calibration'],row['calibration_sha256'],target[row['array_key']],
                labels,f['development_ids'],input_kind=row['input_kind'])
            pin=_json_new(folder/(row['head']+'-'+row['method']+'.json'),report)
            reports.append(dict(head=row['head'],method=row['method'],report=pin))
        for pin in pins+[v['labels'],frozen]:_read(pin)
        require_frozen_code(code)
        return _json_new(folder/'receipt.json',dict(status='passed',code=code,freeze=frozen,inputs=pins+[v['labels']],
            reports=reports,head_count=3,method_count=3,model_selection=False,threshold_retuning=False,
            evaluation_labels_read=True,label_role='development_evaluation',query=0,optimizer_updates=0,
            producer_truth_verified=False,elapsed_seconds=time.monotonic()-start,
            limits=['precomputed development arrays; not authenticated feature production or unseen-class training',
                    'no real open-set or closed-set stability claim']))
    except BaseException as exc:
        _json_new(folder/'failure.json',dict(status='failed',type=type(exc).__name__,message=str(exc),freeze=frozen))
        raise
