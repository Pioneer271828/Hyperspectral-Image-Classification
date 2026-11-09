"""Strict known-only full-context preparation; not proof of a model's history.

Requires complete permitted annotations for source and calibration rectangles.
No query labels, auto-search of a whole GT image, or retrospective certification.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from research_runtime import require_frozen_code
from transfer.registered_development import _read, _pin, _json_new
from .open_set import _semantics, _array_hash, digest
from .registered_open_set import _code, _deferred, _output
from .open_set_producer import ROLES


def _raw(pin, role, known):
    raw = _read(pin)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.namelist()) != len(set(z.namelist())) or sum(x.file_size for x in z.infolist()) > 64*1024*1024:
            raise ValueError('Bounded unique observation entries required')
    fields = {'spectral', 'wavelengths_nm', 'indices'}
    if role != ROLES[-1]: fields.add('context_labels')
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        if set(f.files) != fields: raise ValueError('Complete permitted context annotation required; evaluation labels forbidden')
        a = {k: f[k].copy() for k in fields}
    x, w, ix = (a[k] for k in ('spectral', 'wavelengths_nm', 'indices'))
    if (x.ndim != 3 or x.dtype != np.float32 or not np.isfinite(x).all() or
            min(x.shape[:2]) < 2 or x.shape[0]*x.shape[1] > 4096 or
            not 2 <= x.shape[2] <= 512 or w.shape != (x.shape[2],) or w.dtype != np.float32 or
            not np.isfinite(w).all() or not (w > 0).all() or not (np.diff(w) > 0).all()):
        raise ValueError('Finite bounded FP32 physical-band observation required')
    if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or not 0 < len(ix) <= x.shape[0]*x.shape[1] or
            len(np.unique(ix)) != len(ix) or ix.min() < 0 or ix.max() >= x.shape[0]*x.shape[1]):
        raise ValueError('Unique in-context supervised/evaluation indices required')
    if 'context_labels' in a:
        y = a['context_labels']
        if y.shape != x.shape[:2] or y.dtype.kind not in 'iu' or not np.isin(y, known).all():
            raise ValueError('Every visible source/calibration pixel must be annotated known; zero is unverified, not unknown')
        if set(y.ravel()[ix].tolist()) != set(known):
            raise ValueError('Every known class needs selected source/calibration support')
    return a


def _load(registration):
    v = json.loads(_read(registration))
    fields = {'version', 'kind', 'known', 'unknown', 'dataset', 'observations', 'labels'}
    if (set(v) != fields or v['version'] != 1 or v['kind'] != 'known_full_context_preparation_v1' or
            not isinstance(v['dataset'], str) or not v['dataset'] or
            not isinstance(v['observations'], list) or len(v['observations']) != 3):
        raise ValueError('Explicit three-role context preparation required')
    _semantics(v['known'], v['unknown']); _deferred(v['labels'])
    arrays = []; contexts = []; pins = [registration]
    for role, obs in zip(ROLES, v['observations']):
        if (set(obs) != {'role', 'scene_uid', 'scene_shape', 'origin', 'superpixel_scale', 'input'} or
                obs['role'] != role or not isinstance(obs['scene_uid'], str) or not 0 < len(obs['scene_uid']) <= 120 or
                not isinstance(obs['scene_shape'], list) or len(obs['scene_shape']) != 2 or
                any(type(i) is not int or not 2 <= i < 2**31 for i in obs['scene_shape']) or
                not isinstance(obs['origin'], list) or len(obs['origin']) != 2 or
                any(type(i) is not int or not 0 <= i < 2**31 for i in obs['origin']) or
                type(obs['superpixel_scale']) is not int or not 4 <= obs['superpixel_scale'] <= 4096):
            raise ValueError('Explicit bounded spatial identity required')
        a = _raw(obs['input'], role, [k['id'] for k in v['known']])
        h, w = a['spectral'].shape[:2]; r, c = obs['origin']
        if r+h > obs['scene_shape'][0] or c+w > obs['scene_shape'][1]: raise ValueError('Context outside scene')
        for old in contexts:
            if old['scene_uid'] == obs['scene_uid']:
                if old['scene_shape'] != obs['scene_shape']: raise ValueError('Same scene has conflicting geometry')
                r0, r1, c0, c1 = old['box']
                if max(r, r0) < min(r+h, r1) and max(c, c0) < min(c+w, c1):
                    raise ValueError('Full source/calibration/evaluation contexts overlap')
        if arrays and not np.array_equal(a['wavelengths_nm'], arrays[0]['wavelengths_nm']):
            raise ValueError('First contract requires identical ordered physical bands; no fabricated resampling')
        contexts.append(dict(scene_uid=obs['scene_uid'], scene_shape=obs['scene_shape'], box=[r,r+h,c,c+w]))
        arrays.append(a); pins.append(obs['input'])
    paths = [str(Path(p['path']).resolve()) for p in pins+[v['labels']]]
    if len(paths) != len(set(paths)): raise ValueError('Distinct raw input, registration and deferred labels required')
    return v, arrays, contexts, pins


def prepare_known_contexts(registration, *, out, expected_commit):
    code = _code(expected_commit)
    v, arrays, contexts, pins = _load(registration)
    folder = _output(out, pins+[v['labels']])
    source = arrays[0]['spectral']
    lo = source.min(axis=(0,1)); hi = source.max(axis=(0,1))
    scale = np.where(hi > lo, hi-lo, np.float32(1))
    if not np.isfinite(scale).all(): raise ValueError('Normalization range overflow')
    packets = []; checks = []
    for obs, a, context in zip(v['observations'], arrays, contexts):
        x = a['spectral']; h,w = x.shape[:2]; r,c = obs['origin']; sh,sw = obs['scene_shape']
        yy,xx = np.mgrid[r:r+h,c:c+w]
        coordinates = np.stack((yy/(sh-1),xx/(sw-1)), axis=-1).astype(np.float32)
        normalized = (x-lo)/scale
        if not np.isfinite(normalized).all(): raise ValueError('Normalization overflow')
        packet = dict(x=np.concatenate((normalized, coordinates),axis=-1),
                      wavelengths_nm=a['wavelengths_nm'], indices=a['indices'])
        annotation = None
        if 'context_labels' in a:
            packet['labels'] = a['context_labels'].ravel()[a['indices']]
            annotation = _array_hash(a['context_labels'])
        packets.append(packet)
        checks.append(dict(role=obs['role'], context=context, annotated_pixels=0 if annotation is None else h*w,
            annotation_sha256=annotation, selected_pixels=len(a['indices']),
            array_hashes={k:_array_hash(val) for k,val in packet.items()}))
    # Validate all roles, including source-only normalization, before writing anything.
    for pin in pins: _read(pin)
    require_frozen_code(code)
    folder.mkdir(parents=True, exist_ok=False)
    outputs=[]
    for obs, packet in zip(v['observations'],packets):
        path=folder/(obs['role']+'.npz')
        with path.open('xb') as f: np.savez(f,**packet)
        outputs.append({k:obs[k] for k in ('role','scene_uid','origin','superpixel_scale')} | {'input':_pin(path)})
    stats=dict(minimum=lo.tolist(),maximum=hi.tolist(),fit_role='source_train',
        fit_pixels=int(np.prod(source.shape[:2])),source_pin=v['observations'][0]['input'],clipping=False)
    for pin in pins: _read(pin)
    require_frozen_code(code)
    return _json_new(folder/'receipt.json',dict(status='passed',version=1,code=code,
        registration=registration,inputs=pins,known=v['known'],unknown=v['unknown'],dataset=v['dataset'],
        observations=outputs,labels=v['labels'],statistics=stats,statistics_hash=digest(stats),checks=checks,
        full_context_known_only=True,preprocessing_source_only=True,training_history_verified=False,
        evaluation_labels_read=False,query=0,optimizer_updates=0,
        limits=['annotation truth and scene identity require external dataset provenance',
                'full source/calibration annotation is an additional protocol requirement and must be budgeted',
                'no automatic block search, erased-unknown masking, model training, or historical model certification',
                'strict annotated rectangles may be infeasible in real scenes; retain that failure',
                'same physical-band vector only; not cross-sensor admissibility']))
