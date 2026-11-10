"""E4c-3: authenticated CPU M8 feature production, not training-history proof."""
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from model.checkpoint_io import load_model_checkpoint
from research_runtime import require_frozen_code
from transfer.graph_route_context import route_state_hash
from transfer.registered_development import _read, _pin, _json_new
from .open_set import _semantics, _labels, digest
from .registered_open_set import HEADS, _code, _deferred, _output, _load, _rows


ROLES = ['source_train', 'source_validation', 'development_evaluation']


def _observation(pin, role, known):
    raw = _read(pin)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if (len(z.namelist()) != len(set(z.namelist())) or
                sum(e.file_size for e in z.infolist()) > 64*1024*1024):
            raise ValueError('Bounded unique NPZ entries required')
    fields = {'x', 'wavelengths_nm', 'indices'} | ({'labels'} if role != ROLES[-1] else set())
    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
        if set(f.files) != fields:
            raise ValueError('Observation fields differ; evaluation labels forbidden')
        a = {k: f[k].copy() for k in fields}
    x, w, ix = (a[k] for k in ('x', 'wavelengths_nm', 'indices'))
    if (x.ndim != 3 or x.dtype != np.float32 or not np.isfinite(x).all() or
            min(x.shape[:2]) < 2 or x.shape[0]*x.shape[1] > 4096 or
            w.ndim != 1 or w.dtype != np.float32 or not 2 <= len(w) <= 512 or
            not np.isfinite(w).all() or not (w > 0).all() or not (np.diff(w) > 0).all() or
            x.shape[2] != len(w)+2):
        raise ValueError('Bounded finite FP32 image with two extra channels and ordered nm required')
    if (ix.ndim != 1 or ix.dtype.kind not in 'iu' or not 0 < len(ix) <= x.shape[0]*x.shape[1] or
            len(np.unique(ix)) != len(ix) or ix.min() < 0 or ix.max() >= x.shape[0]*x.shape[1]):
        raise ValueError('Unique in-context pixel indices required')
    a['indices'] = ix.astype(np.int64)
    if 'labels' in a:
        a['labels'] = _labels(a['labels'], len(ix))
        if not set(a['labels'].tolist()) <= set(known):
            raise ValueError('Only declared known labels allowed in source/calibration')
    return a


def _inputs(registration):
    v = json.loads(_read(registration))
    fields = {'version', 'kind', 'known', 'unknown', 'dataset', 'checkpoint', 'sidecar',
              'training_declaration', 'observations', 'labels', 'temperature', 'known_coverage'}
    if (set(v) != fields or v['version'] != 1 or v['kind'] != 'm8_open_set_development_production_v1' or
            not isinstance(v['dataset'], str) or not v['dataset'] or
            not isinstance(v['observations'], list) or len(v['observations']) != 3):
        raise ValueError('Complete development producer registration required')
    _semantics(v['known'], v['unknown']); _deferred(v['labels'])
    if v['sidecar']['path'] != str(Path(v['checkpoint']['path']+'.json').resolve()):
        raise ValueError('Authenticate the actual adjacent checkpoint sidecar')
    metadata = json.loads(_read(v['sidecar'])); _read(v['checkpoint'])
    if (metadata.get('format_version') != 1 or metadata.get('class_counts') != {v['dataset']:len(v['known'])}):
        raise ValueError('Single complete native head required; no column deletion or extra heads')
    declaration = json.loads(_read(v['training_declaration']))
    expected = dict(version=1, kind='open_set_training_semantics_declaration_v1',
        checkpoint_sha256=v['checkpoint']['sha256'], sidecar_sha256=v['sidecar']['sha256'],
        dataset=v['dataset'], known=v['known'], source_training_class_ids=[r['id'] for r in v['known']],
        unknown_training_excluded=True)
    if declaration != expected:
        raise ValueError('Bound training semantic declaration required (not proof of historical exclusion)')
    pins = [registration, v['checkpoint'], v['sidecar'], v['training_declaration']]
    arrays, contexts = [], []
    for role, obs in zip(ROLES, v['observations']):
        if (set(obs) != {'role','scene_uid','origin','superpixel_scale','input'} or obs['role'] != role or
                not isinstance(obs['scene_uid'],str) or not 0<len(obs['scene_uid'])<=120 or
                not isinstance(obs['origin'],list) or len(obs['origin'])!=2 or
                any(type(i) is not int or not 0<=i<2**31 for i in obs['origin']) or
                type(obs['superpixel_scale']) is not int or not 4<=obs['superpixel_scale']<=4096):
            raise ValueError('Ordered roles with global scene/coordinate context required')
        a = _observation(obs['input'],role,expected['source_training_class_ids'])
        h,w = a['x'].shape[:2];r,c = obs['origin']
        context = dict(scene_uid=obs['scene_uid'],box=[r,r+h,c,c+w])
        for old in contexts:
            r0,r1,c0,c1=old['box']
            if (old['scene_uid']==context['scene_uid'] and max(r,r0)<min(r+h,r1) and max(c,c0)<min(c+w,c1)):
                raise ValueError('Source/calibration/evaluation full contexts overlap')
        a['sample_ids'] = np.array([json.dumps([obs['scene_uid'],r+int(i)//w,c+int(i)%w],
                                             separators=(',',':')) for i in a['indices']])
        arrays.append(a); contexts.append(context); pins.append(obs['input'])
    paths=[p['path'] for p in pins]+[v['labels']['path']]
    if len(set(paths))!=len(paths): raise ValueError('Distinct input/deferred files required')
    return v, arrays, contexts, pins


def _model(v):
    # Construct only a private copy. fork_rng includes model initialization during strict load.
    saved = torch.load(io.BytesIO(_read(v['checkpoint'])), map_location='cpu', weights_only=True)
    if any(torch.is_tensor(t) and t.is_floating_point() and t.dtype!=torch.float32 for t in saved.values()):
        raise ValueError('Checkpoint must already be FP32; no silent precision conversion')
    model = load_model_checkpoint(v['checkpoint']['path'], device='cpu')
    b = model.backbone
    b.adapter.clear_attention_cache()
    b.adapter.cache_attention = False  # Private read-only copy; no checkpoint/config modification.
    if (b.graph_mode!='learned' or b.norm_type!='group_norm' or b.extra_channels!=2 or
            b.adapter.cache_attention or b.graph_assignment_mode!='learned' or b.graph_adjacency_mode!='learned' or
            any(getattr(b,k) is not None for k in ('interaction','region_residual','coverage_modulation','graph_router')) or
            any(p.device.type!='cpu' or p.dtype!=torch.float32 or not torch.isfinite(p).all()
                for p in model.parameters()) or any(m.training for m in model.modules())):
        raise ValueError('First producer requires uncoupled learned-graph GroupNorm CPU FP32 cache-off M8')
    if (list(model.heads)!=[v['dataset']] or
            any(family[v['dataset']].out_features!=len(v['known']) for family in
                (model.heads,model.hgnn_heads,model.cnn_heads))):
        raise ValueError('Native complete head column counts differ')
    return model


def _forward(model, v, a, obs):
    x=torch.from_numpy(a['x'].copy()); wave=torch.from_numpy(a['wavelengths_nm'].copy())
    x_before=x.clone();wave_before=wave.clone();state=route_state_hash(model);rng=torch.get_rng_state().clone()
    kw=dict(wavelengths=wave,wavelength_unit='nm',superpixel_scale=obs['superpixel_scale'])
    with torch.no_grad():
        features=model.backbone.forward_features(x,dataset_name=v['dataset'],**kw)
        logits=model.predict_features(v['dataset'],*features,return_logits=True)
        direct=model(v['dataset'],x,**kw,return_logits=True)
        probabilities=model(v['dataset'],x,**kw)
    for f,z,original,p in zip(features,logits,direct,probabilities):
        if (f.shape!=(x.shape[0]*x.shape[1],64) or z.shape!=(len(f),len(v['known'])) or
                not torch.isfinite(f).all() or not torch.isfinite(z).all() or
                not torch.equal(z,original) or not torch.equal(z.softmax(-1),p)):
            raise ValueError('Native three-head feature/logit/probability equivalence failed')
    if (route_state_hash(model)!=state or not torch.equal(rng,torch.get_rng_state()) or
            not torch.equal(x,x_before) or not torch.equal(wave,wave_before) or
            any(p.grad is not None for p in model.parameters())):
        raise ValueError('Read-only inference changed state, RNG, input or gradient')
    data={'sample_ids':a['sample_ids']}
    for h,f,z in zip(HEADS,features,logits):
        data['features_'+h]=f[a['indices']].numpy().copy()
        if obs['role']!='source_train': data['logits_'+h]=z[a['indices']].numpy().copy()
    if 'labels' in a:data['labels']=a['labels']
    return data,dict(role=obs['role'],all_context_pixels=len(features[0]),selected_pixels=len(a['indices']),
                     state_sha256=state,three_heads_exact=True,state_rng_input_unchanged=True)


def produce_open_set(registration,*,out,expected_commit):
    code=_code(expected_commit);v,arrays,contexts,pins=_inputs(registration)
    folder=_output(out,pins+[v['labels']]);folder.mkdir(parents=True,exist_ok=False)
    _json_new(folder/'started.json',dict(status='running',registration=registration,code=code))
    started=time.monotonic();rng=torch.get_rng_state().clone()
    try:
        outputs={};checks=[]
        with torch.random.fork_rng(devices=[]):
            model=_model(v)
            for name,a,obs in zip(('source','calibration','evaluation'),arrays,v['observations']):
                data,check=_forward(model,v,a,obs);checks.append(check)
                path=folder/(name+'.npz')
                with path.open('xb') as f: np.savez(f,**data)
                outputs[name]=_pin(path)
        if not torch.equal(rng,torch.get_rng_state()):raise ValueError('Caller RNG changed')
        ev=dict(version=1,kind='open_set_producer_declaration_v1',model_sha256=v['checkpoint']['sha256'],
            known=v['known'],heads=HEADS,feature_dimensions=[64]*3,
            representations=['M8/native/'+h+'/64' for h in HEADS],source_training_class_ids=[r['id'] for r in v['known']],
            unknown_training_excluded=True,logits_kind='raw_logits')
        evidence=_json_new(folder/'producer-declaration.json',ev)
        generated=dict(version=1,kind='open_set_development_arrays_v1',known=v['known'],unknown=v['unknown'],
            heads=HEADS,temperature=v['temperature'],known_coverage=v['known_coverage'],
            model=v['checkpoint'],model_evidence=evidence,**outputs,labels=v['labels'],calibration_role='source_validation')
        reg=_json_new(folder/'arrays-registration.json',generated)
        # Validate exactly the existing consumer schema and all nine calibrations, never eval labels.
        loaded=_load(reg);_rows(*loaded[:4])
        for pin in pins:_read(pin)
        require_frozen_code(code)
        return _json_new(folder/'receipt.json',dict(status='passed',code=code,registration=registration,
            inputs=pins,arrays_registration=reg,outputs=outputs,producer_declaration=evidence,contexts=contexts,
            checks=checks,network_output_verified=True,training_history_verified=False,
            real_scene_identity_verified=False,evaluation_labels_read=False,query=0,optimizer_updates=0,
            elapsed_seconds=time.monotonic()-started,
            limits=['CPU learned-graph GroupNorm M8 only; preprocessing and scene identities are declared',
                    'Checkpoint and declaration cannot independently prove historical unknown-class exclusion',
                    'No real closed-set stability or open-set benefit conclusion']))
    except BaseException as exc:
        _json_new(folder/'failure.json',dict(status='failed',type=type(exc).__name__,message=str(exc)))
        raise

