"""Consume accepted historical detached stages, never current model inference."""
from pathlib import Path
import torch
from evaluation.frozen_representation_capture import POLICY as CAPTURE_POLICY, authenticate
from g4_support_context import check_files, read, validate_overlay
from loadData.dataset_registry import DATASETS, file_hash
from loadData.target_label_protocol import phase_view
from model.region_pixel_alignment import broadcast_regions
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code, tensor_hash, write_json
from target_bn_calibration import OBS_KEYS

CAPTURE_COMMIT = 'f090ca302192a065de852c43e07e4f3beaeea6b4'
CAPTURE_SOURCE = '996bd5580863976662010288d2b67cc05569781c6d3a422f5e3bc3cb0c61f324'
POLICY = 'authenticated_full54_train_val_probe_pack_v1'
OUTPUTS = {f'stage{s}-{kind}.{ext}' for s in (1,2) for kind,ext in (('features','pt'),('summary','json'))}


def _authenticate(capture_path, capture_sha256, freeze_path, analysis_path):
    path = Path(capture_path).resolve(strict=True)
    if file_hash(path) != capture_sha256: raise ValueError('Capture receipt changed')
    receipt = read(path);producer = receipt['code']
    if (receipt['status'] != 'passed' or receipt['policy'] != CAPTURE_POLICY
            or producer['commit'] != CAPTURE_COMMIT or producer['source_hash'] != CAPTURE_SOURCE
            or producer['dirty'] is not False or digest(producer['files']) != CAPTURE_SOURCE
            or set(receipt['outputs']) != OUTPUTS or receipt['model_forwards'] != 1
            or receipt['optimizer_updates'] != 0 or receipt['bn_calibration_updates'] != 0
            or receipt['raw_labels_deserialized'] is not False or receipt['new_query_release'] is not False
            or receipt['channel_selection'] is not False
            or receipt['evidence']['registered_state_flags_rng_inputs_unchanged'] is not True):
        raise ValueError('Accepted single-case historical capture required')
    # Validate the HISTORICAL artifact producer. No claim that current model code
    # matches it, and no model is loaded or executed by this consumer.
    frozen, _, _, case, record, _, pins = authenticate(freeze_path, receipt['freeze_sha256'],
        analysis_path, receipt['analysis_sha256'], receipt['case'], producer)
    if receipt['protected_files'] != pins: raise ValueError('Capture does not bind complete accepted inputs')
    if receipt['evidence']['observation'] != {k:record['prepared_fields'][k] for k in OBS_KEYS}:
        raise ValueError('Captured observation differs from accepted case')
    def pin(p, sha):
        key = str(Path(p).resolve(strict=True))
        if key in pins and pins[key] != sha: raise ValueError('Conflicting input identity')
        pins[key] = sha
    pin(path, capture_sha256)
    for name, sha in receipt['outputs'].items(): pin(path.parent/name, sha)
    check_files(pins)  # All stage bytes, including the second stage, before decode.
    family_path = Path(case['inputs']['family']).resolve(strict=True)
    if pins.get(str(family_path)) != case['inputs']['family_sha256']:
        raise ValueError('Label family is outside accepted frozen inputs')
    family = read(family_path);target = case['target'];classes = DATASETS[target][2]
    if (family['target'] != target or family['seed'] != 64 or family['classes'] != list(range(1,classes+1))
            or family['shape'] != [record['prepared_fields'][k] for k in ('height','width')]):
        raise ValueError('Target/support family identity differs')
    releases = {p:phase_view(family,10,p,expected_hash=family['family_hash']) for p in ('adapt','validation')}
    overlay = {}
    for phase,role in (('adapt','train'),('validation','val')):
        overlay[role+'_indices'] = torch.tensor(releases[phase]['indices'],dtype=torch.long)
        overlay[role+'_labels'] = torch.tensor(releases[phase]['labels'],dtype=torch.long)-1
    validate_overlay(overlay,target,10,family['shape'])
    selected = next(m for m in frozen['models'] if m['name']==case['name'])
    return path, receipt, record, family, releases, overlay, selected, pins


def _pack(bound):
    path, receipt, record, family, releases, overlay, selected, pins = bound
    features = {};pixels = family['shape'][0]*family['shape'][1]
    for stage,width in (('stage1',128),('stage2',64)):
        values = torch.load(path.parent/(stage+'-features.pt'),map_location='cpu',weights_only=True)
        if not isinstance(values,dict) or set(values) != {'cnn','graph','Q'}: raise ValueError('Exact stage archive required')
        q = values['Q']
        if (not torch.is_tensor(q) or q.ndim != 2 or q.shape[0] != pixels
                or tensor_hash(q) != record['prepared_fields']['Q']): raise ValueError('Captured Q identity differs')
        for branch,count in (('cnn',pixels),('graph',q.shape[1])):
            x = values[branch]
            if (not torch.is_tensor(x) or x.layout != torch.strided or x.shape != (count,width)
                    or x.dtype != q.dtype or not x.is_floating_point() or not torch.isfinite(x).all()):
                raise ValueError('Captured stage feature schema differs')
            aligned = x if branch=='cnn' else broadcast_regions(q,x)
            if not torch.isfinite(aligned).all(): raise ValueError('Nonfinite aligned features')
            features[stage+'_'+branch] = {role:aligned[overlay[role+'_indices']].detach().clone() for role in ('train','val')}
    check_files(pins)
    return {'features':features,'labels':{r:overlay[r+'_labels'] for r in ('train','val')},
        'row_ids':{r:overlay[r+'_indices'] for r in ('train','val')},
        'identity':{'policy':POLICY,'capture_receipt_sha256':file_hash(path),'case':receipt['case'],
            'selected_model_sha256':selected['primary_sha256'],'selected_step':selected['selected_step'],
            'freeze_sha256':receipt['freeze_sha256'],'analysis_sha256':receipt['analysis_sha256'],
            'family_hash':family['family_hash'],'phase_hashes':{k:digest(v) for k,v in releases.items()},
            'query_labels_accessed':False,'label_authorization':'accepted_case_family_K10_V5_phases',
            'scope':'same_scene_transductive_train_val; reused selection validation, not independent spatial confirmation'}}


def export(capture_path,capture_sha256,freeze_path,analysis_path,out,code):
    require_frozen_code(code)
    bound = _authenticate(capture_path,capture_sha256,freeze_path,analysis_path);pins = bound[-1]
    out = external_directory(out)
    if out.exists(): raise FileExistsError('Fresh external probe-pack output required')
    for p in map(Path,pins):
        if out==p or out in p.parents or p.parent in out.parents:
            raise ValueError('Probe output must be outside protected input directories')
    pack = _pack(bound)
    require_frozen_code(code);check_files(pins);out.mkdir(parents=True,exist_ok=False)
    torch.save(pack,out/'pack.pt')
    result = {'status':'passed','policy':POLICY,'code':code,'case':bound[1]['case'],'identity':pack['identity'],
        'protected_files':pins,'outputs':{'pack.pt':file_hash(out/'pack.pt')},'model_loads':0,'model_forwards':0,
        'backbone_updates':0,'probe_updates':0,'query_labels_accessed':False,
        'train_rows':len(pack['labels']['train']),'val_rows':len(pack['labels']['val']),'layers':4}
    check_files(pins);write_json(out/'receipt.json',result);return result
