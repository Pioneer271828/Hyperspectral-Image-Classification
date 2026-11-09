"""External, authenticated development-only inputs and E3 attempt execution."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import canonical_json, digest
from research_runtime import external_directory, code_snapshot, require_frozen_code
from transfer.container_runtime import training_container_kind
from transfer.curriculum_exposure import ExposurePlan
from transfer.development_training import prepare_scene, validate
from transfer.graph_route_context import RouteContextBank, route_state_hash
from transfer.graph_route_training import GraphRouteRun
from transfer.observation_policy import scene_identity


def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    external_directory(path)
    return dict(path=str(path), sha256=_sha(path.read_bytes()))


def _read(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('Explicit file path/hash required')
    path = Path(pin['path']).resolve(strict=True); external_directory(path)
    raw = path.read_bytes()
    if _sha(raw) != pin['sha256']: raise ValueError('Input file identity changed: '+str(path))
    return raw


def _json_new(path, value):
    with Path(path).open('xb') as f:
        f.write((canonical_json(value)+'\n').encode()); f.flush(); os.fsync(f.fileno())
    return _pin(path)


def _released_record(record):
    # This transport accepts only the released development protocol schema.
    # Reject extra label arrays even when a caller has recomputed its hash.
    fields = {'schema_version','kind','parent_plan_hash','shape','classes','roles','heldout_tile_ids',
              'training_candidates','validation_candidates','train_support','validation_support','training_labels',
              'validation_labels','total_training_selection_labels','selection','test_unchanged','rule',
              'oracle_disclosure','normalization_contract','runner_compatible','validation_hash'}
    if set(record) != fields or record['schema_version'] != 1 or record['kind'] != 'oracle_spatial_development_v1':
        raise ValueError('Only the released development record may be packaged')
    for name in ('train_support', 'validation_support'):
        if set(record[name]) != {'indices','labels','seed'}:
            raise ValueError('Unregistered support payload')

def export_development(scene, record, out):
    """Never index or serialize non-development observations or original GT."""
    _released_record(record)
    identity = scene_identity(scene, record)
    out = external_directory(out); out.mkdir(parents=True, exist_ok=False); rows = []
    for tile in record['roles']:
        if tile['role'] not in ('devtrain', 'devval'): continue
        a = np.ascontiguousarray(scene.cube[tile_slice(tile)])
        if a.dtype != np.float32 or not np.isfinite(a).all(): raise ValueError('Finite FP32 development block required')
        path = out/f'tile-{tile["id"]}.npy'
        with path.open('xb') as f: np.save(f, a, allow_pickle=False)
        rows.append(dict(id=tile['id'], role=tile['role'], name=path.name, sha256=_sha(path.read_bytes())))
    manifest = dict(version=1, dataset=scene.dataset, plan=deepcopy(scene.plan), record=deepcopy(record),
                    bands=asdict(scene.bands), statistics=deepcopy(scene.stats), identity=identity, blocks=rows,
                    excluded='all_non_development_observations_and_original_GT')
    return _json_new(out/'development.json', manifest)


def load_development(pin):
    v = json.loads(_read(pin)); folder = Path(pin['path']).resolve().parent
    if (set(v) != {'version','dataset','plan','record','bands','statistics','identity','blocks','excluded'}
            or v['version'] != 1 or v['excluded'] != 'all_non_development_observations_and_original_GT'):
        raise ValueError('Unsupported development package')
    _released_record(v['record'])
    validate(v['record'], v['plan']); bands = BandMetadata(**v['bands'])
    permitted = [r for r in v['record']['roles'] if r['role'] in ('devtrain', 'devval')]
    if len(v['blocks']) != len(permitted): raise ValueError('Development block coverage differs')
    cube = np.zeros((*v['plan']['shape'], len(bands.band_ids)), dtype=np.float32)
    files = [deepcopy(pin)]
    for row, tile in zip(v['blocks'], permitted):
        if (set(row) != {'id','role','name','sha256'} or row['id'] != tile['id'] or row['role'] != tile['role']
                or row['name'] != f'tile-{tile["id"]}.npy'):
            raise ValueError('Block identity/order mismatch')
        path = (folder/row['name']).resolve(strict=True)
        if path.parent != folder: raise ValueError('Block escapes package directory')
        item = dict(path=str(path), sha256=row['sha256']); raw = _read(item)
        a = np.load(io.BytesIO(raw), allow_pickle=False)
        y0,y1,x0,x1 = tile['box']
        if a.dtype != np.float32 or a.shape != (y1-y0,x1-x0,len(bands.band_ids)) or not np.isfinite(a).all():
            raise ValueError('Invalid development block array')
        cube[tile_slice(tile)] = a; files.append(item)
    scene = prepare_scene(cube, bands, v['plan'], v['record']); scene.dataset = v['dataset']
    if scene.stats != v['statistics'] or scene_identity(scene, v['record']) != v['identity']:
        raise ValueError('Development normalization/identity changed')
    return scene, v['record'], files


def _load_source(weight, metadata):
    raw = _read(weight); config = json.loads(_read(metadata))
    if set(config) != {'format_version','class_counts','backbone_kwargs','training_config'} or config['format_version'] != 1:
        raise ValueError('Explicit checkpoint architecture/training metadata required')
    with torch.random.fork_rng(devices=[]):
        model = MultiDatasetVCASAHGCN(config['class_counts'], **config['backbone_kwargs']).cpu().eval()
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), strict=True)
    model.training_config = deepcopy(config['training_config']); model.backbone.adapter.cache_attention = False
    return model


def register_route(source_path, development, plan, out, *, lr, torch_seed, expected_commit,
                   bank_path=None, expected_bank_hash=None):
    """Explicit preparation only: no optimizer step, no auto-enqueue."""
    weight = _pin(source_path); metadata = _pin(str(source_path)+'.json')
    source = _load_source(weight, metadata); scene, record, files = load_development(development)
    if not isinstance(plan, ExposurePlan): raise ValueError('Pre-registered exposure plan required')
    bank_file = None if bank_path is None else _pin(bank_path)
    bank = None if bank_file is None else RouteContextBank(json.loads(_read(bank_file)), expected_hash=expected_bank_hash)
    if bank is None and expected_bank_hash is not None: raise ValueError('Unexpected fixed bank hash')
    run = GraphRouteRun(source, scene, record, plan=plan, bank=bank, expected_bank_hash=expected_bank_hash,
                        lr=lr, torch_seed=torch_seed, expected_source_state=route_state_hash(source), expected_plan_hash=plan.plan_hash)
    if not isinstance(expected_commit, str) or len(expected_commit) != 40 or any(c not in '0123456789abcdef' for c in expected_commit):
        raise ValueError('Full production commit required')
    value = dict(version=1, expected_commit=expected_commit, source=weight, metadata=metadata,
                 development=deepcopy(development), plan=plan.to_dict(), bank=bank_file, bank_hash=expected_bank_hash,
                 lr=lr, torch_seed=torch_seed, contract_hash=run.contract_hash, spec=run.spec)
    out = external_directory(out)
    for f in [weight, metadata, *files, *([] if bank_file is None else [bank_file])]:
        if out == Path(f['path']): raise ValueError('Registration cannot overwrite an input')
    out.parent.mkdir(parents=True, exist_ok=True)
    return _json_new(out, value)


def _load_registration(pin):
    v = json.loads(_read(pin))
    if set(v) != {'version','expected_commit','source','metadata','development','plan','bank','bank_hash',
                  'lr','torch_seed','contract_hash','spec'} or v['version'] != 1:
        raise ValueError('Unsupported route registration')
    scene, record, files = load_development(v['development']); source = _load_source(v['source'], v['metadata'])
    plan = ExposurePlan(scene, record, v['plan'], expected_plan_hash=v['plan']['plan_hash'])
    bank = None if v['bank'] is None else RouteContextBank(json.loads(_read(v['bank'])), expected_hash=v['bank_hash'])
    run = GraphRouteRun(source, scene, record, plan=plan, bank=bank, expected_bank_hash=v['bank_hash'],
            lr=v['lr'], torch_seed=v['torch_seed'], expected_source_state=route_state_hash(source), expected_plan_hash=plan.plan_hash)
    if run.contract_hash != v['contract_hash'] or canonical_json(run.spec) != canonical_json(v['spec']): raise ValueError('Registered execution contract changed')
    return v, run, [deepcopy(pin), v['source'], v['metadata'], *files, *([] if v['bank'] is None else [v['bank']])]


def execute_route(registration, *, out, to_step, expected_commit, resume=None):
    """New attempt directory for each segment. Caller supplies exact resume bytes."""
    code = code_snapshot()
    if not training_container_kind() or sys.flags.optimize or code['dirty'] or code['commit'] != expected_commit:
        raise ValueError('Clean exact registered code in authorized Linux container required')
    v, run, files = _load_registration(registration)
    if v['expected_commit'] != expected_commit: raise ValueError('Registration/code commit mismatch')
    if type(to_step) is not int or not 0 <= to_step <= run.plan.updates: raise ValueError('Absolute target exceeds frozen budget')
    if resume is not None:
        _read(resume); files.append(deepcopy(resume))
        run = GraphRouteRun.load(run.source, run.scene, run.record, resume['path'], plan=run.plan,
            bank=run.bank, expected_bank_hash=v['bank_hash'], expected_sha256=resume['sha256'], expected_contract_hash=v['contract_hash'])
    if to_step < run.completed: raise ValueError('Cannot rewind an accepted prefix')
    out = external_directory(out)
    # Protect complete input directories, including the previous attempt if resuming.
    for pin in files:
        parent = Path(pin['path']).resolve().parent
        if out == parent or out in parent.parents or parent in out.parents:
            raise ValueError('Attempt output overlaps an input/history directory')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); initial = run.completed
    status = dict(status='running', code=code, registration=deepcopy(registration), contract_hash=run.contract_hash,
                  initial_updates=initial, completed_updates=initial, target_updates=to_step, total_budget=run.plan.updates,
                  new_updates=0, checkpoint=None, query=0, inputs=files, command=list(sys.argv))
    def save_status():
        temp = out/'status.tmp'
        with temp.open('xb') as f:
            f.write((canonical_json(status)+'\n').encode()); f.flush(); os.fsync(f.fileno())
        os.replace(temp, out/'status.json')
    def checkpoint():
        path = out/f'step-{run.completed}.pt'; pin = dict(path=str(path), sha256=run.save(path))
        status.update(checkpoint=pin, completed_updates=run.completed, new_updates=run.completed-initial)
        save_status()
    try:
        checkpoint()
        while run.completed < to_step:
            run.step(); checkpoint()
        for pin in files: _read(pin)
        require_frozen_code(code)
        status.update(status='completed' if run.completed == run.plan.updates else 'partial',
                      acceptance=run.finish() if run.completed == run.plan.updates else None,
                      independent_acceptance=False, elapsed_seconds=time.monotonic()-started)
        save_status(); return deepcopy(status)
    except BaseException as e:
        if not run.failed:
            run.failed = True; run.failure = dict(type=type(e).__name__, message=str(e), attempted_step=run.completed+1)
        failure = out/'failure.pt'; pin = dict(path=str(failure), sha256=run.save_failure(failure))
        status.update(status='failed', failure=pin, error=repr(e), elapsed_seconds=time.monotonic()-started)
        save_status(); raise
