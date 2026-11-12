"""Train a fresh known-class model using one prepared source context."""

import json
from pathlib import Path

import numpy as np
import torch

from model.checkpoint_io import save_model_checkpoint
from research_runtime import require_frozen_code
from train_vca_multidataset import supervised_loss
from transfer.g3_training import _optimizer, build_model, state_hash
from transfer.registered_development import _json_new, _pin, _read
from transfer.online_tta_stream import parse_json
from .known_context_preparation import _raw
from .known_development_authority import read_authority, verify_producer
from .open_set import _semantics, digest
from .open_set_producer import ROLES, _observation
from .registered_open_set import _code, _deferred, _output


def _source_inputs(prepared):
    receipt = json.loads(_read(prepared))
    if (receipt['status'] != 'passed' or receipt['version'] != 1 or
            receipt['full_context_known_only'] is not True or receipt['preprocessing_source_only'] is not True):
        raise ValueError('Known-only context preparation receipt required')
    raw_registration = receipt['registration']
    raw = json.loads(_read(raw_registration))
    if (raw['kind'] != 'known_full_context_preparation_v1' or
            any(raw[key] != receipt[key] for key in ('known', 'unknown', 'dataset', 'labels')) or
            [obs['role'] for obs in raw['observations']] != ROLES or
            [obs['role'] for obs in receipt['observations']] != ROLES):
        raise ValueError('Known-class, scene and role identities must match preparation')
    _semantics(receipt['known'], receipt['unknown'])
    _deferred(raw['labels'])
    known = [row['id'] for row in receipt['known']]
    original, packet = raw['observations'][0], receipt['observations'][0]
    if any(original[key] != packet[key] for key in ('role', 'scene_uid', 'origin', 'superpixel_scale')):
        raise ValueError('Source observation identity changed after preparation')
    a = _raw(original['input'], ROLES[0], known)
    source = _observation(packet['input'], ROLES[0], known)
    low, high = a['spectral'].min(axis=(0, 1)), a['spectral'].max(axis=(0, 1))
    stats = dict(minimum=low.tolist(), maximum=high.tolist(), fit_role='source_train',
                 fit_pixels=int(np.prod(a['spectral'].shape[:2])), source_pin=original['input'], clipping=False)
    if receipt['statistics'] != stats or receipt['statistics_hash'] != digest(stats):
        raise ValueError('Training preprocessing must reproduce source-only statistics')
    h, w = a['spectral'].shape[:2]
    r, c = original['origin']
    sh, sw = original['scene_shape']
    if min(r, c) < 0 or min(sh, sw) < 2 or r + h > sh or c + w > sw:
        raise ValueError('Source context must lie inside its scene')
    yy, xx = np.mgrid[r:r+h, c:c+w]
    coordinates = np.stack((yy / (sh-1), xx / (sw-1)), axis=-1).astype(np.float32)
    scale = np.where(high > low, high-low, np.float32(1))
    if not np.isfinite(scale).all():
        raise ValueError('Normalization range overflow')
    expected_x = np.concatenate(((a['spectral'] - low) / scale, coordinates), axis=-1)
    if (not np.array_equal(source['x'], expected_x) or
            not np.array_equal(source['indices'], a['indices']) or
            not np.array_equal(source['labels'], a['context_labels'].ravel()[a['indices']]) or
            not np.array_equal(source['wavelengths_nm'], a['wavelengths_nm'])):
        raise ValueError('Prepared source packet does not reproduce the authorized raw context')
    used = [prepared, raw_registration, original['input'], packet['input']]
    reserved = [v['input'] for v in (*raw['observations'][1:], *receipt['observations'][1:])] + [raw['labels']]
    paths = [pin['path'] for pin in used + reserved]
    if len(paths) != len(set(paths)):
        raise ValueError('Source and reserved role files must be distinct')
    return receipt, known, packet, source, used, reserved


def train_known_source(registration, *, out, expected_commit):
    code = _code(expected_commit)
    trial = parse_json(_read(registration))
    if not isinstance(trial, dict):
        raise ValueError('Explicit source training registration required')
    protocol = trial.get('kind')
    authorized = protocol == 'known_source_authorized_v2'
    fields = {'version', 'kind', 'prepared', 'config', 'seed', 'updates'}
    if authorized:
        fields.add('authority')
    if (set(trial) != fields or
            (trial['version'], protocol) not in ((1, 'known_source_fixed_budget_v1'), (2, 'known_source_authorized_v2')) or
            type(trial['updates']) is not int or trial['updates'] <= 0 or
            type(trial['seed']) is not int or not 0 <= trial['seed'] < 2**32):
        raise ValueError('Explicit positive update budget and uint32 seed required')
    if authorized:
        authority, _, bands = read_authority(trial['authority'])
        receipt = parse_json(_read(trial['prepared']))
        if (authority['training_commit'] != expected_commit
                or receipt.get('registration') != authority['context_registration']):
            raise ValueError('Source training must bind the authorized preparation and producer')
        verify_producer(receipt['code'], authority['preparation_commit'],
                        ('evaluation/known_context_preparation.py', 'scripts/prepare_open_set_contexts.py'))
    prepared, known, observation, data, used, reserved = _source_inputs(trial['prepared'])
    used += [registration, trial['config']]
    if authorized:
        if not np.array_equal(data['wavelengths_nm'], np.asarray(bands.wavelengths, dtype=np.float32)):
            raise ValueError('Source training physical bands differ from the authority')
        used += [trial['authority'], authority['dataset_evidence']]
    if {pin['path'] for pin in used} & {pin['path'] for pin in reserved}:
        raise ValueError('Training registration and config may not alias reserved role files')
    output = _output(out, used + reserved)
    dataset = prepared['dataset']
    base = json.loads(_read(trial['config']))
    model, config = build_model(base, dataset, len(known), trial['seed'], 'cpu')
    backbone = model.backbone
    if (backbone.extra_channels != 2 or backbone.graph_assignment_mode != 'learned' or
            backbone.graph_adjacency_mode != 'learned' or any(getattr(backbone, name) is not None for name in (
                'interaction', 'region_residual', 'coverage_modulation', 'graph_router', 'edge_updater',
            ))):
        raise ValueError('Known-source model must be the uncoupled learned-graph GroupNorm baseline')
    backbone.adapter.cache_attention = False
    optimizer = _optimizer(model, config)
    initial = state_hash(model)
    x, wavelengths = torch.from_numpy(data['x']), torch.from_numpy(data['wavelengths_nm'])
    ids = torch.from_numpy(data['indices'])
    columns = {label: idx for idx, label in enumerate(known)}
    labels = torch.tensor([columns[int(label)] for label in data['labels']], dtype=torch.long)
    one_hot = x.new_zeros((x.shape[0]*x.shape[1], len(known)))
    one_hot[ids, labels] = 1
    mask = x.new_zeros(one_hot.shape)
    mask[ids] = 1
    inputs = {'train_gt_onehot': one_hot, 'train_label_mask': mask}
    settings = config.research.loss
    weights = dict(compactness=settings.graph_compactness_weight, entropy=settings.graph_entropy_weight,
                   balance=settings.graph_balance_weight)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(trial['seed'])
        for _ in range(trial['updates']):
            optimizer.zero_grad(set_to_none=True)
            values = model(dataset, x, wavelengths=wavelengths, wavelength_unit='nm',
                           superpixel_scale=observation['superpixel_scale'], return_graph_info=True)
            loss = supervised_loss(values, inputs, 'train', class_balance=settings.balanced,
                                   graph_loss_weights=weights, cnn_loss_weight=settings.cnn_loss_weight,
                                   graph_branch_loss_weight=settings.graph_branch_loss_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite source training loss')
            loss.backward()
            optimizer.step()
            history.append(loss.item())
    for pin in used:
        _read(pin)
    require_frozen_code(code)
    model.training_config = dict(phase='known_source', optimizer_updates=len(history), seed=trial['seed'],
                                 selected_step=len(history), selection='fixed_final_step', registration=registration,
                                 context_statistics_hash=prepared['statistics_hash'])
    checkpoint = output / 'model.pt'
    save_model_checkpoint(model, checkpoint)
    model_pin, sidecar = _pin(checkpoint), _pin(Path(str(checkpoint) + '.json'))
    declaration = _json_new(output / 'training-semantics.json', dict(
        version=1, kind='open_set_training_semantics_declaration_v1', checkpoint_sha256=model_pin['sha256'],
        sidecar_sha256=sidecar['sha256'], dataset=dataset, known=prepared['known'], source_training_class_ids=known,
        unknown_training_excluded=True,
    ))
    return _json_new(output / 'receipt.json', dict(
        status='completed', code=code, registration=registration, inputs=used, checkpoint=model_pin, sidecar=sidecar,
        training_declaration=declaration, initial_state_sha256=initial, final_state_sha256=state_hash(model),
        source_annotated_pixels=int(x.shape[0]*x.shape[1]), source_supervised_pixels=len(ids),
        optimizer_updates=len(history), selected_step=len(history), selection='fixed_final_step', loss=history,
        calibration_read=False, evaluation_read=False, query=0,
    ))
