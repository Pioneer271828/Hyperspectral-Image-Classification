"""Frozen G2 target inference and a separate, receipt-gated scoring phase."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import observe_tile, validate_tile_plan
from loadData.transfer_protocol import _pool
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.g2_contract import fields, read_pinned, sha, validate_contract, validate_lineage
from transfer.g2_observations import _pack_observations, _read, source_observations
from transfer.g2_source import require_shared_heads
from transfer.g3_training import head_hash, state_hash


HEADS = ('fused', 'graph', 'cnn')
POLICY = 'g2_exact_common_bands_frozen_cpu_v1'


def pack_target_observations(scene, artifact_sha256, plan, cube, wavelengths_nm):
    return _pack_observations(scene, artifact_sha256, plan, 'test', cube, wavelengths_nm)


def _inputs(registration_path, registration_sha256, expected_commit):
    code = code_snapshot()
    if code['commit'] != expected_commit:
        raise ValueError('Unexpected G2 target commit')
    require_frozen_code(code)
    reg = read_pinned(Path(registration_path), registration_sha256)
    fields(reg, 'policy catalog catalog_sha256 contract contract_sha256 source_result source_result_sha256 '
                'source_commit source_registration spectral_evidence spectral_evidence_sha256 '
                'target_plan target_plan_sha256 target_observations target_observations_sha256 '
                'target_labels_sha256 superpixel_scale', 'G2 target registration')
    if reg['policy'] != POLICY or type(reg['superpixel_scale']) is not int or reg['superpixel_scale'] <= 0:
        raise ValueError('Invalid G2 target protocol or scale')
    sha(reg['target_labels_sha256'])
    catalog = read_pinned(Path(reg['catalog']), reg['catalog_sha256'])
    contract = read_pinned(Path(reg['contract']), reg['contract_sha256'])
    if contract['catalog_sha256'] != reg['catalog_sha256']:
        raise ValueError('Target contract catalog identity differs')
    validate_contract(catalog, contract)
    plan = read_pinned(Path(reg['target_plan']), reg['target_plan_sha256'])
    validate_tile_plan(plan)
    return code, reg, catalog, contract, plan


def _selected_source(reg, catalog, contract):
    result_path = Path(reg['source_result'])
    result = read_pinned(result_path, reg['source_result_sha256'])
    if (result['status'] != 'completed_source_cpu_engineering' or result['commit'] != reg['source_commit']
            or result['target_payload_access'] is not False or result['query_accessed'] is not False
            or type(result['selected_step']) is not int or result['selected_step'] < 1):
        raise ValueError('Source result is not an eligible frozen checkpoint')
    source_reg = read_pinned(Path(reg['source_registration']), result['registration_sha256'])
    if (source_reg['contract_sha256'] != reg['contract_sha256']
            or source_reg['catalog_sha256'] != reg['catalog_sha256']):
        raise ValueError('Frozen source registration does not match target contract')
    root = result_path.parent
    relative = f"step-{result['selected_step']:06d}/model.pt"
    if result['selected_checkpoint'] != relative:
        raise ValueError('Unexpected selected source checkpoint path')
    checkpoint = root / relative
    lineage = read_pinned(root / 'lineage.json', result['lineage_sha256'])
    validate_lineage(contract, lineage, reg['contract_sha256'])
    if (hash_file(checkpoint) != result['checkpoint_sha256'] or lineage['selected'] != result['checkpoint_sha256']
            or hash_file(checkpoint.with_suffix('.pt.json')) != result['sidecar_sha256']):
        raise ValueError('Frozen source checkpoint or sidecar changed')
    model = load_model_checkpoint(checkpoint, 'cpu')
    require_shared_heads(catalog, contract, model)
    run = model.training_config['g2_run']
    if (run['selected_step'] != result['selected_step'] or run['registration_sha256'] != result['registration_sha256']
            or state_hash(model) != result['selected_state_sha256']
            or head_hash(model) != lineage['records'][lineage['selected']]['head']['state_sha256']
            or model.backbone.coverage_mode != 'none'):
        raise ValueError('Frozen source model identity or spectral interface differs')
    return model, source_reg, result['checkpoint_sha256']


def _band_positions(values, size):
    if (not isinstance(values, list) or not values
            or any(type(i) is not int or not 0 <= i < size for i in values)
            or len(set(values)) != len(values)):
        raise ValueError('Band correspondence requires valid distinct indices')
    return values


def _prepare_target(reg, catalog, contract, plan, source_reg):
    target = contract['target']
    target_artifact = catalog['scenes'][target]['artifact_sha256']
    evidence = read_pinned(Path(reg['spectral_evidence']), reg['spectral_evidence_sha256'])
    fields(evidence, 'policy source_reference source_artifact_sha256 target_artifact_sha256 '
                     'source_band_indices target_band_indices wavelength_quality radiometry '
                     'provenance_sha256', 'G2 spectral evidence')
    reference = evidence['source_reference']
    if (evidence['policy'] != 'verified_exact_centers_v1' or reference not in contract['sources']
            or evidence['wavelength_quality'] != 'verified'
            or evidence['radiometry'] != 'verified_same_physical_quantity_and_units'
            or evidence['target_artifact_sha256'] != target_artifact
            or evidence['source_artifact_sha256'] != catalog['scenes'][reference]['artifact_sha256']):
        raise ValueError('Exact bands require reviewed wavelength and radiometric evidence')
    sha(evidence['provenance_sha256'])
    entry = source_reg['sources'][reference]
    source_plan = read_pinned(Path(entry['plan']), entry['plan_sha256'])
    source = source_observations(catalog, contract, reference, 'train', source_plan,
                                 Path(entry['train_observations']).read_bytes())
    wave, cube = _read(Path(reg['target_observations']).read_bytes(), reg['target_observations_sha256'],
                       target, target_artifact, plan, 'test')
    s = _band_positions(evidence['source_band_indices'], source.wavelengths_nm.size)
    t = _band_positions(evidence['target_band_indices'], wave.size)
    if not np.array_equal(source.wavelengths_nm[s], wave[t]):
        raise ValueError('Declared common band centers must match exactly')
    source_stats = source.statistics
    stats = dict(schema_version=1, plan_hash=plan['plan_hash'], fit_partition='train',
                 minimum=[source_stats['minimum'][i] for i in s], maximum=[source_stats['maximum'][i] for i in s])
    stats['statistics_hash'] = digest(stats)
    return cube[:, :, t], wave[t], stats, source_stats['statistics_hash']


def infer_target(registration_path, registration_sha256, expected_commit, out):
    code, reg, catalog, contract, plan = _inputs(registration_path, registration_sha256, expected_commit)
    out = external_directory(out)
    if out.exists():
        raise FileExistsError(out)
    model, source_reg, checkpoint_sha = _selected_source(reg, catalog, contract)
    cube, wave, stats, source_stats_hash = _prepare_target(reg, catalog, contract, plan, source_reg)
    model.eval()
    head_key = require_shared_heads(catalog, contract, model)
    ids = []
    probabilities = {name: [] for name in HEADS}
    with torch.inference_mode():
        for tile in plan['tiles']:
            if tile['partition'] != 'test':
                continue
            obs = observe_tile(cube, plan, stats, tile)
            result = forward_partition(model, head_key, obs, wave,
                                       reg['superpixel_scale'], wavelength_unit='nm')
            ids.append(obs.global_ids.ravel())
            for name, values in zip(HEADS, result[:3]):
                probabilities[name].append(values.detach().cpu().numpy())
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    arrays = {name: np.concatenate(parts)[order] for name, parts in probabilities.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite target probabilities')
    require_frozen_code(code)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'predictions.npz', global_ids=ids[order], **arrays)
    receipt = dict(status='predictions_frozen_without_target_labels', commit=expected_commit,
                   registration_sha256=registration_sha256, checkpoint_sha256=checkpoint_sha,
                   predictions_sha256=hash_file(out/'predictions.npz'), pixel_count=int(ids.size),
                   target=contract['target'], plan_hash=plan['plan_hash'],
                   semantic_classes=contract['semantic_classes'], source_statistics_hash=source_stats_hash,
                   optimizer_updates=0, target_label_access=False)
    write_json(out/'receipt.json', receipt)
    return receipt


def score_target(registration_path, registration_sha256, expected_commit, inference_dir, inference_sha256,
                 label_path, out):
    code, reg, catalog, contract, plan = _inputs(registration_path, registration_sha256, expected_commit)
    out = external_directory(out)
    if out.exists():
        raise FileExistsError(out)
    inference_dir = Path(inference_dir)
    receipt = read_pinned(inference_dir/'receipt.json', inference_sha256)
    target = contract['target']
    classes = list(range(1, len(contract['semantic_classes']) + 1))
    if (receipt['status'] != 'predictions_frozen_without_target_labels'
            or receipt['commit'] != expected_commit or receipt['registration_sha256'] != registration_sha256
            or receipt['target'] != target or receipt['plan_hash'] != plan['plan_hash']
            or receipt['semantic_classes'] != contract['semantic_classes']
            or receipt['target_label_access'] is not False or receipt['optimizer_updates'] != 0):
        raise ValueError('Target labels require a matching frozen prediction receipt')
    predictions = inference_dir/'predictions.npz'
    # Decode the same bytes that were verified, even if the file changes during scoring.
    payload = predictions.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt['predictions_sha256']:
        raise ValueError('Frozen predictions changed')
    expected_ids = _pool(plan, 'test')
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {'global_ids', *HEADS} or len(archive.files) != len(HEADS) + 1:
            raise ValueError('Invalid frozen prediction fields')
        ids = archive['global_ids']
        if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected_ids) or ids.size != receipt['pixel_count']:
            raise ValueError('Frozen predictions do not cover exactly the target test blocks')
        arrays = {name: archive[name] for name in HEADS}
    if any(p.shape != (ids.size, len(classes)) or p.dtype.kind != 'f' or not np.isfinite(p).all()
           or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0) for p in arrays.values()):
        raise ValueError('Invalid frozen target probabilities')

    labels = read_pinned(Path(label_path), reg['target_labels_sha256'])
    fields(labels, 'scene partition plan_hash observation_sha256 indices labels', 'target label release')
    selected, native = labels['indices'], labels['labels']
    if (labels['scene'] != target or labels['partition'] != 'test' or labels['plan_hash'] != plan['plan_hash']
            or labels['observation_sha256'] != reg['target_observations_sha256']
            or not isinstance(selected, list) or not isinstance(native, list) or len(selected) != len(native)
            or any(type(i) is not int or not 0 <= i < np.prod(plan['shape']) for i in selected)
            or selected != sorted(set(selected)) or not np.isin(selected, ids).all()
            or any(type(n) is not int or not 1 <= n <= catalog['scenes'][target]['class_count'] for n in native)):
        raise ValueError('Target labels escape the frozen test role or class table')
    order = dict(zip(contract['semantic_classes'], classes))
    mapping = {int(key): order[value] for key, value in contract['semantic_mapping'][target].items()}
    mapped = np.array([mapping.get(n, 0) for n in native], dtype=np.int64)
    included = mapped > 0
    if not included.any():
        raise ValueError('Target test release has no mapped labels')
    positions = np.searchsorted(ids, np.asarray(selected)[included])
    metrics = {}
    for name, prob in arrays.items():
        p = prob[positions]
        truth = mapped[included]
        metrics[name] = classification_metrics(truth, p.argmax(1) + 1, classes)
        metrics[name]['nll'] = float(-np.log(np.maximum(p[np.arange(truth.size), truth - 1], 1e-8)).mean())
    require_frozen_code(code)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='frozen_target_scored', commit=expected_commit, registration_sha256=registration_sha256,
                  inference_sha256=inference_sha256, target_labels_sha256=reg['target_labels_sha256'],
                  semantic_classes=contract['semantic_classes'], excluded_unmapped=int((~included).sum()),
                  metrics=metrics, optimizer_updates=0)
    write_json(out/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for stage in ('infer', 'score'):
        command = stages.add_parser(stage)
        command.add_argument('--registration', dest='registration_path', type=Path, required=True)
        command.add_argument('--registration-sha256', required=True)
        command.add_argument('--expected-commit', required=True)
        command.add_argument('--out', type=Path, required=True)
        if stage == 'score':
            command.add_argument('--inference-dir', type=Path, required=True)
            command.add_argument('--inference-sha256', required=True)
            command.add_argument('--label-path', type=Path, required=True)
    options = vars(parser.parse_args())
    stage = options.pop('stage')
    (infer_target if stage == 'infer' else score_target)(**options)


if __name__ == '__main__':
    main()
