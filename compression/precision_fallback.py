"""Compare fixed precision assignments on a frozen spatial development cohort."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction

import numpy as np
import torch

from compression.int8_model import ALLOWED_PATHS, calibrate_model, calibration_contract, convert_model
from evaluation.classification_metrics import classification_metrics
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from monitoring.dependency_groups import HEADS, _protection
from research_runtime import tensor_hash
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward
from transfer.observation_policy import scene_identity


LAYER_ORDER = tuple(sorted(ALLOWED_PATHS))
PRECISION_ASSIGNMENTS = ((LAYER_ORDER[0],), (LAYER_ORDER[1],), LAYER_ORDER)
LIMITS = {metric: Fraction(1, 200) for metric in ('oa', 'aa', 'recall')}
SMALL_CLASS_MAX_SUPPORT = 25


def _validation_inputs(scene, record):
    identity = scene_identity(scene, record)
    if list(scene.cube.shape[:2]) != scene.plan['shape']:
        raise ValueError('Scene and frozen development geometry differ')
    labels = sparse_labels(record, scene.plan, 'validation_support')
    support = Counter(record['train_support']['labels'])
    blocks = []
    for tile, role in zip(scene.plan['tiles'], record['roles'], strict=True):
        if role['role'] != 'devval':
            continue
        local = labels[tile_slice(tile)].reshape(-1)
        mask = local > 0
        if mask.any():
            blocks.append((tile, mask, local[mask]))
    specification = {
        'identity': identity,
        'class_ids': list(record['classes']),
        'train_support': {str(label): support[label] for label in record['classes']},
        'tiles': [{'id': tile['id'], 'sha256': tensor_hash(scene.cube[tile_slice(tile)])}
                  for tile, _, _ in blocks],
    }
    return specification, blocks


def register_ptq_fallback(reference, scene, record, calibration_tile_ids):
    """Pin the FP32 reference, training calibration tiles, and validation cohort."""
    calibration = calibration_contract(reference, scene, record, LAYER_ORDER, calibration_tile_ids, qat=False)
    validation, _ = _validation_inputs(scene, record)
    identity = validation['identity']['development_hash']
    if getattr(reference, 'training_config', {}).get('development_hash', identity) != identity:
        raise ValueError('Reference checkpoint was trained on a different development record')
    return {
        'schema': 'wp10_ptq_precision_fallback_v1',
        'calibration': calibration,
        'validation': validation,
        'assignments': [list(paths) for paths in PRECISION_ASSIGNMENTS],
        'guards': {'fused_max_drop': {name: float(limit) for name, limit in LIMITS.items()},
                   'small_class_max_train_support': SMALL_CLASS_MAX_SUPPORT, 'max_new_small_class_errors': 0,
                   'joint_requires_passed_singles': True},
        'selection': 'most_integer_layers_then_fixed_layer_order',
    }


@torch.no_grad()
def _predict(model, scene, blocks, class_ids):
    scores = {head: [] for head in HEADS}
    for tile, mask, _ in blocks:
        outputs = _forward(model, scene, tile)
        for head, value in zip(HEADS, outputs[:3], strict=True):
            if tuple(value.shape) != (len(mask), len(class_ids)) or not torch.isfinite(value).all():
                raise ValueError('Finite, aligned three-head validation output required')
            scores[head].append(np.asarray(class_ids)[value[mask].argmax(1).cpu().numpy()])
    return {head: np.concatenate(parts) for head, parts in scores.items()}


def _evaluate_assignments(reference, scene, blocks, validation, make_candidate):
    class_ids = validation['class_ids']
    truth = np.concatenate([local for _, _, local in blocks])
    if set(truth.tolist()) != set(class_ids):
        raise ValueError('Every class needs fixed validation support')
    small = [label for label in class_ids if validation['train_support'][str(label)] <= SMALL_CLASS_MAX_SUPPORT]
    fp32 = deepcopy(reference).eval()
    fp32.backbone.adapter.cache_attention = False
    before = _predict(fp32, scene, blocks, class_ids)
    baseline = {head: classification_metrics(truth, before[head], class_ids) for head in HEADS}
    rows, singles = [], {}
    selected, selected_paths = fp32, ()
    for paths in PRECISION_ASSIGNMENTS:
        candidate = make_candidate(paths)
        after = _predict(candidate, scene, blocks, class_ids)
        metrics = {head: classification_metrics(truth, after[head], class_ids) for head in HEADS}
        guards = {head: _protection(baseline[head], metrics[head], truth, before[head], after[head],
                                    class_ids, small, LIMITS) for head in HEADS}
        failed_singles = [path for path in paths if len(paths) > 1 and not singles[path]]
        passed = guards['fused']['passes_metric_guards'] and not failed_singles
        if len(paths) == 1:
            singles[paths[0]] = passed
        if passed and len(paths) > len(selected_paths):
            selected, selected_paths = candidate, paths
        rows.append({'low_precision_paths': list(paths), 'metrics': metrics, 'protection': guards,
                     'failed_single_layers': failed_singles, 'passed': passed})
    return selected, {
        'baseline': baseline,
        'trials': rows,
        'protected_small_classes': small,
        'selected_low_precision_paths': list(selected_paths),
        'kept_fp32_paths': [path for path in LAYER_ORDER if path not in selected_paths],
        'optimizer_updates': 0,
        'query_labels_accessed': False,
    }


def run_ptq_fallback(reference, scene, record, *, plan, expected_plan_sha256):
    """Materialize the validation-selected integer/FP32 assignment as an owned model."""
    if digest(plan) != expected_plan_sha256:
        raise ValueError('Precision fallback plan hash mismatch')
    tile_ids = [row['tile_id'] for row in plan['calibration']['inputs']]
    if register_ptq_fallback(reference, scene, record, tile_ids) != plan:
        raise ValueError('Precision fallback input or guard contract changed')
    validation, blocks = _validation_inputs(scene, record)

    def make_candidate(paths):
        contract = {**plan['calibration'], 'paths': list(paths)}
        calibrated = calibrate_model(reference, scene, record, paths=paths, contract=contract,
                                     expected_hash=digest(contract))
        return convert_model(calibrated)

    selected, report = _evaluate_assignments(reference, scene, blocks, validation, make_candidate)
    if register_ptq_fallback(reference, scene, record, tile_ids) != plan:
        raise RuntimeError('Precision fallback inputs changed during validation')
    return selected, {**report, 'plan_sha256': expected_plan_sha256, 'precision': 'cpu_int8_ptq',
                      'evaluation_role': 'devval'}
