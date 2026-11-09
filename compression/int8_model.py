"""Explicit CNN-only quantization bound to existing spatial development roles."""
import copy

import torch
from torch import nn

from compression.int8 import CalibratedMatrix, IntegerMatrix
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.g3_training import _forward, state_hash
from transfer.observation_policy import scene_identity


ALLOWED_PATHS = frozenset({'backbone.CNN_Branch.0.point_conv',
                           'backbone.CNN_Branch.1.point_conv'})


def _paths(paths):
    if not isinstance(paths, (list, tuple)) or not paths or any(type(p) is not str for p in paths):
        raise ValueError('explicit layer path list required')
    if len(set(paths)) != len(paths) or not set(paths) <= ALLOWED_PATHS:
        raise ValueError('only unique CNN pointwise whitelist paths are allowed')
    return sorted(paths)


def _dense(model):
    if type(model) is not MultiDatasetVCASAHGCN or hasattr(model, '_int8_plan'):
        raise ValueError('unquantized MultiDatasetVCASAHGCN required')
    for value in model.state_dict().values():
        if (not torch.is_tensor(value) or value.device.type != 'cpu'
                or not torch.isfinite(value).all()
                or (value.is_floating_point() and value.dtype != torch.float32)):
            raise ValueError('finite CPU FP32 model required')


def calibration_contract(model, scene, record, paths, tile_ids, *, qat=False):
    """Freeze model, existing split/statistics/metadata and raw devtrain tile bytes.

    Does not derive new roles or fit any statistics. The returned record must
    be externally pinned (digest or saved-file identity) before calibration.
    """
    _dense(model)
    paths = _paths(paths)
    if type(qat) is not bool:
        raise ValueError('qat must be boolean')
    if record is not None:
        identity = scene_identity(scene, record)
        roles, fitting_role = record['roles'], 'devtrain'
    else:
        # Original spatial protocols have train/val/test partitions without a
        # nested development split. Never erase a known development boundary.
        validate_tile_plan(scene.plan)
        stats = scene.stats
        if (stats.get('fit_partition') != 'train' or 'fit_role' in stats
                or 'development_hash' in stats
                or getattr(model, 'training_config', {}).get('development_hash')
                or stats.get('plan_hash') != scene.plan['plan_hash']
                or stats.get('statistics_hash') != digest({k:v for k,v in stats.items() if k != 'statistics_hash'})):
            raise ValueError('plain spatial calibration requires original train-only statistics')
        identity = dict(plan_hash=scene.plan['plan_hash'], development_hash=None,
                        statistics_hash=stats['statistics_hash'], metadata_hash=scene.bands.fingerprint,
                        dataset=scene.dataset, classes=scene.classes)
        roles = [{**tile, 'role': tile['partition']} for tile in scene.plan['tiles']]
        fitting_role = 'train'
    heads = model.export_config()['class_counts']
    if scene.dataset not in heads or heads[scene.dataset] != scene.classes:
        raise ValueError('model head differs from scene classes')
    if (not isinstance(tile_ids, (list, tuple)) or not tile_ids
            or any(type(i) is not int or not 0 <= i < len(roles) for i in tile_ids)
            or len(set(tile_ids)) != len(tile_ids)):
        raise ValueError('explicit unique calibration tile IDs required')
    # Validate every role before touching any calibration image data.
    if any(roles[i]['role'] != fitting_role for i in tile_ids):
        raise ValueError('calibration accepts registered training tiles only')
    for path in paths:
        CalibratedMatrix(model.get_submodule(path), qat=qat)
    inputs = [{'tile_id': i, 'sha256': tensor_hash(scene.cube[tile_slice(scene.plan['tiles'][i])])}
              for i in tile_ids]
    return dict(schema='vca-int8-calibration-v1', source_state=state_hash(model),
                architecture=model.export_config(), identity=identity, paths=paths,
                inputs=inputs, qat=qat, engine=torch.backends.quantized.engine,
                input_range=[0, 127], weight_range=[-127, 127], output_range=[0, 255],
                fitting_role=fitting_role, optimizer_updates=0)


def _replace(model, path, module):
    parent, _, leaf = path.rpartition('.')
    setattr(model.get_submodule(parent), leaf, module)


class _ObserveTraining(nn.Module):
    def __init__(self, calibrated):
        super().__init__()
        self.calibrated = calibrated

    def forward(self, x):
        return self.calibrated.observe(x, role='development_train')


def calibrate_model(model, scene=None, record=None, *, paths=(), contract=None, expected_hash=None):
    """Return owned frozen-range PTQ/QAT model; empty selection is exact deepcopy."""
    _dense(model)
    if not paths:
        if contract is not None or expected_hash is not None:
            raise ValueError('disabled quantization must not carry a calibration contract')
        return copy.deepcopy(model)
    paths = _paths(paths)
    if not isinstance(contract, dict) or digest(contract) != expected_hash:
        raise ValueError('calibration contract hash mismatch')
    tile_ids = [row['tile_id'] for row in contract['inputs']]
    current = calibration_contract(model, scene, record, paths, tile_ids, qat=contract['qat'])
    if current != contract:
        raise ValueError('model/split/statistics/bands/input calibration identity differs')
    result = copy.deepcopy(model).eval()
    result.backbone.adapter.cache_attention = False
    layers = {}
    for path in paths:
        layers[path] = CalibratedMatrix(result.get_submodule(path), qat=contract['qat'])
        _replace(result, path, _ObserveTraining(layers[path]))
    with torch.no_grad():
        for tile_id in tile_ids:
            _forward(result, scene, scene.plan['tiles'][tile_id])
    for path, layer in layers.items():
        _replace(result, path, layer.freeze())
    if calibration_contract(model, scene, record, paths, tile_ids, qat=contract['qat']) != contract:
        raise ValueError('calibration source changed during observation')
    result._int8_plan = dict(phase='qat' if contract['qat'] else 'ptq',
                            contract=copy.deepcopy(contract), contract_hash=expected_hash)
    return result.eval()


def validate_plan(model):
    if type(model) is not MultiDatasetVCASAHGCN or not hasattr(model, '_int8_plan'):
        raise ValueError('explicit quantized VCA model required')
    plan = model._int8_plan
    if (set(plan) != {'phase', 'contract', 'contract_hash'}
            or plan['phase'] not in ('ptq', 'qat', 'integer')
            or digest(plan['contract']) != plan['contract_hash']):
        raise ValueError('invalid quantization plan identity')
    contract = plan['contract']
    paths = _paths(contract['paths'])
    if (contract['schema'] != 'vca-int8-calibration-v1' or contract['architecture'] != model.export_config()
            or contract['engine'] != torch.backends.quantized.engine
            or contract['input_range'] != [0, 127] or contract['weight_range'] != [-127, 127]
            or contract['output_range'] != [0, 255]):
        raise ValueError('quantization architecture/engine/range mismatch')
    if plan['phase'] != 'integer' and (plan['phase'] == 'qat') != contract['qat']:
        raise ValueError('QAT phase mismatch')
    expected = IntegerMatrix if plan['phase'] == 'integer' else CalibratedMatrix
    actual = {name for name, module in model.named_modules()
              if isinstance(module, (CalibratedMatrix, IntegerMatrix))}
    if actual != set(paths) or any(type(model.get_submodule(p)) is not expected for p in paths):
        raise ValueError('quantized topology differs from whitelist plan')
    for path in paths:
        module = model.get_submodule(path)
        if expected is CalibratedMatrix and (not bool(module.ranges_frozen)
                or not int(module.batches_seen) or module.qat != contract['qat']
                or any(bool(q.observer_enabled[0]) or not bool(q.fake_quant_enabled[0])
                       for q in module.quantizers())):
            raise ValueError('quantization ranges are not frozen')
        if expected is IntegerMatrix and (module.engine != contract['engine']
                or module.input_scale.numel() != 1 or not torch.isfinite(module.input_scale).all()
                or not (module.input_scale > 0).all() or not 0 <= int(module.input_zero_point) <= 127
                or not 0 < module.kernel.scale < float('inf')
                or not 0 <= module.kernel.zero_point <= 255
                or module.kernel.weight().dtype != torch.qint8):
            raise ValueError('invalid integer kernel parameters')
    return copy.deepcopy(plan)


def convert_model(model):
    plan = validate_plan(model)
    if plan['phase'] == 'integer':
        raise ValueError('already converted')
    result = copy.deepcopy(model).eval()
    for path in plan['contract']['paths']:
        _replace(result, path, result.get_submodule(path).convert())
    plan['phase'] = 'integer'
    result._int8_plan = plan
    return result.eval()
