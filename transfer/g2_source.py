"""Source-side label releases and shared heads for the strict G2 contract."""
import hashlib
import json

import numpy as np
import torch

from loadData.transfer_protocol import _pool
from model.research_config import digest
from train_spatial import SpatialConfig
from transfer.g2_contract import fields, validate_contract


SHARED_TASK = 'g2_shared_semantic'


def source_labels(catalog, contract, scene, phase, plan, payload):
    """Decode only a frozen sparse source release and project it into head order."""
    validate_contract(catalog, contract)
    if scene not in contract['sources'] or phase not in ('source_train', 'source_select'):
        raise ValueError('Labels require a source training or source selection phase')
    partition = 'train' if phase == 'source_train' else 'val'
    release = contract['source_releases'][scene][partition]
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != release['labels_sha256']:
        raise ValueError('Source label release hash mismatch')
    record = json.loads(payload)
    fields(record, 'scene partition plan_hash observation_sha256 indices labels', 'source label release')
    allowed = _pool(plan, partition)
    if (record['scene'] != scene or record['partition'] != partition
            or record['plan_hash'] != plan['plan_hash']
            or record['observation_sha256'] != release['observation_sha256']):
        raise ValueError('Source release role, geometry or observation mismatch')
    indices, labels = record['indices'], record['labels']

    size = plan['shape'][0] * plan['shape'][1]
    class_count = catalog['scenes'][scene]['class_count']
    if (not isinstance(indices, list) or not isinstance(labels, list) or len(indices) != len(labels)
            or any(type(i) is not int or not 0 <= i < size for i in indices)
            or indices != sorted(set(indices)) or not np.isin(indices, allowed).all()
            or any(type(label) is not int or not 1 <= label <= class_count for label in labels)):
        raise ValueError('Source labels escape the frozen role or class table')

    order = {semantic: i for i, semantic in enumerate(contract['semantic_classes'], start=1)}
    mapping = {int(native): order[semantic] for native, semantic in contract['semantic_mapping'][scene].items()}
    projected = np.array([mapping.get(label, 0) for label in labels], dtype=np.int64)
    counts = np.bincount(projected, minlength=len(order) + 1)[1:]
    if (counts == 0).any():
        raise ValueError('Source role lacks a shared class')
    result = np.zeros(plan['shape'], dtype=np.int64)
    result.reshape(-1)[indices] = projected
    result.setflags(write=False)
    return result, counts.tolist()


def _identity(contract):
    return {
        'contract_digest': digest(contract),
        'head_key': SHARED_TASK,
        'ordered_classes': list(contract['semantic_classes']),
    }


def build_source_model(catalog, contract, base, *, seed, device='cpu'):
    """Start from fresh weights with one ordered classifier per branch."""
    validate_contract(catalog, contract)
    config = SpatialConfig({
        **base,
        'dataset': contract['sources'][0],
        'device': device,
        'training': {**base.get('training', {}), 'seed': seed},
    })
    if config.research.model.norm_type != 'group_norm':
        raise ValueError('G2 shared heads require scene-independent GroupNorm')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = config.research.build_model({SHARED_TASK: len(contract['semantic_classes'])})
    model.training_config = {'g2_source': _identity(contract)}
    return model.to(device), config


def require_shared_heads(catalog, contract, model):
    """Check the architecture and semantic identity after strict checkpoint reload."""
    validate_contract(catalog, contract)
    metadata = getattr(model, 'training_config', {}).get('g2_source')
    expected = {SHARED_TASK: len(contract['semantic_classes'])}
    families = (model.heads, model.hgnn_heads, model.cnn_heads)
    if (metadata != _identity(contract)
            or model.backbone_kwargs['norm_type'] != 'group_norm'
            or model.backbone.graph_mode != 'learned'
            or any({key: layer.out_features for key, layer in family.items()} != expected for family in families)):
        raise ValueError('Checkpoint does not carry the registered shared G2 heads')
    return SHARED_TASK
