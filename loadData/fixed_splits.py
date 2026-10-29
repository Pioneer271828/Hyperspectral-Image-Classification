"""Versioned per-class random-pixel manifests; explicitly not spatial isolation."""
import hashlib
import json
from pathlib import Path
import numpy as np

def label_hash(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isfinite(labels).all() or (labels < 0).any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Expected a 2D nonnegative integer label image')
    return hashlib.sha256(np.asarray(labels, dtype='<i8').tobytes()).hexdigest()

def make_split(labels, seed=2026, train_num=30, val_num=10):
    fingerprint = label_hash(labels)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid seed')
    if type(train_num) is not int or type(val_num) is not int or min(train_num, val_num) <= 0:
        raise ValueError('Positive integer per-class counts required')
    rng = np.random.default_rng(seed)
    flat = np.asarray(labels).reshape(-1)
    groups = {k: [] for k in ('train', 'val', 'test')}
    for cls in np.unique(flat[flat > 0]):
        indices = np.flatnonzero(flat == cls)
        if len(indices) <= train_num + val_num:
            raise ValueError(f'Class {cls} has insufficient samples to retain test pixels')
        rng.shuffle(indices)
        for name, values in zip(groups, (indices[:train_num], indices[train_num:train_num+val_num], indices[train_num+val_num:])):
            groups[name].extend(sorted(values.tolist()))
    result = {'schema_version': 1, 'protocol': 'fixed_random_pixel_v1', 'shape': list(labels.shape),
              'label_hash': fingerprint, 'seed': seed, 'train_num': train_num, 'val_num': val_num,
              'indices': groups, 'spatial_isolation': False}
    validate_split(result, labels)
    return result

def validate_split(value, labels):
    if value.get('schema_version') != 1 or value.get('protocol') != 'fixed_random_pixel_v1' or value.get('spatial_isolation') is not False:
        raise ValueError('Unsupported split schema/protocol')
    if value['label_hash'] != label_hash(labels) or value['shape'] != list(labels.shape):
        raise ValueError('Split belongs to different labels')
    flat = np.asarray(labels).reshape(-1)
    if set(value['indices']) != {'train', 'val', 'test'}:
        raise ValueError('Three supervision sets required')
    sets = []
    for name in ('train', 'val', 'test'):
        ids = value['indices'][name]
        if any(type(i) is not int or i < 0 or i >= flat.size for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or repeated pixel index')
        if any(flat[i] == 0 for i in ids):
            raise ValueError('Unlabeled pixels cannot enter supervision')
        if name != 'test':
            for cls in np.unique(flat[flat > 0]):
                if sum(flat[i] == cls for i in ids) != value[name + '_num']:
                    raise ValueError('Per-class sample budget differs')
        sets.append(set(ids))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Supervision sets overlap')
    if set.union(*sets) != set(np.flatnonzero(flat > 0)):
        raise ValueError('Foreground coverage incomplete')

def save_split(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)

def load_split(path, labels):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_split(value, labels)
    return value
