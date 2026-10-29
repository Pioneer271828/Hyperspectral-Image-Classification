"""Explicit, read-only dataset inventory; no machine-specific fallback paths."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat, whosmat

DATASETS = {
    'paviaU': ('PaviaU/PaviaU', 'paviaU', 9, 'ROSIS'),
    'xuzhou': ('Xuzhou/xuzhou', 'xuzhou', 9, 'unknown'),
    'whu-honghu': ('WHU-Hi/WHU-Hi-HongHu/WHU_Hi_HongHu', 'WHU_Hi_HongHu', 22, 'WHU-Hi'),
    'WHUL': ('WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou', 'WHU_Hi_LongKou', 9, 'WHU-Hi'),
    'whu-hanchuan': ('WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan', 'WHU_Hi_HanChuan', 16, 'WHU-Hi'),
    'laoyuhesidi': ('LaoYuHeSidi/LapYuHeSiDi', None, 8, 'unknown'),
}

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def inspect_dataset(root, name):
    if name not in DATASETS:
        raise ValueError(f'Unregistered dataset: {name}')
    stem, key, expected_classes, family = DATASETS[name]
    root = Path(root).resolve(strict=True)
    image_path = root / (stem + ('.mat' if key else '.npy'))
    label_path = root / (stem + ('_gt.mat' if key else '_label.npy'))
    for path in (image_path, label_path):
        if root not in path.resolve(strict=True).parents:
            raise ValueError('Dataset path escapes root')
    if key:
        headers = {k: shape for k, shape, _ in whosmat(image_path)}
        shape = tuple(headers[key])
        labels = loadmat(label_path, variable_names=[key + '_gt'])[key + '_gt']
    else:
        image = np.load(image_path, mmap_mode='r', allow_pickle=False)
        shape = image.shape
        labels = np.load(label_path, mmap_mode='r', allow_pickle=False)
    if len(shape) != 3 or tuple(shape[:2]) != labels.shape:
        raise ValueError('Image/label shape mismatch')
    if not np.isfinite(labels).all() or (labels < 0).any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels must be finite nonnegative integers')
    classes = np.unique(labels[labels > 0]).astype(int).tolist()
    if classes != list(range(1, expected_classes + 1)):
        raise ValueError('Unexpected or noncontiguous foreground labels')
    record = {'schema_version': 1, 'name': name, 'shape': list(shape), 'class_count': expected_classes,
              'sensor_family': family, 'wavelength_quality': 'not_verified', 'fwhm_available': False,
              'unlabeled_value': 0, 'image_path': str(image_path.relative_to(root)),
              'label_path': str(label_path.relative_to(root)), 'image_sha256': file_hash(image_path),
              'label_sha256': file_hash(label_path), 'label_counts': {str(c): int((labels == c).sum()) for c in [0, *classes]}}
    return record, np.asarray(labels)
