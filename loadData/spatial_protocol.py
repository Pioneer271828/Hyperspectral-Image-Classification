"""Rectangular spatial isolation, train-only statistics and label-free model inputs."""
from dataclasses import dataclass
import numpy as np
from model.research_config import digest

PARTITIONS = ('train', 'val', 'test')

def make_spatial_plan(shape, axis=0, fractions=(0.6, 0.2, 0.2), gap=4):
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('Positive integer H/W required')
    if type(axis) is not int or axis not in (0, 1) or type(gap) is not int or gap < 0:
        raise ValueError('Invalid axis/gap')
    if len(fractions) != 3 or not np.isfinite(fractions).all() or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError('Three positive fractions summing to one required')
    length = shape[axis]
    first, second = int(length * fractions[0]), int(length * (fractions[0] + fractions[1]))
    boxes = {}
    for name, (start, stop) in zip(PARTITIONS, ((0, first-gap), (first+gap, second-gap), (second+gap, length))):
        if stop <= start:
            raise ValueError('Partition empty after guard removal')
        box = [0, shape[0], 0, shape[1]]
        box[axis*2:axis*2+2] = [start, stop]
        boxes[name] = box
    value = {'schema_version': 1, 'protocol': 'spatial_stripes_v1', 'shape': list(shape),
             'axis': axis, 'fractions': list(fractions), 'gap': gap, 'boxes': boxes,
             'normalization': 'train_region_per_band_minmax', 'graph_mode': 'learned'}
    value['plan_hash'] = digest(value)
    return value

def validate_plan(plan):
    expected = make_spatial_plan(tuple(plan['shape']), plan['axis'], tuple(plan['fractions']), plan['gap'])
    if plan != expected:
        raise ValueError('Spatial plan altered or unsupported')

def slices(plan, partition):
    validate_plan(plan)
    if partition not in PARTITIONS:
        raise ValueError('Unknown partition')
    y0,y1,x0,x1 = plan['boxes'][partition]
    return slice(y0,y1), slice(x0,x1)

def fit_train_statistics(cube, plan):
    validate_plan(plan)
    if cube.ndim != 3 or list(cube.shape[:2]) != plan['shape']:
        raise ValueError('Cube shape differs from plan')
    train = np.asarray(cube[slices(plan, 'train')], dtype=np.float64)
    if not np.isfinite(train).all():
        raise ValueError('Nonfinite training observation')
    minimum, maximum = train.min(axis=(0,1)), train.max(axis=(0,1))
    value = {'schema_version': 1, 'plan_hash': plan['plan_hash'], 'fit_partition': 'train',
             'minimum': minimum.tolist(), 'maximum': maximum.tolist()}
    value['statistics_hash'] = digest(value)
    return value

def validate_statistics(stats, plan):
    values = {k:v for k,v in stats.items() if k != 'statistics_hash'}
    if (stats.get('schema_version') != 1 or stats.get('fit_partition') != 'train'
            or stats.get('plan_hash') != plan['plan_hash'] or digest(values) != stats.get('statistics_hash')):
        raise ValueError('Statistics identity mismatch')
    low, high = np.asarray(stats['minimum']), np.asarray(stats['maximum'])
    if low.ndim != 1 or low.size == 0 or low.shape != high.shape or not np.isfinite([low,high]).all() or (high < low).any():
        raise ValueError('Invalid normalization statistics')

@dataclass(frozen=True)
class PartitionObservation:
    partition: str
    spectral: np.ndarray
    coordinates: np.ndarray
    global_ids: np.ndarray
    plan_hash: str

def observe_partition(cube, plan, stats, partition, window=None):
    validate_plan(plan)
    validate_statistics(stats, plan)
    if cube.ndim != 3 or list(cube.shape[:2]) != plan['shape'] or cube.shape[2] != len(stats['minimum']):
        raise ValueError('Cube shape differs from plan/statistics')
    ys,xs = slices(plan, partition)
    y0,y1,x0,x1 = ys.start,ys.stop,xs.start,xs.stop
    if window is not None:
        if len(window)!=4 or any(type(v) is not int for v in window):
            raise ValueError('Integer global crop bounds required')
        a,b,c,d = window
        if not (y0 <= a < b <= y1 and x0 <= c < d <= x1):
            raise ValueError('Crop crosses partition boundary')
        y0,y1,x0,x1 = window
    values = np.array(cube[y0:y1,x0:x1], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite observation')
    low,high = np.asarray(stats['minimum'],dtype=np.float32),np.asarray(stats['maximum'],dtype=np.float32)
    # No target-region fitting or clipping: target values may legitimately exceed [0,1].
    values = (values-low)/np.where(high>low,high-low,1)
    yy,xx = np.mgrid[y0:y1,x0:x1]
    coordinates = np.stack((yy/max(plan['shape'][0]-1,1),xx/max(plan['shape'][1]-1,1)),axis=-1).astype(np.float32)
    ids = yy*plan['shape'][1]+xx
    for value in (values,coordinates,ids):value.setflags(write=False)
    return PartitionObservation(partition,values,coordinates,ids,plan['plan_hash'])

def training_labels(labels, plan, observation):
    if observation.partition != 'train' or observation.plan_hash != plan['plan_hash']:
        raise ValueError('Only training observations can request training labels')
    if list(labels.shape) != plan['shape']:
        raise ValueError('Label shape mismatch')
    allowed = np.zeros(labels.shape,dtype=bool)
    allowed[slices(plan,'train')] = True
    ids = observation.global_ids.reshape(-1)
    if not allowed.reshape(-1)[ids].all():
        raise ValueError('Supervision outside training partition')
    return np.array(labels.reshape(-1)[ids].reshape(observation.global_ids.shape),copy=True)

def forward_partition(model, dataset, observation, wavelengths=None, superpixel_scale=100):
    import torch
    if model.backbone.graph_mode != 'learned':
        raise ValueError('Strict spatial protocol requires label-free learned graphs')
    if observation.partition != 'train' and model.training:
        raise ValueError('Validation/test must use eval mode with frozen normalization state')
    device = next(model.parameters()).device
    value = torch.tensor(np.concatenate((observation.spectral,observation.coordinates),axis=-1),device=device)
    if wavelengths is not None:wavelengths=torch.as_tensor(wavelengths,device=device)
    return model(dataset,value,wavelengths=wavelengths,superpixel_scale=superpixel_scale,return_graph_info=True)

def class_coverage(labels, plan):
    validate_plan(plan)
    if list(labels.shape) != plan['shape']:
        raise ValueError('Label shape mismatch')
    classes = [int(c) for c in np.unique(labels) if c>0]
    counts = {p:{str(c):int((labels[slices(plan,p)]==c).sum()) for c in classes} for p in PARTITIONS}
    missing = {p:[int(c) for c,n in counts[p].items() if n==0] for p in PARTITIONS}
    return {'counts':counts,'missing_classes':missing,'all_classes_present':not any(missing.values())}
