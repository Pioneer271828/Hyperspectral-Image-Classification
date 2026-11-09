"""Score one registered G2 source validation release with the shared heads."""
import numpy as np
import torch

from evaluation.classification_metrics import confusion_matrix, metrics_from_confusion
from loadData.spatial_protocol import forward_partition
from transfer.g2_observations import source_observations
from transfer.g2_source import require_shared_heads, source_labels


@torch.inference_mode()
def evaluate_source_validation(catalog, contract, model, scene, plan, train_observations,
                               val_observations, val_labels, *, superpixel_scale):
    head_key = require_shared_heads(catalog, contract, model)
    if any(module.training for module in model.modules()):
        raise ValueError('Source validation requires a frozen eval-mode model')
    if getattr(model.backbone, 'coverage_mode', 'none') != 'none':
        raise ValueError('Source validation release does not carry coverage metadata')
    if type(superpixel_scale) is not int or superpixel_scale <= 0:
        raise ValueError('Source validation requires an explicit positive graph scale')

    view = source_observations(catalog, contract, scene, 'val', plan, train_observations,
                               val_payload=val_observations)
    labels, counts = source_labels(catalog, contract, scene, 'source_select', plan, val_labels)
    classes = list(range(1, len(counts) + 1))
    head_names = ('fused', 'graph', 'cnn')
    matrices = {name: np.zeros((len(classes), len(classes)), dtype=np.int64) for name in head_names}
    nlls = dict.fromkeys(head_names, 0.0)
    scored_tiles = []

    for tile in plan['tiles']:
        if tile['partition'] != 'val':
            continue
        observation = view.tile(tile['id'])
        truth = labels.ravel()[observation.global_ids.ravel()]
        included = truth > 0
        if not included.any():
            continue
        truth = truth[included]
        outputs = forward_partition(model, head_key, observation, view.wavelengths_nm.copy(),
                                    superpixel_scale, wavelength_unit='nm')
        for name, output in zip(head_names, outputs[:3]):
            probs = output.detach().cpu().numpy()[included]
            if probs.shape != (truth.size, len(classes)) or not np.isfinite(probs).all():
                raise ValueError('Invalid shared head validation probabilities')
            matrices[name] += confusion_matrix(truth, probs.argmax(axis=-1) + 1, classes)
            nlls[name] -= float(np.log(np.maximum(probs[np.arange(truth.size), truth - 1], 1e-8)).sum())
        scored_tiles.append(tile['id'])

    metrics = {}
    for name in head_names:
        if matrices[name].sum(axis=1).tolist() != counts:
            raise ValueError('Source validation did not score the complete registered labels')
        metrics[name] = {**metrics_from_confusion(matrices[name], classes), 'nll': nlls[name] / sum(counts)}
    release = contract['source_releases'][scene]
    access = [
        dict(scene=scene, phase='fit_statistics', partition='train',
             observation_sha256=release['train']['observation_sha256'], labels_sha256=None),
        dict(scene=scene, phase='validation_graph', partition='val',
             observation_sha256=release['val']['observation_sha256'], labels_sha256=None),
        dict(scene=scene, phase='source_select', partition='val', **release['val']),
    ]
    return dict(scene=scene, semantic_classes=list(contract['semantic_classes']), metrics=metrics,
                scored_tiles=scored_tiles, superpixel_scale=superpixel_scale,
                statistics_hash=view.statistics['statistics_hash'], access=access)
