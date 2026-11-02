"""Opt-in BASE-97 preparation and bounded validation; legacy trainer stays intact."""
import json
import math
from contextlib import contextmanager
from pathlib import Path
import random
import time

import numpy as np
from scipy.io import loadmat
import torch

from loadData.dataset_registry import DATASETS, file_hash, inspect_dataset
from loadData.fixed_splits import label_hash
from model.checkpoint_io import load_model_checkpoint, save_model_checkpoint
from model.research_config import (ResearchConfig, ModelConfig, DataConfig,
                                   TrainingConfig, LossConfig, AugmentationConfig)
from model.interaction_config import InteractionConfig
from research_runtime import external_directory, tensor_hash, write_json

NAMES = ('paviaU', 'xuzhou', 'WHUL')
COUNTS = {'paviaU': [432, 219, 42125], 'xuzhou': [692, 348, 67837],
          'WHUL': [2051, 1028, 201463]}
PHASES = ('train', 'val')


@contextmanager
def graph_random_state(seed):
    """The legacy hypergraph walker uses Python random in addition to NumPy."""
    previous = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(previous)


def load_config(path):
    config = ResearchConfig.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
    # Protocol identity is explicit: this entry is not a generic hyperparameter search.
    fixed = (config.name == 'base97_joint_v1' and config.device == 'cuda'
             and config.model == ModelConfig(latent_channels=64, adapter_embed_dim=32,
                                             adapter_attention_mode='index', adapter_chunk_pixels=16384)
             and config.data == DataConfig(datasets=list(NAMES))
             and config.training == TrainingConfig(seed=64, steps_per_epoch=3, val_interval=10)
             and config.loss == LossConfig() and config.augmentation == AugmentationConfig()
             and config.interaction == InteractionConfig())
    if not fixed:
        raise ValueError('Configuration does not describe the frozen ACC-1P baseline')
    return config


def ratio_split(labels, seed):
    """Match legacy np.random.seed/shuffle without modifying caller RNG state."""
    fingerprint = label_hash(labels)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid split seed')
    flat = np.asarray(labels).reshape(-1)
    classes = np.unique(flat[flat > 0]).astype(int)
    if not len(classes) or not np.array_equal(classes, np.arange(1, len(classes)+1)):
        raise ValueError('Contiguous nonempty foreground classes required')
    rng = np.random.RandomState(seed)
    groups = {phase: [] for phase in ('train', 'val', 'test')}
    counts = {}
    for cls in classes:
        ids = np.flatnonzero(flat == cls)
        rng.shuffle(ids)
        n, v = math.ceil(len(ids)*.01), math.ceil(len(ids)*.005)
        if n + v >= len(ids):
            raise ValueError('Class cannot retain train, validation and test samples')
        pieces = (ids[:n], ids[n:n+v], ids[n+v:])
        counts[str(cls)] = [len(part) for part in pieces]
        for phase, part in zip(groups, pieces):
            groups[phase].extend(part.tolist())
    return {'schema_version': 1, 'protocol': 'acc1p_legacy_randomstate_v1',
            'shape': list(labels.shape), 'label_hash': fingerprint, 'seed': seed,
            'train_ratio': .01, 'val_ratio': .005, 'spatial_isolation': False,
            'counts': counts, 'indices': groups}


def validate_split(value, labels):
    expected = ratio_split(labels, value.get('seed'))
    if value != expected:
        raise ValueError('Split identity, index order or label budget changed')


def raw_identity(root, name):
    record, labels = inspect_dataset(root, name)
    stem, key = DATASETS[name][:2]
    image = loadmat(Path(root)/(stem+'.mat'), variable_names=[key])[key]
    if image.shape != tuple(record['shape']) or not np.isfinite(image).all():
        raise ValueError('Image shape mismatch or nonfinite input')
    return record, labels, {'scope': 'full_scene_global_minmax',
                            'raw_min': float(image.min()), 'raw_max': float(image.max()),
                            'raw_dtype': str(image.dtype)}


def _item_fingerprint(item):
    return {key: tensor_hash(value) if torch.is_tensor(value) else value
            for key, value in item.items()}


def prepare(config, data_root, out_dir, source):
    from loadData import data_read
    from train_vca_multidataset import prepare_dataset

    out = external_directory(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(data_root).resolve(strict=True)
    if root == Path(__file__).resolve().parent or Path(__file__).resolve().parent in root.parents:
        raise ValueError('Dataset snapshot must be outside checkout')
    records = {}
    previous_root = data_read.DATASET_ROOT
    args = config.trainer_args(str(out/'unused.pt'), str(out/'unused.json'))
    started = time.monotonic()
    try:
        data_read.DATASET_ROOT = str(root)
        for offset, name in enumerate(NAMES):
            record, labels, normalization = raw_identity(root, name)
            split = ratio_split(labels, config.training.seed+offset)
            if [sum(v[j] for v in split['counts'].values()) for j in range(3)] != COUNTS[name]:
                raise ValueError(f'{name} does not match original annotated population')
            print(f'Preparing {name} with frozen ratio indices', flush=True)
            with graph_random_state(split['seed']):
                item = prepare_dataset(name, args, torch.device('cpu'), split['seed'])
            for phase in PHASES:
                released = item[phase+'_samples_gt'].numpy()
                expected = np.zeros(labels.size, dtype=np.float32)
                ids = split['indices'][phase]
                expected[ids] = labels.reshape(-1)[ids]
                if not np.array_equal(released, expected):
                    raise ValueError('Prepared labels differ from frozen split')
            # Test identities remain in the split manifest, never in the short-loop item.
            item = {k: v for k, v in item.items() if not k.startswith('test_')}
            if item['Q'] is None or item['A'] is None:
                raise ValueError('Static LDA-SLIC graph required')
            write_json(out/(name+'-split.json'), split)
            torch.save(item, out/(name+'-prepared.pt'))
            records[name] = {'raw': record, 'normalization': normalization,
                             'graph_python_seed': split['seed'],
                             'graph_numpy_rng': 'legacy split shuffle then graph walk',
                             'split_file': name+'-split.json',
                             'split_sha256': file_hash(out/(name+'-split.json')),
                             'prepared_file': name+'-prepared.pt',
                             'prepared_sha256': file_hash(out/(name+'-prepared.pt')),
                             'prepared_fields': _item_fingerprint(item),
                             'regions': item['A'].shape[0]}
            # Detect raw-data changes during lengthy graph preparation too.
            for role in ('image', 'label'):
                if file_hash(root/record[role+'_path']) != record[role+'_sha256']:
                    raise ValueError('Raw dataset changed during preparation')
    finally:
        data_read.DATASET_ROOT = previous_root
    manifest = {'schema_version': 1, 'protocol': 'acc1p_joint_prepared_v1',
                'config': config.to_dict(), 'config_hash': config.config_hash,
                'code': source, 'data_root': str(root), 'datasets': records,
                'context': 'full_scene_transductive_train_label_lda_slic',
                'test_scoring': False, 'elapsed_seconds': time.monotonic()-started}
    write_json(out/'manifest.json', manifest)
    return manifest


def verify_prepared(directory, expected_hash, config, data_root, source):
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory/'manifest.json'
    if file_hash(manifest_path) != expected_hash:
        raise ValueError('Prepared manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('protocol') != 'acc1p_joint_prepared_v1'
            or manifest.get('schema_version') != 1 or manifest.get('test_scoring') is not False
            or manifest['config'] != config.to_dict() or manifest['config_hash'] != config.config_hash
            or manifest['code']['source_hash'] != source['source_hash']
            or set(manifest['datasets']) != set(NAMES)):
        raise ValueError('Prepared code/config/protocol identity mismatch')
    root = Path(data_root).resolve(strict=True)
    result = {}
    for offset, name in enumerate(NAMES):
        entry = manifest['datasets'][name]
        # Fixed names prevent a manifest from loading arbitrary external tensors.
        if entry['split_file'] != name+'-split.json' or entry['prepared_file'] != name+'-prepared.pt':
            raise ValueError('Unexpected prepared file path')
        record, labels = inspect_dataset(root, name)
        if record != entry['raw']:
            raise ValueError('Raw dataset identity changed')
        for prefix in ('split', 'prepared'):
            if file_hash(directory/entry[prefix+'_file']) != entry[prefix+'_sha256']:
                raise ValueError('Prepared artifact changed')
        split = json.loads((directory/entry['split_file']).read_text(encoding='utf-8'))
        validate_split(split, labels)
        if split['seed'] != config.training.seed+offset:
            raise ValueError('Dataset split seed offset changed')
        item = torch.load(directory/entry['prepared_file'], map_location='cpu', weights_only=True)
        if any(k.startswith('test_') for k in item) or _item_fingerprint(item) != entry['prepared_fields']:
            raise ValueError('Training item fingerprint or held-out access changed')
        result[name] = item
    return result, manifest


def forward(model, name, item):
    return model(name, item['net_input'], item['Q'], item['A'],
                 wavelengths=item['wavelengths'], fwhm=item['fwhm'],
                 superpixel_scale=item['superpixel_scale'])


def readout(outputs, item, phase):
    if phase not in PHASES:
        raise ValueError('Short-loop evaluation permits train/validation only')
    truth = item[phase+'_samples_gt'].long()
    valid = truth > 0
    truth = truth[valid]-1
    if not len(truth):
        raise ValueError('Empty supervised phase')
    metrics, selected = {}, {}
    for head, probability in zip(('fused', 'graph', 'cnn'), outputs[:3]):
        value = probability[valid]
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite predictions')
        pred = value.argmax(1)
        c = item['class_num']
        cm = torch.bincount(truth*c+pred, minlength=c*c).reshape(c,c).cpu().numpy()
        support = cm.sum(1)
        if (support == 0).any():
            raise ValueError('Missing supervised class')
        recall = cm.diagonal()/support
        metrics[head] = {'oa': float(cm.trace()/cm.sum()), 'aa': float(recall.mean()),
                         'recall': recall.tolist(), 'confusion': cm.tolist(),
                         'nll': float(-value.gather(1, truth[:,None]).clamp_min(1e-8).log().mean())}
        selected[head] = value.detach().cpu()
    return metrics, selected


def evaluate(model, datasets):
    model.eval()
    result, values = {}, {}
    with torch.no_grad():
        for name, item in datasets.items():
            outputs = forward(model, name, item)
            result[name], values[name] = {}, {}
            for phase in PHASES:
                result[name][phase], values[name][phase] = readout(outputs, item, phase)
    return result, values


def validation_score(metrics):
    # Only validation fused OA participates, never training metrics or test data.
    return float(np.mean([metrics[name]['val']['fused']['oa'] for name in NAMES]))


def update_step(model, optimizer, name, item, config):
    from train_vca_multidataset import supervised_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {k: p.detach().clone() for k,p in model.named_parameters()}
    outputs = forward(model, name, item)
    components = supervised_loss(outputs, item, 'train', class_balance=config.loss.balanced,
                                 cnn_loss_weight=config.loss.cnn_loss_weight,
                                 graph_branch_loss_weight=config.loss.graph_branch_loss_weight,
                                 return_components=True)
    if not torch.isfinite(components['total']):
        raise ValueError('Nonfinite training loss')
    components['total'].backward()
    groups = {'shared_adapter': 'backbone.adapter.', 'backbone': 'backbone.',
              'fused_head': 'heads.'+name+'.', 'graph_head': 'hgnn_heads.'+name+'.',
              'cnn_head': 'cnn_heads.'+name+'.'}
    gradients = {}
    for group, prefix in groups.items():
        parameters = [p for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
        if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Missing/nonfinite gradient: '+group)
        gradients[group] = math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters))
        if gradients[group] == 0:
            raise ValueError('Zero gradient: '+group)
    optimizer.step()
    changes = {}
    for group, prefix in groups.items():
        changes[group] = sum(int(torch.count_nonzero(p.detach()-before[k]))
                             for k,p in model.named_parameters() if k.startswith(prefix))
        if not changes[group]:
            raise ValueError('No parameter update: '+group)
    for k,p in model.named_parameters():
        if k.startswith(('heads.', 'hgnn_heads.', 'cnn_heads.')) and k.split('.')[1] != name:
            if not torch.equal(p.detach(), before[k]):
                raise ValueError('Inactive task head unexpectedly updated')
    return {'dataset': name, 'loss': {k: float(v.detach()) for k,v in components.items()},
            'gradient_norms': gradients, 'changed_elements': changes}


def short_loop(config, items, out_dir, cycles=2, device=None, provenance=None):
    if type(cycles) is not int or cycles not in (1,2) or set(items) != set(NAMES):
        raise ValueError('Short loop needs all three datasets and one or two cycles')
    out = external_directory(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(device or config.device)
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.training.seed)
        torch.cuda.reset_peak_memory_stats(device)
    items = {name: {k: v.to(device) if torch.is_tensor(v) else v for k,v in item.items()}
             for name,item in items.items()}
    if any(any(k.startswith('test_') for k in item) for item in items.values()):
        raise ValueError('Test supervision must not enter short-loop training items')
    model = config.build_model({name: item['class_num'] for name,item in items.items()}).to(device)
    runtime = {'kind': 'engineering_round_robin', 'cycles': cycles,
               'actual_optimizer_steps': cycles*len(NAMES), 'dataset_order': list(NAMES),
               'selection': 'validation_macro_fused_oa', 'test_scoring': False,
               'config_budget_is_future_reference_not_executed': True}
    model.training_config = {'research_config': config.to_dict(), 'short_loop': runtime,
                             'provenance': provenance or {}}
    heads = list(model.heads.parameters())+list(model.hgnn_heads.parameters())+list(model.cnn_heads.parameters())
    optimizer = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': config.training.lr},
                                  {'params': heads, 'lr': config.training.head_lr}],
                                 weight_decay=config.training.weight_decay)
    registered = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(set(registered)) != len(registered) or set(registered) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer does not enroll trainable parameters exactly once')
    initial, _ = evaluate(model, items)
    trace, history, best, selected = [], [], -1., None
    started = time.monotonic()
    for cycle in range(cycles):
        for name in NAMES:
            step = update_step(model, optimizer, name, items[name], config)
            step.update(cycle=cycle+1, step=len(trace)+1)
            trace.append(step)
            write_json(out/'trace.json', trace)
            print(f'Joint short step {len(trace)} {name} loss={step["loss"]["total"]:.6f}', flush=True)
        metrics, values = evaluate(model, items)
        score = validation_score(metrics)
        history.append({'cycle': cycle+1, 'metrics': metrics, 'validation_macro_oa': score})
        if score > best:
            best, selected = score, values
            selected_metrics, selected_cycle = metrics, cycle+1
            save_model_checkpoint(model, out/'primary.pt')
            torch.save(values, out/'selected-probabilities.pt')
    save_model_checkpoint(model, out/'last.pt')
    del optimizer, model
    restored = load_model_checkpoint(out/'primary.pt', device=device)
    restored_metrics, restored_values = evaluate(restored, items)
    errors = {}
    for name in NAMES:
        errors[name] = {}
        for phase in PHASES:
            errors[name][phase] = {}
            for head in ('fused', 'graph', 'cnn'):
                old, new = selected[name][phase][head], restored_values[name][phase][head]
                torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                if not torch.equal(old.argmax(1),new.argmax(1)):
                    raise ValueError('Restored classifier predictions changed')
                errors[name][phase][head] = float((new-old).abs().max())
    summary = {'status': 'passed', 'runtime': runtime, 'initial': initial, 'history': history,
               'selected_cycle': selected_cycle, 'selected': selected_metrics,
               'restored': restored_metrics, 'reload_max_abs_error': errors,
               'checkpoint_sha256': file_hash(out/'primary.pt'), 'elapsed_seconds': time.monotonic()-started,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
    write_json(out/'result.json', summary)
    return summary
