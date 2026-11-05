"""Frozen M4 development screen, read-only completed-run reuse and per-run review."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from loadData.observation_sampling import BandMetadata
from loadData.spatial_tiles import tile_slice
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import ROOT, external_directory, hash_file, require_frozen_code, tensor_hash, write_json
from train_spatial import SpatialConfig
from transfer import development_training as development
from transfer.g3_training import _forward, head_hash, state_hash
from transfer.run_provenance import verify_run_artifacts

SPEC_PATH = ROOT / 'configs/experiments/m4_b2_target_screen.json'
MODES = ('none', 'capacity_control', 'x_residual', 'residual_only')
SEEDS = (64, 512, 2048)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spec():
    value = json.loads(SPEC_PATH.read_text())
    require(value['modes'] == list(MODES) and value['seeds'] == list(SEEDS), 'unregistered M4 arms/seeds')
    require(value['updates_per_run'] == 5000 and value['validation_every'] == 100
            and value['full_target_updates'] == 60000 and value['test_evaluation'] is False
            and value['method'] == 'target_scratch_full' and value['sampling'] == 'class_first', 'unregistered policy')
    return value


def freeze_plan(code, inputs, output, *, smoke=False):
    settings = spec()
    base = json.loads((ROOT / settings['base_config']).read_text())
    cfg = SpatialConfig(base).research
    require(cfg.training.lr == settings['backbone_lr'] and cfg.training.head_lr == settings['head_lr']
            and cfg.training.weight_decay == settings['weight_decay'], 'base optimizer changed')
    require(cfg.model.graph_mode == 'learned' and cfg.model.norm_type == 'group_norm'
            and cfg.interaction.interaction_mode == cfg.interaction.graph_update_mode == 'none', 'base mechanism changed')
    require(smoke or inputs['development_hash'] == settings['development_hash'], 'wrong development protocol')
    plan = {'schema_version': 1, 'scope': 'two_step_preflight' if smoke else 'registered_development_screen',
            'code': code, 'spec': settings, 'base': base, 'inputs': inputs,
            'output': str(external_directory(output)), 'seeds': [64] if smoke else list(SEEDS),
            'modes': list(MODES), 'updates': 2 if smoke else 5000, 'val_every': 1 if smoke else 100}
    plan['plan_hash'] = digest(plan)
    return plan


def verify_plan(plan):
    require(plan['plan_hash'] == digest({k: v for k, v in plan.items() if k != 'plan_hash'}), 'plan hash mismatch')
    require(plan['spec'] == spec(), 'registered specification changed')
    require(plan['modes'] == list(MODES), 'plan modes changed')
    smoke = plan['scope'] == 'two_step_preflight'
    require(plan['scope'] in ('two_step_preflight', 'registered_development_screen'), 'unknown plan scope')
    require(plan['seeds'] == ([64] if smoke else list(SEEDS)) and plan['updates'] == (2 if smoke else 5000)
            and plan['val_every'] == (1 if smoke else 100), 'plan budget/seeds changed')
    require(plan['base'] == json.loads((ROOT / plan['spec']['base_config']).read_text()), 'base config changed')
    require_frozen_code(plan['code'])
    for path, expected in plan['inputs']['files'].items():
        require(hash_file(path) == expected, 'frozen input changed: ' + path)


def input_identity(development_root, protocol_root, data_root):
    paths = [Path(development_root) / name for name in ('manifest.json', 'development.json', 'parent-target-plan.json')]
    paths.append(Path(protocol_root) / 'task.json')
    manifest, record, plan, task = [json.loads(p.read_text()) for p in paths]
    development.validate(record, plan)
    require(manifest['status'] == 'passed' and manifest['validation_hash'] == record['validation_hash']
            and manifest['parent_task_hash'] == task['task_hash'] and plan['plan_hash'] == task['target_plan_hash'],
            'development task mismatch')
    identity = task['target_identity']; data_root = Path(data_root).resolve(strict=True)
    image = (data_root / identity['image_path']).resolve(strict=True)
    require(identity['name'] == 'xuzhou' and image.is_relative_to(data_root)
            and hash_file(image) == identity['image_sha256'], 'target image identity mismatch')
    return {'development_root': str(Path(development_root).resolve()), 'protocol_root': str(Path(protocol_root).resolve()),
            'data_root': str(data_root), 'image': str(image), 'image_key': 'xuzhou',
            'development_hash': record['validation_hash'],
            'files': {**{str(p.resolve()): hash_file(p) for p in paths}, str(image): identity['image_sha256']}}


def load_scene(plan):
    verify_plan(plan)
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS
    from loadData.data_read import readWavelengths
    inputs = plan['inputs']; directory = Path(inputs['development_root'])
    record = json.loads((directory / 'development.json').read_text())
    tiles = json.loads((directory / 'parent-target-plan.json').read_text())
    cube = loadmat(inputs['image'], variable_names=[DATASETS['xuzhou'][1]])[DATASETS['xuzhou'][1]]
    wave, _ = readWavelengths('xuzhou', cube.shape[-1])
    bands = BandMetadata(inputs['files'][inputs['image']], tuple(range(len(wave))), tuple(wave),
                         'nm', 'approximate', (True,) * len(wave))
    return development.prepare_scene(cube, bands, tiles, record), record


class Participation:
    """Read-only bounded probes; no per-step GPU scalar synchronization."""
    def __init__(self, updates):
        self.steps = {1, updates, max(1, updates // 2)}
        self.state = {'train_calls': 0, 'eval_calls': 0, 'feature_probes': [], 'gradient_probes': []}
        self.handles = []
        self.model = None

    def attach(self, model):
        self.model = model
        self.state['initial_shared_backbone_hash'] = digest({k: tensor_hash(v) for k, v in model.backbone.state_dict().items()
                                                            if not k.startswith('region_residual.')})
        self.state['initial_head_hash'] = head_hash(model)
        self.state['total_parameters'] = sum(p.numel() for p in model.parameters())
        layer = model.backbone.region_residual
        self.initial = {} if layer is None else {k: p.detach().cpu().clone() for k, p in layer.named_parameters()}
        self.state['new_parameters'] = sum(p.numel() for p in layer.parameters()) if layer is not None else 0
        if layer is None:
            return
        def observed(module, args, output):
            self.state['train_calls' if module.training else 'eval_calls'] += 1
            step = self.state['train_calls']
            if module.training and step in self.steps:
                with torch.no_grad():
                    self.state['feature_probes'].append({'step': step, 'max_change': float((output.cnn_input - args[0]).abs().max()),
                                                         'finite': bool(torch.isfinite(output.cnn_input).all())})
        self.handles.append(layer.register_forward_hook(observed))
        for name, p in layer.named_parameters():
            def gradient(grad, name=name):
                step = self.state['train_calls']
                if step in self.steps:
                    self.state['gradient_probes'].append({'step': step, 'parameter': name,
                        'finite': bool(torch.isfinite(grad).all()), 'max_abs': float(grad.abs().max())})
            self.handles.append(p.register_hook(gradient))

    def finish(self, mode, updates):
        for handle in self.handles:
            handle.remove()
        layer = self.model.backbone.region_residual
        self.state['parameter_max_updates'] = {} if layer is None else {
            k: float((p.detach().cpu() - self.initial[k]).abs().max()) for k, p in layer.named_parameters()}
        require(mode == 'none' or self.state['train_calls'] == updates, 'M4 invocation count differs')
        require(all(p['finite'] for p in self.state['feature_probes'] + self.state['gradient_probes']), 'nonfinite M4 probe')
        if self.initial:
            require(max(self.state['parameter_max_updates'].values()) > 0, 'M4 parameters did not update')
            require(any(p['max_abs'] > 0 for p in self.state['gradient_probes']), 'M4 gradients always zero')
        return self.state


def validation_readouts(model, scene, record):
    """Selected checkpoint only; auxiliary heads are not standalone branch models."""
    labels = development.sparse_labels(record, scene.plan, 'validation_support')
    classes = record['classes']; n = len(classes)
    matrices = [np.zeros((n, n), dtype=np.int64) for _ in range(3)]; nll = np.zeros(3)
    targets, predictions = [], []
    model.eval()
    with torch.no_grad():
        for role in record['roles']:
            if role['role'] != 'devval': continue
            local = labels[tile_slice(role)].reshape(-1); valid = local > 0
            if not valid.any(): continue
            truth = local[valid] - 1; outputs = _forward(model, scene, scene.plan['tiles'][role['id']]); columns = []
            for index, probabilities in enumerate(outputs[:3]):
                probs = probabilities.detach().cpu().numpy()[valid]; predicted = probs.argmax(1); columns.append(predicted)
                matrices[index] += np.bincount(truth * n + predicted, minlength=n * n).reshape(n, n)
                nll[index] -= np.log(np.maximum(probs[np.arange(len(truth)), truth], 1e-8)).sum()
            targets.extend(truth.tolist()); predictions.extend(np.stack(columns, axis=1).tolist())
    require(len(targets) == 45 and all(m.sum(1).tolist() == [5] * n for m in matrices), 'validation label budget differs')
    report = {name: {**metrics_from_confusion(matrix, classes), 'nll': float(nll[i] / 45)}
              for i, (name, matrix) in enumerate(zip(('fused', 'graph', 'cnn'), matrices))}
    target = np.asarray(targets); predicted = np.asarray(predictions)
    report['both_branches_wrong_by_class'] = [float(((predicted[target == c, 1] != c)
        & (predicted[target == c, 2] != c)).mean()) for c in range(n)]
    report['boundary'] = {'value': None, 'reason': 'zero valid boundary labels in frozen development protocol'}
    return report


def selected_review(folder, scene, record, device):
    result = json.loads((folder / 'result.json').read_text()); verify_run_artifacts(folder)
    model = load_model_checkpoint(folder / 'model.pt', device); before = state_hash(model)
    metrics = validation_readouts(model, scene, record)
    chosen = min(result['validation'], key=lambda r: r['nll'])
    require(chosen['step'] == result['selected_step'] and abs(metrics['fused']['nll'] - chosen['nll']) < 1e-5,
            'selected checkpoint evaluation differs')
    interventions = {}
    layer = model.backbone.region_residual
    if layer is not None:
        def bypass(_, args, output): return replace(output, cnn_input=args[0])
        handle = layer.register_forward_hook(bypass)
        try: interventions['module_bypass'] = validation_readouts(model, scene, record)['fused']
        finally: handle.remove()
        if layer.mode == 'x_residual':
            def hint_off(module, args, output):
                value = module.projection(torch.cat((args[0], torch.zeros_like(args[0])), dim=1))
                return replace(output, cnn_input=value)
            handle = layer.register_forward_hook(hint_off)
            try: interventions['residual_hint_off'] = validation_readouts(model, scene, record)['fused']
            finally: handle.remove()
    require(before == state_hash(model), 'frozen review changed model state')
    losses = np.asarray([r['loss']['total'] for r in result['trace']]); window = min(100, len(losses))
    require(np.isfinite(losses).all(), 'nonfinite training losses')
    return {'selected_step': result['selected_step'], 'metrics': metrics, 'interventions': interventions,
            'first_window_train_loss': float(losses[:window].mean()), 'last_window_train_loss': float(losses[-window:].mean()),
            'validation_curve': result['validation'], 'decision': 'pending paired batch; no within-batch tuning',
            'test_evaluated': False, 'scope': '45 development labels; auxiliary heads and frozen interventions only'}


def case_path(plan, seed, mode):
    require(seed in plan['seeds'] and mode in plan['modes'], 'unregistered case')
    return Path(plan['output']) / 'cases' / ('seed-' + str(seed)) / mode


def verified_case(plan, seed, mode):
    verify_plan(plan); path = case_path(plan, seed, mode)
    receipt = json.loads((path / 'case.json').read_text())
    require(receipt['status'] == 'completed' and receipt['plan_hash'] == plan['plan_hash']
            and receipt['seed'] == seed and receipt['mode'] == mode, 'case incomplete or wrong identity')
    require(receipt.get('receipt_hash') == digest({k: v for k, v in receipt.items() if k != 'receipt_hash'}),
            'case receipt hash mismatch')
    for name, expected in receipt['artifacts'].items():
        file = (path / name).resolve()
        require(file.is_relative_to(path.resolve()) and hash_file(file) == expected, 'case artifact mismatch')
    verified = verify_run_artifacts(path / 'training')
    result = json.loads((path / 'training/result.json').read_text())
    sidecar = json.loads((path / 'training/model.pt.json').read_text())
    base = copy.deepcopy(plan['base']); base['model']['region_residual_mode'] = mode
    require(verified['optimizer_steps'] == plan['updates'] and result['resolved_config']['model']['region_residual_mode'] == mode
            and result['resolved_config']['training']['seed'] == seed and result['sampler']['mode'] == 'class_first'
            and result['development_hash'] == plan['inputs']['development_hash'] and result['test_evaluated'] is False,
            'case runtime policy mismatch')
    require(result['resolved_config']['model'] == SpatialConfig(base).to_dict()['model']
            and sidecar['backbone_kwargs']['region_residual_mode'] == mode
            and result['runtime_config']['validation_every_updates'] == plan['val_every']
            and result['runtime_config']['seeds'] == {'model': seed, 'head': seed + 2, 'sampler': seed + 4, 'training_torch': seed}
            and result['method'] == plan['spec']['method'], 'case architecture/seeds/selection mismatch')
    training = result['resolved_config']['training']
    require((training['lr'], training['head_lr'], training['weight_decay']) ==
            (plan['spec']['backbone_lr'], plan['spec']['head_lr'], plan['spec']['weight_decay']), 'case optimizer changed')
    require(receipt['initial_head_hash'] == result['initial_head_hash'] and receipt['sample_trace_hash'] ==
            digest([(r['tile_id'], r['anchor_class']) for r in result['trace']]), 'case paired evidence mismatch')
    return receipt


def run_case(plan, scene, record, seed, mode, *, device='cuda', resume=False):
    verify_plan(plan)
    require(record['validation_hash'] == plan['inputs']['development_hash'] and scene.stats['fit_role'] == 'devtrain', 'scene identity mismatch')
    path = case_path(plan, seed, mode)
    if path.exists():
        require(resume, 'existing case; explicit completed-case resume required')
        receipt = verified_case(plan, seed, mode)
        return {**receipt, 'operation': 'verified_existing'}
    path.mkdir(parents=True, exist_ok=False)  # atomic case ownership; never overwrite an old attempt
    receipt = {'status': 'running', 'seed': seed, 'mode': mode, 'plan_hash': plan['plan_hash'],
               'pid': os.getpid(), 'artifact_dir': str(path / 'training'), 'started_unix': time.time(),
               'device': str(device), 'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'torch': torch.__version__, 'cuda': torch.version.cuda}
    write_json(path / 'case.json', receipt)
    observer = Participation(plan['updates']); build = development.build_target
    def builder(*args, **kwargs):
        model, config = build(*args, **kwargs); observer.attach(model); return model, config
    base = copy.deepcopy(plan['base']); base['model']['region_residual_mode'] = mode
    try:
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
        with patch.object(development, 'build_target', side_effect=builder):
            result = development.adapt(scene, record, base, path / 'training', steps=plan['updates'],
                val_every=plan['val_every'], seed=seed, device=device, sampling='class_first',
                progress=lambda event: write_json(path / 'progress.json', event))
        participation = observer.finish(mode, plan['updates'])
        participation['peak_training_allocated_bytes'] = (torch.cuda.max_memory_allocated(device)
                                                         if str(device).startswith('cuda') else None)
        review = selected_review(path / 'training', scene, record, device)
        review['participation'] = participation
        write_json(path / 'review.json', review)
        require([r['step'] for r in result['trace']] == list(range(1, plan['updates'] + 1)), 'incomplete training trace')
        receipt.update(status='completed', optimizer_steps=plan['updates'],
            shared_initialization_hash=participation['initial_shared_backbone_hash'], initial_head_hash=result['initial_head_hash'],
            sample_trace_hash=digest([(r['tile_id'], r['anchor_class']) for r in result['trace']]),
            statistics_hash=scene.stats['statistics_hash'], finished_unix=time.time(),
            artifacts={str(p.relative_to(path)): hash_file(p) for p in path.rglob('*') if p.is_file() and p.name != 'case.json'})
        receipt['receipt_hash'] = digest(receipt)
        write_json(path / 'case.json', receipt); verified_case(plan, seed, mode)
        return {**receipt, 'operation': 'trained'}
    except BaseException as error:
        for handle in observer.handles: handle.remove()
        receipt.update(status='failed', error=repr(error), finished_unix=time.time())
        write_json(path / 'case.json', receipt)
        raise


def summarize(plan):
    verify_plan(plan); rows = []; paired = []
    for seed in plan['seeds']:
        group = [verified_case(plan, seed, mode) for mode in plan['modes']]
        for key in ('shared_initialization_hash', 'initial_head_hash', 'sample_trace_hash', 'statistics_hash'):
            require(len({r[key] for r in group}) == 1, 'paired control mismatch: ' + key)
        for mode in plan['modes']:
            review = json.loads((case_path(plan, seed, mode) / 'review.json').read_text())
            rows.append({'seed': seed, 'mode': mode, **review['metrics']['fused'], 'selected_step': review['selected_step'],
                         'seconds': group[plan['modes'].index(mode)]['finished_unix'] - group[plan['modes'].index(mode)]['started_unix']})
        by_mode = {r['mode']: r for r in rows if r['seed'] == seed}
        paired.append({'seed': seed, **{control: by_mode['x_residual']['oa'] - by_mode[control]['oa']
                                        for control in ('none', 'capacity_control')}})
    summary = {}
    for mode in plan['modes']:
        selected = [r for r in rows if r['mode'] == mode]
        summary[mode] = {metric: {'mean': float(np.mean([r[metric] for r in selected])),
            'sample_sd': float(np.std([r[metric] for r in selected], ddof=1)) if len(selected) > 1 else None}
            for metric in ('oa', 'aa', 'kappa', 'macro_f1')}
    candidate = (plan['scope'] == 'registered_development_screen'
        and all(summary['x_residual']['oa']['mean'] > summary[c]['oa']['mean'] and sum(p[c] > 0 for p in paired) >= 2
                for c in ('none', 'capacity_control'))
        and summary['x_residual']['aa']['mean'] >= summary['none']['aa']['mean'])
    return {'status': 'completed', 'scope': plan['scope'], 'plan_hash': plan['plan_hash'], 'rows': rows,
            'summary': summary, 'paired_oa_differences': paired,
            'decision': ('engineering_preflight_only' if plan['scope'] == 'two_step_preflight' else
                         'development_candidate_pending_confirmation' if candidate else 'no_confirmed_residual_gain_keep_baseline'),
            'test_evaluated': False, 'source_pretraining_updates': 0,
            'target_updates': len(rows) * plan['updates'], 'boundary': None}
