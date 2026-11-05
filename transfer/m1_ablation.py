"""Preregistered full/legacy-zero/real-subset development screen (M1-C)."""
import copy
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch

from loadData.observation_sampling import BandMetadata
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import ROOT, external_directory, hash_file, require_frozen_code, write_json
from train_spatial import SpatialConfig
from transfer import development_training as development
from transfer.m4_ablation import input_identity, require
from transfer.observation_policy import ObservationPolicy, default_conditions, evaluate_conditions
from transfer.run_provenance import verify_run_artifacts

SPEC_PATH = ROOT / 'configs/experiments/m1_c_paired_screen.json'
MODES = ('full', 'legacy_support_zero', 'real_subset')
SEEDS = (64, 512, 2048)
CONDITIONS = ('full', 'gap', 'random_25', 'random_50', 'random_75', 'stride')


def spec():
    s = json.loads(SPEC_PATH.read_text())
    require(s['modes'] == list(MODES) and s['seeds'] == list(SEEDS)
            and s['updates_per_run'] == 5000 and s['validation_every'] == 100
            and s['full_target_updates'] == 45000 and s['test_evaluation'] is False
            and s['validation_observation_seed'] == 3407 and s['observation_seed_offset'] == 6,
            'unregistered M1 budget/seeds/modes')
    return s


def freeze_plan(code, inputs, output, *, smoke=False):
    s = spec(); base = json.loads((ROOT / s['base_config']).read_text()); c = SpatialConfig(base).research
    require(c.model.region_residual_mode == 'none' and c.model.graph_mode == 'learned'
            and c.model.norm_type == 'group_norm' and c.interaction.interaction_mode == c.interaction.graph_update_mode == 'none',
            'M1 baseline mechanism changed')
    require((c.training.lr, c.training.head_lr, c.training.weight_decay) ==
            (s['backbone_lr'], s['head_lr'], s['weight_decay']), 'M1 optimizer changed')
    require(smoke or inputs['development_hash'] == s['development_hash'], 'wrong development protocol')
    return {'schema_version': 1, 'scope': 'two_step_preflight' if smoke else 'registered_development_screen',
            'code': code, 'spec': s, 'base': base, 'inputs': inputs, 'output': str(external_directory(output)),
            'seeds': [64] if smoke else list(SEEDS), 'modes': list(MODES), 'updates': 2 if smoke else 5000,
            'val_every': 1 if smoke else 100}


def prepare_policies(plan, scene, record):
    """Save every schedule/bank before the first training run, outside checkout."""
    require(record['validation_hash'] == plan['inputs']['development_hash'], 'wrong policy scene')
    require_frozen_code(plan['code'])
    root = Path(plan['output']); (root / 'policies').mkdir(parents=True, exist_ok=False)
    conditions = default_conditions(scene.bands)
    require(set(conditions) == set(CONDITIONS), 'condition bank differs')
    configs = [conditions[name] for name in plan['spec']['training_cycle']]
    policies = {}; bank = None
    for seed in plan['seeds']:
        policies[str(seed)] = {}
        for mode in MODES:
            policy = ObservationPolicy.freeze(scene, record, updates=plan['updates'],
                training_configs=[{'mode': 'none'}] if mode == 'full' else configs,
                train_seed=seed + 6, validation_seed=3407, conditions=conditions,
                training_application='support_zero' if mode == 'legacy_support_zero' else 'real_subset')
            value = policy.to_dict(); current = digest(value['validation'])
            require(bank is None or current == bank, 'evaluation bank is not common')
            bank = current
            path = root / 'policies' / f'{seed}-{mode}.json'; write_json(path, value)
            policies[str(seed)][mode] = {'path': str(path), 'sha256': hash_file(path), 'policy_hash': policy.policy_hash}
    plan = {**plan, 'policies': policies, 'evaluation_bank_hash': bank}
    plan['plan_hash'] = digest(plan)
    return plan


def verify_plan(plan):
    require(plan['plan_hash'] == digest({k:v for k,v in plan.items() if k != 'plan_hash'}), 'plan hash mismatch')
    require(plan['spec'] == spec() and plan['modes'] == list(MODES), 'registered M1 specification changed')
    smoke = plan['scope'] == 'two_step_preflight'
    require(plan['scope'] in ('two_step_preflight', 'registered_development_screen')
            and plan['seeds'] == ([64] if smoke else list(SEEDS)) and plan['updates'] == (2 if smoke else 5000)
            and plan['val_every'] == (1 if smoke else 100), 'plan budget/scope changed')
    require(plan['base'] == json.loads((ROOT / plan['spec']['base_config']).read_text()), 'base changed')
    require_frozen_code(plan['code'])
    for p, sha in plan['inputs']['files'].items(): require(hash_file(p) == sha, 'frozen input changed')
    require(set(plan['policies']) == {str(s) for s in plan['seeds']}, 'policy roots differ')
    for seed in plan['seeds']:
        require(set(plan['policies'][str(seed)]) == set(MODES), 'policy modes differ')
        for mode, item in plan['policies'][str(seed)].items():
            path = Path(item['path'])
            require(path == Path(plan['output'])/'policies'/f'{seed}-{mode}.json'
                    and hash_file(path) == item['sha256'], 'frozen policy path/hash changed')


def read_scene(inputs):
    from scipy.io import loadmat
    from loadData.data_read import readWavelengths
    root = Path(inputs['development_root'])
    record = json.loads((root / 'development.json').read_text()); tiles = json.loads((root / 'parent-target-plan.json').read_text())
    cube = loadmat(inputs['image'], variable_names=['xuzhou'])['xuzhou']
    wave, _ = readWavelengths('xuzhou', cube.shape[-1])
    bands = BandMetadata(inputs['files'][inputs['image']], tuple(range(len(wave))), tuple(wave), 'nm', 'approximate', (True,)*len(wave))
    return development.prepare_scene(cube, bands, tiles, record), record


def case_path(plan, seed, mode):
    require(seed in plan['seeds'] and mode in MODES, 'unregistered case')
    return Path(plan['output'])/'cases'/f'seed-{seed}'/mode


def policy_for(plan, seed, mode):
    item = plan['policies'][str(seed)][mode]
    policy = ObservationPolicy(json.loads(Path(item['path']).read_text()))
    require(policy.policy_hash == item['policy_hash'] and digest(policy.to_dict()['validation']) == plan['evaluation_bank_hash'], 'policy bank differs')
    return policy


class SpectralProbe:
    def __init__(self, updates):
        self.points = {1, max(1, updates//2), updates}; self.handles = []
        self.data = {'train_calls':0, 'eval_calls':0, 'train_width_counts':{}, 'gradient_probes':[], 'feature_probes':[]}

    def attach(self, model):
        self.model = model; self.parameter = next(model.backbone.adapter.parameters()); self.initial = self.parameter.detach().cpu().clone()
        self.data['parameters'] = sum(p.numel() for p in model.parameters())
        def observed(module, args, kwargs):
            training = model.training; key = 'train_calls' if training else 'eval_calls'; self.data[key] += 1
            width = args[0].shape[-1]; require(width == len(kwargs['wavelengths']), 'model received wrong wavelength count')
            if training:
                counts = self.data['train_width_counts']; counts[str(width)] = counts.get(str(width),0)+1
                if self.data[key] in self.points:
                    self.data['feature_probes'].append({'step':self.data[key], 'bands':width,
                        'finite':bool(torch.isfinite(args[0]).all()), 'wavelength_unit':kwargs.get('wavelength_unit')})
        def gradient(g):
            if self.data['train_calls'] in self.points:
                self.data['gradient_probes'].append({'step':self.data['train_calls'], 'finite':bool(torch.isfinite(g).all()), 'max_abs':float(g.abs().max())})
        self.handles += [model.backbone.adapter.register_forward_pre_hook(observed,with_kwargs=True), self.parameter.register_hook(gradient)]

    def finish(self, policy, updates):
        for h in self.handles: h.remove()
        self.data['parameter_max_update'] = float((self.parameter.detach().cpu()-self.initial).abs().max())
        expected = {}
        for step in range(1,updates+1):
            b = str(policy.training_record(step)['selected_band_count']); expected[b] = expected.get(b,0)+1
        require(self.data['train_calls'] == updates and self.data['train_width_counts'] == expected, 'actual training B differs from frozen records')
        require(self.data['parameter_max_update'] > 0 and self.data['gradient_probes']
                and all(x['finite'] for x in self.data['gradient_probes']+self.data['feature_probes'])
                and any(x['max_abs'] > 0 for x in self.data['gradient_probes']), 'missing/nonfinite adapter participation')
        return self.data


def verified_case(plan, seed, mode):
    verify_plan(plan); path = case_path(plan,seed,mode); r = json.loads((path/'case.json').read_text())
    require(r['status'] == 'completed' and r['plan_hash'] == plan['plan_hash'] and r['seed'] == seed and r['mode'] == mode
            and r.get('receipt_hash') == digest({k:v for k,v in r.items() if k != 'receipt_hash'}), 'incomplete/corrupt case receipt')
    for p,sha in r['artifacts'].items():
        f = (path/p).resolve(); require(f.is_relative_to(path.resolve()) and hash_file(f) == sha, 'case artifact mismatch')
    actual = verify_run_artifacts(path/'training'); result = json.loads((path/'training/result.json').read_text())
    info = result['runtime_config']; cfg = result['resolved_config']
    require(actual['optimizer_steps'] == plan['updates'] and cfg['model'] == SpatialConfig(plan['base']).to_dict()['model']
            and info['seeds'] == {'model':seed,'head':seed+2,'sampler':seed+4,'training_torch':seed}
            and info['validation_every_updates'] == plan['val_every'] and result['method'] == 'target_scratch_full'
            and result['selection'] == plan['spec']['selection'] and result['sampler']['mode'] == 'class_first'
            and result['development_hash'] == plan['inputs']['development_hash'] and result['test_evaluated'] is False,
            'case runtime/selection mismatch')
    require(all(cfg['training'][k] == plan['spec'][s] for k,s in [('lr','backbone_lr'),('head_lr','head_lr'),('weight_decay','weight_decay')]), 'optimizer changed')
    policy = json.loads((path/'training/observation-policy.json').read_text())
    require(policy == json.loads(Path(plan['policies'][str(seed)][mode]['path']).read_text()), 'case policy differs from preregistration')
    require(r['initial_backbone_hash'] == result['initial_backbone_hash'] and r['initial_head_hash'] == result['initial_head_hash']
            and r['sample_trace_hash'] == digest([(x['tile_id'],x['anchor_class']) for x in result['trace']]), 'paired evidence differs')
    return r


def run_case(plan, scene, record, seed, mode, *, device='cuda', resume=False):
    verify_plan(plan); path = case_path(plan,seed,mode); policy = policy_for(plan,seed,mode); policy.validate(scene,record,plan['updates'])
    if path.exists():
        require(resume, 'existing case requires explicit verified resume')
        return {**verified_case(plan,seed,mode), 'operation':'verified_existing'}
    path.mkdir(parents=True,exist_ok=False)
    receipt = {'status':'running','plan_hash':plan['plan_hash'],'seed':seed,'mode':mode,'artifact_dir':str(path/'training'),
               'started_unix':time.time(),'pid':os.getpid(),'device':str(device),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json(path/'case.json',receipt); observer = SpectralProbe(plan['updates']); original = development.build_target
    def build(*a,**kw):
        model,cfg = original(*a,**kw); observer.attach(model); return model,cfg
    try:
        if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
        with patch.object(development,'build_target',side_effect=build):
            result = development.adapt(scene,record,plan['base'],path/'training',steps=plan['updates'],val_every=plan['val_every'],
                seed=seed,device=device,sampling='class_first',observation_policy=policy,
                progress=lambda event:write_json(path/'progress.json',event))
        participation = observer.finish(policy,plan['updates'])
        participation['peak_training_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
        selected = load_model_checkpoint(path/'training/model.pt',device)
        review = evaluate_conditions(selected,scene,record,policy)
        chosen = min(result['validation'],key=lambda x:x['nll'])
        require(abs(review['conditions']['full']['fused']['nll']-chosen['nll']) < 1e-5, 'selected checkpoint NLL differs')
        loss = [x['loss']['total'] for x in result['trace']]; window = min(100,len(loss))
        review.update(selected_step=result['selected_step'],validation_curve=result['validation'],participation=participation,
            first_window_train_loss=float(np.mean(loss[:window])),last_window_train_loss=float(np.mean(loss[-window:])),
            decision='pending registered paired batch; no interim tuning')
        write_json(path/'review.json',review)
        receipt.update(status='completed',optimizer_steps=plan['updates'],initial_backbone_hash=result['initial_backbone_hash'],
            initial_head_hash=result['initial_head_hash'],sample_trace_hash=digest([(x['tile_id'],x['anchor_class']) for x in result['trace']]),
            statistics_hash=scene.stats['statistics_hash'],evaluation_bank_hash=plan['evaluation_bank_hash'],
            perturbation_sequence_hash=digest([x['record_hash'] for x in policy.to_dict()['training']]),finished_unix=time.time(),
            artifacts={str(p.relative_to(path)):hash_file(p) for p in path.rglob('*') if p.is_file() and p.name != 'case.json'})
        receipt['receipt_hash'] = digest(receipt); write_json(path/'case.json',receipt); verified_case(plan,seed,mode)
        return {**receipt,'operation':'trained'}
    except BaseException as error:
        for h in observer.handles: h.remove()
        receipt.update(status='failed',error=repr(error),finished_unix=time.time());write_json(path/'case.json',receipt);raise


def decision(summary, pairs, scope):
    if scope == 'two_step_preflight': return 'engineering_preflight_only'
    candidate = (summary['real_subset']['full_oa']['mean'] >= summary['full']['full_oa']['mean']
        and summary['real_subset']['full_aa']['mean'] >= summary['full']['full_aa']['mean']
        and sum(x['full_oa_vs_full'] >= 0 for x in pairs) >= 2
        and all(summary['real_subset']['perturbed_oa']['mean'] > summary[c]['perturbed_oa']['mean']
                and sum(x['perturbed_oa_vs_'+c] > 0 for x in pairs) >= 2 for c in ('full','legacy_support_zero')))
    return 'development_robustness_candidate_pending_confirmation' if candidate else 'no_joint_accuracy_robustness_gain_keep_full'


def summarize(plan):
    verify_plan(plan); rows = []; pairs = []
    for seed in plan['seeds']:
        group = [verified_case(plan,seed,m) for m in MODES]
        for k in ('initial_backbone_hash','initial_head_hash','sample_trace_hash','statistics_hash','evaluation_bank_hash'):
            require(len({x[k] for x in group}) == 1,'paired mismatch: '+k)
        require(group[1]['perturbation_sequence_hash'] == group[2]['perturbation_sequence_hash'],'legacy/subset selections differ')
        for mode,r in zip(MODES,group):
            review = json.loads((case_path(plan,seed,mode)/'review.json').read_text()); scores = review['conditions']
            full = scores['full']['fused']; perturbed = [scores[c]['fused']['oa'] for c in CONDITIONS if c != 'full']
            rows.append({'seed':seed,'mode':mode,'full_oa':full['oa'],'full_aa':full['aa'],'full_kappa':full['kappa'],
                'full_macro_f1':full['macro_f1'],'perturbed_oa':float(np.mean(perturbed)), 'worst_condition_oa':min(perturbed),
                'condition_oa':{c:scores[c]['fused']['oa'] for c in CONDITIONS},'selected_step':review['selected_step'],
                'seconds':r['finished_unix']-r['started_unix']})
        by = {x['mode']:x for x in rows if x['seed'] == seed}
        pairs.append({'seed':seed,'full_oa_vs_full':by['real_subset']['full_oa']-by['full']['full_oa'],
            **{'perturbed_oa_vs_'+c:by['real_subset']['perturbed_oa']-by[c]['perturbed_oa'] for c in ('full','legacy_support_zero')}})
    summary = {m:{k:{'mean':float(np.mean([x[k] for x in rows if x['mode']==m])),
            'sample_sd':float(np.std([x[k] for x in rows if x['mode']==m],ddof=1)) if len(plan['seeds'])>1 else None}
            for k in ('full_oa','full_aa','full_kappa','full_macro_f1','perturbed_oa','worst_condition_oa')} for m in MODES}
    return {'status':'completed','scope':plan['scope'],'plan_hash':plan['plan_hash'],'rows':rows,'summary':summary,
            'paired_differences':pairs,'decision':decision(summary,pairs,plan['scope']),
            'target_updates':len(rows)*plan['updates'],'source_pretraining_updates':0,'test_evaluated':False,'boundary':None}
