"""BASE-97B legacy random schedule and explicit resumable training state."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from joint_baseline import NAMES, forward, load_config, readout, verify_prepared
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint,save_model_checkpoint
from model.research_config import digest
from model import utils
from research_runtime import external_directory, tensor_hash, write_json

PREPARATION_COMMIT = '592e24153bdd2e2ea4dcde498da5b1a2f50ad9a1'
PREPARATION_HASH = '414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b'


def load_plan(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = dict(schema_version=1, protocol='base97_honghu_joint_training_v1',
                    baseline_config='configs/experiments/base97_joint.json',
                    preparation_commit=PREPARATION_COMMIT, prepared_manifest_sha256=PREPARATION_HASH,
                    split_root_seed=64, training_seeds=[64,512,2048], steps_per_epoch=3,
                    profiles={'engineering': {'epochs':3,'val_interval':1},
                              'full': {'epochs':500,'val_interval':10}},
                    scheduler={'factor':.5,'patience':20,'threshold':.0001},test_scoring=False)
    if value != expected:
        raise ValueError('Unknown or modified BASE-97B plan; register a new protocol version')
    return value


def options(plan, profile, seed, baseline):
    if profile not in plan['profiles'] or type(seed) is not int or seed not in plan['training_seeds']:
        raise ValueError('Unknown profile or unregistered training seed')
    budget = plan['profiles'][profile]
    config = replace(baseline, name='base97_'+profile+'_s'+str(seed),
                     training=replace(baseline.training,seed=seed,epochs=budget['epochs'],
                                      steps_per_epoch=3,val_interval=budget['val_interval']))
    return {'protocol':plan['protocol'],'profile':profile,'model_seed':seed,'schedule_seed':seed,
            'split_root_seed':plan['split_root_seed'],'epochs':budget['epochs'],
            'val_interval':budget['val_interval'],'steps_per_epoch':3,
            'optimizer_steps':(budget['epochs']+1)*3,'test_scoring':False,
            'scheduler':plan['scheduler'],'config':config.to_dict()}


def schedule_for(settings, *, task_names=NAMES):
    rng = np.random.default_rng(settings['schedule_seed'])
    return [str(name) for _ in range(settings['epochs']+1)
            for name in rng.choice(task_names,size=settings['steps_per_epoch'],replace=True)]


def validation_epochs(settings):
    return [e for e in range(settings['epochs']+1)
            if e % settings['val_interval']==0 or e==settings['epochs']]


def audit_preparation_source(producer, consumer):
    if producer.get('commit') != PREPARATION_COMMIT or not producer.get('files'):
        raise ValueError('Unaccepted preparation producer')
    if producer.get('source_hash') != digest(producer['files']):
        raise ValueError('Corrupt preparation source fingerprint')
    for name, expected in producer['files'].items():
        if consumer['files'].get(name) != expected:
            raise ValueError('Preparation producer file changed: '+name)


def import_preparation(directory, data_root, plan, consumer):
    directory = Path(directory)
    if file_hash(directory/'manifest.json') != plan['prepared_manifest_sha256']:
        raise ValueError('Wrong frozen prepared manifest')
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    audit_preparation_source(manifest['code'],consumer)
    baseline = load_config(plan['baseline_config'])
    # Old validator remains strict; its producer is authenticated before reuse.
    items, manifest = verify_prepared(directory,plan['prepared_manifest_sha256'],baseline,
                                      data_root,manifest['code'])
    return baseline,items,{'prepared_manifest_sha256':plan['prepared_manifest_sha256'],
                          'preparation_code':manifest['code'],'data_root':str(Path(data_root).resolve())}


def _tree_fingerprint(value):
    def simplify(x):
        if torch.is_tensor(x): return {'tensor':tensor_hash(x)}
        if isinstance(x,dict): return {str(k):simplify(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [simplify(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x): return {'nonfinite':str(x)}
        return x
    return digest(simplify(value))


def _rng_state(device):
    state = np.random.get_state()
    return {'torch':torch.get_rng_state(),'python':random.getstate(),
            'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'cuda':torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None}


def _restore_rng(state,device):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    if device.type=='cuda':
        if state['cuda'] is None: raise ValueError('CUDA RNG missing from training state')
        torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def _atomic_torch_save(value,path):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temp=Path(stream.name)
    try:
        torch.save(value,temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _run_lock(directory):
    lock=directory/'.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    try: yield
    finally: lock.unlink()


def validation_snapshot(model,items):
    """Preserve the old fused-loss/OA arithmetic for scheduler and selection."""
    model.eval(); losses=[]; oas={}; metrics={}; probabilities={}
    with torch.no_grad():
        for name,item in items.items():
            outputs=forward(model,name,item)
            losses.append(utils.compute_loss(outputs[0],item['val_gt_onehot'],item['val_label_mask']))
            oas[name]=float(utils.evaluate_performance(outputs[0],item['val_samples_gt'],
                                                     item['val_gt_onehot'],item['zeros']))
            metrics[name]={}; probabilities[name]={}
            for phase in ('train','val'):
                metrics[name][phase],probabilities[name][phase]=readout(outputs,item,phase)
    loss=float(torch.stack(losses).mean().cpu())
    if not math.isfinite(loss): raise ValueError('Nonfinite validation loss')
    return {'val_loss':loss,'macro_oa':float(np.mean(list(oas.values()))),
            'legacy_val_oa':oas,'metrics':metrics},probabilities


def train_step(model,optimizer,name,item,config,*,forward_fn=None):
    from train_vca_multidataset import supervised_loss
    model.train(); optimizer.zero_grad(set_to_none=True)
    graph_before={n:p.detach().clone() for n,p in model.named_parameters()
                  if n.startswith('backbone.HGNN_Branch.')}
    outputs=(forward if forward_fn is None else forward_fn)(model,name,item)
    terms=supervised_loss(outputs,item,'train',class_balance=config.loss.balanced,
                          cnn_loss_weight=config.loss.cnn_loss_weight,
                          graph_branch_loss_weight=config.loss.graph_branch_loss_weight,
                          return_components=True)
    if not torch.isfinite(terms['total']): raise ValueError('Nonfinite training loss')
    terms['total'].backward()
    grad={}
    for key,prefix in [('graph','backbone.HGNN_Branch.'),('adapter','backbone.adapter.'),
                       ('fused_head','heads.'+name+'.'),('graph_head','hgnn_heads.'+name+'.'),
                       ('cnn_head','cnn_heads.'+name+'.')]:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not params or any(not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('Missing/nonfinite gradient: '+key)
        # Legitimate zero gradients near saturation are logged, not called dead units.
        grad[key]=math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
    optimizer.step()
    changed=sum(int(torch.count_nonzero(p.detach()-graph_before[n]))
                for n,p in model.named_parameters() if n in graph_before)
    return {'dataset':name,'loss':{k:float(v.detach()) for k,v in terms.items()},
            'gradient_norms':grad,'graph_changed_elements':changed,
            'learning_rates':[float(g['lr']) for g in optimizer.param_groups]}


def _completed(directory,identity,settings,schedule):
    value=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if (value.get('status')!='completed' or value.get('identity')!=identity
            or value.get('optimizer_steps')!=len(schedule)
            or value.get('schedule_sha256')!=digest(schedule)
            or value.get('settings')!=settings or value.get('test_scoring') is not False):
        raise ValueError('Completed result identity/budget mismatch')
    required={'identity.json','progress.json','primary.pt','primary.pt.json','last.pt','last.pt.json',
              'selected-probabilities.pt','trace.json','history.json'}
    if not required <= value['artifacts'].keys(): raise ValueError('Completed artifact list incomplete')
    for name,h in value['artifacts'].items():
        if Path(name).name!=name or file_hash(directory/name)!=h:
            raise ValueError('Completed artifact changed: '+name)
    trace=json.loads((directory/'trace.json').read_text())
    if [v['dataset'] for v in trace]!=schedule:
        raise ValueError('Completed trace does not match frozen schedule')
    if value['per_dataset_updates']!=dict(Counter(schedule)):
        raise ValueError('Completed per-task exposure differs')
    return {'operation':'verified_completed','updates_this_call':0,'result':value}


def run_training(settings,items,directory,provenance,source,*,resume=False,
                 stop_after=None,device=None,finish_guard=None,runtime_adapter=None):
    """No test scoring; interruptions save a cursor without extra validation."""
    from model.research_config import ResearchConfig
    from joint_calibrated_selection import enabled,validation_candidate,load_selected
    calibrated_selection=(enabled(settings) if runtime_adapter is None else runtime_adapter.validate(settings))
    step_fn=train_step if runtime_adapter is None else runtime_adapter.train_step
    snapshot=validation_snapshot if runtime_adapter is None else runtime_adapter.validation_snapshot
    tasks=NAMES
    if runtime_adapter is not None and hasattr(runtime_adapter,'task_names'):
        tasks=tuple(runtime_adapter.task_names(settings))
        if not tasks or tasks!=tuple(n for n in NAMES if n in tasks):
            raise ValueError('Runtime tasks must be nonempty canonical known distinct tasks')
    if tuple(items)!=tasks or any(any(k.startswith('test_') for k in i) for i in items.values()):
        raise ValueError('Expected ordered train/validation-only prepared tasks')
    if runtime_adapter is not None and hasattr(runtime_adapter, 'validate_inputs'):
        runtime_adapter.validate_inputs(items)
    schedule=schedule_for(settings,task_names=tasks)
    if stop_after is not None and (type(stop_after)is not int or not 0<stop_after<len(schedule)):
        raise ValueError('Interruption cursor must be inside the declared budget')
    execution_device=str(torch.device(device or settings['config']['device']))
    identity={'protocol':settings['protocol'],'settings_hash':digest(settings),'settings':settings,
              'execution_device':execution_device,
              'consumer_source_hash':source['source_hash'],'prepared':provenance,
              'schedule_sha256':digest(schedule)}
    directory=external_directory(directory)
    if resume:
        if not directory.is_dir(): raise ValueError('Resume needs an existing owned run')
    else:
        directory.mkdir(parents=True,exist_ok=False)
    with _run_lock(directory):
        identity_path=directory/'identity.json'
        if resume:
            if json.loads(identity_path.read_text())!=identity: raise ValueError('Resume identity mismatch')
            if (directory/'result.json').exists():
                if finish_guard: finish_guard()
                return _completed(directory,identity,settings,schedule)
        else: write_json(identity_path,identity)
        device=torch.device(execution_device)
        config=ResearchConfig.from_dict(settings['config'])
        torch.manual_seed(settings['model_seed']); np.random.seed(settings['model_seed'])
        random.seed(settings['model_seed'])
        if device.type=='cuda': torch.cuda.manual_seed_all(settings['model_seed'])
        items={n:{k:v.to(device) if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
        counts={n:i['class_num'] for n,i in items.items()}
        model=(config.build_model(counts) if runtime_adapter is None else runtime_adapter.build_model(config,counts)).to(device)
        heads=list(model.heads.parameters())+list(model.hgnn_heads.parameters())+list(model.cnn_heads.parameters())
        optimizer=(runtime_adapter.optimizer(model,config)
                   if runtime_adapter is not None and hasattr(runtime_adapter,'optimizer') else
                   torch.optim.Adam([{'params':model.backbone.parameters(),'lr':config.training.lr},
                                    {'params':heads,'lr':config.training.head_lr}],
                                   weight_decay=config.training.weight_decay))
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        if len(set(ids))!=len(ids) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Optimizer registration differs from all trainable parameters')
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',
                  factor=settings['scheduler']['factor'],patience=settings['scheduler']['patience'],
                  threshold=settings['scheduler']['threshold'],threshold_mode='rel')
        score_fn=(runtime_adapter.selection_score
                  if runtime_adapter is not None and hasattr(runtime_adapter,'selection_score') else None)
        candidate_fn=(runtime_adapter.validation_candidate
                      if runtime_adapter is not None and hasattr(runtime_adapter,'validation_candidate') else validation_candidate)
        calibration_key=getattr(runtime_adapter,'calibration_metadata_key','bn_calibration')
        if calibration_key in ('research_config','joint_runtime','provenance','training_code','checkpoint_optimizer_step'):
            raise ValueError('Calibration metadata key collides with canonical metadata')
        state={'cursor':0,'trace':[],'history':[],'best_score':None if score_fn else -1.,'best_step':None,
               'best_model':None,'best_probabilities':None,'identity':identity,'start_code':source}
        restored_fingerprints=None
        if resume:
            progress=json.loads((directory/'progress.json').read_text())
            state_path=directory/progress['state_file']
            if Path(progress['state_file']).name!=progress['state_file'] or file_hash(state_path)!=progress['state_sha256']:
                raise ValueError('Training-state checkpoint changed')
            state=torch.load(state_path,map_location='cpu',weights_only=True)
            if state['identity']!=identity or [s['dataset'] for s in state['trace']]!=schedule[:state['cursor']]:
                raise ValueError('Training cursor/schedule changed')
            model.load_state_dict(state['model'],strict=True)
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            restored_fingerprints={k:_tree_fingerprint(v) for k,v in
                [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict())]}
            if restored_fingerprints!=state['state_fingerprints']:
                raise ValueError('Restored model/optimizer/scheduler state differs')
            _restore_rng(state['rng'],device)
        start_cursor=state['cursor']
        if stop_after is not None and stop_after<=start_cursor: raise ValueError('Stop cursor already reached')

        def publish(status):
            if finish_guard: finish_guard()
            state.update(model={n:t.detach().cpu().clone() for n,t in model.state_dict().items()},
                         optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=_rng_state(device))
            state['state_fingerprints']={k:_tree_fingerprint(state[k]) for k in ('model','optimizer','scheduler')}
            path=directory/f'state-{state["cursor"]:06d}.pt'
            _atomic_torch_save(state,path)
            write_json(directory/'progress.json',{'status':status,'cursor':state['cursor'],
                       'state_file':path.name,'state_sha256':file_hash(path),
                       'best_step':state['best_step'],'best_score':state['best_score']})

        while state['cursor']<len(schedule):
            i=state['cursor']; epoch=i//settings['steps_per_epoch']
            if runtime_adapter is not None and hasattr(runtime_adapter, 'train_step_at'):
                step=runtime_adapter.train_step_at(i,model,optimizer,schedule[i],items[schedule[i]],config)
            else:
                step=step_fn(model,optimizer,schedule[i],items[schedule[i]],config)
            step.update(step=i+1,epoch=epoch)
            state['trace'].append(step); state['cursor']=i+1
            end_epoch=state['cursor']%settings['steps_per_epoch']==0
            if end_epoch and epoch in validation_epochs(settings):
                candidate=model; receipt=None
                if calibrated_selection:
                    model.eval()
                    candidate,receipt=candidate_fn(model,items,
                        directory/f'raw-validation-{state["cursor"]:06d}.pt')
                metric,prob=snapshot(candidate,items)
                scheduler.step(metric['val_loss'])
                event={'epoch':epoch,'step':state['cursor'],**metric}
                score=metric['macro_oa'] if score_fn is None else float(score_fn(metric))
                if not math.isfinite(score):raise ValueError('Nonfinite checkpoint selection score')
                if score_fn is not None:event['selection_score']=score
                state['history'].append(event)
                if state['best_score'] is None or score>state['best_score']:
                    state.update(best_score=score,best_step=state['cursor'],
                                 best_model={n:t.detach().cpu().clone() for n,t in candidate.state_dict().items()},
                                 best_probabilities=prob)
                    if calibrated_selection: state['best_calibration']=receipt
                publish('running')
                print(f'epoch={epoch} step={state["cursor"]} val_nll={metric["val_loss"]:.6f} macro_oa={metric["macro_oa"]:.6f}',flush=True)
            if stop_after is not None and state['cursor']==stop_after:
                if end_epoch and epoch in validation_epochs(settings):
                    progress=json.loads((directory/'progress.json').read_text())
                    progress['status']='interrupted'; write_json(directory/'progress.json',progress)
                else: publish('interrupted')
                return {'operation':'interrupted','updates_this_call':state['cursor']-start_cursor,
                        'cursor':state['cursor'],'restored_state_fingerprints':restored_fingerprints}
        # Final validation always publishes the final authoritative training state.
        runtime={**settings,'executed_optimizer_steps':len(schedule),'selected_step':state['best_step']}
        extras=(runtime_adapter.checkpoint_metadata(model)
                if runtime_adapter is not None and hasattr(runtime_adapter,'checkpoint_metadata') else {})
        reserved={'research_config','joint_runtime','provenance','training_code','checkpoint_optimizer_step',calibration_key}
        if not isinstance(extras,dict) or reserved.intersection(extras):
            raise ValueError('Runtime checkpoint metadata collides with canonical metadata')
        model.training_config={'research_config':config.to_dict(),'joint_runtime':runtime,
                               'provenance':provenance,'training_code':state['start_code'],
                               'checkpoint_optimizer_step':len(schedule),**extras}
        save_model_checkpoint(model,directory/'last.pt')
        model.load_state_dict(state['best_model'],strict=True)
        model.training_config['checkpoint_optimizer_step']=state['best_step']
        if calibrated_selection: model.training_config[calibration_key]=state['best_calibration']
        save_model_checkpoint(model,directory/'primary.pt')
        _atomic_torch_save(state['best_probabilities'],directory/'selected-probabilities.pt')
        selected_loader=load_selected if runtime_adapter is None else runtime_adapter.load_selected
        restored=(selected_loader(directory/'primary.pt',items,
                    directory/f'raw-validation-{state["best_step"]:06d}.pt',device=device)
                  if calibrated_selection else load_model_checkpoint(directory/'primary.pt',device=device))
        check,values=snapshot(restored,items)
        max_error=0.
        for n in tasks:
            for phase in ('train','val'):
                if set(state['best_probabilities'][n][phase])!=set(values[n][phase]):
                    raise ValueError('Selected active-head set changed')
                for head in state['best_probabilities'][n][phase]:
                    old,new=state['best_probabilities'][n][phase][head],values[n][phase][head]
                    torch.testing.assert_close(new,old,rtol=1e-5,atol=1e-6)
                    if not torch.equal(old.argmax(1),new.argmax(1)): raise ValueError('Selected predictions changed')
                    max_error=max(max_error,float((new-old).abs().max()))
        write_json(directory/'trace.json',state['trace']); write_json(directory/'history.json',state['history'])
        if finish_guard: finish_guard()
        progress=json.loads((directory/'progress.json').read_text())
        progress['status']='completed'; write_json(directory/'progress.json',progress)
        artifacts={p.name:file_hash(p) for p in directory.iterdir() if p.is_file() and p.name not in ('.lock','result.json')}
        result={'status':'completed','identity':identity,'start_code':state['start_code'],
                'settings':settings,'optimizer_steps':len(schedule),'schedule_sha256':digest(schedule),
                'per_dataset_updates':dict(Counter(schedule)),'selected_step':state['best_step'],
                'selected_validation':check,'selected_probability_max_abs_error':max_error,
                'last_model_fingerprint':state['state_fingerprints']['model'],
                'artifacts':artifacts,'test_scoring':False,
                'execution':{'device':str(device),'visible_cuda_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                             'torch_version':torch.__version__,'cuda_version':torch.version.cuda}}
        with (directory/'result.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        return {'operation':'completed','updates_this_call':len(schedule)-start_cursor,
                'restored_state_fingerprints':restored_fingerprints,'result':result}
