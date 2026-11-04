"""G4 support-only short training; no query scoring or implicit full budget."""
import copy,json,math,tempfile
from dataclasses import replace
from pathlib import Path
import torch
from joint_baseline import NAMES,load_config
from joint_training import _atomic_torch_save
from loadData.dataset_registry import DATASETS,file_hash
from model.task_losses import probability_loss_components
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import tensor_hash,write_json
from target_initialization import initialize_target,state_id,METHODS
from target_bn_calibration import (calibrate_target,verify_target_binding,backbone_id,
    observation_id,OBS_KEYS,POLICY,CONTEXT,GRAPH_POLICY)

PROTOCOL='g4_support_training_engineering_v1'
HEAD_ONLY=METHODS[2]


def target_settings(target,method,*,seed=64,support_seed=64,per_class=10,device='cpu'):
    if (target not in NAMES or method not in METHODS or type(seed)is not int or seed not in (64,512,2048)
            or type(support_seed)is not int or support_seed not in (64,512,2048)
            or type(per_class)is not int or per_class not in (1,5,10)):
        raise ValueError('Unregistered target method/seed/label budget')
    cfg=load_config('configs/experiments/base97_joint.json')
    cfg=replace(cfg,name='g4_target_engineering_'+target,device=str(torch.device(device)),
        data=replace(cfg.data,datasets=[target]),training=replace(cfg.training,seed=seed,
            epochs=11,steps_per_epoch=1,val_interval=3,primary_checkpoint='val_loss'))
    cfg.validate()
    return {'protocol':PROTOCOL,'profile':'engineering','target':target,'method':method,
        'model_seed':seed,'head_seed':seed+1000,'schedule_seed':seed,'support_seed':support_seed,
        'support_per_class':per_class,'validation_per_class':5,'epochs':11,'val_interval':3,
        'steps_per_epoch':1,'optimizer_steps':12,'test_scoring':False,
        'scheduler':{'factor':.5,'patience':20,'threshold':.0001},'config':cfg.to_dict(),
        'validation_policy':{'calibration':POLICY,'selection':'minimum_fused_validation_nll_first_tie',
            'context':CONTEXT,'graph_policy':GRAPH_POLICY,'query_scoring':False}}


def observation(model,item):
    device=next(model.parameters()).device
    return {k:item[k].to(device) if torch.is_tensor(item[k]) else item[k] for k in OBS_KEYS}


def features(model,name,item):
    obs=observation(model,item)
    return model.backbone.forward_features(obs['net_input'],obs['Q'],obs['A'],
        wavelengths=obs['wavelengths'],fwhm=obs['fwhm'],dataset_name=name,
        superpixel_scale=obs['superpixel_scale'])[:3]


def selected_outputs(model,name,item,phase,feature_rows=None):
    device=next(model.parameters()).device
    if feature_rows is None:
        ids=item[phase+'_indices'].to(device)
        feature_rows=tuple(v[ids] for v in features(model,name,item))
    else:feature_rows=tuple(v.to(device) for v in feature_rows)
    return model.predict_features(name,*feature_rows)


def loss_terms(outputs,labels,config):
    onehot=torch.nn.functional.one_hot(labels,num_classes=outputs[0].shape[1]).to(outputs[0])
    mask=torch.ones_like(onehot)
    return probability_loss_components(outputs,onehot,mask,class_balance=config.loss.balanced,
        graph_branch_loss_weight=config.loss.graph_branch_loss_weight,cnn_loss_weight=config.loss.cnn_loss_weight)


def read_rows(outputs,labels,classes):
    metrics={};probabilities={}
    for name,value in zip(('fused','graph','cnn'),outputs):
        if not torch.isfinite(value).all():raise ValueError('Nonfinite selected probabilities')
        labels=labels.to(value.device);pred=value.argmax(1)
        cm=torch.bincount(labels*classes+pred,minlength=classes*classes).reshape(classes,classes).cpu()
        support=cm.sum(1)
        if not bool((support>0).all()):raise ValueError('Every target class must have permitted labels')
        recall=cm.diagonal().double()/support
        metrics[name]={'oa':int(cm.trace())/int(cm.sum()),'aa':float(recall.mean()),'recall':recall.tolist(),
            'confusion':cm.tolist(),'nll':float(-value.gather(1,labels[:,None]).clamp_min(1e-8).log().mean())}
        probabilities[name]=value.detach().cpu()
    return metrics,probabilities


class TargetRuntime:
    calibration_metadata_key='target_bn_calibration'

    def __init__(self,settings,items,directory,*,source_spec=None):
        self.settings=copy.deepcopy(settings);self.items=items;self.directory=Path(directory)
        self.source_spec=copy.deepcopy(source_spec or {});self.bank=None;self.bank_manifest=None
        self.validate(settings);self.validate_inputs(items)
        if (settings['method']==METHODS[0])!= (not self.source_spec):
            raise ValueError('Scratch/source specification role mismatch')

    def validate(self,settings):
        expected=target_settings(settings.get('target'),settings.get('method'),seed=settings.get('model_seed'),
            support_seed=settings.get('support_seed'),per_class=settings.get('support_per_class'),device=settings['config']['device'])
        if {k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}!=expected:
            raise ValueError('Target training policy/configuration mismatch')
        return True

    def task_names(self,settings):self.validate(settings);return (settings['target'],)

    def validate_inputs(self,items):
        target=self.settings['target'];classes=DATASETS[target][2]
        if tuple(items)!=(target,):raise ValueError('Single declared target required')
        item=items[target]
        allowed=OBS_KEYS|{'name','height','width','bands','class_num','train_indices','train_labels','val_indices','val_labels'}
        if set(item)!=allowed or item['name']!=target or item['class_num']!=classes:
            raise ValueError('Only sanitized support/validation fields accepted')
        observation_id({k:item[k] for k in OBS_KEYS})
        all_ids=[];n=item['net_input'].shape[0]*item['net_input'].shape[1]
        for phase,k in (('train',self.settings['support_per_class']),('val',5)):
            ids=item[phase+'_indices'];labels=item[phase+'_labels']
            if (ids.dtype!=torch.long or labels.dtype!=torch.long or ids.ndim!=1 or labels.shape!=ids.shape
                    or len(ids)!=k*classes or not bool(((ids>=0)&(ids<n)).all())
                    or not bool(((labels>=0)&(labels<classes)).all())
                    or not torch.equal(torch.bincount(labels.cpu(),minlength=classes),torch.full((classes,),k,dtype=torch.long))):
                raise ValueError('Target support/validation label budget differs')
            all_ids.extend(ids.cpu().tolist())
        if len(all_ids)!=len(set(all_ids)):raise ValueError('Target label roles overlap')

    def build_model(self,config,counts):
        target=self.settings['target']
        if counts!={target:DATASETS[target][2]}:raise ValueError('Target class count mismatch')
        model,self.initialization=initialize_target(target,self.settings['method'],
            model_seed=self.settings['model_seed'],head_seed=self.settings['head_seed'],device=config.device,**self.source_spec)
        model.eval()
        if self.settings['method']==HEAD_ONLY:
            model,self.initial_calibration=calibrate_target(model,observation(model,self.items[target]),mode=POLICY,
                context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))
            self.initialization=copy.deepcopy(model.training_config['target_initialization'])
        return model

    def optimizer(self,model,config):
        frozen=self.settings['method']==HEAD_ONLY
        model.backbone.requires_grad_(not frozen)
        heads=list(model.heads.parameters())+list(model.hgnn_heads.parameters())+list(model.cnn_heads.parameters())
        groups=[] if frozen else [{'params':model.backbone.parameters(),'lr':config.training.lr}]
        groups.append({'params':heads,'lr':config.training.head_lr})
        return torch.optim.Adam(groups,weight_decay=config.training.weight_decay)

    @staticmethod
    def selection_score(metric):return -metric['val_loss']

    def checkpoint_metadata(self,model):
        result={'target_initialization':copy.deepcopy(self.initialization)}
        if self.bank_manifest is not None:result['target_feature_bank_sha256']=file_hash(self.directory/'feature-bank.json')
        return result

    def _bank(self,model,item,*,cursor=None):
        if any(p.requires_grad for p in model.backbone.parameters()) or any(m.training for m in model.backbone.modules()):
            raise ValueError('Feature bank requires frozen eval backbone')
        expected={'backbone_sha256':backbone_id(model),'observation':observation_id(observation(model,item)),
            'rows':{p:{k:tensor_hash(item[p+'_'+k]) for k in ('indices','labels')} for p in ('train','val')}}
        manifest_path=self.directory/'feature-bank.json';bank_path=self.directory/'feature-bank.pt'
        if self.bank is None:
            if cursor is not None and cursor>0:
                progress=json.loads((self.directory/'progress.json').read_text())
                if Path(progress['state_file']).name!=progress['state_file']:raise ValueError('Invalid progress state path')
                state_path=self.directory/progress['state_file']
                if file_hash(state_path)!=progress['state_sha256']:raise ValueError('Changed resume state')
                state=torch.load(state_path,map_location='cpu',weights_only=True)
                if state['cursor']!=cursor or state['trace'][0].get('feature_bank_sha256')!=file_hash(manifest_path):
                    raise ValueError('Resume feature bank identity differs')
            if manifest_path.exists():
                self.bank_manifest=json.loads(manifest_path.read_text())
                if file_hash(bank_path)!=self.bank_manifest['bank_sha256']:raise ValueError('Feature bank payload changed')
                self.bank=torch.load(bank_path,map_location='cpu',weights_only=True)
            else:
                if cursor!=0:raise ValueError('Persisted feature bank missing')
                with torch.no_grad():all_features=features(model,self.settings['target'],item)
                self.bank={p:tuple(v[item[p+'_indices'].to(v.device)].detach().cpu() for v in all_features) for p in ('train','val')}
                _atomic_torch_save(self.bank,bank_path)
                self.bank_manifest={**expected,'bank_sha256':file_hash(bank_path),
                    'initial_calibration':copy.deepcopy(self.initial_calibration),'policy':'g4_frozen_selected_rows_v1'}
                write_json(manifest_path,self.bank_manifest)
        if (any(self.bank_manifest.get(k)!=v for k,v in expected.items())
                or file_hash(bank_path)!=self.bank_manifest['bank_sha256']
                or json.loads(manifest_path.read_text())!=self.bank_manifest):
            raise ValueError('Frozen feature bank binding changed')
        return self.bank

    def train_step_at(self,cursor,model,optimizer,name,item,config):
        frozen=self.settings['method']==HEAD_ONLY
        model.train()
        if frozen:model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('backbone.HGNN_Branch.')}
        bank=self._bank(model,item,cursor=cursor) if frozen else None
        outputs=selected_outputs(model,name,item,'train',None if bank is None else bank['train'])
        terms=loss_terms(outputs,item['train_labels'].to(outputs[0].device),config)
        if not torch.isfinite(terms['total']):raise ValueError('Nonfinite target loss')
        terms['total'].backward();grad={}
        for key,prefix in [('graph','backbone.HGNN_Branch.'),('adapter','backbone.adapter.'),
                ('fused_head','heads.'+name+'.'),('graph_head','hgnn_heads.'+name+'.'),('cnn_head','cnn_heads.'+name+'.')]:
            params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            if (not params and not (frozen and key in ('graph','adapter'))) or any(not torch.isfinite(p.grad).all() for p in params):
                raise ValueError('Missing/nonfinite target gradient: '+key)
            grad[key]=math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
        optimizer.step()
        changed=sum(int(torch.count_nonzero(p.detach()-before[n])) for n,p in model.named_parameters() if n in before)
        result={'dataset':name,'loss':{k:float(v.detach()) for k,v in terms.items()},'gradient_norms':grad,
            'graph_changed_elements':changed,'learning_rates':[float(g['lr']) for g in optimizer.param_groups]}
        if frozen:
            if changed or any(p.grad is not None for p in model.backbone.parameters()):raise ValueError('Frozen backbone updated')
            result['feature_bank_sha256']=file_hash(self.directory/'feature-bank.json')
        return result

    def train_step(self,*args):raise RuntimeError('Target training requires cursor-aware entry')

    def validation_snapshot(self,model,items):
        model.eval();name=self.settings['target'];item=items[name];metrics={name:{}};values={name:{}}
        frozen=self.settings['method']==HEAD_ONLY
        bank=self._bank(model,item) if frozen else None
        with torch.no_grad():
            # One backbone pass per candidate; classifier rows remain phase-specific.
            full=None if frozen else features(model,name,item)
            for phase in ('train','val'):
                rows=bank[phase] if frozen else tuple(v[item[phase+'_indices'].to(v.device)] for v in full)
                outputs=selected_outputs(model,name,item,phase,rows)
                metrics[name][phase],values[name][phase]=read_rows(outputs,item[phase+'_labels'],item['class_num'])
        val=metrics[name]['val']['fused']
        if not math.isfinite(val['nll']):raise ValueError('Nonfinite target validation NLL')
        return {'val_loss':val['nll'],'macro_oa':val['oa'],'legacy_val_oa':{name:val['oa']},'metrics':metrics},values

    def validation_candidate(self,model,items,raw_source_path):
        path=Path(raw_source_path)
        raw={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            try:torch.save(raw,stream)
            except Exception:
                stream.close();temporary.unlink(missing_ok=True);raise
        try:temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)
        name=self.settings['target'];obs=observation(model,items[name]);raw_id=state_id(model)
        if self.settings['method']==HEAD_ONLY:
            # Heads never feed BN. Rebind the unchanged, verified frozen calibrated
            # backbone; recomputing FP32 moments would invalidate persisted features.
            self._bank(model,items[name]);candidate=copy.deepcopy(model)
            receipt=copy.deepcopy(self.bank_manifest['initial_calibration'])
            if receipt['backbone_sha256']!=backbone_id(candidate) or receipt['observation']!=observation_id(obs):
                raise ValueError('Head-only calibration backbone changed')
            receipt['initial_calibration_source_state_sha256']=receipt['source_state_sha256']
            receipt.update(source_state_sha256=raw_id,state_sha256=state_id(candidate),
                head_only_rebind='unchanged_frozen_backbone_no_moment_recomputation')
        else:
            candidate,receipt=calibrate_target(model,obs,mode=POLICY,context=CONTEXT,
                graph_policy=GRAPH_POLICY,source_state_sha256=raw_id)
        receipt['raw_source_file_sha256']=file_hash(path)
        candidate.training_config['target_bn_calibration']=copy.deepcopy(receipt)
        return candidate,receipt

    def load_selected(self,path,items,raw_source_path,*,device='cpu'):
        model=load_model_checkpoint(path,device=device);meta=model.training_config
        self.validate(meta['joint_runtime'])
        if ({k:v for k,v in meta['joint_runtime'].items() if k not in ('executed_optimizer_steps','selected_step')}!=self.settings
                or meta['research_config']!=self.settings['config']
                or meta['target_initialization']!=self.initialization):raise ValueError('Selected target runtime/initialization differs')
        receipt=meta['target_bn_calibration']
        if file_hash(raw_source_path)!=receipt['raw_source_file_sha256']:raise ValueError('Selected raw source file changed')
        raw=torch.load(raw_source_path,map_location='cpu',weights_only=True)
        if digest({k:tensor_hash(v) for k,v in raw.items()})!=receipt['source_state_sha256']:
            raise ValueError('Selected raw source state differs')
        frozen=self.settings['method']==HEAD_ONLY;model.backbone.requires_grad_(not frozen);model.eval()
        verify_target_binding(model,observation(model,items[self.settings['target']]),context=CONTEXT,
            graph_policy=GRAPH_POLICY,source_state_sha256=receipt['source_state_sha256'])
        if frozen:
            if meta.get('target_feature_bank_sha256')!=file_hash(self.directory/'feature-bank.json'):
                raise ValueError('Selected feature bank metadata changed')
            self._bank(model,items[self.settings['target']])
        return model
