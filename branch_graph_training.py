"""BG-ABL1 runtime adapter; existing trainer owns scheduling and continuation."""
import copy
import math
import numpy as np
import torch
from joint_baseline import readout
from joint_calibrated_selection import VALIDATION_POLICY
from model.branch_graph_ablation import BranchGraphAblation,variant_spec,load_ablation_selected
from model import utils

PROTOCOL='branch_graph_joint_ablation_v1'


def ablation_settings(settings,variant):
    if settings.get('protocol')!='base97_honghu_joint_training_v1' or 'ablation' in settings:
        raise ValueError('Original BASE97 settings required')
    value=copy.deepcopy(settings);value.update(protocol=PROTOCOL,ablation=variant_spec(variant),
        validation_policy=copy.deepcopy(VALIDATION_POLICY))
    return value


def forward(model,name,item):
    return model(name,item['net_input'],item.get('Q'),item.get('A'),wavelengths=item['wavelengths'],
                 fwhm=item['fwhm'],superpixel_scale=item['superpixel_scale'])


def loss_terms(outputs,item,config):
    if set(outputs) not in ({'fused'},{'fused','graph','cnn'}):raise ValueError('Unknown active-head set')
    terms={head+'_supervised':utils.compute_loss(q,item['train_gt_onehot'],item['train_label_mask'],
        class_balance=config.loss.balanced) for head,q in outputs.items()}
    if len(outputs)==3:
        total=terms['fused_supervised']+config.loss.graph_branch_loss_weight*terms['graph_supervised']+config.loss.cnn_loss_weight*terms['cnn_supervised']
    else:total=terms['fused_supervised']
    return {**terms,'total':total}


def parameter_manifest(model):
    groups={}
    for name,p in model.named_parameters():
        group=('adapter' if name.startswith('backbone.adapter.') else 'denoise' if name.startswith('backbone.CNN_denoise.')
               else 'cnn' if name.startswith('backbone.CNN_Branch.') else 'graph' if name.startswith('backbone.HGNN_Branch.')
               else 'primary_head' if name.startswith('heads.') else 'graph_head' if name.startswith('hgnn_heads.') else 'cnn_head')
        groups[group]=groups.get(group,0)+p.numel()
    return {'groups':groups,'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}


class AblationRuntime:
    def __init__(self,variant):self.spec=variant_spec(variant)

    def validate(self,settings):
        if (settings.get('protocol')!=PROTOCOL or settings.get('ablation')!=self.spec
            or settings.get('validation_policy')!=VALIDATION_POLICY):
            raise ValueError('Explicit matching ablation protocol/policy required')
        return True

    def build_model(self,config,counts):return BranchGraphAblation(counts,self.spec['variant'],**config.model_kwargs())

    def train_step(self,model,optimizer,name,item,config):
        if model.ablation!=self.spec:raise ValueError('Runtime/model variant mismatch')
        model.train();optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        terms=loss_terms(forward(model,name,item),item,config)
        if not torch.isfinite(terms['total']):raise ValueError('Nonfinite ablation loss')
        terms['total'].backward()
        gradients={};changes={};active=[]
        for key,prefix in [('adapter','backbone.adapter.'),('denoise','backbone.CNN_denoise.'),
                           ('cnn','backbone.CNN_Branch.'),('graph','backbone.HGNN_Branch.'),
                           ('primary_head','heads.'+name+'.'),('graph_head','hgnn_heads.'+name+'.'),
                           ('cnn_head','cnn_heads.'+name+'.')]:
            params=[(n,p) for n,p in model.named_parameters() if n.startswith(prefix)]
            if not params:continue
            # Other-domain affine parameters are registered but inactive this step.
            contributing=[(n,p) for n,p in params if p.grad is not None]
            if not contributing or any(not torch.isfinite(p.grad).all() for _,p in contributing):
                raise ValueError('Missing/nonfinite active gradient: '+key)
            active.append((key,params));gradients[key]=math.sqrt(sum(float(p.grad.double().square().sum()) for _,p in contributing))
        optimizer.step()
        for key,params in active:
            changes[key]=sum(int(torch.count_nonzero(p.detach()-before[n])) for n,p in params)
        for n,p in model.named_parameters():
            if n.startswith(('heads.','hgnn_heads.','cnn_heads.')) and n.split('.')[1]!=name and not torch.equal(p.detach(),before[n]):
                raise ValueError('Inactive task classifier changed')
        return {'dataset':name,'loss':{k:float(v.detach()) for k,v in terms.items()},
                'gradient_norms':gradients,'changed_elements':changes,
                'graph_changed_elements':changes.get('graph',0),
                'learning_rates':[float(g['lr']) for g in optimizer.param_groups]}

    def validation_snapshot(self,model,items):
        model.eval();losses=[];oas={};metrics={};probabilities={}
        with torch.no_grad():
            for name,item in items.items():
                output=forward(model,name,item);q=output['fused']
                losses.append(utils.compute_loss(q,item['val_gt_onehot'],item['val_label_mask']))
                oas[name]=float(utils.evaluate_performance(q,item['val_samples_gt'],item['val_gt_onehot'],item['zeros']))
                metrics[name]={};probabilities[name]={}
                for phase in ('train','val'):
                    metrics[name][phase]={};probabilities[name][phase]={}
                    for head,value in output.items():
                        # Reuse established arithmetic without inventing stored
                        # inactive heads: only this one head's readout is retained.
                        m,p=readout((value,value,value),item,phase)
                        metrics[name][phase][head]=m['fused'];probabilities[name][phase][head]=p['fused']
        loss=float(torch.stack(losses).mean().cpu())
        if not math.isfinite(loss):raise ValueError('Nonfinite validation loss')
        return {'val_loss':loss,'macro_oa':float(np.mean(list(oas.values()))),'legacy_val_oa':oas,'metrics':metrics},probabilities

    def load_selected(self,path,items,raw_source_path,*,device='cpu'):
        return load_ablation_selected(path,items,raw_source_path,device=device,expected_variant=self.spec['variant'])
