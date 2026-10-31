"""Source-only pretraining and support-only G3 target adaptation."""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import numpy as np
import torch
from loadData.spatial_tiles import observe_tile,tile_slice,fit_tile_statistics,validate_tile_plan
from loadData.spatial_protocol import forward_partition
from loadData.observation_sampling import BandMetadata,sample
from loadData.transfer_protocol import _validate_record,target_labels_for_tile
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from model.research_config import digest
from research_runtime import tensor_hash,hash_file,write_json
from train_spatial import SpatialConfig,evaluate
from train_vca_multidataset import supervised_loss
from model.task_losses import detached_loss_values

METHODS=('target_scratch_full','source_pretrained_head_only','source_pretrained_full_finetune')


@dataclass
class Scene:
    dataset: str
    cube: np.ndarray
    plan: dict
    bands: BandMetadata
    classes: int
    stats: dict

    @classmethod
    def prepare(cls,dataset,cube,plan,bands,classes):
        validate_tile_plan(plan)
        if bands.wavelength_unit!='nm' or bands.wavelength_quality=='index_only' or not all(bands.valid):
            raise ValueError('First G3 runner requires full delivered physical-center metadata (quality may be approximate)')
        if list(cube.shape[:2])!=plan['shape'] or cube.shape[-1]!=len(bands.band_ids):raise ValueError('Scene/metadata mismatch')
        return cls(dataset,cube,plan,bands,classes,fit_tile_statistics(cube,plan))


def state_hash(model):
    return digest({name:tensor_hash(value) for name,value in model.state_dict().items()})


def head_hash(model):
    return digest({name:state_hash(getattr(model,name)) for name in ('heads','hgnn_heads','cnn_heads')})


def build_model(base,dataset,classes,seed,device):
    value=copy.deepcopy(base);value.update(dataset=dataset,device=device)
    config=SpatialConfig(value)
    if config.research.model.norm_type!='group_norm' or config.research.interaction.interaction_mode!='none':
        raise ValueError('Frozen G3 baseline uses GroupNorm and no interaction')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=config.research.build_model({dataset:classes})
    return model.to(device),config


def build_target(base,scene,method,source,*,model_seed,head_seed,device):
    if method not in METHODS:raise ValueError('Unknown target method')
    model,config=build_model(base,scene.dataset,scene.classes,model_seed,device)
    # Reset all three target heads independently of source label semantics.
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
        torch.manual_seed(head_seed)
        for name in ('heads','hgnn_heads','cnn_heads'):getattr(model,name)[scene.dataset].reset_parameters()
    if method!='target_scratch_full':
        if source is None:raise ValueError('Selected source checkpoint required')
        model.backbone.load_state_dict(source.backbone.state_dict(),strict=True)
    frozen=method=='source_pretrained_head_only'
    model.backbone.requires_grad_(not frozen)
    if frozen:model.backbone.eval()
    return model,config


def _optimizer(model,config):
    backbone=[p for p in model.backbone.parameters() if p.requires_grad]
    heads=[p for name in ('heads','hgnn_heads','cnn_heads') for p in getattr(model,name).parameters() if p.requires_grad]
    groups=([{'params':backbone,'lr':config.training.lr}] if backbone else [])+[{'params':heads,'lr':config.training.head_lr}]
    return torch.optim.Adam(groups,weight_decay=config.training.weight_decay)


def _forward(model,scene,tile):
    obs=observe_tile(scene.cube,scene.plan,scene.stats,tile)
    view,_=sample(obs,scene.bands,{'mode':'none'})
    return forward_partition(model,scene.dataset,view.observation,np.asarray(view.bands.wavelengths,dtype=np.float32),
                             50 if scene.dataset=='paviaU' else 100)


def _update(model,scene,tile,local_labels,config,optimizer,head_only=False):
    model.train()
    if head_only:model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    outputs=_forward(model,scene,tile)
    local=torch.as_tensor(np.array(local_labels,copy=True).reshape(-1),device=next(model.parameters()).device,dtype=torch.long)
    valid=local>0
    if not valid.any() or (local[valid]>scene.classes).any():raise ValueError('Training tile needs valid released labels')
    mask=valid[:,None].expand(-1,scene.classes)
    onehot=torch.nn.functional.one_hot((local-1).clamp_min(0),scene.classes).float()*mask
    cfg=config.research
    components=supervised_loss(outputs,{'train_gt_onehot':onehot,'train_label_mask':mask.float()},'train',
        class_balance=cfg.loss.balanced,cnn_loss_weight=cfg.loss.cnn_loss_weight,graph_branch_loss_weight=cfg.loss.graph_branch_loss_weight,
        graph_loss_weights={'compactness':cfg.loss.graph_compactness_weight,'entropy':cfg.loss.graph_entropy_weight,
                            'balance':cfg.loss.graph_balance_weight},return_components=True)
    if not torch.isfinite(components['total']):raise ValueError('Nonfinite training loss')
    components['total'].backward();optimizer.step()
    return detached_loss_values(components)


def source_train_counts(scene,labels):
    if list(np.shape(labels))!=scene.plan['shape']:raise ValueError('Source label shape mismatch')
    counts=np.zeros(scene.classes,dtype=np.int64)
    for tile in scene.plan['tiles']:
        if tile['partition']!='train':continue
        local=np.asarray(labels[tile_slice(tile)])
        if not np.isfinite(local).all() or not np.equal(local,np.floor(local)).all() or (local<0).any() or (local>scene.classes).any():
            raise ValueError('Invalid source training labels')
        counts+=np.bincount(local.astype(np.int64).ravel(),minlength=scene.classes+1)[1:]
    if (counts==0).any():raise ValueError('Source train classes missing; no split redraw')
    return counts.tolist()

def train_source(scene,source_labels,base,folder,*,steps,val_every,model_seed,sampler_seed,device,task_hash,progress=None):
    """Only this source scene and its labels are available to this stage."""
    if type(steps) is not int or steps<=0 or type(val_every) is not int or val_every<=0:raise ValueError('Positive source budget required')
    counts=source_train_counts(scene,source_labels)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model,config=build_model(base,scene.dataset,scene.classes,model_seed,device)
    config=SpatialConfig(config.to_dict()|{'training':config.to_dict()['training']|{'epochs':0,'steps_per_epoch':steps}})
    model.training_config={'task_hash':task_hash,'phase':'source','plan_hash':scene.plan['plan_hash'],'statistics_hash':scene.stats['statistics_hash']}
    initial_backbone=state_hash(model.backbone)
    optimizer=_optimizer(model,config);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=20)
    tiles=[t for t in scene.plan['tiles'] if t['partition']=='train' and (source_labels[tile_slice(t)]>0).any()]
    if not tiles:raise ValueError('No source training labels')
    rng=np.random.default_rng(sampler_seed);torch.manual_seed(model_seed);best=None;history=[];trace=[]
    for step in range(1,steps+1):
        tile=tiles[int(rng.integers(len(tiles)))];loss=_update(model,scene,tile,source_labels[tile_slice(tile)],config,optimizer)
        if step==1 or step%10==0 or step==steps:
            event={'phase':'source_train','step':step,'budget':steps,'tile_id':tile['id'],'loss':loss}
            trace.append(event)
            if progress:progress(event)
        if step%val_every==0 or step==steps:
            val=evaluate(model,scene.cube,source_labels,scene.plan,scene.stats,'val',scene.dataset,scene.classes,
                         np.asarray(scene.bands.wavelengths,dtype=np.float32),50 if scene.dataset=='paviaU' else 100)
            scheduler.step(val['nll']);history.append({'step':step,'val':val})
            if best is None or val['nll']<best:
                best=val['nll'];model.training_config['selected_step']=step;save_model_checkpoint(model,folder/'model.pt')
            if progress:progress({'phase':'source_val','step':step,'budget':steps,'val_nll':val['nll']})
    selected=load_model_checkpoint(folder/'model.pt',device)
    result={'status':'completed','optimizer_steps':steps,'selection':'source_val_nll','selected_step':min(history,key=lambda x:x['val']['nll'])['step'],
            'history':history,'trace':trace,'source_train_counts':counts,'initial_backbone_hash':initial_backbone,'source_test_evaluated':False,'checkpoint_sha256':hash_file(folder/'model.pt'),
            'sidecar_sha256':hash_file(folder/'model.pt.json'),'backbone_hash':state_hash(selected.backbone),'resolved_config':config.to_dict()}
    write_json(folder/'result.json',result);write_json(folder/'statistics.json',scene.stats)
    return selected,result


def adapt_target(scene,support,base,folder,*,method,source,steps,model_seed,head_seed,sampler_seed,device,task_hash,progress=None):
    """No full target label image is accepted; all supervision comes from frozen support."""
    if type(steps) is not int or steps<=0:raise ValueError('Positive target budget required')
    _validate_record(support,scene.plan)
    if support['classes']!=list(range(1,scene.classes+1)):raise ValueError('Target head/support class mismatch')
    sparse=np.zeros(scene.plan['shape'],dtype=np.int64);sparse.reshape(-1)[support['indices']]=support['labels']
    tiles=[t for t in scene.plan['tiles'] if t['partition']=='train' and (sparse[tile_slice(t)]>0).any()]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model,config=build_target(base,scene,method,source,model_seed=model_seed,head_seed=head_seed,device=device)
    config=SpatialConfig(config.to_dict()|{'training':config.to_dict()['training']|{'epochs':0,'steps_per_epoch':steps}})
    initial_heads=head_hash(model);initial_backbone=state_hash(model.backbone)
    model.training_config={'task_hash':task_hash,'phase':'target','method':method,'support_hash':support['support_hash'],
                           'plan_hash':scene.plan['plan_hash'],'statistics_hash':scene.stats['statistics_hash'],
                           'selection':'fixed_final_step','optimizer_steps':steps}
    optimizer=_optimizer(model,config);rng=np.random.default_rng(sampler_seed);torch.manual_seed(model_seed)
    trace=[];frozen=method=='source_pretrained_head_only'
    for step in range(1,steps+1):
        tile=tiles[int(rng.integers(len(tiles)))];labels=target_labels_for_tile(sparse,scene.plan,support,tile['id'],phase='adapt')
        loss=_update(model,scene,tile,labels,config,optimizer,head_only=frozen)
        if step==1 or step%10==0 or step==steps:
            event={'phase':'target_adapt','method':method,'step':step,'budget':steps,'tile_id':tile['id'],'loss':loss}
            trace.append(event)
            if progress:progress(event)
    unchanged=state_hash(model.backbone)==initial_backbone
    if frozen and not unchanged:raise AssertionError('Frozen backbone parameters/buffers changed')
    save_model_checkpoint(model,folder/'model.pt')
    result={'status':'completed','method':method,'optimizer_steps':steps,'selection':'fixed_final_step','trace':trace,
            'support_hash':support['support_hash'],'target_support_labels':len(support['indices']),'target_validation_labels':0,
            'initial_head_hash':initial_heads,'final_head_hash':head_hash(model),'initial_backbone_hash':initial_backbone,'backbone_unchanged':unchanged,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'checkpoint_sha256':hash_file(folder/'model.pt'),'sidecar_sha256':hash_file(folder/'model.pt.json'),'resolved_config':config.to_dict()}
    write_json(folder/'result.json',result);write_json(folder/'statistics.json',scene.stats)
    return result


def evaluate_target(scene,support,labels,checkpoint,*,expected_checkpoint_hash,expected_sidecar_hash,task_hash,method,device):
    if hash_file(checkpoint)!=expected_checkpoint_hash or hash_file(str(checkpoint)+'.json')!=expected_sidecar_hash:raise ValueError('Final checkpoint/sidecar changed')
    model=load_model_checkpoint(checkpoint,device)
    info=model.training_config
    if any(info.get(k)!=v for k,v in {'task_hash':task_hash,'method':method,'phase':'target',
             'support_hash':support['support_hash'],'plan_hash':scene.plan['plan_hash'],
             'statistics_hash':scene.stats['statistics_hash'],'selection':'fixed_final_step'}.items()):
        raise ValueError('Final checkpoint/task identity mismatch')
    released=np.zeros(scene.plan['shape'],dtype=np.int64)
    for tile in scene.plan['tiles']:
        if tile['partition']=='test':released[tile_slice(tile)]=target_labels_for_tile(labels,scene.plan,support,tile['id'],phase='final_test',final_checkpoint_sha256=expected_checkpoint_hash)
    return evaluate(model,scene.cube,released,scene.plan,scene.stats,'test',scene.dataset,scene.classes,
                    np.asarray(scene.bands.wavelengths,dtype=np.float32),100)




