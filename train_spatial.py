"""Strict independent-block training using C0 attempts and the existing dual branch model."""
from dataclasses import asdict
from pathlib import Path
import argparse
import contextlib
import json
import traceback
import numpy as np
import torch
from scipy.io import loadmat
from model.research_config import ResearchConfig, digest
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from loadData.dataset_registry import DATASETS, inspect_dataset
from loadData.data_read import readWavelengths
from loadData.spatial_protocol import forward_partition
from loadData.spatial_tiles import make_tile_plan, validate_tile_plan, fit_tile_statistics, observe_tile, tile_slice, tile_coverage
from research_runtime import (RunRecord, AlreadyCompleted, code_snapshot, require_frozen_code, tensor_hash, write_json)
from train_vca_multidataset import supervised_loss

class SpatialConfig:
    """Separate explicit schema; unused legacy data/augmentation switches cannot enter silently."""
    def __init__(self,value):
        defaults={'schema_version':1,'name':'spatial','dataset':'paviaU','device':'cpu','model':{},
                  'interaction':{},'loss':{},'training':{},'tile_size':32,'gap':2,'split_seed':2026}
        if not isinstance(value,dict) or set(value)-set(defaults):raise ValueError('Unknown spatial config fields')
        values={**defaults,**value}
        if type(values['schema_version']) is not int or values['schema_version']!=1:raise ValueError('Unsupported schema')
        if values['dataset'] not in DATASETS:raise ValueError('Unregistered dataset')
        self.research=ResearchConfig.from_dict({k:values[k] for k in ('name','device','model','interaction','loss','training')}
                                             | {'data':{'datasets':[values['dataset']]}})
        if self.research.model.graph_mode!='learned':raise ValueError('Strict training requires learned graph')
        if self.research.training.steps_per_epoch<=0:raise ValueError('Explicit positive steps_per_epoch required')
        if self.research.training.eval_checkpoint!='primary':raise ValueError('Strict adapter evaluates validation-selected primary only')
        make_tile_plan((values['tile_size']*4,values['tile_size']*4),values['tile_size'],values['gap'],values['split_seed'])
        self.values={**values,'model':asdict(self.research.model),'interaction':asdict(self.research.interaction),
                     'loss':asdict(self.research.loss),'training':asdict(self.research.training)}
        self.name=self.research.name
        self.training=self.research.training
    def to_dict(self):return json.loads(json.dumps(self.values))
    @property
    def config_hash(self):return digest(self.to_dict())
    @property
    def budget(self):return self.research.budget

def evaluate(model,cube,labels,plan,stats,partition,dataset,classes,wave,scale):
    """Streaming confusion and NLL; missing classes stay explicit."""
    model.eval()
    confusion=np.zeros((classes,classes),dtype=np.int64)
    nll=0.0
    with torch.no_grad():
        for tile in plan['tiles']:
            if tile['partition']!=partition:continue
            obs=observe_tile(cube,plan,stats,tile)
            output=forward_partition(model,dataset,obs,wave,scale)[0].cpu().numpy()
            target=np.asarray(labels[tile_slice(tile)]).reshape(-1).astype(np.int64)
            valid=target>0
            if not valid.any():continue
            truth=target[valid]-1;prediction=output[valid].argmax(axis=1)
            confusion+=np.bincount(truth*classes+prediction,minlength=classes*classes).reshape(classes,classes)
            nll-=float(np.log(np.maximum(output[valid,truth],1e-8)).sum())
    support=confusion.sum(axis=1);total=int(support.sum())
    if total==0:raise ValueError(f'{partition} contains no supervised pixels')
    oa=float(np.trace(confusion)/total)
    expected=float(np.dot(support.astype(float),confusion.sum(axis=0).astype(float))/(total*total))
    return {'nll':nll/total,'oa':oa,'aa_supported':float(np.mean(np.diag(confusion)[support>0]/support[support>0])),
            'kappa':(oa-expected)/(1-expected) if expected<1 else None,'count':total,
            'support':support.tolist(),'missing_classes':(np.flatnonzero(support==0)+1).tolist(),'confusion':confusion.tolist()}

def train_blocks(config,cube,labels,raw_identity,record,wave=None):
    cfg=config.research;v=config.values;dataset=v['dataset']
    classes=DATASETS[dataset][2]
    if cube.ndim!=3 or labels.shape!=cube.shape[:2]:raise ValueError('Image/label shape mismatch')
    plan=make_tile_plan(tuple(cube.shape[:2]),v['tile_size'],v['gap'],v['split_seed'])
    stats=fit_tile_statistics(cube,plan)
    train_tiles=[tile for tile in plan['tiles'] if tile['partition']=='train' and (labels[tile_slice(tile)]>0).any()]
    counts=np.zeros(classes,dtype=np.int64)
    for tile in train_tiles:
        local=np.asarray(labels[tile_slice(tile)])
        if not np.isfinite(local).all() or (local<0).any() or (local>classes).any() or not np.equal(local,np.floor(local)).all():
            raise ValueError('Invalid training labels')
        counts+=np.bincount(local.astype(np.int64).ravel(),minlength=classes+1)[1:]
    if (counts==0).any():raise ValueError(f'Train region missing classes {(np.flatnonzero(counts==0)+1).tolist()}; no automatic redraw')
    scale=cfg.data.pavia_superpixel_scale if dataset=='paviaU' else cfg.data.superpixel_scale
    if cfg.model.norm_type in ('domain_bn','batch_norm'):
        for tile in train_tiles:
            y0,y1,x0,x1=tile['box']
            gh,gw=LearnableSuperpixelGraph._grid_shape(y1-y0,x1-x0,scale)
            if gh*gw<2:raise ValueError('BatchNorm requires at least two graph nodes per train block; increase tile size or explicitly choose group_norm')
    record.record_protocol_data({'raw':raw_identity,'plan_hash':plan['plan_hash'],
                                 'statistics_hash':stats['statistics_hash'],'wavelength_hash':tensor_hash(wave)})
    write_json(record.attempt/'protocol.json',plan);write_json(record.attempt/'statistics.json',stats)
    torch.manual_seed(cfg.training.seed)
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    model=cfg.build_model({dataset:classes}).to(cfg.device)
    record.record_model(model)
    model.training_config={'spatial_config':config.to_dict(),'plan_hash':plan['plan_hash'],
                           'statistics_hash':stats['statistics_hash'],'run_id':record.run_id}
    heads=list(model.heads.parameters())+list(model.hgnn_heads.parameters())+list(model.cnn_heads.parameters())
    optimizer=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':cfg.training.lr},
                                {'params':heads,'lr':cfg.training.head_lr}],weight_decay=cfg.training.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=20)
    rng=np.random.default_rng(cfg.training.seed)
    history=[];best=None;updates=0
    for epoch in range(cfg.training.epochs+1):
        for _ in range(cfg.training.steps_per_epoch):
            tile=train_tiles[int(rng.integers(len(train_tiles)))];obs=observe_tile(cube,plan,stats,tile)
            model.train();optimizer.zero_grad()
            outputs=forward_partition(model,dataset,obs,wave,scale)
            local=torch.as_tensor(np.array(labels[tile_slice(tile)],copy=True).reshape(-1),device=cfg.device,dtype=torch.long)
            mask=(local>0)[:,None].expand(-1,classes)
            onehot=torch.nn.functional.one_hot((local-1).clamp_min(0),classes).float()*mask
            loss=supervised_loss(outputs,{'train_gt_onehot':onehot,'train_label_mask':mask.float()},'train',
                class_balance=cfg.loss.balanced,cnn_loss_weight=cfg.loss.cnn_loss_weight,
                graph_branch_loss_weight=cfg.loss.graph_branch_loss_weight,
                graph_loss_weights={'compactness':cfg.loss.graph_compactness_weight,
                    'entropy':cfg.loss.graph_entropy_weight,'balance':cfg.loss.graph_balance_weight})
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();updates+=1
        if epoch%cfg.training.val_interval==0 or epoch==cfg.training.epochs:
            val=evaluate(model,cube,labels,plan,stats,'val',dataset,classes,wave,scale)
            scheduler.step(val['nll'])
            criterion=-val['oa'] if cfg.training.primary_checkpoint=='macro_oa' else val['nll']
            history.append({'epoch':epoch,'optimizer_steps':updates,'val':val})
            if best is None or criterion<best:
                best=criterion;save_model_checkpoint(model,record.save_path)
    save_model_checkpoint(model,record.attempt/'last.pt')
    selected,_,_=restore_spatial_checkpoint(record.save_path,record.attempt/'protocol.json',record.attempt/'statistics.json',cfg.device)
    # First semantic test-label access for evaluation occurs after checkpoint selection.
    test=evaluate(selected,cube,labels,plan,stats,'test',dataset,classes,wave,scale)
    result={'schema_version':1,'protocol':'spatial_tiles_v1','run_id':record.run_id,'optimizer_steps':updates,
            'selection':cfg.training.primary_checkpoint,'history':history,'test':test,
            'coverage':tile_coverage(labels,plan,classes),'plan_hash':plan['plan_hash'],
            'statistics_hash':stats['statistics_hash'],'scope':'independent block contexts; not scene-wide inference'}
    write_json(record.metrics_path,result)
    return result

def restore_spatial_checkpoint(checkpoint,protocol,statistics,device='cpu'):
    plan=json.loads(Path(protocol).read_text(encoding='utf-8'))
    stats=json.loads(Path(statistics).read_text(encoding='utf-8'))
    validate_tile_plan(plan)
    identity={k:v for k,v in stats.items() if k!='statistics_hash'}
    if stats.get('plan_hash')!=plan['plan_hash'] or stats.get('fit_partition')!='train' or digest(identity)!=stats.get('statistics_hash'):
        raise ValueError('Protocol/statistics identity mismatch')
    model=load_model_checkpoint(checkpoint,device)
    if model.training_config.get('plan_hash')!=plan['plan_hash'] or model.training_config.get('statistics_hash')!=stats['statistics_hash']:
        raise ValueError('Checkpoint belongs to a different spatial protocol/statistics')
    return model,plan,stats

def execute(config,root,code,data_loader,resume=False):
    record=RunRecord(root,config,code,resume=resume)
    with record:
        with (record.attempt/'training.log').open('w',encoding='utf-8') as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:
                    require_frozen_code(code)
                    try:
                        cube,labels,identity,wave=data_loader()
                        train_blocks(config,cube,labels,identity,record,wave=wave)
                    finally:require_frozen_code(code)
                except AlreadyCompleted:raise
                except BaseException:
                    traceback.print_exc();raise
    return record.manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--out-root',required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=SpatialConfig(json.loads(Path(a.config).read_text(encoding='utf-8')))
    code=code_snapshot()
    if code['dirty']:p.error('Commit and synchronize current source first')
    print({'config_hash':config.config_hash,'budget':config.budget,'geometry':{k:config.values[k] for k in ('tile_size','gap','split_seed')}},flush=True)
    if a.dry_run:return
    def loader():
        name=config.values['dataset'];record,labels=inspect_dataset(a.data_root,name)
        _,key,_,_=DATASETS[name]
        path=Path(a.data_root)/record['image_path']
        cube=loadmat(path,variable_names=[key])[key] if key else np.load(path,mmap_mode='r',allow_pickle=False)
        # Approximate metadata remains explicitly unverified; original adapter semantics retained.
        wave,_=readWavelengths(name,cube.shape[2])
        return cube,labels,record,wave
    print(execute(config,a.out_root,code,loader,a.resume)['status'],flush=True)

if __name__=='__main__':main()
