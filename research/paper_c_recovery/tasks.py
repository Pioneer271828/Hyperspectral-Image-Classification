"""Adapters to pinned native and external DSNet components, not their trainers."""
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

METHODS=('native','dsnet')
NATIVE='395caef57fba8ab6a09223538f748179dab552f2'
BANK='49bce8c5fa2d8b79c0e58b2a70693c7b897d4eb7'
AUTHOR='e910795073b52373ce0771ac85139a17f250b236'
PILOT='bddd9f59daf2a755c719de647a429d08d0932337547e65eb2e91a8070fea58b9'
OPTIMIZER={'name':'Adam','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False}
MODEL={'latent_channels':64,'adapter_embed_dim':32,'norm_type':'group_norm','graph_mode':'learned',
       'alph':.8,'region_residual_mode':'none','coverage_mode':'none','interaction_mode':'none'}
LOSSES={'class_balance':True,'graph_loss_weights':{'compactness':.01,'entropy':.001,'balance':.01},
        'cnn_loss_weight':1.,'graph_branch_loss_weight':1.}


def prior_module(root,name):
    """Read this repo's frozen prior adapter, never import an upstream trainer."""
    path=Path(root)/'research'/'paper1_c'/f'{name}.py'
    if name=='pilot':
        # The frozen pilot uses a sibling import; leave the pinned project
        # dependency first so its transfer/inference packages keep priority.
        sys.path.insert(1,str(path.parent))
    spec=importlib.util.spec_from_file_location('paper_c_previous_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)


def metric(probabilities,labels,classes,*,logits=False):
    cm=torch.zeros((classes,classes),dtype=torch.int64);total_nll=0.;n=0
    for output,y in zip(probabilities,labels):
        if y.ndim!=1 or y.dtype!=torch.long or not len(y) or int(y.min())<0 or int(y.max())>=classes:
            raise ValueError('Released class labels invalid')
        if output.shape!=(len(y),classes) or not torch.isfinite(output).all():raise ValueError('Output shape or finiteness differs')
        if logits:total_nll+=float(F.cross_entropy(output,y,reduction='sum'))
        else:
            if (output<0).any() or not torch.allclose(output.sum(1),torch.ones(len(y)),atol=1e-5,rtol=1e-5):
                raise ValueError('Native probability contract differs')
            total_nll-=float(output[torch.arange(len(y)),y].clamp_min(1e-8).log().sum())
        pred=output.argmax(1);cm+=torch.bincount(y*classes+pred,minlength=classes**2).reshape(classes,classes);n+=len(y)
    support=cm.sum(1)
    if n==0 or (support==0).any():raise ValueError('Selection role missing classes')
    recall=cm.diag().double()/support
    return {'nll':total_nll/n,'oa':float(cm.diag().sum().double()/n),'aa':float(recall.mean()),
            'confusion':cm.tolist(),'support':support.tolist(),'recall':recall.tolist(),'points':n}


class NativeTask:
    def __init__(self,data,scene,classes):
        from model.VCASAHGCN import MultiDatasetVCASAHGCN
        self.model=MultiDatasetVCASAHGCN({scene:classes},**MODEL)
        self.optimizer=make_optimizer(self.model);self.data=data;self.scene=scene;self.classes=classes

    def _forward(self,row):
        from loadData.spatial_protocol import forward_partition
        return forward_partition(self.model,self.scene,row['observation'],row['wavelengths'],
                                 50 if self.scene=='paviaU' else 100,wavelength_unit='nm')

    def loss(self,block):
        from model.task_losses import probability_loss_components
        row=self.data['devtrain'][block];out=self._forward(row)
        points=row['flat_points'];labels=row['labels'];pixels=out[0].shape[0]
        mask=torch.zeros((pixels,self.classes));mask[points]=1.
        onehot=torch.zeros_like(mask);onehot[points]=F.one_hot(labels,self.classes).float()
        return probability_loss_components(out,onehot,mask,**LOSSES)['total']

    def evaluate(self):
        rows=list(self.data['devval'].values())
        return metric([self._forward(r)[0][r['flat_points']] for r in rows],
                      [r['labels'] for r in rows],self.classes)


class DSNetTask:
    def __init__(self,data,scene,classes,bands,adapter,author):
        self.model=adapter.Adapter(author,bands,classes);self.optimizer=make_optimizer(self.model)
        self.data=data;self.classes=classes;self.objective=adapter.objective

    def loss(self,block):
        r=self.data['devtrain'][block]
        return self.objective(self.model(r['patches']),r['patches'],r['labels'])

    def evaluate(self):
        rows=list(self.data['devval'].values())
        return metric([self.model(r['patches'])[2] for r in rows],[r['labels'] for r in rows],self.classes,logits=True)


def materialize(bank,adapter):
    """Exactly admitted devtrain/devval rows; no raw image or GT/query reader."""
    result={};identities=[]
    for role in ('devtrain','devval'):
        rows={}
        for row in bank.role_rows(role):
            if not row['samples']:continue
            _,_,view,context=bank.materialize(row,{'mode':'none'})
            obs=view.observation;y0,_,x0,_=row['box'];width=bank.release.shape[1]
            points=[(int(i)//width-y0,int(i)%width-x0) for i,_ in row['samples']]
            labels=torch.tensor([label-1 for _,label in row['samples']],dtype=torch.long)
            flat=torch.tensor([y*obs.spectral.shape[1]+x for y,x in points],dtype=torch.long)
            x=torch.tensor(np.array(obs.spectral,copy=True))
            patches=adapter.patches(x,points)
            # Independent coordinate clamping, not the adapter's F.pad path.
            reference=np.stack([np.stack([np.stack([obs.spectral[min(max(y+dy,0),x.shape[0]-1),
                min(max(z+dx,0),x.shape[1]-1)] for dx in range(-3,4)]) for dy in range(-3,4)]).transpose(2,0,1)
                for y,z in points])
            if not np.array_equal(patches.numpy(),reference):raise ValueError('Patch bytes differ from independent local coordinates')
            rows[row['id']]={'observation':obs,'wavelengths':np.array(view.bands.wavelengths,dtype=np.float32),
                             'flat_points':flat,'labels':labels,'patches':patches}
            identities.append({'role':role,'block':row['id'],'sample_count':len(labels),
                               'context_sha256':hashlib.sha256(json_bytes(context)).hexdigest(),
                               'native_input_sha256':hashlib.sha256(np.concatenate((obs.spectral,obs.coordinates),-1).tobytes()).hexdigest(),
                               'patch_sha256':hashlib.sha256(patches.numpy().tobytes()).hexdigest()})
        result[role]=rows
    return result,identities


def json_bytes(value):
    import json
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def synthetic_data(adapter,bands,classes,*,height=28,width=28):
    from loadData.spatial_protocol import PartitionObservation
    rng=np.random.default_rng(7000+bands)
    yy,xx=np.mgrid[:height,:width]
    coord=np.stack((yy/(height-1),xx/(width-1)),-1).astype(np.float32)
    ids=yy*width+xx
    rows={}
    for role,partition in (('devtrain','train'),('devval','val')):
        # Each role has one synthetic point per class; never sampled real labels.
        spectral=rng.uniform(.1,1.,(height,width,bands)).astype(np.float32)
        points=[(i//width,i%width) for i in range(classes)]
        x=torch.tensor(spectral)
        rows[role]={0:{'observation':PartitionObservation(partition,spectral,coord,ids,'synthetic-cost-only'),
            'wavelengths':np.linspace(450,2500,bands,dtype=np.float32),
            'flat_points':torch.arange(classes),'labels':torch.arange(classes),
            'patches':adapter.patches(x,points)}}
    return rows
