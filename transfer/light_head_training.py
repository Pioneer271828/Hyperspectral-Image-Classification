"""CPU independent light-head training; frozen source/prefix and explicit selection."""
from copy import deepcopy
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from inference.light_cnn import LightCNN,validate_source
from loadData.spatial_tiles import observe_tile,tile_slice
from loadData.observation_sampling import sample
from model.research_config import digest
from model.utils import compute_loss
from transfer.observation_policy import scene_identity
from transfer.development_training import sparse_labels,development_counts
from transfer.tile_sampling import SupportTileSampler
from transfer.g3_training import state_hash


def _finite(value):return type(value) in (int,float) and math.isfinite(value)


class LightHeadRun:
    """An opt-in head-only runner, not a calibrated or deployable cascade.

    Full-model parameters and buffers are checked but never registered with Adam.
    Local sampler RNG is the only training RNG; eval prefix has no dropout.
    """
    def __init__(self,source,scene,record,*,updates,seed,lr,class_balance):
        validate_source(source);identity=scene_identity(scene,record)
        if type(updates) is not int or updates<1 or type(seed) is not int or not 0<=seed<2**32:
            raise ValueError('Positive budget and uint32 sampler seed required')
        if not _finite(lr) or lr<=0 or type(class_balance) is not bool:raise ValueError('Explicit positive Adam lr and class balance required')
        if scene.dataset not in source.cnn_heads or source.cnn_heads[scene.dataset].out_features!=scene.classes:raise ValueError('Matching source task head required')
        self.source=source;self.scene=scene;self.record=deepcopy(record);self.path=LightCNN(source)
        self.head=deepcopy(self.path.heads[scene.dataset]).requires_grad_(True)
        self.prefix_state=state_hash(self.path);self.optimizer=torch.optim.Adam(self.head.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False)
        self.sampler=SupportTileSampler(scene.plan,record['roles'],sparse_labels(record,scene.plan,'train_support'),record['classes'],'class_first')
        self.spec=dict(schema_version=1,source_state=state_hash(source),prefix_state=self.prefix_state,initial_head_state=state_hash(self.head),
            identity=identity,architecture=source.export_config(),updates=updates,seed=seed,lr=lr,class_balance=class_balance,
            optimizer_config=deepcopy(self.optimizer.state_dict()['param_groups']),
            sampler_hash=self.sampler.metadata['sampler_hash'],method='independent_frozen_prefix_head_only',
            loss='original_single_CNN_probability_loss',selection='first_min_full_devval_CNN_NLL_each_update')
        self.contract_hash=digest(self.spec);self.rng=np.random.default_rng(seed);self.completed=0;self.history=[];self.selected_step=None;self.selected_head=None;self.failed=False

    def _check(self):
        if self.failed:raise ValueError('Failed run must be preserved and diagnosed')
        validate_source(self.source)
        if (state_hash(self.source)!=self.spec['source_state'] or state_hash(self.path)!=self.prefix_state
                or scene_identity(self.scene,self.record)!=self.spec['identity'] or digest(self.spec)!=self.contract_hash
                or any(m.training for m in self.path.modules()) or any(p.requires_grad for p in self.path.parameters())):
            raise ValueError('Frozen source/prefix/data/contract changed')
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 or not torch.isfinite(p).all() or not p.requires_grad for p in self.head.parameters()):
            raise ValueError('Finite trainable CPU FP32 head required')
        group=self.optimizer.param_groups[0]
        if (len(self.optimizer.param_groups)!=1 or [id(p) for p in group['params']]!=[id(p) for p in self.head.parameters()]
                or self.optimizer.state_dict()['param_groups']!=self.spec['optimizer_config']):
            raise ValueError('Registered head-only Adam changed')

    def _features(self,tile):
        if (type(tile.get('id')) is not int or not 0<=tile['id']<len(self.scene.plan['tiles'])
                or tile!=self.scene.plan['tiles'][tile['id']] or self.record['roles'][tile['id']]['role'] not in ('devtrain','devval')):
            raise ValueError('Only registered development tiles allowed')
        obs=observe_tile(self.scene.cube,self.scene.plan,self.scene.stats,tile);view,_=sample(obs,self.scene.bands,{'mode':'none'})
        x=torch.tensor(np.concatenate((view.observation.spectral,view.observation.coordinates),-1),dtype=torch.float32)
        wave=torch.tensor(view.bands.wavelengths,dtype=torch.float32)
        return self.path.frozen_features(self.scene.dataset,x,wavelengths=wave)

    def evaluate(self):
        self._check();labels=sparse_labels(self.record,self.scene.plan,'validation_support');matrix=np.zeros((self.scene.classes,self.scene.classes),dtype=np.int64);nll=0.;count=0
        with torch.no_grad():
            for tile in self.scene.plan['tiles']:
                if self.record['roles'][tile['id']]['role']!='devval':continue
                target=labels[tile_slice(tile)].ravel();mask=target>0
                if not mask.any():continue
                probs=self.head(self._features(tile)).softmax(-1).numpy()[mask];target=target[mask]-1
                if not np.isfinite(probs).all():raise ValueError('Nonfinite validation')
                prediction=probs.argmax(1);matrix+=np.bincount(target*self.scene.classes+prediction,minlength=self.scene.classes**2).reshape(matrix.shape)
                nll-=float(np.log(np.maximum(probs[np.arange(len(target)),target],1e-8)).sum());count+=len(target)
        if count!=development_counts(self.record)[1]:raise ValueError('Fixed per-class validation budget required')
        return dict(nll=nll/count,oa=float(np.trace(matrix)/count),count=count,confusion=matrix.tolist(),support=matrix.sum(1).tolist(),recall=(np.diag(matrix)/matrix.sum(1)).tolist())

    def step(self):
        self._check()
        if self.completed==self.spec['updates']:raise ValueError('Completed budget; zero further optimizer updates')
        try:
            tile,anchor=self.sampler.draw(self.rng);features=self._features(tile)
            target=torch.tensor(sparse_labels(self.record,self.scene.plan,'train_support')[tile_slice(tile)].ravel())
            mask=(target>0)[:,None].expand(-1,self.scene.classes).float()
            onehot=torch.nn.functional.one_hot((target-1).clamp_min(0),self.scene.classes).float()*mask
            self.optimizer.zero_grad(set_to_none=True);probs=self.head(features).softmax(-1)
            loss=compute_loss(probs,onehot,mask,class_balance=self.spec['class_balance']);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):raise ValueError('Invalid head loss/gradient')
            gradient=sum(float(p.grad.abs().sum()) for p in self.head.parameters());self.optimizer.step();self.completed+=1
            validation=self.evaluate();row=dict(step=self.completed,tile_id=tile['id'],anchor=anchor,loss=float(loss.detach()),head_gradient_l1=gradient,validation=validation)
            self.history.append(row)
            if self.selected_step is None or validation['nll']<self.history[self.selected_step-1]['validation']['nll']:
                self.selected_step=self.completed;self.selected_head=deepcopy(self.head.state_dict())
            return deepcopy(row)
        except Exception:
            self.failed=True
            raise

    def advance(self,to_step):
        if type(to_step) is not int or not self.completed<=to_step<=self.spec['updates']:raise ValueError('Invalid absolute progress target')
        self._check()
        while self.completed<to_step:self.step()
        return deepcopy(self.history)

    def save(self,path):
        self._check()
        packet=dict(format='vca-light-head-run-v1',spec=deepcopy(self.spec),contract_hash=self.contract_hash,completed=self.completed,
            head=deepcopy(self.head.state_dict()),optimizer=deepcopy(self.optimizer.state_dict()),rng=deepcopy(self.rng.bit_generator.state),
            history=deepcopy(self.history),selected_step=self.selected_step,selected_head=deepcopy(self.selected_head))
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                temp=Path(f.name);torch.save(packet,f);f.flush();os.fsync(f.fileno())
            os.link(temp,path)
        finally:
            if temp is not None:temp.unlink(missing_ok=True)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls,source,scene,record,path,*,expected_sha256,expected_contract_hash):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Run file hash mismatch')
        packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        keys={'format','spec','contract_hash','completed','head','optimizer','rng','history','selected_step','selected_head'}
        if not isinstance(packet,dict) or set(packet)!=keys or packet['format']!='vca-light-head-run-v1' or packet['contract_hash']!=expected_contract_hash:
            raise ValueError('Invalid registered run packet')
        spec=packet['spec']
        if not isinstance(spec,dict) or any(k not in spec for k in ('updates','seed','lr','class_balance')):raise ValueError('Invalid training specification')
        run=cls(source,scene,record,**{k:spec[k] for k in ('updates','seed','lr','class_balance')})
        if spec!=run.spec or digest(spec)!=expected_contract_hash:raise ValueError('Run source/architecture/training contract mismatch')
        n=packet['completed'];history=packet['history']
        if type(n) is not int or not 0<=n<=spec['updates'] or not isinstance(history,list) or len(history)!=n:raise ValueError('Invalid completed budget/history')
        rng=np.random.default_rng(spec['seed']);validation_count=development_counts(record)[1]
        for step,row in enumerate(history,1):
            tile,anchor=run.sampler.draw(rng)
            if (set(row)!={'step','tile_id','anchor','loss','head_gradient_l1','validation'} or type(row['step']) is not int
                    or (row['step'],row['tile_id'],row['anchor'])!=(step,tile['id'],anchor)
                    or not _finite(row['loss']) or row['loss']<0 or not _finite(row['head_gradient_l1']) or row['head_gradient_l1']<0):raise ValueError('Invalid training history')
            v=row['validation'];c=np.asarray(v['confusion'])
            if (set(v)!={'nll','oa','count','confusion','support','recall'} or not _finite(v['nll']) or v['nll']<0
                    or c.shape!=(scene.classes,scene.classes) or c.dtype.kind not in 'iu' or (c<0).any() or v['count']!=validation_count
                    or c.sum()!=validation_count or not np.array_equal(c.sum(1),np.full(scene.classes,5))
                    or v['support']!=c.sum(1).tolist() or v['recall']!=(np.diag(c)/c.sum(1)).tolist() or v['oa']!=float(np.trace(c)/validation_count)):
                raise ValueError('Invalid fixed validation history')
        selected=None if n==0 else min(range(n),key=lambda i:history[i]['validation']['nll'])+1
        if packet['selected_step']!=selected or packet['rng']!=rng.bit_generator.state:raise ValueError('Wrong selection or sampler RNG')
        def check_head(state):
            ref=run.head.state_dict()
            if (not isinstance(state,dict) or set(state)!=set(ref) or any(not torch.is_tensor(v) or v.shape!=ref[k].shape or v.dtype!=ref[k].dtype or not torch.isfinite(v).all() for k,v in state.items())):
                raise ValueError('Invalid finite FP32 head state')
        check_head(packet['head'])
        if n:check_head(packet['selected_head'])
        elif packet['selected_head'] is not None:raise ValueError('Selection before first update')
        opt=packet['optimizer'];ref=run.optimizer.state_dict()
        if not isinstance(opt,dict) or set(opt)!=set(ref) or opt['param_groups']!=ref['param_groups']:raise ValueError('Optimizer configuration changed')
        if set(opt['state'])!=(set(ref['param_groups'][0]['params']) if n else set()):raise ValueError('Optimizer state count mismatch')
        for i,p in zip(ref['param_groups'][0]['params'],run.head.parameters()):
            if not n:continue
            slot=opt['state'][i]
            if set(slot)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Invalid Adam state')
            if not torch.is_tensor(slot['step']) or slot['step'].shape!=torch.Size([]) or slot['step'].item()!=n:raise ValueError('Adam update budget mismatch')
            for k in ('exp_avg','exp_avg_sq'):
                v=slot[k]
                if not torch.is_tensor(v) or v.shape!=p.shape or v.dtype!=p.dtype or not torch.isfinite(v).all() or (k=='exp_avg_sq' and (v<0).any()):raise ValueError('Invalid Adam moment')
        run.head.load_state_dict(packet['head']);run.optimizer.load_state_dict(opt);run.rng.bit_generator.state=packet['rng']
        run.completed=n;run.history=deepcopy(history);run.selected_step=selected;run.selected_head=deepcopy(packet['selected_head'])
        if n:
            if run.evaluate()!=history[-1]['validation']:raise ValueError('Final head does not reproduce frozen validation')
            run.head.load_state_dict(packet['selected_head'])
            if run.evaluate()!=history[selected-1]['validation']:raise ValueError('Selected head does not reproduce selected validation')
            run.head.load_state_dict(packet['head'])
        elif state_hash(run.head)!=spec['initial_head_state']:raise ValueError('Zero-update head differs from source initialization')
        return run
