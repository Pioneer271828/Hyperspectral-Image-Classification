"""Bounded controlled signals through the existing two-stage regional graph path.

Topping et al., arXiv:2111.14522v4 section 2 motivates node sensitivity.
This independent diagnostic neither implements SDRF nor proves oversquashing.
"""
from copy import deepcopy
import math
import torch
from model.VCASAHGCN import SharedHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from evaluation.linear_probe import fingerprint


def _state(model):
    return fingerprint({'state':model.state_dict(),'flags':[m.training for m in model.modules()],
                        'grad':[p.grad for p in model.parameters()]})


def _matrix(adjacency,n):
    if (not torch.is_tensor(adjacency) or adjacency.device.type!='cpu' or adjacency.dtype!=torch.float32
            or adjacency.shape!=(n,n) or adjacency.layout not in (torch.strided,torch.sparse_coo)):
        raise ValueError('Bounded CPU float32 dense/COO adjacency required')
    a=adjacency.detach().to_dense() if adjacency.is_sparse else adjacency.detach().clone()
    if (not torch.isfinite(a).all() or (a<0).any() or (a.diag()<=0).any()
            or not torch.allclose(a.sum(1),torch.ones(n),rtol=1e-5,atol=1e-6)):
        raise ValueError('Nonnegative row-normalized graph with positive self loops required')
    return a


def _pairs(value,n,width,label,limit):
    if (not isinstance(value,list) or not 1<=len(value)<=limit or any(not isinstance(p,list) or len(p)!=2
            or any(type(i) is not int for i in p) or not 0<=p[0]<n or not 0<=p[1]<width for p in value)
            or value!=[list(p) for p in sorted(set(map(tuple,value)))]):
        raise ValueError('Sorted unique bounded '+label+' node/channel pairs required')


def _validate(model,x,adjacency,request,dataset):
    if type(model) is not SharedHGCN or any(k in model.__dict__ for k in ('forward_stage1','forward_stage2','forward')):
        raise ValueError('Unintercepted existing SharedHGCN required')
    if (not torch.is_tensor(x) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.layout!=torch.strided
            or x.ndim!=2 or not 1<=x.shape[0]<=256 or not 1<=x.shape[1]<=128 or not torch.isfinite(x).all()
            or x.shape[1]!=model.hgc1.weight.shape[0]):
        raise ValueError('Finite CPU float32 regional inputs <=256x128 required')
    if any(t.device.type!='cpu' or t.is_floating_point() and (t.dtype!=torch.float32 or not torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError('Finite CPU float32 model state required')
    if not (dataset is None or isinstance(dataset,str) and dataset):raise ValueError('Explicit dataset name or None required')
    if not isinstance(request,dict) or set(request)!={'sources','targets','epsilon','cut_nodes'}:
        raise ValueError('Exact controlled-signal request required')
    n=x.shape[0];_pairs(request['sources'],n,x.shape[1],'source',32)
    targets=request['targets']
    if (not isinstance(targets,list) or not 1<=len(targets)<=16
            or any(not isinstance(t,list) or len(t)!=3 or any(type(i) is not int for i in t)
                   or t[0] not in (1,2) or not 0<=t[1]<n
                   or not 0<=t[2]<(model.hgc1.weight.shape[1] if t[0]==1 else model.hgc2.weight.shape[1]) for t in targets)
            or targets!=[list(t) for t in sorted(set(map(tuple,targets)))]):
        raise ValueError('Sorted unique stage/node/channel targets required')
    cut=request['cut_nodes']
    if not isinstance(cut,list) or any(type(i) is not int or not 0<=i<n for i in cut) or cut!=sorted(set(cut)):
        raise ValueError('Sorted unique in-range intermediate cut nodes required')
    eps=request['epsilon']
    if type(eps) not in (int,float) or not math.isfinite(eps) or not 1e-5<=eps<=.1:
        raise ValueError('Explicit bounded finite-difference epsilon required')
    return _matrix(adjacency,n)


def _distances(a,source):
    # A[receiver,sender]: do not silently treat directed supports as undirected.
    n=a.shape[0];d=[None]*n;d[source]=0;queue=[source]
    for j in queue:
        for i in torch.where(a[:,j]>0)[0].tolist():
            if d[i] is None:d[i]=d[j]+1;queue.append(i)
    return d


def _identity(model,x,a,request,dataset):
    return {'version':1,'model_state':_state(model),'input_hash':tensor_hash(x),
            'adjacency_hash':tensor_hash(a),'request':deepcopy(request),'dataset':dataset}


class SignalPlan:
    def __init__(self,value,*,expected_hash):
        if (not isinstance(value,dict) or set(value)!={'version','model_state','input_hash','adjacency_hash','request','dataset','plan_hash'}
                or value['version']!=1 or value['plan_hash']!=expected_hash
                or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash):
            raise ValueError('Frozen controlled-signal identity differs')
        self._value=deepcopy(value);self.plan_hash=expected_hash
    def to_dict(self):return deepcopy(self._value)
    @classmethod
    def freeze(cls,model,x,adjacency,*,request,dataset=None):
        a=_validate(model,x,adjacency,request,dataset);v=_identity(model,x,a,request,dataset)
        v['plan_hash']=digest(v);return cls(v,expected_hash=v['plan_hash'])


def review_signals(model,x,adjacency,plan,*,expected_plan_hash):
    if not isinstance(plan,SignalPlan):raise ValueError('Frozen SignalPlan required')
    v=SignalPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict();request=v['request'];dataset=v['dataset']
    a=_validate(model,x,adjacency,request,dataset)
    if _identity(model,x,a,request,dataset)!={k:z for k,z in v.items() if k!='plan_hash'}:
        raise ValueError('Signal model/input/graph/request changed')
    before=_state(model);input_hash=tensor_hash(x);original_graph_hash=tensor_hash(adjacency)
    private=deepcopy(model).eval();private.clear_graph_cache()
    for p in private.parameters():p.requires_grad_(False)
    distances={s[0]:_distances(a,s[0]) for s in request['sources']};rows=[];calls=0
    def forward(value,cut):
        nonlocal calls
        calls+=1;first=private.forward_stage1(value,a,dataset)
        if cut:
            first=first.clone();first[request['cut_nodes']]=0
        second=private.forward_stage2(first,a,dataset)
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():raise ValueError('Nonfinite graph response')
        return first,second
    with torch.random.fork_rng(devices=[]),torch.enable_grad():
        for arm in ('original','cut_intermediate'):
            leaf=x.detach().clone().requires_grad_(True);outputs=forward(leaf,arm!='original')
            for target in request['targets']:
                stage,node,channel=target;score=outputs[stage-1][node,channel]
                grad=torch.autograd.grad(score,leaf,retain_graph=True)[0]
                if not torch.isfinite(grad).all():raise ValueError('Nonfinite graph Jacobian')
                responses=[]
                for source in request['sources']:
                    i,c=source;plus=x.detach().clone();minus=x.detach().clone();eps=request['epsilon']
                    plus[i,c]+=eps;minus[i,c]-=eps
                    if plus[i,c]==x[i,c] or minus[i,c]==x[i,c]:raise ValueError('Signal perturbation rounded away')
                    with torch.no_grad():
                        high=forward(plus,arm!='original')[stage-1][node,channel]
                        low=forward(minus,arm!='original')[stage-1][node,channel]
                    # Use represented perturbation spacing; retain asymmetry diagnostics.
                    fd=float((high-low)/(plus[i,c]-minus[i,c]));jac=float(grad[i,c]);distance=distances[i][node]
                    responses.append({'source':source,'hop_distance':distance,'within_stage_reach':distance is not None and distance<=stage,
                        'jacobian':jac,'finite_difference':fd,'absolute_difference':abs(jac-fd),
                        'positive_response':float(high-score.detach()),'negative_response':float(low-score.detach()),
                        'positive_input_delta':float(plus[i,c]-x[i,c]),'negative_input_delta':float(minus[i,c]-x[i,c])})
                rows.append({'arm':arm,'target':target,'baseline':float(score.detach()),'responses':responses})
    if before!=_state(model) or tensor_hash(x)!=input_hash or tensor_hash(adjacency)!=original_graph_hash:
        raise RuntimeError('Original model/input/graph mutated')
    return {'plan_hash':expected_plan_hash,'rows':rows,'graph_path_calls':calls,'optimizer_updates':0,'query_labels_accessed':False,
            'parameter_gradients_accumulated':False,'original_inputs_unchanged':True,
            'scope':'Caller-bound regional graph path only; no CNN, Q derivative, interaction, classification or learned recovery task',
            'limits':['Direct node sensitivity, not proof of oversquashing','Intermediate zeroing can shift representations and ReLU regions',
                      'Finite differences are diagnostics, not a relaxed numerical acceptance gate','Topology distances refer to original directed support']}
