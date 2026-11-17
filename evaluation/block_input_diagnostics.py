"""Bounded frozen-block spectral sensitivity and within-block interventions."""
from copy import deepcopy
import numpy as np
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from loadData.spatial_tiles import observe_tile
from transfer.graph_route_context import _context,route_state_hash
from evaluation.graph_edge_intervention import _selected


DEFAULT_SPECTRAL_CAP = 65_536
XUZHOU_SPECTRAL_CAP = 28 * 28 * 436


def _spectral(value, *, spectral_cap=DEFAULT_SPECTRAL_CAP):
    if type(spectral_cap) is not int or spectral_cap not in (DEFAULT_SPECTRAL_CAP, XUZHOU_SPECTRAL_CAP):
        raise ValueError('Expected a registered spectral block bound')
    if (not torch.is_tensor(value) or value.device.type!='cpu' or value.dtype not in (torch.float32,torch.float64)
            or value.ndim!=3 or min(value.shape)<=0 or value.numel()>spectral_cap or not torch.isfinite(value).all()):
        raise ValueError(f'Finite bounded CPU floating H,W,B block required (<= {spectral_cap} elements)')
    if spectral_cap == XUZHOU_SPECTRAL_CAP and (value.shape[2]!=436 or max(value.shape[:2])>28 or value.dtype!=torch.float32):
        raise ValueError('The registered Xuzhou bound requires FP32, 436 bands, and sides at most 28 pixels')


def _box(box,shape):
    if not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(x) is not int for x in box):raise ValueError('Integer local rectangle required')
    y0,y1,x0,x1=box
    if not (0<=y0<y1<=shape[0] and 0<=x0<x1<=shape[1]):raise ValueError('Rectangle outside block')
    return slice(y0,y1),slice(x0,x1)


def intervene_spectrum(original,spec,*,spectral_cap=DEFAULT_SPECTRAL_CAP):
    _spectral(original,spectral_cap=spectral_cap)
    if not isinstance(spec,dict):raise ValueError('Explicit intervention dictionary required')
    kind=spec.get('kind')
    keys={'kind'} if kind=='identity' else {'kind','box'} if kind=='occlusion_min' else {'kind','box','donor_box'} if kind=='replace' else set()
    if not keys or set(spec)!=keys:raise ValueError('Unknown intervention/schema')
    value=original.clone()
    if kind=='identity':return value
    target=_box(spec['box'],original.shape)
    if kind=='occlusion_min':value[target]=0
    else:
        donor=_box(spec['donor_box'],original.shape)
        if value[target].shape!=original[donor].shape:raise ValueError('Replacement rectangles must have equal shape')
        value[target]=original[donor].clone()
    return value


def input_gradients(forward,value,*,pixel_indices,class_id,spectral_cap=DEFAULT_SPECTRAL_CAP):
    """Three scalar probability targets, not a full pixel-to-pixel Jacobian."""
    _spectral(value,spectral_cap=spectral_cap);n=value.shape[0]*value.shape[1]
    if (not isinstance(pixel_indices,(list,tuple)) or not pixel_indices
            or any(type(i) is not int or not 0<=i<n for i in pixel_indices)
            or list(pixel_indices)!=sorted(set(pixel_indices)) or type(class_id) is not int or class_id<1):
        raise ValueError('Sorted explicit local target pixels and one-based class required')
    with torch.random.fork_rng(devices=[]),torch.enable_grad():
        leaf=value.detach().clone().requires_grad_(True);outputs=forward(leaf)
        if (not isinstance(outputs,(list,tuple)) or len(outputs)<3
                or any(not torch.is_tensor(p) or p.device.type!='cpu' or p.dtype!=value.dtype or p.ndim!=2 or p.shape[0]!=n
                       or p.shape[1]<class_id or not torch.isfinite(p).all() or (p<0).any() or (p>1).any()
                       or not torch.allclose(p.sum(1),torch.ones(n,dtype=p.dtype),rtol=1e-5,atol=1e-6) for p in outputs[:3])
                or len({p.shape[1] for p in outputs[:3]})!=1):
            raise ValueError('Three finite CPU probability arrays required')
        scores=[];gradients=[];connected=[]
        for head,p in enumerate(outputs[:3]):
            score=p[list(pixel_indices),class_id-1].mean()
            grad=torch.autograd.grad(score,leaf,retain_graph=head<2,allow_unused=True)[0] if score.requires_grad else None
            connected.append(grad is not None);grad=torch.zeros_like(leaf) if grad is None else grad
            if not torch.isfinite(grad).all():raise ValueError('Nonfinite input gradient')
            scores.append(float(score.detach()));gradients.append(grad.detach())
    return dict(scores=scores,gradients=torch.stack(gradients),connected=connected,
                probabilities=tuple(p.detach() for p in outputs[:3]))


def _value(run,tile_id,target_ids,class_id,interventions,*,spectral_cap=DEFAULT_SPECTRAL_CAP):
    model=_selected(run);context=_context(run.scene,run.record,tile_id)
    obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][tile_id])
    spectrum=torch.tensor(obs.spectral.copy());_spectral(spectrum,spectral_cap=spectral_cap)
    if (not isinstance(target_ids,(list,tuple)) or not target_ids or any(type(i) is not int for i in target_ids)
            or list(target_ids)!=sorted(set(target_ids)) or not set(target_ids)<=set(obs.global_ids.ravel().tolist())
            or type(class_id) is not int or not 1<=class_id<=run.scene.classes):raise ValueError('Explicit in-block global targets and class required')
    if not isinstance(interventions,(list,tuple)) or not 1<=len(interventions)<=32:raise ValueError('One to 32 predeclared interventions required')
    rows=[dict(spec=deepcopy(spec),spectrum_hash=tensor_hash(intervene_spectrum(spectrum,spec,spectral_cap=spectral_cap))) for spec in interventions]
    value=dict(version=1,contract_hash=run.contract_hash,selected_step=run.selected_step,selected_state_hash=route_state_hash(model),
        context=context,normalized_spectrum_hash=tensor_hash(spectrum),coordinates_hash=tensor_hash(torch.tensor(obs.coordinates.copy())),
        target_ids=list(target_ids),class_id=class_id,interventions=rows,
        units='derivative of mean class probability w.r.t. training-minmax normalized spectral input; coordinates fixed',
        occlusion_baseline='zero normalized spectrum means training minimum, not mean or physical missing bands',route_probe_step=0)
    value['plan_hash']=digest(value);return value,model,obs,spectrum


class BlockInputPlan:
    def __init__(self,value,*,expected_hash):
        if not isinstance(value,dict) or value.get('plan_hash')!=expected_hash or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash:
            raise ValueError('Frozen input-probe plan identity mismatch')
        self._value=deepcopy(value);self.plan_hash=expected_hash
    def to_dict(self):return deepcopy(self._value)
    @classmethod
    def freeze(cls,run,*,tile_id,target_ids,class_id,interventions):
        value,*_=_value(run,tile_id,target_ids,class_id,interventions)
        return cls(value,expected_hash=value['plan_hash'])


def _probe(run,model,tile_id,spectrum,expected_spectrum_hash):
    # Original input/context is authenticated before replacing spectral values on a private copy.
    targets=[(model,'forward'),(model.backbone.learnable_graph,'build_graph'),(model.backbone.graph_router,'forward')]
    if any(name in owner.__dict__ for owner,name in targets):raise ValueError('Model already intercepted')
    original=model.forward;build=model.backbone.learnable_graph.build_graph;route=model.backbone.graph_router.forward
    captured={};calls=[]
    def forward(dataset,value,*args,**kwargs):
        b=spectrum.shape[-1]
        if value.shape[:2]!=spectrum.shape[:2] or value.shape[-1]!=b+2 or tensor_hash(value[...,:b])!=expected_spectrum_hash:
            raise ValueError('Original normalized input differs')
        calls.append(1)
        # A fixed-bank validation forward ordinarily uses no_grad. Only this
        # private diagnostic leaf needs a graph; no parameter .grad is written.
        with torch.set_grad_enabled(spectrum.requires_grad):
            return original(dataset,torch.cat((spectrum,value[...,b:]),dim=-1),*args,**kwargs)
    def graph(*args,**kwargs):
        state=build(*args,**kwargs);captured['q_hash']=tensor_hash(state.Q);captured['a0_hash']=tensor_hash(state.A0);return state
    def routing(*args,**kwargs):
        a,info=route(*args,**kwargs);ix=a.coalesce().indices();captured.update(final_a_hash=tensor_hash(a),nodes=a.shape[0],edges=(ix[0]*a.shape[0]+ix[1]).tolist());return a,info
    model.forward=forward;model.backbone.learnable_graph.build_graph=graph;model.backbone.graph_router.forward=routing
    try:outputs=run._predict(model,run.scene.plan['tiles'][tile_id],0)
    finally:
        for owner,name in reversed(targets):delattr(owner,name)
    if len(calls)!=1 or set(captured)!={'q_hash','a0_hash','final_a_hash','nodes','edges'}:raise ValueError('Unexpected probe call structure')
    return outputs,captured


def review_block(run,plan,*,expected_plan_hash):
    if not isinstance(plan,BlockInputPlan):raise ValueError('Frozen BlockInputPlan required')
    value=BlockInputPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict()
    fresh,model,obs,spectrum=_value(run,value['context']['tile_id'],value['target_ids'],value['class_id'],[r['spec'] for r in value['interventions']])
    if fresh!=value:raise ValueError('Plan model/context/input/targets differ')
    before=route_state_hash(model);tid=value['context']['tile_id'];indices=[obs.global_ids.ravel().tolist().index(i) for i in value['target_ids']];captured=[]
    def forward(x):
        result,support=_probe(run,model,tid,x,value['normalized_spectrum_hash']);captured.append(support);return result
    baseline=input_gradients(forward,spectrum,pixel_indices=indices,class_id=value['class_id']);base_support=captured[0]
    rows=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        direct=run._predict(model,run.scene.plan['tiles'][tid],0)
        if any(not torch.equal(a,b) for a,b in zip(direct[:3],baseline['probabilities'])):raise ValueError('Input tracing changed original prediction')
        for row in value['interventions']:
            x=intervene_spectrum(spectrum,row['spec']);outputs,support=_probe(run,model,tid,x,value['normalized_spectrum_hash'])
            scores=[float(p[indices,value['class_id']-1].mean()) for p in outputs[:3]]
            if row['spectrum_hash']==value['normalized_spectrum_hash'] and any(not torch.equal(a,b) for a,b in zip(outputs[:3],baseline['probabilities'])):
                raise ValueError('Identity intervention changed probabilities')
            rows.append(dict(spec=row['spec'],spectrum_hash=tensor_hash(x),scores=scores,
                delta=[a-b for a,b in zip(scores,baseline['scores'])],probability_hashes=[tensor_hash(p) for p in outputs[:3]],
                support=support,added_edges=sorted(set(support['edges'])-set(base_support['edges'])),removed_edges=sorted(set(base_support['edges'])-set(support['edges']))))
    grad=baseline['gradients'];pixels=torch.linalg.vector_norm(grad,dim=-1);bands=torch.linalg.vector_norm(grad,dim=(1,2))
    if before!=route_state_hash(model):raise ValueError('Input probe changed selected state')
    run._check()
    return dict(plan_hash=expected_plan_hash,selected_state_hash=before,baseline_scores=baseline['scores'],
        baseline_probability_hashes=[tensor_hash(p) for p in baseline['probabilities']],baseline_support=base_support,
        connected=baseline['connected'],gradient_hash=tensor_hash(grad),gradients=grad.tolist(),pixel_norms=pixels.tolist(),band_norms=bands.tolist(),
        interventions=rows,optimizer_updates=0,parameter_gradient_accumulation=False,query=0,original_ground_truth_read=False,target_labels_used=False,
        limitations=['piecewise local derivative under discrete routing','gradient saturation and intervention distribution shift','selection weights fixed, no receptive-field benefit or causal accuracy conclusion'])
