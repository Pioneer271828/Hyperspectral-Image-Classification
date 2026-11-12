"""Authenticated full-network local signal responses on development blocks only."""
from copy import deepcopy
import math
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.graph_route_context import _context,route_state_hash
from loadData.spatial_tiles import observe_tile
from .block_input_diagnostics import DEFAULT_SPECTRAL_CAP,XUZHOU_SPECTRAL_CAP,_value,_probe,input_gradients


def _bind(run,*,tile_id,target_ids,class_id,sources,epsilon,spectral_cap=DEFAULT_SPECTRAL_CAP):
    context=_context(run.scene,run.record,tile_id)  # Query rejected before prediction.
    obs=observe_tile(run.scene.cube,run.scene.plan,run.scene.stats,run.scene.plan['tiles'][tile_id])
    ids=obs.global_ids.ravel().tolist();bands=obs.spectral.shape[-1]
    if (not isinstance(sources,list) or not 1<=len(sources)<=16
            or any(not isinstance(s,list) or len(s)!=2 or any(type(i) is not int for i in s)
                   or s[0] not in ids or not 0<=s[1]<bands for s in sources)
            or sources!=[list(s) for s in sorted(set(map(tuple,sources)))]):
        raise ValueError('One to16 sorted unique in-block global-pixel/physical-band sources required')
    if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or not 1e-5<=epsilon<=.1:
        raise ValueError('Finite normalized-spectrum epsilon in [1e-5,.1] required')
    if not isinstance(target_ids,list) or not 1<=len(target_ids)<=16:
        raise ValueError('One to16 explicit target pixels required')
    if spectral_cap!=DEFAULT_SPECTRAL_CAP and (spectral_cap!=XUZHOU_SPECTRAL_CAP or run.scene.dataset!='xuzhou'):
        raise ValueError('Only the explicit Xuzhou spectral bound is supported')
    base,model,obs,x=_value(run,tile_id,target_ids,class_id,[{'kind':'identity'}],spectral_cap=spectral_cap)
    if base['context']!=context:raise ValueError('Context changed while binding signal')
    v={'version':1,'base':base,'sources':deepcopy(sources),'epsilon':epsilon,
       'coordinate_policy':'fixed coordinates; physical-band perturbations in training-minmax units, unclipped',
       'route_policy':'native input-dependent Q/A/routing; report changes, do not freeze or replace supports'}
    if spectral_cap!=DEFAULT_SPECTRAL_CAP:
        v.update(version=2,spectral_cap=spectral_cap)
    v['plan_hash']=digest(v);return v,model,obs,x


class SelectedSignalPlan:
    def __init__(self,value,*,expected_hash):
        fields={'version','base','sources','epsilon','coordinate_policy','route_policy','plan_hash'}
        if (isinstance(value,dict) and value.get('version')==2 and type(value.get('spectral_cap')) is int
                and value['spectral_cap']==XUZHOU_SPECTRAL_CAP):
            fields.add('spectral_cap')
        elif not isinstance(value,dict) or value.get('version')!=1:
            raise ValueError('Selected signal plan identity differs')
        if (set(value)!=fields or value['plan_hash']!=expected_hash
                or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash):
            raise ValueError('Selected signal plan identity differs')
        self._value=deepcopy(value);self.plan_hash=expected_hash
    def to_dict(self):return deepcopy(self._value)
    @classmethod
    def freeze(cls,run,**kwargs):
        v,*_=_bind(run,**kwargs);return cls(v,expected_hash=v['plan_hash'])


def review_selected_signals(run,plan,*,expected_plan_hash):
    if not isinstance(plan,SelectedSignalPlan):raise ValueError('Frozen SelectedSignalPlan required')
    v=SelectedSignalPlan(plan.to_dict(),expected_hash=expected_plan_hash).to_dict();base=v['base']
    cap=v.get('spectral_cap',DEFAULT_SPECTRAL_CAP)
    fresh,model,obs,x=_bind(run,tile_id=base['context']['tile_id'],target_ids=base['target_ids'],
        class_id=base['class_id'],sources=v['sources'],epsilon=v['epsilon'],spectral_cap=cap)
    if fresh!=v:raise ValueError('Selected model/data/role/source/request changed')
    before=route_state_hash(model);ids=obs.global_ids.ravel().tolist();target_indices=[ids.index(i) for i in base['target_ids']]
    tid=base['context']['tile_id'];captured=[]
    def forward(value):
        probabilities,support=_probe(run,model,tid,value,base['normalized_spectrum_hash'])
        captured.append(support);return probabilities
    baseline=input_gradients(forward,x,pixel_indices=target_indices,class_id=base['class_id'],spectral_cap=cap);support=captured[0]
    rows=[];class_index=base['class_id']-1
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        direct=run._predict(model,run.scene.plan['tiles'][tid],0)
        if any(not torch.equal(a,b) for a,b in zip(direct[:3],baseline['probabilities'])):
            raise ValueError('Gradient tracing changed native baseline probabilities')
        for global_id,band in v['sources']:
            local=ids.index(global_id);y,col=divmod(local,x.shape[1]);out=[];deltas=[];supports=[];hashes=[]
            for sign in (1,-1):
                changed=x.detach().clone();changed[y,col,band]+=sign*v['epsilon']
                delta=float(changed[y,col,band]-x[y,col,band])
                if delta==0 or not math.isfinite(delta):raise ValueError('Perturbation rounded away/nonfinite')
                outputs,current=_probe(run,model,tid,changed,base['normalized_spectrum_hash'])
                if any(not torch.isfinite(p).all() for p in outputs[:3]):raise ValueError('Nonfinite signal output')
                out.append([float(p[target_indices,class_index].mean()) for p in outputs[:3]])
                deltas.append(delta);supports.append(current);hashes.append(tensor_hash(changed))
            gradients=baseline['gradients'][:,y,col,band].tolist()
            fd=[(plus-minus)/(deltas[0]-deltas[1]) for plus,minus in zip(*out)]
            rows.append({'source':[global_id,band],'local_pixel':[y,col],'gradient':gradients,'finite_difference':fd,
                'absolute_difference':[abs(a-b) for a,b in zip(gradients,fd)],'input_deltas':deltas,'spectrum_hashes':hashes,
                'probability_scores':out,'score_deltas':[[a-b for a,b in zip(values,baseline['scores'])] for values in out],
                'supports':supports,'topology_changed':[current['edges']!=support['edges'] for current in supports],
                'q_changed':[current['q_hash']!=support['q_hash'] for current in supports],
                'graph_values_changed':[current['final_a_hash']!=support['final_a_hash'] for current in supports]})
    if before!=route_state_hash(model):raise ValueError('Private selected model changed registered state')
    run._check()
    return {'plan_hash':expected_plan_hash,'context':base['context'],'selected_state_hash':before,
            'baseline_scores':baseline['scores'],'baseline_probability_hashes':[tensor_hash(p) for p in baseline['probabilities']],
            'baseline_support':support,'rows':rows,'whole_model_forwards':2+2*len(rows),'optimizer_updates':0,'query_labels_accessed':False,
            'target_labels_used':False,'scope':'native full-path local spectral sensitivity; selected weights and development context',
            'limits':['Reused selection validation, not independent confirmation','Nonsmooth ReLU/top-k and FP32 can differ from finite differences',
                      'No derivative threshold certifies long-range learning or accuracy benefit','Perturbations are unclipped normalized values, not physical missing-band simulation']}
