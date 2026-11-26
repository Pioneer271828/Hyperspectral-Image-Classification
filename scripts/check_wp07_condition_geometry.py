"""Missing wide-weight isolation and GPU initializer RNG checks; stop before classifier heads."""
import argparse
import json
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from joint_training import _rng_state
from research_runtime import code_snapshot,require_frozen_code,tensor_hash
from model.research_config import digest
from transfer import wp07_observed_conditions as cond
from transfer import wp07_graph_range as graph
from transfer import registered_development as dev
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_scale_campaign import read,packet
from transfer.xuzhou_multiscale_audit import _exact
from scripts import run_native_released_points as worker
from scripts.run_wp07_b_pair import observation_only_resources
from scripts.probe_wp07_observed_conditions import Provider,pin


def main():
    graph.require_checks();torch.set_num_threads(1);p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','native','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--expected-commit',required=True);a=p.parse_args()
    code=code_snapshot();require_frozen_code(code);graph.require(code['commit']==a.expected_commit,'Wrong clean geometry checker')
    graph.require(os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.root in a.out.parents and not a.out.exists(),'Assigned GPU and fresh external output required')
    a.out.mkdir();plan=read(pin(a.native/'pair-v2/plan.json'),a.root);case=next(c for c in plan['cases'] if c['id']=='xuzhou/512/2/content')
    graph.require(case['gpu']==a.gpu,'Only the original GPU is qualified')
    dest=Path(case['destination']);spec=read(pin(dest/'registration.json'),a.root);checkpoint=read(pin(dest/'completed.json'),a.root)['checkpoint'];state=packet(checkpoint,a.root)
    with observation_only_resources(),worker.arithmetic(a.root,a.gpu) as execution,torch.no_grad():
        graph.require(execution.numeric==spec['numeric'],'Original arithmetic identity differs')
        provider=Provider(a.root,read(case['preflight'],a.root),512)
        model=MultiscaleTrial(cnn_multiscale='content',graph_multiscale='content').initialize(provider.baseline(512),spec['module_seed']).to(execution.device)
        model.load_state_dict(state['selected']['state'],strict=True);model.backbone.adapter.cache_attention=False
        rng=_rng_state(execution.device);cond.install_encoders(model,spec['module_seed']+41017);model.eval()
        graph.require(_exact(rng,_rng_state(execution.device)),'Adapter insertion changed CPU or CUDA RNG')
        before={n:tensor_hash(v) for n,v in model.state_dict().items()};rows=[]
        for event in spec['validation']:
            actual,labels,targets=provider.materialize('devval',event,spec['producer']);provider.validate_observation(spec,actual,labels,targets,event)
            reference=None
            for intervention in ('matched','swap_band','unknown_wavelength','missing_gsd'):
                binding=cond.bind_condition(actual,provider.master,provider.catalog,intervention)
                entry=graph.capture_geometry(model,actual,spec['producer'],'devval',spec['registered_graph_scale'],condition=binding)
                values={k:graph.sparse_digest(entry[k]) for k in ('q','near','wide')}
                if reference is None:reference=values
                graph.require(values==reference,'A condition changed Q or either candidate adjacency weights')
                near,wide=entry['near'].coalesce(),entry['wide'].coalesce()
                rows.append(dict(view_id=digest((provider.catalog,event['tile'],event['band']))[:20],band=event['band'],condition=intervention,
                    q=values['q'],near=values['near'],wide=values['wide'],
                    near_and_wide_indices_exact=torch.equal(near.indices(),wide.indices()),
                    near_and_wide_weights_exact=torch.equal(near.values(),wide.values())))
        graph.require(len(rows)==40 and all(tensor_hash(model.state_dict()[n])==h for n,h in before.items())
            and _exact(rng,_rng_state(execution.device)),'Geometry checks changed state or missed a view')
    result=dict(status='passed',executor=code['commit'],source_hash=code['source_hash'],checkpoint=checkpoint,
        original_checkpoint_producer=spec['producer'],views=10,condition_geometry_captures=40,rows=rows,
        CPU_and_CUDA_rng_unchanged_by_adapter_insertion=True,CPU_and_CUDA_rng_unchanged_by_capture=True,
        both_candidate_indices_and_values_fixed_across_conditions=True,source_and_new_model_tensors_unchanged=True,
        classifier_heads_evaluated=0,optimizer_updates=0,raw_GT_opened=False,query_accessed=False,
        limitation='The historical factor2 near/wide degeneracy remains; no graph-range efficacy inference')
    dev._json_new(a.out/'receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
