"""Zero-update condition-consumption probes of existing released-development checkpoints."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from model.research_config import digest
from research_runtime import code_snapshot,require_frozen_code,tensor_hash
from transfer import wp07_observed_conditions as cond
from transfer import wp07_fine16_training as frozen
from transfer import wp07_graph_range as graph
from transfer import native_released_campaign as campaign
from transfer import registered_development as dev
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_scale_training import evaluate_validation
from transfer.native_scale_observation import forward as native_forward
from transfer.native_scale_campaign import read,packet
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact
from scripts import run_native_released_points as worker
from scripts.run_wp07_b_pair import observation_only_resources
from scripts.run_wp07_condition_integration import table


def pin(p):return dev._pin(Path(p).resolve(strict=True))


class Provider(frozen.BankProvider):
    def __init__(self,root,receipt,seed):
        super().__init__(root,receipt,seed)
        self.master=len(self._bank.release.bands.band_ids)
        self.catalog=digest({'bank':receipt['bank'],'processed_parent_band_count':self.master})
        self.override='matched'

    def selection_condition(self,actual,spec):
        return cond.bind_condition(actual,self.master,self.catalog,self.override)


def main():
    graph.require_checks();torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','core','native','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--expected-commit',required=True)
    a=p.parse_args();code=code_snapshot();require_frozen_code(code)
    graph.require(code['commit']==a.expected_commit and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu,'Wrong adapter/GPU')
    graph.require(a.root in a.out.parents and not a.out.exists(),'Fresh external probe directory required')
    a.out.mkdir();private=a.out/'_private';private.mkdir()
    cases=[]
    for scene in ('paviaU','xuzhou'):
        planpath=a.core/scene/'pair/plan.json';plan=read(pin(planpath),a.root)
        for c in plan['cases']:
            if c['arm']=='content':cases.append(dict(scene=scene,seed=c['seed'],regime='fine16_graph',case=c,plan=plan,plan_pin=pin(planpath)))
    native_plan=read(pin(a.native/'pair-v2/plan.json'),a.root)
    old=next(c for c in native_plan['cases'] if c['id']=='xuzhou/512/2/content')
    cases.append(dict(scene='xuzhou',seed=512,regime='native_factor2_both',case=old,plan=native_plan,plan_pin=pin(a.native/'pair-v2/plan.json')))
    registration=dev._json_new(a.out/'registration.json',dict(executor=code['commit'],source_hash=code['source_hash'],
        program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models=7,physical_gpu=a.gpu,
        primary_controls=['matched','swap_band','unknown_wavelength','missing_gsd'],neutral_insert=True,
        donor_scope='actual same-scene processed full/even condition only; content and VCA input held fixed',
        condition_parameters='new fixed-seed initialization; never trained on these index descriptors',
        interpretation='consumption/isolation pressure test, not trained-condition efficacy',optimizer_updates=0,
        plans=[r['plan_pin'] for r in cases],new_labels=0,query_authorized=False,raw_GT_authorized=False))
    endpoints=[];classes=[];public_points=[];view_rows=[];costs=[];results=[]
    started=time.perf_counter()
    with observation_only_resources(),worker.arithmetic(a.root,a.gpu) as execution,torch.no_grad():
        for item in cases:
            c=item['case'];scene=item['scene'];seed=item['seed'];regime=item['regime'];dest=Path(c['destination'])
            graph.require(c['gpu']==a.gpu,'Only checkpoints originally executed on assigned GPU are usable')
            spec=read(pin(dest/'registration.json'),a.root);statepin=read(pin(dest/'completed.json'),a.root)['checkpoint'];state=packet(statepin,a.root)
            graph.require(state['selected']['step']==spec['steps'] and _exact(state['selected']['state'],state['model']),'Only previously checked selected=last endpoints')
            preflight=item['plan']['preflight'] if regime=='fine16_graph' else c['preflight'];receipt=read(preflight,a.root)
            provider=Provider(a.root,receipt,seed)
            if regime=='fine16_graph':
                entries=packet(item['plan']['cache'],a.root)['entries'];engine=frozen.RangeRun(spec,provider,entries,spec['producer'],execution);engine.restore(state,spec['steps'])
                model=engine.model;native=False
                historical=read(pin(dest/'independent.json'),a.root)['endpoints']['selected']['original_point_predictions']
            else:
                model=MultiscaleTrial(cnn_multiscale='content',graph_multiscale='content').initialize(provider.baseline(seed),spec['module_seed']).to(execution.device)
                graph.require(set(model.state_dict())==set(state['model']),'Original candidate checkpoint architecture differs')
                model.load_state_dict(state['model'],strict=True);model.backbone.adapter.cache_attention=False;native=True
                historical=read(pin(a.native/'independent-gpu3/xuzhou-512-2-content.json'),a.root)['endpoints']['selected']['points']
            graph.require(execution.numeric==spec['numeric'],'Original arithmetic/device identity differs')
            model.eval();old_state={n:tensor_hash(v) for n,v in model.state_dict().items()};old_model_state=state_hash(model)
            initial_model_parameters=sum(p.numel() for p in model.parameters())
            cond.install_encoders(model,spec['module_seed']+41017)
            model.eval()
            graph.require(all(tensor_hash(model.state_dict()[n])==h for n,h in old_state.items()),'Insertion changed an old tensor')
            parameters=sum(p.numel() for p in model.parameters());adapter_initial=state_hash(model)
            # Adding inference-only tensors does not extend the optimizer or authorize continuation.
            before={n:tensor_hash(v) for n,v in model.state_dict().items()}
            original_rng=__import__('joint_training',fromlist=['_rng_state'])._rng_state(execution.device)
            records=[]
            def materialize(event):
                if native:
                    actual,labels,targets=provider.materialize('devval',event,spec['producer'])
                    provider.validate_observation(spec,actual,labels,targets,event);return actual,labels,targets
                return engine._materialize('devval',event)

            def infer(actual,role,intervention):
                if native:
                    condition=provider.selection_condition(actual,spec)
                    output,trace=native_forward(model,actual,spec['producer'],role,condition=condition,execution=execution,graph_scale=spec['registered_graph_scale'])
                    return output,trace,condition
                return engine._forward(actual,role,intervention)

            def evaluate(scope,control,neutral=False):
                points=[];provider.override=control
                policy={'reference':('content','content'),'cnn':('joint','content'),'graph':('content','joint'),'both':('joint','joint')}[scope]
                def observe(event,actual,output,targets):
                    y0,_,x0,_=actual.lattice.box;width=actual.lattice.scene_shape[1];f=actual.lattice.factor;gw=actual.lattice.shape[1]
                    for head,values in zip(frozen.HEADS,output[:3],strict=True):
                        probs=values.detach().cpu().numpy()
                        for ordinal,(pixel,label) in enumerate(targets.points):
                            y,x=divmod(pixel,width);cell=((y-y0)//f)*gw+(x-x0)//f
                            points.append(dict(tile=event['tile'],released_ordinal=ordinal,head=head,band=event['band'],class_id=label,probabilities=probs[cell].astype(float).tolist()))
                with cond.selection_policy(model,*policy):
                    measured=evaluate_validation(model,spec['validation'],spec['steps'],materialize,infer,released=provider,detailed=True,observe=observe)
                graph.require(len(points)==270,'Original 45-point/two-band/three-head denominator differs')
                return dict(scope=scope,control=control,neutral=neutral,metrics=campaign.scoring(measured,9),validation=measured,points=points)

            reference=evaluate('reference','missing');key=lambda r:(r['tile'],r['released_ordinal'],r['head'],r['band'])
            oldmap={key(r):r for r in historical};refmap={key(r):r for r in reference['points']}
            graph.require(set(refmap)==set(oldmap),'Original released point identity differs')
            graph.require(all(refmap[k]['probabilities']==oldmap[k]['probabilities'] and refmap[k]['class_id']==oldmap[k]['class_id'] for k in refmap),'Missing-condition graft changed historical point probabilities')
            records.append(reference)
            scopes=('graph',) if regime=='fine16_graph' else ('cnn','graph','both')
            for scope in scopes:
                for control in ('matched','swap_band','unknown_wavelength','missing_gsd'):
                    measured=evaluate(scope,control)
                    graph.require([(r['observation'],r['released'],r['graph']['q'],r['graph']['a0']) for r in measured['validation']['rows']]==
                        [(r['observation'],r['released'],r['graph']['q'],r['graph']['a0']) for r in reference['validation']['rows']],
                        'Condition changed content/VCA observation/released points/Q/near-A')
                    for r,ref in zip(measured['validation']['rows'],reference['validation']['rows'],strict=True):
                        graph.require(r['graph']['range_nnz']==ref['graph']['range_nnz'],'Graph candidate pool changed')
                        if scope=='graph':graph.require(r['heads']['cnn']==ref['heads']['cnn'],'Graph condition affected CNN branch')
                        if scope=='cnn':graph.require(r['heads']['graph']==ref['heads']['graph'],'CNN condition affected graph branch')
                    records.append(measured)
            # The unknown approximate wavelength and absent GSD controls must be exact negatives on this real catalog.
            for scope in scopes:
                by={r['control']:r for r in records if r['scope']==scope}
                graph.require(by['matched']['points']==by['unknown_wavelength']['points']==by['missing_gsd']['points'],
                    'Unknown wavelengths or absent GSD were silently promoted to physical evidence')
            for owner,name in ((model.backbone,'range_metadata'),(model.backbone.CNN_Branch[0],'condition_selector')):
                layer=getattr(owner,name,None)
                if layer is not None:layer.weight.zero_()
            neutral_record=evaluate('both' if native else 'graph','matched',neutral=True)
            graph.require(neutral_record['points']==reference['points'],'Neutral metadata insertion did not preserve the trained predictor')
            records.append(neutral_record)
            # Timing uses the neutral graft, a fixed lowest tile, both actual bands, and the same original physical GPU.
            first_tile=min(e['tile'] for e in spec['validation'])
            for event in (e for e in spec['validation'] if e['tile']==first_tile):
                actual,_,_=materialize(event);provider.override='matched';samples=[]
                with cond.selection_policy(model,'joint' if native else 'content','joint'):
                    for i in range(7):
                        if i==2:torch.cuda.reset_peak_memory_stats()
                        execution.sync();t0=time.perf_counter();output,trace,_=infer(actual,'devval','matched');execution.sync()
                        if i>=2:samples.append((time.perf_counter()-t0)*1000)
                costs.append(dict(scene=scene,seed=seed,regime=regime,band=event['band'],warmups=2,samples=5,
                    p50_ms=statistics.median(samples),p95_ms=float(np.percentile(samples,95)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_parameters=parameters,
                    source_parameters=initial_model_parameters,condition_parameters_trained=False,
                    cnn_candidates=3 if native else 1,graph_candidates=2,
                    scope='same original GPU; resident neutral-grafted model/host view plus prepare-transfer-forward-diagnostics; no kernel-only claim'))
            # Only newly inserted metadata tensors may change during explicit neutralization.
            graph.require(all(tensor_hash(model.state_dict()[n])==h for n,h in old_state.items()),'Zero-update probe changed a source tensor')
            neutral_keys=('backbone.range_metadata.','backbone.CNN_Branch.0.condition_selector.')
            graph.require(all(tensor_hash(model.state_dict()[n])==h for n,h in before.items() if not n.startswith(neutral_keys)),
                'Inference changed an inserted encoder or non-neutralized tensor')
            graph.require(_exact(original_rng,__import__('joint_training',fromlist=['_rng_state'])._rng_state(execution.device)), 'Zero-update probe changed RNG')
            graph.require(hashlib.sha256(Path(statepin['path']).read_bytes()).hexdigest()==statepin['sha256'],'Original completed checkpoint bytes changed')
            group=f'{regime}-{scene}-{seed}'
            for record in records:
                common=dict(regime=regime,scene=scene,seed=seed,scope=record['scope'],condition=record['control'],neutral=record['neutral'],condition_parameters_trained=False)
                for head,bands in record['metrics'].items():
                    for band,m in bands.items():
                        ref=reference['metrics'][head][band]
                        endpoints.append(dict(**common,head=head,band=band,**{k:m[k] for k in ('count','oa','aa','nll')},nll_delta=m['nll']-ref['nll'],oa_delta=m['oa']-ref['oa']))
                        for j,c_ in enumerate(m['per_class']):classes.append(dict(**common,head=head,band=band,class_id=c_['class_id'],support=c_['support'],correct=m['confusion_matrix'][j][j],reference_correct=ref['confusion_matrix'][j][j],recall_delta=c_['recall']-ref['per_class'][j]['recall']))
                for r in record['points']:
                    ref=refmap[key(r)];label=r['class_id'];pred=max(range(9),key=r['probabilities'].__getitem__)+1;pred0=max(range(9),key=ref['probabilities'].__getitem__)+1
                    public_points.append(dict(**common,point_id=digest((provider.catalog,r['tile'],r['released_ordinal']))[:20],
                        view_id=digest((provider.catalog,r['tile'],r['band']))[:20],head=r['head'],band=r['band'],class_id=label,
                        nll_delta=-math.log(max(r['probabilities'][label-1],1e-8))+math.log(max(ref['probabilities'][label-1],1e-8)),
                        prediction=pred,reference_prediction=pred0,correctness_change=int(pred==label)-int(pred0==label)))
                for v in record['validation']['rows']:
                    view_rows.append(dict(**common,view_id=digest((provider.catalog,v['tile'],v['band']))[:20],band=v['band'],released_points=v['released']['points'],
                        cnn_weights=json.dumps(v['graph']['weights']['cnn']['weights']),graph_weights=json.dumps(v['graph']['weights']['graph']['weights']),
                        near_nnz=v['graph']['range_nnz'][0],wide_nnz=v['graph']['range_nnz'][1]))
            result=dev._json_new(private/(group+'.json'),dict(registration=registration,checkpoint=statepin,checkpoint_producer=spec['producer'],
                executor=code['commit'],old_model_state=old_model_state,random_adapter_initial_state=adapter_initial,
                metadata_initialization_seed=spec['module_seed']+41017,source_tensors_unchanged=True,original_file_unchanged=True,
                RNG_unchanged=True,optimizer_updates=0,condition_parameters_trained=False,records=records))
            results.append(result);print(json.dumps({'case':group,'status':'passed','controls':len(records),'optimizer_updates':0}),flush=True)
    for name,rows in [('endpoints.csv',endpoints),('per-class.csv',classes),('points.csv',public_points),('view-weights.csv',view_rows),('resident-cost.csv',costs)]:table(a.out/name,rows)
    dev._json_new(a.out/'summary.json',dict(status='passed',executor=code['commit'],source_hash=code['source_hash'],program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        models=7,cases=results,optimizer_updates=0,condition_parameters_trained=False,new_labels=0,query_accessed=False,raw_GT_opened=False,
        all_historical_missing_graft_point_probabilities_bitwise_equal=True,all_neutral_grafts_bitwise_equal=True,
        original_source_tensors_and_checkpoint_files_unchanged=True,unknown_wavelength_and_absent_GSD_are_exact_negative_controls=True,
        table_rows={'endpoints':len(endpoints),'classes':len(classes),'points':len(public_points),'views':len(view_rows),'cost':len(costs)},
        interpretation='Fixed-seed untrained-condition stress probes; no trained-condition accuracy benefit or sensor causal effect',elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
