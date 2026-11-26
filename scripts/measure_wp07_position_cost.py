"""Score-free cost of an existing predictor with untrained position encoders; no new real views."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from joint_training import _rng_state
from research_runtime import code_snapshot, require_frozen_code
from scripts.probe_wp07_observed_conditions import Provider, pin
from scripts.run_wp07_b_pair import observation_only_resources
from scripts.run_wp07_condition_integration import table
from scripts import run_native_released_points as worker
from transfer import wp07_exact_positions as position
from transfer import wp07_observed_conditions as condition
from transfer import wp07_fine16_training as frozen
from transfer import wp07_graph_range as graph
from transfer import registered_development as dev
from transfer.native_scale_campaign import read, packet
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact


def main():
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','core','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--representation-family',choices=('legacy','balanced'),default='legacy')
    a=p.parse_args();code=code_snapshot();require_frozen_code(code)
    graph.require(code['commit']==a.expected_commit and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu,'Wrong frozen source or device')
    a.out.mkdir()
    plan=read(pin(a.core/'xuzhou/pair/plan.json'),a.root)
    case=next(c for c in plan['cases'] if c['seed']==512 and c['arm']=='content')
    graph.require(case['gpu']==a.gpu,'Only original physical GPU')
    dest=Path(case['destination']);spec=read(pin(dest/'registration.json'),a.root)
    state=packet(read(pin(dest/'completed.json'),a.root)['checkpoint'],a.root)
    rows=[]
    with observation_only_resources(),worker.arithmetic(a.root,a.gpu) as execution,torch.no_grad():
        graph.require(execution.numeric==spec['numeric'],'Changed arithmetic')
        provider=Provider(a.root,read(plan['preflight'],a.root),512)
        engine=frozen.RangeRun(spec,provider,packet(plan['cache'],a.root)['entries'],spec['producer'],execution)
        engine.restore(state,spec['steps']);engine.optimizer=None;model=engine.model;model.eval()
        condition.install_encoders(model,spec['module_seed']+41017)
        position.extend(model,spec['module_seed']+27183)
        before=state_hash(model);rng=_rng_state(execution.device)
        tile=min(e['tile'] for e in spec['validation'])
        for event in (e for e in spec['validation'] if e['tile']==tile):
            started=time.perf_counter();actual,_,_=engine._materialize('devval',event);materialization=time.perf_counter()-started
            representations=position.REPRESENTATIONS if a.representation_family=='legacy' else position.BALANCED_REPRESENTATIONS
            for representation in representations:
                for control in ('missing','matched'):
                    provider.override=control;samples=[]
                    with position.representation(model,representation),condition.selection_policy(model,'content','joint'):
                        for i in range(7):
                            if i==2:torch.cuda.reset_peak_memory_stats()
                            execution.sync();start=time.perf_counter();_,trace,_=engine._forward(actual,'devval','matched');execution.sync()
                            if i>=2:samples.append((time.perf_counter()-start)*1000)
                    rows.append(dict(scene='xuzhou',seed=512,band=event['band'],representation=representation,condition=control,
                        warmups=2,samples=5,p50_ms=statistics.median(samples),maximum_ms=max(samples),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_parameters=sum(p.numel() for p in model.parameters()),
                        input_materialization_seconds=materialization,cnn_candidates=1,graph_candidates=2,
                        scope='shared original GPU, resident old graph-only model with untrained descriptor encoder; prepare/transfer/frozen-QA/forward/diagnostics included; no scoring, kernel-time or speed claim'))
        graph.require(state_hash(model)==before and _exact(rng,_rng_state(execution.device)),'Cost observation changed weights or RNG')
    table(a.out/'real-resident-cost.csv',rows)
    dev._json_new(a.out/'summary.json',dict(status='passed',producer=code['commit'],source_hash=code['source_hash'],rows=len(rows),
        classifier_forwards=56,optimizer_updates=0,validation_scoring_calls=0,new_real_views=0,
        real_condition_parameters_trained=False,weights_and_RNG_unchanged=True,physical_gpu=a.gpu))


if __name__=='__main__':main()
