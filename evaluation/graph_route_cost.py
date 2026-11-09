"""Resident raw-block E3 CPU costs; private backward probes never optimize."""
from contextlib import contextmanager
from copy import deepcopy
import math
import os
import statistics
import time
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.graph_route_context import _context,route_state_hash
from transfer.graph_route_training import GraphRouteRun
from transfer.sample_curriculum import curriculum_loss


def _summary(samples):
    if len(samples)<4:raise ValueError('At least four raw timing samples required')
    for sample in samples:
        if set(sample)!={'wall_ms','process_cpu_ms'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in sample.values()):
            raise ValueError('Finite positive measured wall/CPU costs required')
    wall=[r['wall_ms'] for r in samples];cpu=[r['process_cpu_ms'] for r in samples]
    return dict(p50_ms=statistics.median(wall),p95_ms=statistics.quantiles(wall,n=100,method='inclusive')[94],
        mean_ms=statistics.mean(wall),sample_sd_ms=statistics.stdev(wall),process_cpu_p50_ms=statistics.median(cpu))


@contextmanager
def _stages(model):
    """Sequential forward subsets; separate from the uninstrumented timings."""
    b=model.backbone
    targets=[('graph_construction',b.learnable_graph,'build_graph'),('route_score_select_sparse',b.graph_router,'forward'),
             ('graph_stage1',b,'graph_stage1'),('graph_stage2',b,'graph_stage2')]
    if any(method in owner.__dict__ for _,owner,method in targets):raise ValueError('Cost probe already intercepted')
    records={name:[] for name,_,_ in targets};installed=[]
    try:
        for name,owner,method in targets:
            original=getattr(owner,method)
            def wrapper(*args,_fn=original,_name=name,**kwargs):
                cpu=time.process_time_ns();wall=time.perf_counter_ns()
                result=_fn(*args,**kwargs)
                records[_name].append(dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,process_cpu_ms=(time.process_time_ns()-cpu)/1e6))
                return result
            setattr(owner,method,wrapper);installed.append((owner,method))
        yield records
    finally:
        for owner,method in reversed(installed):delattr(owner,method)


def _once(run,model,tile,step,backward):
    # Fingerprints and RNG preservation are outside measured compute scope.
    with torch.random.fork_rng(devices=[]),torch.set_grad_enabled(backward):
        cpu=time.process_time_ns();wall=time.perf_counter_ns()
        model.train(backward);model.zero_grad(set_to_none=True)
        outputs=run._predict(model,tile,step)
        loss=None
        if backward:
            loss=curriculum_loss(outputs,run.scene,run.record,tile,policy=None,class_balance=True)['total']
            loss.backward()
        timing=dict(wall_ms=(time.perf_counter_ns()-wall)/1e6,process_cpu_ms=(time.process_time_ns()-cpu)/1e6)
    heads=[tensor_hash(x.detach()) for x in outputs[:3]]
    gradients={k:None if p.grad is None else tensor_hash(p.grad) for k,p in model.named_parameters()}
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite diagnostic gradient')
    signature=dict(probabilities=heads,loss=None if loss is None else float(loss.detach()),gradients=gradients,
        route={k:outputs[3]['interaction'][k] for k in GraphRouteRun.ROUTE_FIELDS})
    return timing,signature


def benchmark_route(run,*,state_kind,tile_ids,route_step,warmups=2,repeats=8):
    """No I/O/load/optimizer time; raw-block normalization and all compute included."""
    started=time.perf_counter()
    if (type(run) is not GraphRouteRun or state_kind not in ('last','selected')
            or torch.get_num_threads()!=1 or len(os.sched_getaffinity(0))!=1
            or type(route_step) is not int or route_step<0 or type(warmups) is not int or warmups<1
            or type(repeats) is not int or repeats<4 or repeats%2):
        raise ValueError('Completed CPU run, one core/thread, explicit step, warmups and even repeats>=4 required')
    run.finish()
    if not isinstance(tile_ids,(list,tuple)) or not tile_ids or any(type(i) is not int for i in tile_ids) or len(set(tile_ids))!=len(tile_ids):
        raise ValueError('Unique explicit development training tiles required')
    contexts=[_context(run.scene,run.record,i) for i in tile_ids]
    if any(c['role']!='devtrain' for c in contexts):raise ValueError('Backward costs only on released development training blocks')
    source=route_state_hash(run.source);original=route_state_hash(run.model)
    model=deepcopy(run.model)
    if state_kind=='selected':model.load_state_dict(run.selected_state)
    state=route_state_hash(model);arms=['forward','forward_backward']
    plan=dict(version=1,contract_hash=run.contract_hash,state_kind=state_kind,state_hash=state,
        optimization_step=run.selected_step if state_kind=='selected' else run.completed,contexts=contexts,
        route_step=route_step,warmups=warmups,repeats=repeats,order=[arms if i%2==0 else arms[::-1] for i in range(repeats)],
        threads=1,affinity=sorted(os.sched_getaffinity(0)),torch_version=str(torch.__version__),
        scope='resident_raw_block_normalization_model_mode_grad_clear_full_forward_and_optional_original_loss_backward',
        excluded=['file_io','checkpoint_load','initial_identity_validation','model_copy','output_gradient_hashing','optimizer_step','selection_validation'],
        stage_scope='separate instrumented forward subsets; route contains scoring/selection/COO build together; no remote search',
        memory=None,energy=None)
    plan['plan_hash']=digest(plan);setup=time.perf_counter()-started;rows=[]
    for context in contexts:
        tile=run.scene.plan['tiles'][context['tile_id']];baseline={};samples=[]
        for arm in arms:
            _,baseline[arm]=_once(run,model,tile,route_step,arm=='forward_backward')
        for _ in range(warmups):
            for arm in arms:
                _,signature=_once(run,model,tile,route_step,arm=='forward_backward')
                if signature!=baseline[arm]:raise ValueError('Warmup output or gradient changed')
        for order in plan['order']:
            row=dict(order=order,timings={})
            for arm in order:
                timing,signature=_once(run,model,tile,route_step,arm=='forward_backward')
                if signature!=baseline[arm]:raise ValueError('Timed output or gradient changed')
                row['timings'][arm]=timing
            samples.append(row)
        profiles={}
        for arm in arms:
            timings=[];stage_rows=[]
            for _ in range(repeats):
                with _stages(model) as stages:
                    timing,signature=_once(run,model,tile,route_step,arm=='forward_backward')
                if signature!=baseline[arm] or any(len(v)!=1 for v in stages.values()):raise ValueError('Instrumentation changed computation or call counts')
                if sum(v[0]['wall_ms'] for v in stages.values())>timing['wall_ms']:raise ValueError('Stage intervals overlap or escape measured call')
                timings.append(timing);stage_rows.append({k:v[0] for k,v in stages.items()})
            profiles[arm]=dict(samples=timings,summary=_summary(timings),stage_samples=stage_rows,
                stage_summary={k:_summary([r[k] for r in stage_rows]) for k in stage_rows[0]})
        rows.append(dict(tile_id=context['tile_id'],samples=samples,
            summary={arm:_summary([r['timings'][arm] for r in samples]) for arm in arms},
            profiles=profiles,baseline=baseline,baseline_hash=digest(baseline)))
    if state!=route_state_hash(model) or original!=route_state_hash(run.model) or source!=route_state_hash(run.source):
        raise ValueError('Benchmark changed a bound model')
    run._check()
    return dict(status='passed',plan=plan,plan_hash=plan['plan_hash'],initial_setup_seconds=setup,rows=rows,
        forward_calls=len(contexts)*2*(1+warmups+2*repeats),backward_calls=len(contexts)*(1+warmups+2*repeats),
        optimizer_updates=0,query=0,scientific_speed_claim=False,
        interpretation='CPU repeated resident-block diagnostics; backward includes loss/grad clear, not complete optimizer update; instrumented samples kept separate')
