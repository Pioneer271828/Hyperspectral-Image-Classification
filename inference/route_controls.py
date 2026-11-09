"""Label-free stratified random routing and resident-input paired CPU cost."""
import hashlib
import math
import os
import statistics
import time
import torch
from inference.tile_cascade import validate_requests
from inference.trained_light import TrainedLight,trained_cascade
from model.research_config import digest
from transfer.g3_training import state_hash


def _tensor(t):
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest())


def request_hash(requests):
    return digest(sorted([dict(**{k:v for k,v in r.items() if k not in ('x','wavelengths')},x=_tensor(r['x']),wavelengths=_tensor(r['wavelengths'])) for r in requests],key=lambda r:r['id']))


def _check(full,published,requests,min_coverage):
    if type(published) is not TrainedLight:raise ValueError('Authenticated trained-light pair required')
    dataset=published.validate(full);validate_requests(published.model,requests)
    if (any(r['dataset']!=dataset for r in requests) or type(min_coverage) not in (int,float)
            or not math.isfinite(min_coverage) or not 0<=min_coverage<=1):raise ValueError('Wrong task or coverage rule')


def random_plan(full,published,requests,reference,*,seed,min_coverage):
    """Match fast counts per task/shape/wavelength-count/graph-scale stratum.

    Hash ranking uses only registered IDs and seed; actual latency may differ.
    Reference route provenance must be pinned by the experiment registration.
    """
    _check(full,published,requests,min_coverage)
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Registered uint32 random seed required')
    if not isinstance(reference,list) or len(reference)!=len(requests):raise ValueError('Complete reference route list required')
    byid={r['id']:r for r in requests};routes={}
    for r in reference:
        if (not isinstance(r,dict) or set(r)!={'id','route'} or r['id'] not in byid or r['id'] in routes
                or r['route'] not in ('fast','full')):raise ValueError('Exact reference routes required')
        routes[r['id']]=r['route']
    groups={}
    for r in requests:
        known=r['coverage_known'] and r['coverage_fraction']>=min_coverage
        if not known and routes[r['id']]=='fast':raise ValueError('Reference bypasses mandatory full fallback')
        key=(r['dataset'],*r['x'].shape,len(r['wavelengths']),r['scale'])
        groups.setdefault(key,dict(eligible=[],fast=0,all=[]));g=groups[key];g['all'].append(r['id'])
        if known:g['eligible'].append(r['id'])
        g['fast']+=routes[r['id']]=='fast'
    chosen=[];strata=[]
    for key,g in sorted(groups.items()):
        ranked=sorted(g['eligible'],key=lambda i:(digest(dict(seed=seed,id=i,stratum=key)),i))
        selected=ranked[:g['fast']];chosen.extend(selected)
        strata.append(dict(key=list(key),ids=sorted(g['all']),eligible=sorted(g['eligible']),fast_count=g['fast'],selected=sorted(selected)))
    return dict(format='dy-stratified-random-v1',source_state=state_hash(full),light_state=state_hash(published.model),
        training_contract=published.specification['contract_hash'],input_hash=request_hash(requests),seed=seed,min_coverage=min_coverage,
        reference=sorted(reference,key=lambda r:r['id']),strata=strata,selected=sorted(chosen),matching='per_stratum_fast_counts_not_measured_cost')


def random_cascade(full,published,requests,plan,*,expected_plan_hash):
    if not isinstance(plan,dict) or digest(plan)!=expected_plan_hash:raise ValueError('External random plan hash mismatch')
    rebuilt=random_plan(full,published,requests,plan['reference'],seed=plan['seed'],min_coverage=plan['min_coverage'])
    if rebuilt!=plan:raise ValueError('Random plan/source/input changed')
    selected=set(plan['selected']);outputs=[];counts=dict(light_calls=0,full_calls=0,accepted_tiles=0,total_tiles=len(requests))
    with torch.no_grad():
        for r in requests:
            known=r['coverage_known'] and r['coverage_fraction']>=plan['min_coverage'];accept=r['id'] in selected
            if known:
                # Adaptive gates pay light inference even for rejected known blocks.
                p=published.model(r['dataset'],r['x'],wavelengths=r['wavelengths']);counts['light_calls']+=1
            if not accept:
                p=full(r['dataset'],r['x'],wavelengths=r['wavelengths'],superpixel_scale=r['scale'])[0];counts['full_calls']+=1
            else:counts['accepted_tiles']+=1
            if not torch.isfinite(p).all():raise ValueError('Nonfinite random control output')
            outputs.append(dict(id=r['id'],route='fast' if accept else 'full',probabilities=p))
    return dict(outputs=outputs,counts=counts,mode='stratified_random',calibrated=False,plan_hash=expected_plan_hash)


def _output_hash(result):
    return digest([dict(id=r['id'],route=r['route'],probabilities=_tensor(r['probabilities'])) for r in result['outputs']])


def benchmark_controls(full,published,requests,*,adaptive_kwargs,seed,warmups=2,pairs=8):
    """Single-core CPU paired cost, including public validation/routing overhead.

    Excludes model/input loading, preprocessing, initial plan creation and output
    hashing. Random-plan revalidation is included. Memory/energy are not measured.
    """
    if (type(warmups) is not int or warmups<1 or type(pairs) is not int or pairs<4 or pairs%4
            or torch.get_num_threads()!=1 or len(os.sched_getaffinity(0))!=1):
        raise ValueError('One CPU core/thread, warmup and multiple-of-four paired rounds required')
    if not isinstance(adaptive_kwargs,dict) or adaptive_kwargs.get('mode')!='adaptive':raise ValueError('Explicit frozen adaptive configuration required')
    if not requests:raise ValueError('Nonempty matched benchmark required')
    source=state_hash(full);light=state_hash(published.model);inputs=request_hash(requests);start=time.perf_counter()
    adaptive=trained_cascade(full,published,requests,**adaptive_kwargs)
    reference=[dict(id=r['id'],route=r['route']) for r in adaptive['outputs']]
    coverage=adaptive_kwargs.get('min_coverage',1.)
    plan=random_plan(full,published,requests,reference,seed=seed,min_coverage=coverage);pin=digest(plan);setup_seconds=time.perf_counter()-start
    arms=dict(full=lambda:trained_cascade(full,published,requests),
              fast=lambda:trained_cascade(full,published,requests,mode='fast',min_coverage=coverage),
              adaptive=lambda:trained_cascade(full,published,requests,**adaptive_kwargs),
              random=lambda:random_cascade(full,published,requests,plan,expected_plan_hash=pin))
    baseline={k:fn() for k,fn in arms.items()};before={k:_output_hash(v) for k,v in baseline.items()};samples=[];keys=list(arms)
    if baseline['adaptive']['counts']!=baseline['random']['counts']:raise AssertionError('Matched control calls differ')
    for _ in range(warmups):
        for fn in arms.values():fn()
    for round_id in range(pairs):
        order=keys[round_id%4:]+keys[:round_id%4];row=dict(order=order,timings={})
        for name in order:
            cpu=time.process_time_ns();wall=time.perf_counter_ns();output=arms[name]()
            wall_ms=(time.perf_counter_ns()-wall)/1e6;cpu_ms=(time.process_time_ns()-cpu)/1e6
            if _output_hash(output)!=before[name] or output['counts']!=baseline[name]['counts']:raise RuntimeError('Benchmark output/route changed')
            row['timings'][name]=dict(wall_ms=wall_ms,process_cpu_ms=cpu_ms)
        samples.append(row)
    if state_hash(full)!=source or state_hash(published.model)!=light or request_hash(requests)!=inputs:raise RuntimeError('Benchmark changed frozen inputs')
    summary={}
    for name in arms:
        ms=[r['timings'][name]['wall_ms'] for r in samples];cpu=[r['timings'][name]['process_cpu_ms'] for r in samples]
        summary[name]=dict(p50_ms=statistics.median(ms),p95_ms=statistics.quantiles(ms,n=100,method='inclusive')[94],
            mean_ms=statistics.mean(ms),sample_sd_ms=statistics.stdev(ms),process_cpu_p50_ms=statistics.median(cpu),
            blocks_per_second=len(requests)/(statistics.median(ms)/1000),calls=baseline[name]['counts'])
    return dict(status='passed',scope='resident_cpu_batch_public_api_validation_and_compute',affinity=sorted(os.sched_getaffinity(0)),threads=1,
        source_state=source,light_state=light,input_hash=inputs,random_plan=plan,random_plan_hash=pin,adaptive_kwargs=adaptive_kwargs,
        initial_reference_and_plan_seconds=setup_seconds,warmups=warmups,pairs=pairs,samples=samples,summary=summary,
        output_hashes=before,matching='same_stratified_call_counts_not_latency',optimizer_updates=0,query=0,
        peak_memory=None,energy=None,scientific_speed_claim=False)
