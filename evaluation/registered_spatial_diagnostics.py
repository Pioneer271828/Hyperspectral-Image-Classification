"""Freeze all local development diagnostics before exporting selected-model probes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from model.research_config import digest
from research_runtime import code_snapshot,require_frozen_code,external_directory,tensor_hash
from transfer.container_runtime import training_container_kind
from transfer.registered_development import _load_registration,_read,_json_new
from transfer.graph_route_training import GraphRouteRun
from transfer.graph_route_context import route_state_hash
from transfer.development_training import sparse_labels
from .block_input_diagnostics import BlockInputPlan,review_block
from .annotated_components import ComponentPlan,score_components
from .graph_edge_intervention import _selected
from .classification_metrics import classification_metrics
from .spatial_metrics import boundary_bands
from .spatial_diagnostics import spatial_support
from .branch_diagnostics import branch_diagnostics


def _code(expected_commit):
    code=code_snapshot()
    if not training_container_kind() or sys.flags.optimize or code['dirty'] or code['commit']!=expected_commit:
        raise ValueError('Clean exact code in authorized Linux container required')
    return code


def _load(registration,completed):
    v,base,files=_load_registration(registration);_read(completed)
    run=GraphRouteRun.load(base.source,base.scene,base.record,completed['path'],plan=base.plan,bank=base.bank,
        expected_bank_hash=v['bank_hash'],expected_sha256=completed['sha256'],expected_contract_hash=v['contract_hash'])
    run.finish()  # Partial/failed artifacts cannot enter selected-state diagnostics.
    return v,run,files+[deepcopy(completed)]


def _labels(run,tid):
    if type(tid) is not int or not 0<=tid<len(run.record['roles']):raise ValueError('Integer development tile required')
    role=run.record['roles'][tid]['role']
    if role not in ('devtrain','devval'):raise ValueError('Query/non-development diagnostics forbidden')
    key='train_support' if role=='devtrain' else 'validation_support'
    y0,y1,x0,x1=run.scene.plan['tiles'][tid]['box']
    labels=sparse_labels(run.record,run.scene.plan,key)[y0:y1,x0:x1]
    return labels,np.ones(labels.shape,dtype=bool),role


def _plans(run,requests):
    if (not isinstance(requests,dict) or set(requests)!={'version','blocks'} or requests['version']!=1
            or not isinstance(requests['blocks'],list) or not 1<=len(requests['blocks'])<=16):
        raise ValueError('One to sixteen explicit development block requests required')
    rows=[];ids=[]
    for item in requests['blocks']:
        if not isinstance(item,dict) or set(item)!={'tile_id','target_ids','class_id','interventions','small_area','thin_depth'}:
            raise ValueError('Unknown request fields')
        labels,access,role=_labels(run,item['tile_id']);ids.append(item['tile_id'])
        block=BlockInputPlan.freeze(run,**{k:item[k] for k in ('tile_id','target_ids','class_id','interventions')})
        geometry=ComponentPlan.freeze(labels,run.record['classes'],geometry_mask=access,score_mask=access,
            small_area=item['small_area'],thin_depth=item['thin_depth'])
        rows.append(dict(tile_id=item['tile_id'],role=role,input_plan=block.to_dict(),geometry_plan=geometry.to_dict()))
    if ids!=sorted(set(ids)):raise ValueError('Block ids must be unique and ordered')
    return rows


def _output(out,files):
    out=external_directory(out)
    for pin in files:
        parent=Path(pin['path']).resolve(strict=True).parent
        if out==parent or out in parent.parents or parent in out.parents:raise ValueError('Output overlaps protected input directory')
    if out.exists():raise FileExistsError('New diagnostic output directory required')
    return out


def freeze_diagnostics(registration,completed,requests,*,out,expected_commit):
    # Existing loader rechecks selected AND last-state validation metrics. This
    # is artifact authentication, not a new diagnostic/selection decision.
    code=_code(expected_commit);request_value=json.loads(_read(requests));v,run,files=_load(registration,completed)
    files.append(deepcopy(requests));rows=_plans(run,request_value);folder=_output(out,files)
    for pin in files:_read(pin)
    require_frozen_code(code)
    value=dict(version=1,code=code,training_commit=v['expected_commit'],registration=deepcopy(registration),completed=deepcopy(completed),
        requests=deepcopy(requests),inputs=files,contract_hash=run.contract_hash,selected_step=run.selected_step,
        blocks=rows,optimizer_updates=0,query=0,original_ground_truth_read=False,
        validation_recheck_states=['selected','last'],diagnostic_scores_computed=False)
    value['freeze_hash']=digest(value);folder.mkdir(parents=True,exist_ok=False)
    return _json_new(folder/'freeze.json',value)


def export_diagnostics(frozen,*,out,expected_commit):
    code=_code(expected_commit);value=json.loads(_read(frozen))
    fields={'version','code','training_commit','registration','completed','requests','inputs','contract_hash','selected_step',
            'blocks','optimizer_updates','query','original_ground_truth_read','freeze_hash',
            'validation_recheck_states','diagnostic_scores_computed'}
    if (set(value)!=fields or value['version']!=1 or value['freeze_hash']!=digest({k:v for k,v in value.items() if k!='freeze_hash'})
            or value['code']!=code or value['optimizer_updates']!=0 or value['query']!=0 or value['original_ground_truth_read'] is not False
            or value['validation_recheck_states']!=['selected','last'] or value['diagnostic_scores_computed'] is not False):
        raise ValueError('Frozen diagnostic schema/code differs')
    for pin in value['inputs']:_read(pin)
    v,run,files=_load(value['registration'],value['completed']);files.append(value['requests'])
    if (files!=value['inputs'] or v['expected_commit']!=value['training_commit'] or run.contract_hash!=value['contract_hash']
            or run.selected_step!=value['selected_step'] or _plans(run,json.loads(_read(value['requests'])))!=value['blocks']):
        raise ValueError('Frozen requests/model/geometry changed')
    folder=_output(out,files+[frozen]);folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    _json_new(folder/'started.json',dict(status='running',freeze=frozen,code=code,query=0,optimizer_updates=0))
    reports=[];before=route_state_hash(run.model)
    try:
        model=_selected(run);selected_before=route_state_hash(model)
        for row in value['blocks']:
            tid=row['tile_id'];labels,access,role=_labels(run,tid);ids=run.record['classes']
            ip=row['input_plan'];gp=row['geometry_plan']
            input_result=review_block(run,BlockInputPlan(ip,expected_hash=ip['plan_hash']),expected_plan_hash=ip['plan_hash'])
            with torch.no_grad():probabilities=run._predict(model,run.scene.plan['tiles'][tid],0)
            if [tensor_hash(p) for p in probabilities[:3]]!=input_result['baseline_probability_hashes']:
                raise ValueError('Scored predictions differ from input diagnostic baseline')
            predictions=[(p.argmax(1)+1).numpy().reshape(labels.shape) for p in probabilities[:3]]
            bands=boundary_bands(labels,access,(1,2,3));heads={}
            for name,pred in zip(('fused','graph','cnn'),predictions):
                heads[name]=dict(components=score_components(labels,pred,ids,geometry_mask=access,score_mask=access,
                    plan=ComponentPlan(gp,expected_hash=gp['plan_hash']),expected_plan_hash=gp['plan_hash']),
                    boundary={str(w):dict(boundary=classification_metrics(labels,pred,ids,mask),
                        observed_nonboundary=classification_metrics(labels,pred,ids,access&~mask)) for w,mask in bands.items()})
            result=dict(tile_id=tid,role=role,input_diagnostics=input_result,heads=heads,
                spatial_support=spatial_support(labels,ids,geometry_mask=access,score_mask=access),
                branches=branch_diagnostics(labels,predictions[2],predictions[1],predictions[0],ids,access),
                labels='released sparse development support only; not complete object geometry')
            reports.append(_json_new(folder/f'tile-{tid}.json',result))
        for pin in files+[frozen]:_read(pin)
        if before!=route_state_hash(run.model) or selected_before!=route_state_hash(model):raise ValueError('Diagnostic changed model state')
        require_frozen_code(code)
        receipt=dict(status='passed',code=code,training_commit=value['training_commit'],freeze=frozen,inputs=files,selected_step=run.selected_step,
            reports=reports,optimizer_updates=0,query=0,original_ground_truth_read=False,elapsed_seconds=time.monotonic()-started,
            independent_scientific_acceptance=False,scope='CPU registered nine-class 90/45 development diagnostic only')
        return _json_new(folder/'receipt.json',receipt)
    except BaseException as exc:
        _json_new(folder/'failure.json',dict(status='failed',error=repr(exc),completed_reports=reports,freeze=frozen,query=0,optimizer_updates=0))
        raise
