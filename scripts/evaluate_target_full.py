"""Freeze/audit all27 models before releasing any real target query labels."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.g4_target import (freeze_for_query,read_query_labels,load_case,predict_query,query_metrics,
    paired_summary,read_json,check_files,POLICY)
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def completed(directory,identity):
    value=read_json(directory/'result.json')
    if value.get('status')!='completed' or value.get('identity')!=identity or value.get('policy')!=POLICY:
        raise ValueError('Completed evaluation identity differs')
    check_files(value['protected_training_files']);check_files(value['raw_label_pins'])
    for name,sha in value['artifacts'].items():
        if Path(name).name!=name or file_hash(directory/name)!=sha:raise ValueError('Completed evaluation artifact changed')
    return {'operation':'verified_completed','evaluation_updates_this_call':0,'result':value}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','training-root','launch-sha256','training-acceptance-sha256','prepared-manifest',
              'prepared-manifest-sha256','data-root','out-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or code['dirty'] or code['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected evaluator SHA required')
    out=external_directory(a.out_dir);identity={'code':code,'training_root':str(Path(a.training_root).resolve()),
        'launch_sha256':a.launch_sha256,'training_acceptance_sha256':a.training_acceptance_sha256,
        'prepared_manifest':str(Path(a.prepared_manifest).resolve()),'prepared_manifest_sha256':a.prepared_manifest_sha256,
        'data_root':str(Path(a.data_root).resolve()),'device':'cuda','policy':POLICY}
    if a.resume:
        if a.audit_only or not (out/'result.json').exists():raise ValueError('Resume only verifies a completed evaluation')
        result=completed(out,identity);require_frozen_code(code)
        print(json.dumps({'operation':result['operation'],'evaluation_updates_this_call':0}));return
    start=time.monotonic()
    # freeze_for_query creates owned output and retains failures without GT access.
    stage,freeze=freeze_for_query(a.training_root,a.launch_sha256,a.training_acceptance_sha256,code,out,device='cuda')
    write_json(out/'identity.json',identity)
    if a.audit_only:
        print('G4_ALL_NATIVE_PREQUERY_AUDIT_PASSED',flush=True);return
    rows=[];raw_pins={str(Path(a.prepared_manifest).resolve()):a.prepared_manifest_sha256};max_memory=0
    try:
        for row in stage['rows']:
            case=row['case'];release,pins=read_query_labels(freeze,case,a.data_root,a.prepared_manifest,a.prepared_manifest_sha256)
            raw_pins.update(pins);model,runtime,items=load_case(row,device='cuda')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            probabilities=predict_query(model,items[case['target']],release['indices']);torch.cuda.synchronize()
            seconds=time.monotonic()-t;peak=torch.cuda.max_memory_allocated();max_memory=max(max_memory,peak)
            values={h:p.numpy() for h,p in probabilities.items()};metrics,branch=query_metrics(release['labels'],values)
            name=case['name'];payload=out/(name+'-query.pt')
            # All output files are new and outside the producer checkout.
            with payload.open('xb') as stream:torch.save({'target':case['target'],'seed':case['seed'],'method':case['method'],
                'checkpoint_sha256':release['checkpoint_sha256'],'family_hash':release['parent_hash'],
                'indices':torch.tensor(release['indices'],dtype=torch.long),'labels':torch.tensor(release['labels'],dtype=torch.long),
                'probabilities':probabilities,'freeze_sha256':file_hash(out/'freeze.json')},stream)
            result={'target':case['target'],'seed':case['seed'],'method':case['method'],'name':name,
                'selected_step':row['result']['selected_step'],'family_hash':release['parent_hash'],
                'checkpoint_sha256':release['checkpoint_sha256'],'metrics':metrics,'branch':branch,
                'query_payload':payload.name,'query_payload_sha256':file_hash(payload),
                'query_feature_forward_heads_and_cpu_copy_seconds':seconds,'gpu_peak_allocated_bytes':peak,
                'cost_excludes_loading_binding_and_initialization':True,'optimizer_updates':0}
            rows.append(result);write_json(out/(name+'-metrics.json'),result)
            print('G4_QUERY_CASE_PASSED',name,flush=True)
            del model,runtime,items,probabilities,values;torch.cuda.empty_cache()
        check_files(freeze['protected_files']);check_files(raw_pins);require_frozen_code(code)
        summary=paired_summary(rows);write_json(out/'summary.json',summary)
        artifacts={p.name:file_hash(p) for p in out.iterdir() if p.is_file()}
        result={'status':'completed','policy':POLICY,'identity':identity,'rows':rows,'summary':summary,
            'protected_training_files':freeze['protected_files'],'raw_label_pins':raw_pins,
            'artifacts':artifacts,'elapsed_seconds':time.monotonic()-start,'gpu_peak_allocated_bytes':max_memory,
            'optimizer_updates':0,'target_training_changed':False,'query_used_for_selection':False}
        write_json(out/'result.json',result)
        print('G4_ALL_QUERY_SCORED',len(rows),flush=True)
    except Exception as error:
        write_json(out/'query-failure.json',{'status':'failed','error':repr(error),'completed_cases':len(rows),
            'query_stage_started_after_global_native_gate':True,'optimizer_updates':0})
        traceback.print_exc();raise


if __name__=='__main__':main()
