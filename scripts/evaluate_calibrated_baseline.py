"""BN-ALIGN3 frozen three-arm evaluation; all choices precede test readout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from evaluation.branch_diagnostics import branch_diagnostics
from evaluation.calibrated_baseline import (ARMS, SEEDS, compact_readout, require_reuse_receipts, scheduler_reuse)
from joint_baseline import COUNTS, NAMES, forward
from joint_calibrated_selection import load_selected
from joint_training import import_preparation, load_plan
from loadData.dataset_registry import file_hash, inspect_dataset
from model.bn_calibration import state_digest
from model.checkpoint_io import load_model_checkpoint
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def files(directory):
    return {str(p):file_hash(p) for p in Path(directory).rglob('*') if p.is_file()}


def assert_metrics(actual, expected):
    matrix = expected.get('confusion_matrix',expected.get('confusion'))
    if actual['confusion_matrix'] != matrix: raise ValueError('Frozen confusion matrix changed')
    for key in ('oa','aa','kappa','nll'):
        if key in expected and not np.isclose(actual[key],expected[key],rtol=0,atol=1e-6 if key=='nll' else 1e-12):
            raise ValueError('Frozen metric changed: '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','campaign-dir','prepared-dir','data-root','original-analysis-dir',
                'bn-align1-dir','bn-align2-dir','cpu-reuse-dir','cuda-reuse-dir','out-dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    code=code_snapshot()
    if sys.platform!='linux' or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean published exact SHA required')
    out=external_directory(a.out_dir);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:
        roots={key:Path(getattr(a,key+'_dir')) for key in ('campaign','bn_align1','bn_align2','original_analysis')}
        campaign,bn1,bn2,old=(roots[k] for k in ('campaign','bn_align1','bn_align2','original_analysis'))
        accepted=load(bn2/'result.json')
        if (accepted['status']!='passed' or accepted['code']['commit']!='7e843210429be3bc81bc2568e975fcde7fa10e88'
            or accepted['replayed_states']!=153 or load(bn1/'result.json')['code']['commit']!='dc5da4a9318f9e77febe848ac7cc28c4c9d19885'
            or load(campaign/'launch.json')['code']['commit']!='8daa65b116ba585b33a84e3af93f3741716b710a'):
            raise ValueError('Expected accepted frozen campaigns')
        for name,h in accepted['code']['files'].items():
            if code['files'].get(name)!=h: raise ValueError('Accepted production source changed: '+name)
        protected={}
        for directory in (bn1,bn2):
            receipt=load(directory/'independent-receipt.json')
            hashes=receipt.get('artifact_hashes',receipt.get('files',{}))
            if not hashes: raise ValueError('Accepted independent artifact audit missing')
            for name,h in hashes.items():
                if file_hash(directory/name)!=h: raise ValueError('Prior calibration artifact changed: '+name)
            protected.update(files(directory))
        reuse={};reuse_hashes={}
        for device in ('cpu','cuda'):
            directory=Path(getattr(a,device+'_reuse_dir'));reuse[device]=load(directory/'receipt.json')
            reuse_hashes[str(directory/'receipt.json')]=file_hash(directory/'receipt.json')
            for name,h in reuse[device]['artifacts'].items():
                if file_hash(directory/name)!=h: raise ValueError('Synthetic reuse artifact changed')
        require_reuse_receipts(reuse,code)
        _,items,provenance=import_preparation(a.prepared_dir,a.data_root,
            load_plan('configs/experiments/base97_training.json'),code)
        input_files=load(bn2/'plan.json')['input_files']
        for path,h in input_files.items():
            if file_hash(path)!=h: raise ValueError('Frozen dataset/preparation changed')
        frozen={};scheduler={};records={}
        for seed in SEEDS:
            folder=campaign/('seed-'+str(seed));result=load(folder/'result.json');records[seed]=result
            if result['identity']['prepared']!=provenance: raise ValueError('Preparation identity changed')
            expected=load(bn2/'plan.json')['source_files'][str(seed)]
            if expected!={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}:
                raise ValueError('Historical training files changed')
            protected.update(files(folder))
            hist=load(folder/'history.json');cal=load(bn2/('seed-'+str(seed))/'history.json')['calibrated']
            if len(hist)!=51 or len(cal)!=51: raise ValueError('All51 scheduler events required')
            states=(torch.load(folder/f'state-{h["step"]:06d}.pt',map_location='cpu',weights_only=True) for h in hist)
            # Small generator interface avoids retaining all training tensors.
            scheduler[str(seed)]=scheduler_reuse(hist,cal,list({'cursor':s['cursor'],'scheduler':s['scheduler']} for s in states),result['settings'])
            selected=load(bn2/('seed-'+str(seed))/'review.json')['calibrated_reselected']
            arm_paths={ARMS[0]:(folder/'primary.pt',None),
                       ARMS[1]:(bn1/('seed-'+str(seed))/'calibrated.pt',folder/'primary.pt'),
                       ARMS[2]:(bn2/('seed-'+str(seed))/'selected-calibrated.pt',folder/selected['source_file'])}
            frozen[str(seed)]={}
            for arm,(path,raw_path) in arm_paths.items():
                model=(load_model_checkpoint(path,device=a.device) if raw_path is None
                       else load_selected(path,items,raw_path,device=a.device))
                step=result['selected_step'] if arm!=ARMS[2] else selected['step']
                if model.training_config['checkpoint_optimizer_step']!=step: raise ValueError('Selected metadata cursor changed')
                frozen[str(seed)][arm]={'path':str(path),'sha256':file_hash(path),'sidecar_sha256':file_hash(str(path)+'.json'),
                    'source':str(raw_path) if raw_path else None,'selected_step':step,'state_sha256':state_digest(model)}
                del model
        write_json(out/'reuse-gate.json',{'status':'passed','code':code,'synthetic_receipts':reuse_hashes,
            'scheduler_events':scheduler,'events_verified':153,'real_optimizer_updates':0,
            'scope':'audited historical raw-weight reuse; no new full training claimed'})
        write_json(out/'frozen-before-test.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'code':code,'models':frozen,'protected_files':protected,'input_files':input_files,
            'reuse_gate_sha256':file_hash(out/'reuse-gate.json'),'arms':list(ARMS),'seeds':list(SEEDS),
            'selection':'already frozen by validation; no test-based choices',
            'primary':'macro fused test OA','aa_guard_max_drop_fraction':.005,'test_was_previously_inspected':True,
            'context':'acc1p_full_scene_transductive','real_optimizer_updates':0})
        print('ALL_MODELS_AND_REUSE_FROZEN_BEFORE_TEST',flush=True)
        # Only now assemble scoring labels/indices; calibration/selection never receives them.
        labels={};splits={}
        for name in NAMES:
            _,image_labels=inspect_dataset(Path(a.data_root),name);labels[name]=image_labels.reshape(-1)
            splits[name]=load(Path(a.prepared_dir)/(name+'-split.json'))
            if [len(splits[name]['indices'][phase]) for phase in ('train','val','test')]!=COUNTS[name]:
                raise ValueError('Frozen supervision counts changed')
        rows=[]
        for seed in SEEDS:
            row={'seed':seed,'arms':{}};packets={}
            historical=load(old/('seed-'+str(seed)+'-analysis.json'))
            one=load(bn1/('seed-'+str(seed))/'review.json')
            two=load(bn2/('seed-'+str(seed))/'review.json')
            for arm in ARMS:
                bound=frozen[str(seed)][arm];path=Path(bound['path'])
                model=(load_model_checkpoint(path,device=a.device) if bound['source'] is None else
                       load_selected(path,items,Path(bound['source']),device=a.device))
                before=state_digest(model);row['arms'][arm]={'selected_step':bound['selected_step'],'datasets':{}};packets[arm]={}
                for name in NAMES:
                    item={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in items[name].items()}
                    with torch.no_grad(): outputs=forward(model,name,item)
                    task={};packets[arm][name]={};classes=item['class_num']
                    for phase in ('train','val','test'):
                        ids=np.asarray(splits[name]['indices'][phase],dtype=np.int64);truth=labels[name][ids]
                        packet={'pixel_indices':torch.as_tensor(ids),'labels':torch.as_tensor(truth.astype(np.int16)),'heads':{}};metrics={}
                        for head,q in zip(('fused','graph','cnn'),outputs[:3]):
                            metrics[head],packet['heads'][head]=compact_readout(truth,q[torch.as_tensor(ids,device=a.device)],classes)
                            if arm==ARMS[0]: expected=historical['datasets'][name][phase]['heads'][head]
                            elif phase!='test':
                                expected=(one['metrics'][name][phase]['calibrated_reload'][head] if arm==ARMS[1]
                                          else two['calibrated_reselected']['metrics'][name][phase][head])
                            else: expected=None
                            if expected is not None: assert_metrics(metrics[head],expected)
                        diag=branch_diagnostics(truth,*[packet['heads'][h]['prediction'].numpy() for h in ('cnn','graph','fused')],list(range(1,classes+1)))
                        task[phase]={'heads':metrics,'complementarity':diag};packets[arm][name][phase]=packet
                    row['arms'][arm]['datasets'][name]=task
                    del outputs,item
                if state_digest(model)!=before or before!=bound['state_sha256']: raise ValueError('Readout changed classifier state')
                del model
            torch.save(packets,out/('predictions-'+str(seed)+'.pt'));write_json(out/('seed-'+str(seed)+'-analysis.json'),row);rows.append(row)
            print('EVALUATED_SEED',seed,flush=True)
        summary={}
        for arm in ARMS:
            values=[float(np.mean([r['arms'][arm]['datasets'][n]['test']['heads']['fused']['oa'] for n in NAMES]))*100 for r in rows]
            summary[arm]={'macro_oa_percent':{'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))},'datasets':{}}
            for name in NAMES:
                summary[arm]['datasets'][name]={}
                for metric in ('oa','aa','kappa','nll'):
                    v=[r['arms'][arm]['datasets'][name]['test']['heads']['fused'][metric] for r in rows]
                    summary[arm]['datasets'][name][metric]={'values':v,'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1))}
        base=summary[ARMS[0]];candidate=summary[ARMS[2]]
        paired=[b-a for a,b in zip(base['macro_oa_percent']['values'],candidate['macro_oa_percent']['values'])]
        aa_ok=all(candidate['datasets'][n]['aa']['mean']>=base['datasets'][n]['aa']['mean']-.005 for n in NAMES)
        for path,h in {**protected,**input_files,**reuse_hashes}.items():
            if file_hash(path)!=h: raise ValueError('Protected source/input/reuse receipt changed: '+path)
        require_frozen_code(code)
        artifacts={p.name:file_hash(p) for p in out.iterdir() if p.is_file()}
        write_json(out/'result.json',{'status':'passed','code':code,'summary':summary,'paired_macro_oa_delta_pp':paired,
            'gain_gate_passed':bool(np.mean(paired)>0 and sum(v>0 for v in paired)>=2 and aa_ok),
            'per_task_aa_guard_passed':bool(aa_ok),'metric_cases':243,'real_optimizer_updates':0,
            'new_test_readout':True,'historical_files_unchanged':len(protected),'inputs_unchanged':len(input_files),
            'elapsed_seconds':time.monotonic()-started,'artifacts':artifacts,
            'scope':'frozen matched benchmark; previously inspected split; not independent unseen-sensor confirmation'})
        print(json.dumps({'status':'passed','macro_test_oa_percent':{k:v['macro_oa_percent'] for k,v in summary.items()},
                          'paired_delta_pp':paired}),flush=True)
    except Exception as error:
        write_json(out/'failure.json',{'error':str(error),'type':type(error).__name__,'code':code});raise


if __name__=='__main__':main()
