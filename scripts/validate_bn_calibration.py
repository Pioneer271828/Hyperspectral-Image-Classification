"""BN-ALIGN1: frozen three-seed, train/val-only calibration acceptance."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from joint_baseline import NAMES,forward,readout
from joint_training import load_plan,import_preparation
from loadData.dataset_registry import file_hash
from model.bn_calibration import CONTEXT,POLICY,calibrate_denoise,state_digest,verify_calibration_binding
from model.checkpoint_io import load_model_checkpoint,save_model_checkpoint
from research_runtime import code_snapshot,external_directory,require_frozen_code,write_json


def double_reference_check(model,items,checkpoint_hash):
    """Strict formula control; production execution and artifacts stay FP32."""
    source=copy.deepcopy(model).double()
    obs={n:{k:v.double() if torch.is_tensor(v) else v for k,v in i.items()
            if k in ('net_input','wavelengths','fwhm')} for n,i in items.items()}
    calibrated,_=calibrate_denoise(source,obs,mode=POLICY,context=CONTEXT,
        source_checkpoint_sha256=checkpoint_hash)
    live=copy.deepcopy(source)
    for block in live.backbone.CNN_denoise:
        for bn in block.norm.norm.norms.values():bn.training=True;bn.track_running_stats=False
    maximum=0.
    with torch.no_grad():
        for name,raw in items.items():
            item={k:v.double() if torch.is_tensor(v) else v for k,v in raw.items()}
            # Legacy dense conversion fixes FP32. Sparse representation preserves
            # the exact same coefficients for this numerical-only FP64 control.
            dense=item['A'];item['A']=dense.to_sparse().coalesce()
            if not torch.equal(item['A'].to_dense(),dense):raise ValueError('Graph values changed')
            a,b=forward(calibrated,name,item),forward(live,name,item)
            for phase in ('train','val'):
                _,q=readout(a,item,phase);_,z=readout(b,item,phase)
                for h in q:
                    torch.testing.assert_close(q[h],z[h],rtol=1e-9,atol=1e-10)
                    if not torch.equal(q[h].argmax(1),z[h].argmax(1)):raise ValueError('FP64 predictions differ')
                    maximum=max(maximum,float((q[h]-z[h]).abs().max()))
    return maximum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','campaign-dir','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean checkout at published expected full SHA required')
    out=external_directory(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();batch=Path(a.campaign_dir);load=lambda path:json.loads(Path(path).read_text())
    try:
        launch=load(batch/'launch.json')
        if (launch['code']['commit']!='8daa65b116ba585b33a84e3af93f3741716b710a'
                or load(batch/'status.json')['status']!='training_completed'):
            raise ValueError('Accepted BASE97C completed campaign required')
        for name,h in launch['code']['files'].items():
            if source['files'].get(name)!=h:raise ValueError('Existing trained implementation changed: '+name)
        _,items,provenance=import_preparation(Path(a.prepared_dir),a.data_root,
            load_plan('configs/experiments/base97_training.json'),source)
        observations={n:{k:item[k] for k in ('net_input','wavelengths','fwhm')} for n,item in items.items()}
        original_files={};inputs={}
        for seed in (64,512,2048):
            r=batch/('seed-'+str(seed));v=load(r/'result.json')
            if v['status']!='completed' or v['optimizer_steps']!=1503 or v['identity']['prepared']!=provenance:
                raise ValueError('Wrong source run identity/budget')
            for name,h in v['artifacts'].items():
                if file_hash(r/name)!=h:raise ValueError('Source artifact changed: '+name)
            original_files[str(seed)]={f.name:file_hash(f) for f in r.iterdir() if f.is_file()}
        for f in Path(a.prepared_dir).iterdir():
            if f.is_file():inputs[str(f)]=file_hash(f)
        manifest=load(Path(a.prepared_dir)/'manifest.json')
        for value in manifest['datasets'].values():
            for role in ('image','label'):
                f=Path(a.data_root)/value['raw'][role+'_path'];inputs[str(f)]=file_hash(f)
        write_json(out/'plan.json',{'code':source,'comparisons':['baseline','calibrated_reload','live_reference','wrong_domain'],
            'source_files':original_files,'input_files':inputs,'calibration_context':CONTEXT,
            'seed_order':[64,512,2048],'optimizer_updates':0,'test_scoring':False})
        rows=[]
        device_items={n:{k:v.to(a.device) if torch.is_tensor(v) else v for k,v in item.items()} for n,item in items.items()}
        for seed in (64,512,2048):
            r=batch/('seed-'+str(seed));source_result=load(r/'result.json');model=load_model_checkpoint(r/'primary.pt',device=a.device)
            untouched=state_digest(model)
            candidate,receipt=calibrate_denoise(model,observations,mode=POLICY,context=CONTEXT,
                source_checkpoint_sha256=file_hash(r/'primary.pt'))
            folder=out/('seed-'+str(seed));folder.mkdir()
            save_model_checkpoint(candidate,folder/'calibrated.pt')
            reloaded=load_model_checkpoint(folder/'calibrated.pt',device=a.device)
            verify_calibration_binding(reloaded,observations,context=CONTEXT,source_checkpoint_sha256=file_hash(r/'primary.pt'))
            if state_digest(reloaded)!=state_digest(candidate):raise ValueError('Reloaded calibration state changed')
            live=copy.deepcopy(model)
            for block in live.backbone.CNN_denoise:
                for bn in block.norm.norm.norms.values():bn.training=True;bn.track_running_stats=False
            wrong=copy.deepcopy(candidate)
            with torch.no_grad():
                for block,original in zip(wrong.backbone.CNN_denoise,candidate.backbone.CNN_denoise):
                    for i,name in enumerate(NAMES):
                        other=original.norm.norm.norms[NAMES[(i+1)%3]]
                        block.norm.norm.norms[name].running_mean.copy_(other.running_mean)
                        block.norm.norm.norms[name].running_var.copy_(other.running_var)
            try:verify_calibration_binding(wrong,observations,context=CONTEXT,source_checkpoint_sha256=file_hash(r/'primary.pt'))
            except ValueError:pass
            else:raise ValueError('Wrong-domain control passed checkpoint binding')
            metrics={};max_reload=0.;max_reference=0.;saved={}
            stable={key:state_digest(m) for key,m in [('candidate',candidate),('reload',reloaded),('live',live),('wrong',wrong)]}
            with torch.no_grad():
                for name,item in device_items.items():
                    outputs={key:forward(m,name,item) for key,m in [('baseline',model),('calibrated',candidate),
                        ('calibrated_reload',reloaded),('live_reference',live),('wrong_domain',wrong)]}
                    metrics[name]={};saved[name]={}
                    for phase in ('train','val'):
                        metrics[name][phase]={};saved[name][phase]={}
                        for key,q in outputs.items():
                            metrics[name][phase][key],saved[name][phase][key]=readout(q,item,phase)
                        for head in ('fused','cnn','graph'):
                            if metrics[name][phase]['baseline'][head]['confusion']!=source_result['selected_validation']['metrics'][name][phase][head]['confusion']:
                                raise ValueError('Original selected classifier metrics changed')
                            c=saved[name][phase]['calibrated'][head];z=saved[name][phase]['calibrated_reload'][head]
                            live_q=saved[name][phase]['live_reference'][head]
                            torch.testing.assert_close(c,z,rtol=1e-5,atol=1e-6)
                            torch.testing.assert_close(c,live_q,rtol=1e-4,atol=1e-4)
                            if not torch.equal(c.argmax(1),z.argmax(1)) or not torch.equal(c.argmax(1),live_q.argmax(1)):
                                raise ValueError('Calibration/reload/reference supervised predictions differ')
                            max_reload=max(max_reload,float((c-z).abs().max()));max_reference=max(max_reference,float((c-live_q).abs().max()))
            if state_digest(model)!=untouched:raise ValueError('Source model changed')
            for key,m in [('candidate',candidate),('reload',reloaded),('live',live),('wrong',wrong)]:
                if state_digest(m)!=stable[key]:raise ValueError('Inference changed state: '+key)
            double_error=double_reference_check(model,device_items,file_hash(r/'primary.pt'))
            torch.save(saved,folder/'probabilities.pt')
            row={'seed':seed,'metrics':metrics,'calibration':receipt,'source_state_unchanged':True,
                 'reload_max_abs_error':max_reload,'reference_max_abs_error':max_reference,
                 'fp64_reference_max_abs_error':double_error}
            write_json(folder/'review.json',row);rows.append(row)
            print('SEED_COMPLETE',seed,'macro_val', {k:float(np.mean([metrics[n]['val'][k]['fused']['oa'] for n in NAMES]))
                   for k in ('baseline','calibrated_reload','live_reference','wrong_domain')},flush=True)
        summary={}
        for phase in ('train','val'):
            summary[phase]={}
            for key in ('baseline','calibrated_reload','live_reference','wrong_domain'):
                values=[float(np.mean([row['metrics'][n][phase][key]['fused']['oa'] for n in NAMES])) for row in rows]
                summary[phase][key]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        before=summary['val']['baseline']['values'];after=summary['val']['calibrated_reload']['values']
        aa_ok=all(np.mean([r['metrics'][n]['val']['calibrated_reload']['fused']['aa'] for r in rows])>=
                  np.mean([r['metrics'][n]['val']['baseline']['fused']['aa'] for r in rows]) for n in NAMES)
        decision=sum(after)/3>=sum(before)/3 and sum(b>a for a,b in zip(before,after))>=2 and aa_ok
        for seed,files in original_files.items():
            r=batch/('seed-'+seed)
            if files!={f.name:file_hash(f) for f in r.iterdir() if f.is_file()}:raise ValueError('Historical training files changed')
        for path,h in inputs.items():
            if file_hash(path)!=h:raise ValueError('Raw/prepared input changed')
        require_frozen_code(source)
        write_json(out/'result.json',{'status':'passed','code':source,'summary':summary,'candidate_gate_passed':bool(decision),
            'optimizer_updates':0,'test_scoring':False,'source_training_files_unchanged':sum(map(len,original_files.values())),
            'raw_prepared_files_unchanged':len(inputs),'elapsed_seconds':time.monotonic()-started,
            'scope':'reused_validation_frozen_primary_transductive_only'})
        print(json.dumps({'status':'passed','summary':summary,'candidate_gate_passed':bool(decision)}),flush=True)
    except Exception as error:
        write_json(out/'failure.json',{'type':type(error).__name__,'error':str(error),'code':source});raise


if __name__=='__main__':main()
