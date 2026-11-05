import json,sys,os
from pathlib import Path
import numpy as np
import torch
root=Path('/home/lbx/hsi-pioneer271828');repo=root/'checkouts/g4g-issue159'
os.chdir(repo);sys.path.insert(0,str(repo))
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from model.research_config import digest
from research_runtime import write_json,code_snapshot
from target_training import TargetRuntime,target_settings,HEAD_ONLY
from target_data_context import load_target_context

out=root/'runs/g4g-issue159-short01';receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='passed'
launch=json.loads((out/'launch.json').read_text());assert code_snapshot()==launch['code']
protected={str(p):file_hash(p) for case in launch['cases'] for p in Path(case['directory']).iterdir() if p.is_file()}
checks=[];metrics_count=recall_count=0
for case in launch['cases']:
    directory=Path(case['directory']);target=case['target'];method=case['method']
    cmd=case['command'];args=dict(zip(cmd[2::2],cmd[3::2]))
    # Parse named pairs explicitly; command has only scalar named options.
    assert all(k.startswith('--') for k in args)
    data,provenance,guard=load_target_context(target,args['--observation'],args['--observation-record'],
        args['--observation-record-sha256'],args['--label-family'],args['--label-family-sha256'],per_class=10,support_seed=64)
    opts=target_settings(target,method,seed=64,support_seed=64,per_class=10,device='cuda')
    spec={} if '--source-path' not in args else dict(source_path=args['--source-path'],
        source_weight_sha256=args['--source-weight-sha256'],source_sidecar_sha256=args['--source-sidecar-sha256'])
    runtime=TargetRuntime(opts,data,directory,source_spec=spec)
    result=json.loads((directory/'result.json').read_text());step=result['selected_step']
    model=runtime.load_selected(directory/'primary.pt',data,directory/f'raw-validation-{step:06d}.pt',device='cuda')
    metric,values=runtime.validation_snapshot(model,data)
    stored=torch.load(directory/'selected-probabilities.pt',map_location='cpu',weights_only=True)
    cm_checks=[]
    for phase in ('train','val'):
        truth=data[target][phase+'_labels'].numpy();classes=data[target]['class_num']
        for head,p in values[target][phase].items():
            torch.testing.assert_close(p,stored[target][phase][head],rtol=1e-5,atol=1e-6)
            assert torch.equal(p.argmax(1),stored[target][phase][head].argmax(1))
            v=p.numpy();pred=v.argmax(1);cm=np.zeros((classes,classes),np.int64);np.add.at(cm,(truth,pred),1)
            recalls=cm.diagonal()/cm.sum(1);oa=float(cm.trace()/cm.sum());aa=float(recalls.mean())
            nll=float(-np.log(np.clip(v[np.arange(len(truth)),truth],1e-8,1)).mean())
            expected=result['selected_validation']['metrics'][target][phase][head]
            assert expected['confusion']==cm.tolist()
            np.testing.assert_allclose(expected['recall'],recalls,rtol=0,atol=1e-12)
            np.testing.assert_allclose([expected['oa'],expected['aa'],expected['nll']],[oa,aa,nll],rtol=1e-5,atol=1e-6)
            metrics_count+=1;recall_count+=classes;cm_checks.append({'phase':phase,'head':head,'oa':oa,'aa':aa,'nll':nll})
    history=json.loads((directory/'history.json').read_text());trace=json.loads((directory/'trace.json').read_text())
    assert len(trace)==12 and [e['step'] for e in history]==[1,4,7,10,12]
    assert min(history,key=lambda e:e['val_loss'])['step']==step
    assert all(e['selection_score']==-e['val_loss'] for e in history)
    if method==HEAD_ONLY:
        assert all(not p.requires_grad for p in model.backbone.parameters())
        assert all(t['graph_changed_elements']==0 and t['gradient_norms']['adapter']==0 and t['gradient_norms']['graph']==0 for t in trace)
        assert all(t['feature_bank_sha256']==file_hash(directory/'feature-bank.json') for t in trace)
    else:assert all(t['graph_changed_elements']>0 and t['gradient_norms']['graph']>0 and t['gradient_norms']['adapter']>0 for t in trace)
    guard();checks.append({'target':target,'method':method,'selected_step':step,'metrics':cm_checks})
    print('G4G_INDEPENDENT_CASE_PASSED',target,method,flush=True)
    del model,runtime;torch.cuda.empty_cache()
assert all(file_hash(p)==h for p,h in protected.items())
assert all(file_hash(p)==h for p,h in receipt['protected_files'].items());assert code_snapshot()==launch['code']
write_json(out/'independent-receipt.json',{'status':'passed','cases':checks,'head_metric_cases':metrics_count,
    'class_recalls':recall_count,'training_files_unchanged':len(protected),
    'protected_inputs_unchanged':len(receipt['protected_files']),'analysis_optimizer_updates':0,'model_test_scoring':False})
print('G4G_INDEPENDENT_ALL_PASSED',metrics_count,recall_count,flush=True)
