import os,sys,json
from pathlib import Path
import torch
root=Path('/home/lbx/hsi-pioneer271828');repo=root/'checkouts/m4g4a-issue165';os.chdir(repo);sys.path.insert(0,str(repo))
from source_data_context import load_source_context
from target_data_context import load_target_context
from joint_training import _tree_fingerprint,load_plan
from target_initialization import initialize_target
from m4_transfer import initialize_m4_target
from evaluation.g4_target import independent_trainval
from research_runtime import code_snapshot,write_json
from loadData.dataset_registry import file_hash

out=root/'runs/m4g4a-issue165-real01';receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='passed'
code=code_snapshot();assert code==receipt['code'];launch=json.loads((out/'launch.json').read_text());targets=json.loads((out/'target-launch.json').read_text())['targets']
folders=[Path(c['directory']) for c in launch['sources']+targets]
protected={str(p):file_hash(p) for d in folders for p in d.iterdir() if p.is_file()}
prepared=root/'runs/base97a-honghu-issue97-20261103-prepared01';data_root=root/'datasets/snapshots/joint-honghu-20261103-issue97/dataset'
plan=load_plan('configs/experiments/base97_training.json');contexts={};count=recalls=0;checks=[]
for case in launch['sources']+targets:
    folder=Path(case['directory']);result=json.loads((folder/'result.json').read_text());metadata=json.loads((folder/'primary.pt.json').read_text())['training_config']
    state=torch.load(folder/json.loads((folder/'progress.json').read_text())['state_file'],map_location='cpu',weights_only=True)
    primary=torch.load(folder/'primary.pt',map_location='cpu',weights_only=True);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    saved=torch.load(folder/'selected-probabilities.pt',map_location='cpu',weights_only=True)
    assert _tree_fingerprint(primary)==_tree_fingerprint(state['best_model'])
    assert _tree_fingerprint(last)==state['state_fingerprints']['model']
    assert _tree_fingerprint(saved)==_tree_fingerprint(state['best_probabilities'])
    assert metadata['research_config']==result['settings']['config'] and metadata['training_code']==code
    assert metadata['checkpoint_optimizer_step']==result['selected_step']
    if case['kind']=='source':
        target=case['target']
        if target not in contexts:contexts[target]=load_source_context(prepared,data_root,held_out=target,consumer=code,manifest_sha256=plan['prepared_manifest_sha256'])
        data=contexts[target].items
        assert target not in data and not any(target in key.split('.') for key in primary)
        views={}
        for name,item in data.items():
            views[name]={'class_num':item['class_num']}
            for phase in ('train','val'):
                gt=item[phase+'_samples_gt'];views[name][phase+'_labels']=gt[gt>0].long()-1
    else:
        command=case['command'];args=dict(zip(command[2::2],command[3::2]))
        data,prov,guard=load_target_context(case['target'],args['--observation'],args['--observation-record'],args['--observation-record-sha256'],
            args['--label-family'],args['--label-family-sha256'],per_class=10,support_seed=64)
        views=data;kwargs={}
        if '--source-path' in args:kwargs=dict(source_path=args['--source-path'],source_weight_sha256=args['--source-weight-sha256'],source_sidecar_sha256=args['--source-sidecar-sha256'])
        initial,record=initialize_m4_target(case['target'],case['method'],case['mode'],model_seed=64,head_seed=1064,device='cpu',**kwargs)
        assert record['initial_state_sha256']==metadata['target_initialization']['initial_state_sha256'];del initial;guard()
        assert tuple(metadata['research_config']['data']['datasets'])==(case['target'],)
    for name,item in views.items():
        checked=independent_trainval(saved[name],item,result['selected_validation']['metrics'][name])
        count+=checked['head_metrics'];recalls+=checked['class_recalls']
    checks.append({'kind':case['kind'],'name':case['name'],'state_and_metrics':'passed'})
    print('M4G4A_INDEPENDENT_CASE_PASSED',case['kind'],case['name'],flush=True)
    del state,primary,last,saved
assert all(file_hash(p)==h for p,h in protected.items())
assert all(file_hash(p)==h for p,h in receipt['protected_inputs'].items())
assert code_snapshot()==code
write_json(out/'independent-receipt.json',{'status':'passed','code':code,'cases':checks,'head_metric_cases':count,'class_recalls':recalls,
    'training_files_unchanged':len(protected),'protected_inputs_unchanged':len(receipt['protected_inputs']),
    'analysis_optimizer_updates':0,'model_test_scoring':False,'query_labels_accessed':False})
print('M4G4A_INDEPENDENT_ALL_PASSED',count,recalls,flush=True)
