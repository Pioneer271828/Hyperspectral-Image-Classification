import os,sys,json,time,subprocess
from pathlib import Path
root=Path('/home/lbx/hsi-pioneer271828');repo=root/'checkouts/m4g4a-issue165'
os.chdir(repo);sys.path.insert(0,str(repo))
from joint_baseline import NAMES
from joint_training import load_plan
from loadData.dataset_registry import file_hash
from source_data_context import load_source_context
from target_data_context import load_target_context
from m4_transfer import MODES,source_options,target_options
from target_initialization import METHODS
from research_runtime import code_snapshot,write_json

SHA='77e2a3f45ad5c7e8e319221da69d61af0f2ecf80'
code=code_snapshot();assert code['commit']==SHA and not code['dirty']
out=root/'runs/m4g4a-issue165-real01';out.mkdir(exist_ok=False)
for name in ('sources','targets','logs','reviews'):(out/name).mkdir()
prepared=root/'runs/base97a-honghu-issue97-20261103-prepared01'
data_root=root/'datasets/snapshots/joint-honghu-20261103-issue97/dataset'
observations=root/'runs/g4d-issue151-observations02';families=root/'runs/g4c-issue149-labels01'
plan=load_plan('configs/experiments/base97_training.json');sources=[];targets=[];pins={}
for target in NAMES:
    context=load_source_context(prepared,data_root,held_out=target,consumer=code,manifest_sha256=plan['prepared_manifest_sha256'])
    pins.update(context.record['protected_files']);del context
    graph=observations/target;family=families/(target+'-64')/'curator-family.json'
    items,prov,guard=load_target_context(target,graph/'observation.pt',graph/'record.json',file_hash(graph/'record.json'),family,file_hash(family),per_class=10,support_seed=64)
    pins.update(prov['protected_files']);del items
    for mode in MODES:
        name=target+'-'+mode;folder=out/'sources'/name
        command=[sys.executable,'scripts/train_m4_source_short.py','--expected-commit',SHA,'--prepared-dir',str(prepared),
            '--data-root',str(data_root),'--held-out',target,'--mode',mode,'--seed','64','--out-dir',str(folder)]
        sources.append({'kind':'source','name':name,'target':target,'mode':mode,'directory':str(folder),'command':command})
        for method in METHODS[:2]:
            target_name=name+'-'+method;folder_t=out/'targets'/target_name
            cmd=[sys.executable,'scripts/train_m4_target_short.py','--expected-commit',SHA,'--target',target,'--method',method,
                '--mode',mode,'--model-seed','64','--support-seed','64','--support-per-class','10','--device','cuda',
                '--observation',str(graph/'observation.pt'),'--observation-record',str(graph/'record.json'),
                '--observation-record-sha256',file_hash(graph/'record.json'),'--label-family',str(family),
                '--label-family-sha256',file_hash(family),'--out-dir',str(folder_t)]
            targets.append({'kind':'target','name':target_name,'target':target,'mode':mode,'method':method,
                'source_directory':str(folder),'directory':str(folder_t),'command':cmd})
write_json(out/'launch.json',{'source':code,'sources':sources,'targets':targets,'protected_inputs':pins,
    'declared_source_updates':72,'declared_target_updates':216,'query_labels_accessed':False,'model_test_scoring':False})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
reviews=[];started=time.monotonic()
def run(case):
    start=time.monotonic();command=case['command'];folder=Path(case['directory']);log=out/'logs'/(case['kind']+'-'+case['name']+'.log')
    with log.open('x') as stream:
        split=(case['kind']=='source' and case['target']=='xuzhou' and case['mode']=='x_residual') or (
            case['kind']=='target' and case['mode']=='x_residual' and case['method']==METHODS[1])
        if split:
            subprocess.run(command+['--stop-after','3' if case['kind']=='source' else '5'],stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
            subprocess.run(command+['--resume'],stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
        else:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
        before={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
        subprocess.run(command+['--resume'],stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
        assert before=={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}
    result=json.loads((folder/'result.json').read_text());trace=json.loads((folder/'trace.json').read_text());history=json.loads((folder/'history.json').read_text())
    expected=8 if case['kind']=='source' else 12
    assert result['status']=='completed' and result['optimizer_steps']==len(trace)==expected and result['start_code']==code
    selected=(max(history,key=lambda r:r['macro_oa']) if case['kind']=='source' else min(history,key=lambda r:r['val_loss']))
    assert result['selected_step']==selected['step']
    if case['mode']!='none':assert sum(r['m4_changed_elements'] for r in trace)>0 and any(r['m4_parameter_gradient_norm']>0 for r in trace)
    review={'status':'completed','kind':case['kind'],'name':case['name'],'optimizer_updates':expected,
        'selected_step':result['selected_step'],'first_loss':trace[0]['loss']['total'],'last_loss':trace[-1]['loss']['total'],
        'first_validation_nll':history[0]['val_loss'],'last_validation_nll':history[-1]['val_loss'],
        'selected_validation':result['selected_validation'],'native_reload_max_abs_error':result['selected_probability_max_abs_error'],
        'elapsed_seconds':time.monotonic()-start,'completed_files_unchanged':len(before)}
    reviews.append(review);write_json(out/'reviews'/(case['kind']+'-'+case['name']+'.json'),review)
    write_json(out/'progress.json',{'completed_source':sum(r['kind']=='source' for r in reviews),'completed_target':sum(r['kind']=='target' for r in reviews)})
    print('M4G4A_REAL_CASE_PASSED',case['kind'],case['name'],flush=True)
for case in sources:run(case)
source_files={str(p):file_hash(p) for c in sources for p in Path(c['directory']).iterdir() if p.is_file()}
for case in targets:
    if case['method']!=METHODS[0]:
        d=Path(case['source_directory']);r=json.loads((d/'result.json').read_text())
        case['command']+=['--source-path',str(d/'primary.pt'),'--source-weight-sha256',r['artifacts']['primary.pt'],
            '--source-sidecar-sha256',r['artifacts']['primary.pt.json']]
write_json(out/'target-launch.json',{'targets':targets,'source_files':source_files,'code':code,'optimizer_updates':0})
for case in targets:run(case)
assert all(file_hash(p)==h for p,h in pins.items());assert all(file_hash(p)==h for p,h in source_files.items());assert code_snapshot()==code
write_json(out/'receipt.json',{'status':'passed','code':code,'cases':reviews,'optimizer_updates':288,
    'source_updates':72,'target_updates':216,'elapsed_seconds':time.monotonic()-started,
    'protected_inputs':pins,'source_files_unchanged':source_files,'query_labels_accessed':False,'model_test_scoring':False})
print('M4G4A_REAL_ALL_PASSED',len(reviews),flush=True)
