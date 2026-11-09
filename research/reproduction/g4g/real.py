import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/home/lbx/hsi-pioneer271828');repo=root/'checkouts/g4g-issue159'
sys.path.insert(0,str(repo));os.chdir(repo)
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from research_runtime import write_json,code_snapshot
from target_initialization import METHODS

SHA='b29f7b9f1a99c1245d593dcd167404d3f953d450'
code=code_snapshot();assert code['commit']==SHA and not code['dirty']
out=root/'runs/g4g-issue159-short01';out.mkdir(exist_ok=False)
cases=[];pins={}
for name in NAMES:
    graph=root/'runs/g4d-issue151-observations02'/name
    family=root/'runs/g4c-issue149-labels01'/(name+'-64')/'curator-family.json'
    source=root/'runs/g4p1-issue153-full01/cases'/(name+'-s64')
    result=json.loads((source/'result.json').read_text());assert result['status']=='completed' and result['optimizer_steps']==1002
    assert result['start_code']['commit']=='9c66e20f85d26e88ecb9e6790d8c1e2c62ae8ed1'
    for key in ('primary.pt','primary.pt.json'):assert file_hash(source/key)==result['artifacts'][key]
    for p in (graph/'observation.pt',graph/'record.json',family,source/'primary.pt',source/'primary.pt.json'):pins[str(p)]=file_hash(p)
    for method in METHODS:
        command=[sys.executable,'scripts/train_target_short.py','--expected-commit',SHA,
            '--target',name,'--method',method,'--model-seed','64','--support-seed','64','--support-per-class','10',
            '--device','cuda','--out-dir',str(out/(name+'-'+method)),
            '--observation',str(graph/'observation.pt'),'--observation-record',str(graph/'record.json'),
            '--observation-record-sha256',pins[str(graph/'record.json')],
            '--label-family',str(family),'--label-family-sha256',pins[str(family)]]
        if method!=METHODS[0]:command+=['--source-path',str(source/'primary.pt'),
            '--source-weight-sha256',result['artifacts']['primary.pt'],'--source-sidecar-sha256',result['artifacts']['primary.pt.json']]
        cases.append({'target':name,'method':method,'command':command,'directory':str(out/(name+'-'+method))})
write_json(out/'launch.json',{'code':code,'cases':cases,'protected_files':pins,'query_label_access':False,'declared_updates':108})
start=time.monotonic();receipts=[]
for case in cases:
    t=time.monotonic();log=out/(case['target']+'-'+case['method']+'.log')
    with log.open('w') as stream:
        cmd=case['command']
        if case['method']==METHODS[2]:
            subprocess.run(cmd+['--stop-after','5'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run(cmd+['--resume'],stdout=stream,stderr=subprocess.STDOUT,check=True)
        else:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
        directory=Path(case['directory']);before={str(p):file_hash(p) for p in directory.iterdir() if p.is_file()}
        subprocess.run(cmd+['--resume'],stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert before=={str(p):file_hash(p) for p in directory.iterdir() if p.is_file()}
    result=json.loads((directory/'result.json').read_text());assert result['status']=='completed' and result['optimizer_steps']==12
    receipts.append({'target':case['target'],'method':case['method'],'elapsed_seconds':time.monotonic()-t,
        'selected_step':result['selected_step'],'max_reload_error':result['selected_probability_max_abs_error'],
        'selected_validation':result['selected_validation'],'completed_files_unchanged':len(before)})
    print('G4G_REAL_CASE_PASSED',case['target'],case['method'],round(receipts[-1]['elapsed_seconds'],2),flush=True)
    write_json(out/'progress.json',{'completed':len(receipts),'total':len(cases),'receipts':receipts})
assert all(file_hash(p)==h for p,h in pins.items());assert code_snapshot()==code
write_json(out/'receipt.json',{'status':'passed','code':code,'cases':receipts,'elapsed_seconds':time.monotonic()-start,
    'optimizer_updates':108,'protected_files':pins,'protected_files_unchanged':True,'query_label_accessed':False,'model_test_scoring':False})
print('G4G_REAL_ALL_PASSED',len(receipts),flush=True)
