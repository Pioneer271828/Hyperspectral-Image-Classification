"""Standalone CPU reader of this B package's copied packets and predictions; never build a model."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
torch.set_num_threads(1)
if int(np.__version__.split('.')[0]) < 2:
    import numpy.core, numpy.core.multiarray, numpy.core.numeric, numpy.core._multiarray_umath
    for suffix in ('', '.multiarray', '.numeric', '._multiarray_umath'):
        sys.modules['numpy._core'+suffix] = sys.modules['numpy.core'+suffix]


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
def tensor_hash(value):
    tensor=value.detach().cpu();header={'dtype':str(tensor.dtype),'shape':list(tensor.shape)}
    result=hashlib.sha256(canonical(header).encode())
    if tensor.numel():result.update(memoryview(tensor.contiguous().numpy()).cast('B'))
    return result.hexdigest()


def exact(left,right):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype!=right.dtype or left.shape!=right.shape:return False
        if left.is_sparse:return exact(left.coalesce().indices(),right.coalesce().indices()) and exact(left.coalesce().values(),right.coalesce().values())
        return torch.equal(left,right)
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(exact(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):return type(left)==type(right) and len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    return left==right


def recount(points,metrics,classes):
    grouped={}
    for point in points:grouped.setdefault((point['head'],point['band']),[]).append(point)
    maximum=0.
    for (head,band),rows in grouped.items():
        matrix=[[0]*classes for _ in range(classes)]
        for row in rows:matrix[row['class_id']-1][max(range(classes),key=row['probabilities'].__getitem__)]+=1
        nll=math.fsum(-math.log(max(row['probabilities'][row['class_id']-1],1e-8)) for row in rows)/len(rows)
        assert matrix==metrics[head][band]['confusion_matrix']
        error=abs(nll-metrics[head][band]['nll']);assert error<=1e-12;maximum=max(maximum,error)
    return len(grouped),maximum


def main():
    root,old,out=map(Path,sys.argv[1:4]);surface=sys.argv[4]
    packets=cases=matrices=buffer_hashes=0;maximum=0.
    for folder in ('matrix-v1-output','balanced-v2-output'):
        summary=load(root/folder/'summary.json');assert summary['status']=='passed' and len(summary['cases'])==48
        for case in summary['cases']:
            directory=root/folder/case['case'];spec=load(directory/'registration.json')
            assert spec['synthetic'] and spec['steps']==24 and spec['events']==[0,12,24]
            states={}
            for phase,updates in (('full',24),('prefix',12),('resume',12),('completed',0)):
                receipt=load(directory/(phase+'.json'));file=directory/(phase+'.pt')
                assert receipt['updates']==updates and sha(file)==receipt['checkpoint']['sha256']
                value=torch.load(file,map_location='cpu',weights_only=False);states[phase]=value;packets+=1
                for name,expected in spec.get('fixed_buffers',{}).items():
                    assert tensor_hash(value['model'][name])==expected;buffer_hashes+=1
            assert exact(states['full'],states['resume']) and exact(states['full'],states['completed'])
            assert exact(states['full']['history'][:12],states['prefix']['history'])
            assert states['full']['cursor']==24 and all(int(v['step'])==24 for v in states['full']['optimizer']['state'].values())
            audit=load(directory/'independent.json')
            for endpoint in audit['endpoints'].values():
                count,error=recount(endpoint['original_point_predictions'],endpoint['metrics'],case['classes'])
                matrices+=count;maximum=max(maximum,error)
            cases+=1
    real=load(root/'real-probe-v1-output/_private/point-records.json')
    historical=load(old/'wp07-b-fine16-20261126/xuzhou/pair/execution/xuzhou-512-content/independent.json')['endpoints']['selected']['original_point_predictions']
    key=lambda p:(p['tile'],p['released_ordinal'],p['head'],p['band'])
    index={key(p):p for p in historical};reference=real['records'][0]['points']
    assert len(index)==len(reference)==270
    assert all(row['probabilities']==index[key(row)]['probabilities'] and row['class_id']==index[key(row)]['class_id'] for row in reference)
    for record in real['records']:
        count,error=recount(record['points'],record['metrics'],9);matrices+=count;maximum=max(maximum,error)
        if record['neutral']:assert record['points']==reference
    assert packets==384 and cases==96 and matrices==2364 and buffer_hashes==384
    result=dict(status='passed_actual_copy_CPU_read_and_independent_recount',execution_surface=surface,
        synthetic_cases=cases,actual_model_Adam_RNG_packets_loaded=packets,full_split_completed_exact_cases=cases,
        fixed_rotation_buffer_hashes_verified=buffer_hashes,endpoint_confusions_recounted=matrices,
        NLL_max_absolute_error=maximum,historical_real_probability_vectors_bitwise_matched=270,
        old_real_checkpoint_models_represented=1,classifier_forwards=0,optimizer_steps=0,
        python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
        NumPy2_pickle_aliases_process_local=int(np.__version__.split('.')[0])<2,reader_sha256=sha(Path(__file__)),
        original_LXC_files_read=surface=='source-LXC',
        scope='Actual saved-copy readability, exact already-saved recovery-route comparison and scalar prediction recount. No cross-host trainer continuation or GPU numerical qualification.')
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
