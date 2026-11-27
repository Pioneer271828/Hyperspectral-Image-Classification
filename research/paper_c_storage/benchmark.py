"""Bounded existing-synthetic corpus measurement; no neural calls or updates."""
import argparse
import copy
import io
import json
from pathlib import Path
import pickle
import shutil
import statistics
import subprocess
from time import perf_counter
import zipfile
import zlib

import numpy as np
import torch

from .store import Store
from .codec import hashed,require
from .bridge import ROOT,import_case

BASE=Path('/home/lbx/hsi-pioneer271828')


def clean(expected):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==expected
            and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Clean expected source SHA required')


def tensor_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if torch.is_tensor(value):
        storage=value.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
        if key in seen:return 0
        seen.add(key);return storage.nbytes()
    if isinstance(value,dict):return sum(tensor_bytes(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v,seen) for v in value)
    return 0


def components(path):
    raw=path.read_bytes();state=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        pickle_bytes=next(i.file_size for i in z.infolist() if i.filename.endswith('/data.pkl'))
        storage_bytes=sum(i.file_size for i in z.infolist() if '/data/' in i.filename)
    seen=set();parts={k:tensor_bytes(state[k],seen) for k in ['model','optimizer','rng']}
    require(sum(parts.values())==storage_bytes,'Tensor storage category partition differs')
    # Isolated pickle probes are explicitly NOT additive parts of data.pkl.
    probes={k:len(pickle.dumps(state[k],protocol=2)) for k in ['best','selection_events','module_modes','step']}
    #9 selections and5 dev centers/class: generated metadata only, no scoring.
    classes=len(state['selection_events'][-1]['metrics']['support']);rng=np.random.default_rng(1439)
    events=[]
    for i in range(9):
        truth=np.repeat(np.arange(classes),5);pred=rng.integers(0,classes,len(truth));cm=np.bincount(truth*classes+pred,minlength=classes**2).reshape(classes,classes)
        events.append({'step':i*200,'metrics':{'nll':float(2+i/100),'oa':float(np.trace(cm)/len(truth)),
                       'confusion':cm.tolist(),'support':cm.sum(1).tolist(),'recall':(np.diag(cm)/cm.sum(1)).tolist(),
                       'prediction':pred.tolist(),'points':len(truth)}})
    probe=copy.copy(state);probe['selection_events']=events;probe['best']={'step':0,'nll':2.};probe['step']=4400
    out=io.BytesIO();torch.save(probe,out)
    with zipfile.ZipFile(out) as z:new_pickle=next(i.file_size for i in z.infolist() if i.filename.endswith('/data.pkl'))
    return {'file_bytes':len(raw),'model_tensor_storage_bytes':parts['model'],'Adam_tensor_storage_bytes':parts['optimizer'],
            'RNG_tensor_storage_bytes':parts['rng'],'all_tensor_storage_bytes':storage_bytes,'data_pickle_bytes':pickle_bytes,
            'zip_headers_padding_other_bytes':len(raw)-storage_bytes-pickle_bytes,'isolated_nonadditive_pickle_probes':probes,
            'future_metadata_only_size_probe_pickle_bytes':new_pickle,'future_metadata_growth_raw_bytes':max(0,new_pickle-pickle_bytes)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();clean(a.expected_commit);torch.set_num_threads(1)
    source=Path(a.source).resolve();out=Path(a.out).resolve()
    require(source==BASE/'runs/paper-c-spatial-baseline-20261127-01/profile03','Only already-generated C13 synthetic shape corpus admitted')
    require(out.is_relative_to(BASE/'runs') and not out.exists(),'New authorized run output required')
    require(shutil.disk_usage(out.parent).free>=768*1024**2,'Insufficient bounded pilot workspace')
    cases=sorted(p for p in source.iterdir() if p.is_dir());require(len(cases)==12,'Expected12 frozen shape cases')
    out.mkdir(mode=0o700);inventory=[];components_rows=[];whole={};baseline_zlib=0
    for case in cases:
        c=json.loads((case/'contract.json').read_text());require(c['scope']=='synthetic cost profile only' and c['updates']==4 and c['query_reads']==0,'Non-synthetic source rejected')
        for step in range(5):
            folder=case/'stages'/f'{step:06d}'
            for f in sorted(folder.iterdir()):
                require(f.name in ['state.pt','record.json','intent.json'],'Unexpected stage member')
                raw=f.read_bytes();inventory.append({'case':case.name,'step':step,'name':f.name,'sha256':hashed(raw),'bytes':len(raw)})
                if f.name=='state.pt':whole.setdefault(hashed(raw),len(raw));baseline_zlib+=len(zlib.compress(raw,1))
            if step in [0,4]:components_rows.append({'case':case.name,'step':step,**components(folder/'state.pt')})
    results={};projected={}
    for layout in ['chunk-zlib','zip-xor']:
        store=Store(out/layout,layout=layout,max_bytes=512*1024**2);rows=[];t=perf_counter();verify_seconds=0.;read_before=store.read_bytes
        for case in cases:
            row=import_case(store,case,case.name);meta=json.loads((case/'contract.json').read_text())
            last=store.frames(case.name);order=np.random.default_rng(991).permutation(5)
            for step in order:
                t0=perf_counter()
                for name in ['state.pt','record.json','intent.json']:
                    require(store.decode_file(last[step],name)==(case/'stages'/f'{step:06d}'/name).read_bytes(),'Random-access original bytes differ')
                verify_seconds+=perf_counter()-t0
            row['scene']=meta['scene'];row['method']=meta['method'];rows.append(row)
        object_stats=[{'codec':c,'depth':d,'objects':n,'payload_bytes':b} for c,d,n,b in store.db.execute('SELECT codec,depth,count(*),sum(length(payload)) FROM objects GROUP BY codec,depth')]
        store.checkpoint();physical=store.bytes();allocated=store.allocated();elapsed=perf_counter()-t
        # Per-shape marginal growth after initial Adam creation, not a
        # guarantee that future mature-training entropy follows four updates.
        projections=[]
        class_map={'paviaU':9,'xuzhou':9,'whu-honghu':22}
        for row in rows:
            sizes=[x['after_checkpoint_bytes'] for x in row['timings']]
            slopes=[max(0,sizes[i]-sizes[i-1]) for i in range(2,5)]
            n=class_map[row['scene']]*5*40
            growth=next(x['future_metadata_growth_raw_bytes'] for x in components_rows if x['case']==row['name'] and x['step']==4)
            projections.append({'scene':row['scene'],'method':row['method'],'warm_step_db_growth_bytes':slopes,
               'three_seed_updates':3*n,'shape_only_low_bytes':3*(n+1)*min(slopes),
               'shape_only_high_plus_metadata_bytes':3*(n+1)*(max(slopes)+growth),
               'encode_commit_checkpoint_median_seconds':statistics.median(x['encode_commit_checkpoint_seconds'] for x in row['timings'][1:])})
        result={'logical_store_bytes_with_shm':physical,'filesystem_allocated_bytes':allocated,'elapsed_seconds':elapsed,'original_byte_verification_seconds':verify_seconds,
                'encoded_payload_bytes':store.write_bytes,'object_payload_bytes_read':store.read_bytes-read_before,'objects_by_codec_depth':object_stats,'cases':rows,
                'peak_before_journal_checkpoint_bytes':max(x['before_journal_checkpoint_bytes'] for r in rows for x in r['timings'])}
        store.close();result['closed_store_bytes']=sum(f.stat().st_size for f in (out/layout).iterdir() if f.is_file())
        results[layout]=result;projected[layout]={'rows':projections,'low_bytes':sum(x['shape_only_low_bytes'] for x in projections),
             'high_plus_raw_metadata_growth_bytes':sum(x['shape_only_high_plus_metadata_bytes'] for x in projections),
             'encode_seconds_projection':sum(x['three_seed_updates']*x['encode_commit_checkpoint_median_seconds'] for x in projections),
             'uncertainty':'four early synthetic updates only; not a real-training bound or capacity guarantee'}
    for r in inventory:require(hashed((source/r['case']/'stages'/f"{r['step']:06d}"/r['name']).read_bytes())==r['sha256'],'Original source changed')
    value={'schema':'c14_bounded_existing_synthetic_storage_v1','producer':a.expected_commit,'source_states':60,'source_files':len(inventory),
       'source_state_bytes':sum(r['bytes'] for r in inventory if r['name']=='state.pt'),'exact_whole_file_unique_bytes':sum(whole.values()),
       'whole_state_zlib_payload_baseline_bytes':baseline_zlib,'components':components_rows,'strategies':results,'future36_case_scenarios':projected,
       'real_updates':0,'synthetic_optimizer_updates':0,'new_query_scoring':0,'all_original_source_hashes_unchanged':True,
       'no_large_archive_created':True,'approved_destination':'same authorized LXC only','current_free_bytes':shutil.disk_usage(out).free}
    (out/'source-inventory.json').write_text(json.dumps(inventory,indent=2));(out/'result.json').write_text(json.dumps(value,indent=2))
    print(json.dumps({k:v for k,v in value.items() if k not in ['components','strategies','future36_case_scenarios']}))
    print(json.dumps({k:{q:v for q,v in s.items() if q!='rows'} for k,s in projected.items()}))


if __name__=='__main__':main()
