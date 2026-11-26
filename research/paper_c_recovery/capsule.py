"""Prioritize complete initial/selected/current states plus every history record."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from time import perf_counter

from audit import audit


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    suite=Path(a.suite).resolve();run=Path(a.run).resolve();out=Path(a.out).resolve()
    if out.exists() or out.is_relative_to(suite) or not run.is_relative_to(suite):raise ValueError('New external capsule and own run required')
    reg=json.loads((run/'registration.json').read_text());summary=json.loads((run/'summary.json').read_text())
    if not summary['complete']:raise ValueError('Only completely committed batches may be compacted')
    started=perf_counter();recover=[];keep=set()
    for case in reg['cases']:
        folder=run/case['id'];r=audit(folder);paths={name:folder/'stages'/f'{step:06d}'/'state.pt'
            for name,step in (('initial',0),('selected',r['selected_step']),('current',r['committed_optimizer_updates']))}
        keep.update(paths.values());recover.append({'case':case['id'],'updates':r['committed_optimizer_updates'],'selected_step':r['selected_step'],
            'checkpoints':{k:{'name':str(f.relative_to(suite)),'sha256':sha(f),'bytes':f.stat().st_size} for k,f in paths.items()}})
    full=[];rows=[];objects={};source={};omitted=[]
    for f in sorted(suite.rglob('*')):
        if f.is_symlink():raise ValueError('Symlink not admitted in own artifact suite')
        if not f.is_file() or f.name=='lock':continue
        if f.suffix not in ('.pt','.json','.log','.csv','.npz'):raise ValueError('Unlisted capsule file type')
        row={'name':str(f.relative_to(suite)),'sha256':sha(f),'bytes':f.stat().st_size};full.append(row)
        if f.suffix=='.pt' and f not in keep:omitted.append(row);continue
        rows.append(row);objects[row['sha256']]={'bytes':row['bytes']};source.setdefault(row['sha256'],f)
    history={'schema':'paper_c_full_source_history_index_v1','files':full,'omitted_intermediate_states':omitted,
        'scope':'All original file identities, not a claim that omitted state tensors are physically backed up. Initial/selected/current full states are included for every real case. Full historical reentry requires the full archive or separately validated reconstruction.'}
    raw=json.dumps(history,sort_keys=True,indent=2).encode();h=hashlib.sha256(raw).hexdigest()
    rows.append({'name':'recovery/full-source-history-index.json','sha256':h,'bytes':len(raw)})
    objects[h]={'bytes':len(raw)};source[h]=raw
    index={'schema':'paper_c_priority_recovery_capsule_v1','files':rows,'objects':objects,'recovery':recover,
        'original_suite_files':len(full),'original_suite_bytes':sum(x['bytes'] for x in full),
        'omitted_intermediate_checkpoint_files':len(omitted),'omitted_intermediate_checkpoint_bytes':sum(x['bytes'] for x in omitted),
        'raw_bank_GT_query_arrays_copied':False,'producer':reg['producer'],
        'limit':'Every current/selected/initial complete model/optimizer/RNG/cursor and all JSON/log/probability evidence are physically present. Other historical state.pt tensors remain source-only unless full-archive transfer also verifies. Original input/environment/path bindings still apply.'}
    def clean(info):info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;return info
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        raw_index=json.dumps(index,sort_keys=True,indent=2).encode();info=tarfile.TarInfo('object-index01.json');info.size=len(raw_index);t.addfile(info,io.BytesIO(raw_index))
        for h,f in source.items():
            if isinstance(f,bytes):info=tarfile.TarInfo('objects/'+h);info.size=len(f);t.addfile(info,io.BytesIO(f))
            else:t.add(f,arcname='objects/'+h,recursive=False,filter=clean)
    for row in full:
        f=suite/row['name']
        if f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise ValueError('Source changed during capsule seal')
    receipt={'schema':index['schema'],'status':'SEALED_SOURCE_HASHES_UNCHANGED','archive':out.name,'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),
        'logical_files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),'unique_objects':len(objects),'unique_bytes':sum(r['bytes'] for r in objects.values()),
        'cases':len(recover),'omitted_intermediate_checkpoint_files':len(omitted),'omitted_intermediate_checkpoint_bytes':sum(r['bytes'] for r in omitted),
        'elapsed_seconds':perf_counter()-started,'new_optimizer_updates':0,'raw_bank_GT_query_arrays_copied':False,'target_verification':'PENDING'}
    with out.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
