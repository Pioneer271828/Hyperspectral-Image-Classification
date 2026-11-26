"""Seal finished C-only outputs as exact-SHA objects, without bank/input arrays."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    suite=Path(a.suite).resolve();out=Path(a.out).resolve();run=suite/'engineering3-execution01'
    if out.exists() or out.is_relative_to(suite):raise ValueError('New checkout-external archive required')
    summary=json.loads((run/'summary.json').read_text());reg=json.loads((run/'registration.json').read_text())
    if not summary['complete']:raise ValueError('Cannot seal incomplete engineering batch')
    started=perf_counter();recover=[]
    for case in reg['cases']:
        folder=run/case['id'];r=audit(folder);last=folder/'stages'/f'{r["committed_optimizer_updates"]:06d}'/'state.pt'
        selected=folder/'stages'/f'{r["selected_step"]:06d}'/'state.pt'
        recover.append({'case':case['id'],'contract_sha256':r['contract_sha256'],'updates':r['committed_optimizer_updates'],
            'last_checkpoint':str(last.relative_to(suite)),'last_sha256':sha(last),
            'selected_checkpoint':str(selected.relative_to(suite)),'selected_sha256':sha(selected),
            'complete_optimizer_RNG_BN_cursor':True})
    rows=[];objects={};sources={}
    for f in sorted(suite.rglob('*')):
        if f.is_symlink():raise ValueError('Unexpected symlink in own output suite')
        if not f.is_file() or f.name=='lock':continue
        # The suite contains model states and JSON/log receipts, never source banks.
        if f.suffix not in ('.pt','.json','.log','.csv'):raise ValueError('Unexpected output type in backup whitelist')
        h=sha(f);rows.append({'name':str(f.relative_to(suite)),'sha256':h,'bytes':f.stat().st_size})
        objects[h]={'bytes':f.stat().st_size};sources.setdefault(h,f)
    index={'schema':'paper_c_recovery_object_backup_v1','files':rows,'objects':objects,'recovery':recover,
        'raw_bank_GT_query_arrays_copied':False,'producer':reg['producer'],
        'recovery_dependencies':'Original pinned code, author checkout, compatible runtime and already authorized bank releases remain required; absolute-path/container mapping is explicit, never silent.'}
    raw=json.dumps(index,sort_keys=True,indent=2,allow_nan=False).encode()
    def clean(info):info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;return info
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        info=tarfile.TarInfo('object-index01.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        for h,f in sources.items():t.add(f,arcname='objects/'+h,recursive=False,filter=clean)
    for row in rows:
        f=suite/row['name']
        if f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise ValueError('Finished source changed during archive')
    receipt={'schema':'paper_c_recovery_seal_v1','status':'SEALED_SOURCE_HASHES_UNCHANGED','archive':out.name,
        'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),'logical_files':len(rows),'logical_bytes':sum(x['bytes'] for x in rows),
        'unique_objects':len(objects),'unique_bytes':sum(x['bytes'] for x in objects.values()),'cases':recover,
        'elapsed_seconds':perf_counter()-started,'new_optimizer_updates':0,'new_neural_forwards':0,
        'raw_bank_GT_query_arrays_copied':False,'target_verification':'PENDING; source seal is not a verified independent backup'}
    with out.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('status','archive_bytes','archive_sha256','logical_files','logical_bytes','elapsed_seconds')}))


if __name__=='__main__':main()
