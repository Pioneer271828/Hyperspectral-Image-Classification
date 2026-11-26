"""Seal completed routing outputs with exact file-SHA deduplication."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True);p.add_argument('--collection',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    suite=Path(a.suite).resolve();out=Path(a.out).resolve();collection=json.loads(Path(a.collection).read_text())
    if collection['cases']!=15 or collection['stages']!=540 or out.exists() or out.is_relative_to(suite):raise ValueError('Complete fifteen-case collection and new external archive required')
    cases=sorted({r['case'] for r in collection['rows']})
    for case in cases:
        root=suite/case;s=json.loads((root/'summary.json').read_text())
        for name,h in s['files'].items():
            if sha(root/name)!=h:raise ValueError('Completed route source differs')
    files=[];objects={};sources={}
    for f in sorted(suite.rglob('*')):
        if f.is_symlink():raise ValueError('Unlisted output symlink')
        if not f.is_file():continue
        if f.suffix not in ('.npz','.json','.csv','.log','.png','.svg'):raise ValueError('Unexpected routing artifact type')
        h=sha(f);files.append({'name':str(f.relative_to(suite)),'sha256':h,'bytes':f.stat().st_size});objects[h]={'bytes':f.stat().st_size};sources.setdefault(h,f)
    index={'schema':'paper_c_all_routing_outputs_object_backup_v1','files':files,'objects':objects,'cases':cases,
           'deduplication':'exact whole-file SHA only; every original logical output retained byte-identically','query_or_raw_bank_inputs_copied':False}
    def clean(info):info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;return info
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        raw=json.dumps(index,sort_keys=True,indent=2).encode();info=tarfile.TarInfo('object-index01.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        for h,f in sources.items():t.add(f,arcname='objects/'+h,recursive=False,filter=clean)
    for row in files:
        if sha(suite/row['name'])!=row['sha256']:raise ValueError('Routing source changed during seal')
    receipt={'status':'SEALED_SOURCE_HASHES_UNCHANGED','archive':out.name,'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),
        'logical_files':len(files),'logical_bytes':sum(r['bytes'] for r in files),'unique_objects':len(objects),
        'unique_bytes':sum(r['bytes'] for r in objects.values()),'target_verification':'PENDING','new_optimizer_updates':0,'query_reads':0}
    with out.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
