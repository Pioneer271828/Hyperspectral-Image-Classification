"""Small consistent SQLite snapshot and aggregate-only decision delivery.

No large archive, original-file cleanup, new destination or optimizer call.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import zipfile
import zlib

import numpy
import torch

from .benchmark import BASE,ROOT,clean
from .codec import hashed,require


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();clean(a.expected_commit);run=Path(a.run).resolve();out=Path(a.out).resolve()
    require(run==BASE/'runs/paper-c-storage-20261127-01' and out.is_relative_to(BASE/'backups') and not out.exists(),'New authorized small backup required')
    out.mkdir(mode=0o700);small=out/'small-store';small.mkdir();source=run/'cli05-store/objects.sqlite3'
    require(source.stat().st_size<16*1024**2,'Small fixture snapshot cap exceeded')
    # Standard SQLite online-backup API; source is quiescent for this proof.
    src=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);dest=sqlite3.connect(small/'objects.sqlite3')
    src.backup(dest);dest.execute('PRAGMA journal_mode=WAL');dest.execute('PRAGMA synchronous=FULL');dest.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    dest.close();src.close()
    bundle=out/'source.bundle';subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
    public=out/'decision-package';public.mkdir();review=public/'review';review.mkdir()
    for name in ['README.md','one-page.md','contracts.json']:shutil.copy2(ROOT/'research/paper_c_storage'/name,public/name)
    for srcpath,name in [(run/'measurement02/result.json','measurement.json'),(run/'measurement02/source-inventory.json','source-inventory.json'),
                         (run/'recovery02/receipt.json','random-recovery.json'),(run/'tests01.log','tests01.log'),(run/'tests03.log','tests03.log'),
                         (run/'tests06.log','tests06.log'),(run/'pipeline04.json','pipeline04.json'),(run/'pipeline05.json','pipeline05.json'),
                         (run/'decision04.log','retained-title-overflow.log'),(run/'repository05.log','repository.log'),
                         (run/'decision06/render-receipt.json','render-receipt.json')]:shutil.copy2(srcpath,review/name)
    for name in ['storage-decision-one-page.pdf','storage-decision-preview.png']:shutil.copy2(run/'decision06'/name,public/name)
    shutil.copy2(bundle,public/'source.bundle')
    env={'hostname':socket.gethostname(),'virtualization':subprocess.check_output(['systemd-detect-virt'],text=True).strip(),'python':sys.version,
         'executable':sys.executable,'platform':platform.platform(),'numpy':numpy.__version__,'torch':torch.__version__,'sqlite':sqlite3.sqlite_version,
         'zlib':zlib.ZLIB_RUNTIME_VERSION,'CPU_threads':1,'current_free_bytes':shutil.disk_usage(BASE).free,
         'measurement_cache':'warm OS/database after writes; no cold-device cache eviction','real_training':False,'new_query_scoring':False}
    (review/'environment.json').write_text(json.dumps(env,indent=2))
    # Receipts retain distinct producers; certify only unchanged encoder bodies.
    def prior(revision,name):return subprocess.check_output(['git','show',revision+':research/paper_c_storage/'+name],cwd=ROOT)
    def body(raw,cls,fn):
        nodes=ast.parse(raw).body
        parent=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls) if cls else None
        return ast.dump(next(n for n in (parent.body if parent else nodes) if isinstance(n,ast.FunctionDef) and n.name==fn),include_attributes=False)
    measured='d23fad3ad0e218913842157c132ca636a6d4d860';equal=[]
    for name,cls,fn in [('store.py','Store','put'),('store.py','Store','transaction'),('store.py','Transaction','commit'),('bridge.py',None,'import_case')]:
        same=body(prior(measured,name),cls,fn)==body((ROOT/'research/paper_c_storage'/name).read_bytes(),cls,fn)
        require(same,'Measured encoder changed');equal.append({'file':name,'class':cls,'function':fn,'AST_equal':same})
    (review/'source-equivalence.json').write_text(json.dumps({'final_source':a.expected_commit,'measurement_producer':measured,'unchanged_encoder_bodies':equal,
        'later_changes':'restore readiness guard and decode size refusal; no compression/commit encoder change'},indent=2))
    (review/'failures.json').write_text(json.dumps({'sync01':'First incremental fetch was issued from home directory and refused; corrected by separate clean checkout. No existing checkout was changed.',
        'decision04':'Title rectangle short by0.45pt for embedded CJK font; overflow refused before PDF delivery. Fixed, rerendered and visually reviewed.',
        'repository05':'Inherited current-status Markdown stale, exit1; no whole-repository or CI pass claimed',
        'intentional_faults':'Process exits71 at four publication boundaries and corruption/partial-restore injections are expected tests; no real update replay'},indent=2))
    snapshot_rows=[]
    for f in [bundle,small/'objects.sqlite3']:
        snapshot_rows.append({'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':hashed(f.read_bytes())})
    restored=out/'fresh-restored';restored.mkdir();(restored/'small-store').mkdir()
    for row in snapshot_rows:
        f=out/row['path'];target=restored/row['path'];shutil.copy2(f,target)
        require(hashed(target.read_bytes())==row['sha256'],'Fresh small snapshot copy differs')
    # Decode the copied database without reading original state paths.
    from .store import Store
    copy=Store(restored/'small-store');rows=copy.frames('fixture');verified=0
    for frame in rows:
        for name in frame['files']:copy.decode_file(frame,name);verified+=1
    copy.checkpoint();copy.close()
    require(len(rows)==5 and verified==15,'Expected small complete snapshot missing')
    for row in snapshot_rows:require(hashed((out/row['path']).read_bytes())==row['sha256'],'Frozen backup source changed')
    receipt={'schema':'c14_small_sqlite_snapshot_receipt_v1','producer':a.expected_commit,'snapshot_files':snapshot_rows,'fresh_copy_differences':0,
             'decoded_original_files':verified,'committed_stages':len(rows),'does_not_require_original_state_paths':True,
             'backup_api':'sqlite3.Connection.backup; quiescent source in this test','same_authorized_LXC_not_offhost':True,
             'large_training_archive_created':False,'old_assets_deleted_or_moved':False,'new_real_updates':0,'new_query_scoring':0}
    (out/'preservation-receipt.json').write_text(json.dumps(receipt,indent=2));(public/'preservation-receipt.json').write_text(json.dumps(receipt,indent=2))
    index=[{'path':str(f.relative_to(public)),'bytes':f.stat().st_size,'sha256':hashed(f.read_bytes())} for f in sorted(public.rglob('*')) if f.is_file()]
    (public/'file-index.json').write_text(json.dumps(index,indent=2))
    with zipfile.ZipFile(out/'decision-package.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(public.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(public)))
    print(json.dumps({'public_files_indexed':len(index),'zip_bytes':(out/'decision-package.zip').stat().st_size,'zip_sha256':hashed((out/'decision-package.zip').read_bytes()),
                      'small_snapshot_bytes':sum(r['bytes'] for r in snapshot_rows),'original_files_decoded_without_source':verified}))


if __name__=='__main__':main()
