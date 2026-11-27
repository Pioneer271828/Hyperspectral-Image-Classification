"""Privately preserve exact released HSI bytes and verify a fresh Linux restore."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from bank_reader import ReleasedBank
from inspect_bank import guard


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--root',required=True);p.add_argument('--run-root',required=True);p.add_argument('--out',required=True);p.add_argument('--restore',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out);restore=guard(a.expected_commit,a.restore);root=Path(a.root).resolve();run=Path(a.run_root).resolve();repo=Path(__file__).resolve().parents[2]
    if out.is_relative_to(restore) or restore.is_relative_to(out):raise ValueError('Separate archive/restore paths required')
    source_bundle=out.with_suffix('.source.bundle')
    if source_bundle.exists():raise ValueError('New private source bundle path required')
    subprocess.run(['git','bundle','create',str(source_bundle),'HEAD'],cwd=repo,check=True,capture_output=True)
    source_bundle.chmod(0o600)
    subprocess.run(['git','bundle','verify',str(source_bundle)],cwd=repo,check=True,capture_output=True)
    files=[(source_bundle,'source.bundle','source')]
    files.append((root/'index.json','inputs/index.json','original_portable_authority'))
    for scene in ('xuzhou','whu-honghu'):
        bank=ReleasedBank(root/scene,scene,portable_index=root/'index.json')
        paths={'bank.json','registration.json'}
        for identity in bank.members:paths.update((f'tile-{identity}.json',f'tile-{identity}.npy'))
        for name in sorted(paths):files.append((root/scene/name,f'inputs/{scene}/{name}','released_input'))
    for folder in ('runs','references01'):
        for path in sorted((run/folder).rglob('*')):
            if path.is_symlink():raise ValueError('Private evidence alias rejected')
            if path.is_file():files.append((path,'evidence/'+str(path.relative_to(run)),'private_evidence'))
    records=[]
    for path,name,kind in files:
        if path.is_symlink() or not path.is_file():raise ValueError('Regular original file required')
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':digest(path),'kind':kind})
    if len({r['path'] for r in records})!=len(records) or sum(r['bytes'] for r in records)>256*1024**2:raise ValueError('Duplicate member or private snapshot budget exceeded')
    manifest={'schema':'paper_c_C6_private_preservation_v1','source':a.expected_commit,'files':records,'scope':'Private authorized released development inputs, source and evidence only. Original numeric labels/metadata and full reference PDF stay on Linux. No new GT/query/weights.'}
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        for (path,name,_),row in zip(files,records):
            if digest(path)!=row['sha256']:raise ValueError('Source changed before seal')
            t.add(path,arcname=name,recursive=False)
        raw=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('snapshot-index.json');info.size=len(raw);info.mode=0o600;t.addfile(info,io.BytesIO(raw))
    out.chmod(0o600);archive_sha=digest(out);restore.mkdir(parents=True,mode=0o700)
    with tarfile.open(out,'r:gz') as t:
        members=t.getmembers();names={m.name for m in members}
        if len(names)!=len(members) or names!={r['path'] for r in records}|{'snapshot-index.json'}:raise ValueError('Sealed member roster differs')
        archived_manifest=json.load(t.extractfile('snapshot-index.json'))
        if archived_manifest!=manifest:raise ValueError('Sealed manifest differs')
        for member in members:
            target=(restore/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(restore) or target.exists():raise ValueError('Unsafe/private restore member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
    mismatches=[r['path'] for r in records if (restore/r['path']).stat().st_size!=r['bytes'] or digest(restore/r['path'])!=r['sha256']]
    if mismatches or digest(out)!=archive_sha:raise ValueError('Full archive/restore verification failed')
    for scene in ('xuzhou','whu-honghu'):ReleasedBank(restore/'inputs'/scene,scene,portable_index=restore/'inputs/index.json')
    checkout=restore/'checkout';subprocess.run(['git','init',str(checkout)],check=True,capture_output=True);subprocess.run(['git','fetch',str(restore/'source.bundle'),'HEAD'],cwd=checkout,check=True,capture_output=True);subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=checkout,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if head!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=checkout).strip():raise ValueError('Restored source identity/worktree differs')
    receipt={'schema':'paper_c_C6_preservation_redacted_v1','source':a.expected_commit,'archive_bytes':out.stat().st_size,'archive_sha256':archive_sha,'source_files':len(records),'restored_verified_files':len(records),'mismatches':0,'uncompressed_bytes':sum(r['bytes'] for r in records),'restored_source_complete_SHA':head,'restored_both_bank_metadata_admissions':True,'payload_arrays_copied_to_Windows':0,'off_host_backup_claim':False,'backup_location_scope':'Second verified private copy on authorized rmvision; not an independent physical host.','new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'raw_GT_or_query_reads':0}
    out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
