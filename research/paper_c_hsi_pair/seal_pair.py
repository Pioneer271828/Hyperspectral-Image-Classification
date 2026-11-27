"""Seal this pair's17 original spectral tiles and replay from a verified restore."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from c6_bridge import ReleasedBank,canonical,sha,frozen_source,check_source


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1024**2),b''):h.update(raw)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--probe',required=True);p.add_argument('--out',required=True);p.add_argument('--restore',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);restored=frozen_source(a.expected_commit,a.restore);root=Path(a.run_root);inputs=Path(a.inputs);repo=Path(__file__).resolve().parents[2]
    envelope=json.loads(Path(a.plan).read_text());plan=envelope['plan'];admission=json.loads((Path(__file__).parent/'admission.json').read_text());original=json.loads(Path(a.probe).read_text())
    if sha(canonical(plan))!=admission['plan_sha256'] or original['plan_sha256']!=admission['plan_sha256'] or not original['passed']:raise ValueError('Current admitted plan/probe required')
    bundle=out.with_suffix('.source.bundle')
    if bundle.exists():raise ValueError('Fresh source bundle required')
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=repo,check=True,capture_output=True);bundle.chmod(0o600);files=[(bundle,'source.bundle','source')]
    files.append((inputs/'index.json','inputs/index.json','original_authority'))
    selected_npys=0;payload_bytes=0
    for scene in plan['scenes']:
        name=scene['scene'];bank=ReleasedBank(inputs/name,name,portable_index=inputs/'index.json');selected={p['tile_id'] for points in scene['roles'].values() for p in points}
        for member in ('bank.json','registration.json'):files.append((inputs/name/member,f'inputs/{name}/{member}','original_authority'))
        for identity in sorted(bank.members):files.append((inputs/name/f'tile-{identity}.json',f'inputs/{name}/tile-{identity}.json','original_labeled_tile_manifest'))
        for identity in sorted(selected):
            path=inputs/name/f'tile-{identity}.npy'
            if digest(path)!=bank.members[identity]['content']['npy_sha256']:raise ValueError('Selected original payload changed')
            files.append((path,f'inputs/{name}/tile-{identity}.npy','selected_spectral_payload'));selected_npys+=1;payload_bytes+=path.stat().st_size
    if selected_npys!=17 or payload_bytes!=19602176:raise ValueError('Frozen selected input footprint differs')
    folders=('metadata01','custody01','legacy01','review02','plan03','validation04','validation05','plan06','validation07','catalog07','check07')
    for folder in folders:
        for path in sorted((root/folder).rglob('*')):
            if path.is_symlink():raise ValueError('Private evidence alias rejected')
            if path.is_file():files.append((path,'evidence/'+str(path.relative_to(root)),'private_evidence'))
    original_archive=root/'xuzhou-existing-provenance01.tar.gz'
    if digest(original_archive)!='288639b47f44f233a568ce11160aeb6771ea0bb8d92770a9f415089b5306f613':raise ValueError('Original private custody archive changed')
    files.append((original_archive,'evidence/xuzhou-existing-provenance01.tar.gz','original_custody_archive'))
    records=[]
    for path,name,kind in files:
        if path.is_symlink() or not path.is_file():raise ValueError('Regular private snapshot source required')
        records.append({'path':name,'bytes':path.stat().st_size,'sha256':digest(path),'kind':kind})
    if len({r['path'] for r in records})!=len(records) or sum(r['bytes'] for r in records)>64*1024**2:raise ValueError('Duplicate member or private snapshot64MiB budget exceeded')
    manifest={'schema':'paper_c_C7_private_pair_snapshot_v1','source':a.expected_commit,'plan_sha256':admission['plan_sha256'],'files':records,'scope':'Complete C7 pair input/recovery scope:17 selected NPY payloads, all original required labeled-tile JSON authority, source and private evidence. Not a full112-spectral-array C6 snapshot. Academic data/point IDs/public metadata HTML stay private on Linux.'}
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        for (path,name,_),record in zip(files,records):
            if digest(path)!=record['sha256']:raise ValueError('Snapshot source drifted')
            t.add(path,arcname=name,recursive=False)
        raw=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('snapshot-index.json');info.size=len(raw);info.mode=0o600;t.addfile(info,io.BytesIO(raw))
    out.chmod(0o600);archive_sha=digest(out);restored.mkdir(parents=True,mode=0o700)
    with tarfile.open(out,'r:gz') as t:
        members=t.getmembers()
        if len({m.name for m in members})!=len(members) or {m.name for m in members}!={r['path'] for r in records}|{'snapshot-index.json'}:raise ValueError('Private archive roster differs')
        if json.load(t.extractfile('snapshot-index.json'))!=manifest:raise ValueError('Private archived manifest differs')
        for member in members:
            target=(restored/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(restored) or target.exists():raise ValueError('Unsafe/private restore member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
    mismatches=[r['path'] for r in records if (restored/r['path']).stat().st_size!=r['bytes'] or digest(restored/r['path'])!=r['sha256']]
    if mismatches or digest(out)!=archive_sha:raise ValueError('Whole archive/restore byte verification failed')
    checkout=restored/'source';subprocess.run(['git','init',str(checkout)],check=True,capture_output=True);subprocess.run(['git','fetch',str(restored/'source.bundle'),'HEAD'],cwd=checkout,check=True,capture_output=True);subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=checkout,check=True,capture_output=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=checkout).strip():raise ValueError('Restored complete source identity differs')
    restored_evidence=restored/'evidence';relative_plan=Path(a.plan).resolve().relative_to(root.resolve());probe_out=restored/'recovery-probe01.json';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cmd=[sys.executable,str(checkout/'research/paper_c_hsi_pair/probe_pair.py'),'--expected-commit',a.expected_commit,'--run-root',str(restored_evidence),'--inputs',str(restored/'inputs'),'--plan',str(restored_evidence/relative_plan),'--out',str(probe_out)]
    with (restored/'recovery-probe01.log').open('x') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    recovery=json.loads(probe_out.read_text())
    if not recovery['passed'] or recovery['directions']!=original['directions'] or recovery['unique_original_points']!=40 or recovery['point_presentations_two_directions']!=80:raise ValueError('Restored pair inputs differ from original production')
    check_source(a.expected_commit);receipt={'schema':'paper_c_C7_private_recovery_redacted_v1','source':a.expected_commit,'plan_sha256':admission['plan_sha256'],'archive_bytes':out.stat().st_size,'archive_sha256':archive_sha,'source_files':len(records),'restored_verified_files':len(records),'mismatches':0,'selected_spectral_arrays':selected_npys,'selected_NPY_payload_bytes':payload_bytes,'original_labeled_tile_manifests':112,'restored_complete_source_SHA':a.expected_commit,'restored_source_clean':True,'restored_real_input_directions':2,'restored_unique_original_points':40,'restored_point_presentations':80,'restored_all_phase_input_outputs_match_original':True,'payload_files_on_Windows':0,'off_host_verified':False,'location_scope':'Second private copy and complete replay on same authorized rmvision host; no independent physical-host disaster recovery claim.','new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'new_raw_GT_or_query_reads':0};out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
