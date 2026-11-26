"""Build one immutable SHA-object increment for an already completed position package."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(path.read_text())


def main():
    root=Path(sys.argv[1]).resolve(strict=True);published=sys.argv[2]
    stage=root/'preservation008';assert not stage.exists();stage.mkdir(mode=0o700)
    snapshot=stage/'snapshot008';snapshot.mkdir();objects=snapshot/'objects';objects.mkdir()
    rows=[];recovery=[];continuous=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or stage in path.parents:continue
        assert not path.is_symlink()
        checksum=sha(path);obj=objects/checksum
        if not obj.exists():shutil.copyfile(path,obj)
        assert sha(obj)==checksum
        rows.append(dict(source=root.name+'/'+path.relative_to(root).as_posix(),sha256=checksum,bytes=path.stat().st_size))
    files={row['source']:row for row in rows}
    for folder in ('matrix-v1-output','balanced-v2-output'):
        summary=load(root/folder/'summary.json');assert summary['status']=='passed'
        for case in summary['cases']:
            directory=root/folder/case['case'];spec=load(directory/'registration.json')
            relative=root.name+'/'+directory.relative_to(root).as_posix()
            phases=[]
            for phase in ('prefix','resume','completed'):
                receipt=load(directory/(phase+'.json'));checkpoint=relative+'/'+phase+'.pt'
                assert receipt['checkpoint']['sha256']==files[checkpoint]['sha256']
                phases.append(dict(phase=phase,checkpoint=checkpoint,sha256=files[checkpoint]['sha256'],updates=receipt['updates']))
            recovery.append(dict(case=relative,registration=relative+'/registration.json',producer=spec['producer'],steps=24,synthetic=True,phases=phases))
            continuous.append(dict(case=relative,checkpoint=relative+'/full.pt',sha256=files[relative+'/full.pt']['sha256'],updates=24))
    assert len(recovery)==len(continuous)==96
    manifest=dict(schema='wp07_B_exact_position_incremental_backup_v1',files=rows,source_bytes=sum(row['bytes'] for row in rows),
        recovery_index=recovery,continuous_routes=continuous,new_synthetic_cases=96,new_real_cases=0,
        main_synthetic_updates_both_routes=4608,new_real_optimizer_updates=0,final_publication_commit=published,
        restore_dependencies=['snapshot002','snapshot003','snapshot004','snapshot005','snapshot006','snapshot007','objects','RESTORE-INDEX.json'],
        final_paths={'legacy':'matrix-v1-output','balanced':'balanced-v2-output','authority':'admission-v3-output',
            'real':'real-probe-v1-output','wp12':'wp12-v2-output','report':'render-v2-output','source':'source-final.bundle'},
        excluded=['Existing real weights/banks already in snapshots002-007; no recopy','Raw GT/sealed query/credentials','Unrelated A/C/D artifacts'])
    (snapshot/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.monotonic();archive=stage/'snapshot008.tar.xz'
    with tarfile.open(archive,'x:xz',preset=9) as stream:stream.add(snapshot,arcname='snapshot008')
    result=dict(status='built',archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        logical_files=len(rows),unique_objects=len({row['sha256'] for row in rows}),mapped_bytes=manifest['source_bytes'],
        new_synthetic_cases=96,new_real_cases=0,committed_split_phases=288,continuous_packets=96,
        elapsed_archive_seconds=time.monotonic()-started,source_root='[authorized-LXC]/runs/'+root.name)
    (stage/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
