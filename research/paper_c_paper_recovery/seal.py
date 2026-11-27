"""Seal recovery controls privately and export only explicit aggregate artifacts."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

from recover import BASE, read, relative, require, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--source-bundle',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    require(git('rev-parse','HEAD')==a.expected_commit and not git('status','--porcelain'), 'Clean published sealer required')
    b=a.backup.resolve()
    require(b.is_relative_to(BASE/'backups'), 'Authorized private backup required')
    r=read(b/'review/reconciliation.json')
    c=read(b/'review/rebuild-receipt.json')
    require(c['status']=='PASSED' and c['aggregate_differences']==0 and r['increment']['fresh_restore_differences']==0,
            'Verified reconciliation and reconstruction required')
    m=read(b/'increment/manifest.json')
    missing=m['missing_sources']
    write(b/'review/dependency-summary.json', {'verified_paths':len(m['files']),
          'missing_source_paths':len(missing),'missing_suffix_counts':{suffix:sum(Path(v['path']).suffix==suffix for v in missing) for suffix in sorted({Path(v['path']).suffix for v in missing})},
          'private_path_and_point_id_exported':False,'manifest_sha256':sha(b/'increment/manifest.json')})
    source=b/'source'
    source.mkdir(mode=0o700)
    bundles=[(a.source_bundle,'C11.bundle'),(BASE/'backups/paper-c-paper-recovery-A-source01.bundle','A-frozen.bundle'),
             (BASE/'backups/oss-slic-portability-20261127/source.bundle','A-SLIC.bundle')]
    source_rows=[]
    for path,name in bundles:
        require(path.resolve().is_relative_to(BASE/'backups'), 'Source bundle outside authorized project')
        dest=source/name
        shutil.copyfile(path,dest)
        require(sha(path)==sha(dest), 'Source bundle copy differs')
        source_rows.append({'name':name,'bytes':dest.stat().st_size,'sha256':sha(dest)})
    repo=b/'source-restored.git'
    subprocess.run(['git','clone','-q','--bare',str(source/'C11.bundle'),str(repo)],check=True)
    for name in ('A-frozen.bundle','A-SLIC.bundle'):
        subprocess.run(['git','-C',str(repo),'fetch','-q',str(source/name),'+refs/*:refs/recovery/'+name+'/*'],check=True)
    revisions=[a.expected_commit,r['producer'],'b36f1c976120eb8abd7d74e7f44295227550e9ea',
               '9d5318803d926eacebf8984bc4e6a7bed388bbc4','7e2d40cf593af3d5ae94f5ae20a5e198c978f650',
               '395caef57fba8ab6a09223538f748179dab552f2','6bb51b7d2b1e01e579fb07aad521f23d1c2ccc17',
               'cbc7e1b9bdc561b2714ea5b2498d34ae8cd5b369','2d14fc6b2d8f1ee40f7bc6c1ebdb1f4542e3392f',
               '540a97c84ffbeaf7278473966eb80da6be96c8bc']
    for revision in revisions:
        subprocess.run(['git','-C',str(repo),'cat-file','-e',revision+'^{commit}'],check=True)
    subprocess.run(['git','-C',str(repo),'fsck','--full'],check=True,stdout=subprocess.DEVNULL)
    current=b/'source-restored-checkout'
    subprocess.run(['git','clone','-q',str(repo),str(current)],check=True)
    require(subprocess.check_output(['git','-C',str(current),'rev-parse','HEAD'],text=True).strip()==a.expected_commit
            and not subprocess.check_output(['git','-C',str(current),'status','--porcelain'],text=True).strip(), 'Fresh source checkout differs')
    write(b/'review/source-recovery.json',{'status':'PASSED','bundles':source_rows,'commit_objects_verified':revisions,
          'fresh_HEAD':a.expected_commit,'fresh_checkout_clean':True,'full_git_fsck':True,'new_optimizer_updates':0})
    archive=b/'recovery-controls.tar.gz'
    names=['increment/manifest.json','available-private.json','layout.json','s2-retention.json']
    names += ['source/'+row['name'] for row in source_rows]
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for name in names:
            tar.add(b/name,arcname=name,recursive=False)
    control=b/'controls-restored'
    control.mkdir(mode=0o700)
    expected={name:sha(b/name) for name in names}
    with tarfile.open(archive) as tar:
        seen=set()
        for member in tar:
            relative(member.name)
            require(member.isfile() and member.name in expected and member.name not in seen,'Unexpected control archive member')
            dest=control/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as stream,dest.open('xb') as target:
                shutil.copyfileobj(stream,target)
            require(sha(dest)==expected[member.name], 'Restored recovery control differs')
            seen.add(member.name)
        require(seen==set(expected),'Missing recovery control')
    write(b/'review/preservation.json',{'status':'PASSED','control_archive_bytes':archive.stat().st_size,
          'control_archive_sha256':sha(archive),'control_files_restored':len(names),'control_differences':0,
          'new_dependency_objects':r['increment']['new_unique_objects'],'new_dependency_bytes':r['increment']['new_unique_bytes'],
          'storage_scope':'SAME_LXC_HOST_ONLY','source_sealer':a.expected_commit,'new_neural_calls':0,'optimizer_updates':0,
          'limit':'Opaque archival copies include frozen historical S2 input files. No GT/query array is decoded or scored. Existing critical archives remain required alongside the incremental store.'})
    review=b/'review'
    allow={'reconciliation.json','rebuild-receipt.json','rebuilt-routing.csv','rebuilt-paper1-historical.csv',
           'dependency-summary.json','source-recovery.json','preservation.json'}
    require({p.name for p in review.iterdir()}==allow,'Explicit deidentified export allowlist differs')
    package=b/'review.zip'
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(allow):z.write(review/name,name)
    manifest={'files':{name:{'bytes':(review/name).stat().st_size,'sha256':sha(review/name)} for name in sorted(allow)},
              'zip_bytes':package.stat().st_size,'zip_sha256':sha(package),'private_weights_arrays_point_ids_exported':False}
    write(b/'review.receipt.json',manifest)
    print(json.dumps({'status':'PASSED','export_files':len(allow),'zip_bytes':manifest['zip_bytes'],'zip_sha256':manifest['zip_sha256']}))


if __name__=='__main__':main()
