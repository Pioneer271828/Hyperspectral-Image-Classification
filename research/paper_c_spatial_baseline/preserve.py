"""Seal C13 source, synthetic evidence and historical aggregate dependencies.

Private payloads remain on the authorized LXC. The public packet contains only
source, license notices, generated-plan summaries and deidentified aggregates.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from time import perf_counter
import zipfile

from .protocol import sha, require
from .tasks import clean
from .run import ROOT, new_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-commit',required=True);p.add_argument('--run',required=True)
    p.add_argument('--out',required=True);p.add_argument('--source-packet',required=True)
    a=p.parse_args();clean(ROOT,a.expected_commit);start=perf_counter()
    run=Path(a.run).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False)
    capsule=out/'capsule';capsule.mkdir();source=capsule/'source';source.mkdir()
    shutil.copytree(run,capsule/'run')
    for name in ['Native.bundle','DSNet.commit','DSNet.tree','DSNet-model.py','DSNet-LICENSE']:
        shutil.copy2(Path(a.source_packet)/name,source/name)
    subprocess.run(['git','-C',str(ROOT),'bundle','create',str(source/'C13-final.bundle'),'HEAD'],check=True)
    historical=capsule/'historical-inputs';historical.mkdir()
    root=run.parents[1]
    for name,path in {
        's2-launch.json':root/'runs/m4s2-issue135-full01/launch.json',
        's2-results.json':root/'backups/paper-c-paper-figures-20261127-02/for-D/frozen/s2-results.json',
        'A-registration.json':root/'runs/paper-a-external-pavia-20261127/registration-authorized.json',
        'A-probabilities.json':root/'runs/paper-a-external-pavia-20261127/full-probabilities.json',
        'xz-shape-source.json':root/'runs/wp02-bank-xuzhou03/registration.json',
        'hh-shape-source.json':root/'runs/wp02-bank-honghu03/registration.json',
    }.items():shutil.copy2(path,historical/name)
    files=[{'path':str(f.relative_to(capsule)),'sha256':sha(f),'bytes':f.stat().st_size}
           for f in sorted(capsule.rglob('*')) if f.is_file()]
    manifest={'schema':'c13_preservation_manifest_v1','producer':a.expected_commit,'files':files,
              'source_and_synthetic_only_new_runs':True,'real_fits':0,'new_query_scoring':0}
    new_json(capsule/'manifest.json',manifest)
    archive=out/'preservation01.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(capsule,arcname='capsule')
    restored=out/'restored01';restored.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            require((restored/member.name).resolve().is_relative_to(restored),'Archive path escapes restore root')
            require(member.isfile() or member.isdir(),'Unexpected archive link/special member')
        tar.extractall(restored)
    differences=[r['path'] for r in files if sha(restored/'capsule'/r['path'])!=r['sha256']]
    require(not differences,'Fresh restore content differs')
    public=out/'public01';public.mkdir();review=public/'review';review.mkdir()
    names=['history03.json','author-packet-equivalence05.json','independent-audit05.json',
           'pipeline02.json','pipeline03.json','pipeline04.json','pipeline05.json',
           'tests01.log','tests02.log','tests03.log','tests05.log','history02.log',
           'repository05.log','zero-forward-reentry05.log','denied-generated-future-fit03.log',
           'denied-no-budget03.log','profile03/profile.json']
    for name in names:shutil.copy2(run/name,review/Path(name).name)
    for name in ['README.md','source-review.json','future-comparison.json','paper-ready.md']:
        shutil.copy2(ROOT/'research/paper_c_spatial_baseline'/name,public/name)
    shutil.copytree(source,public/'source')
    future=json.loads((run/'GENERATED-future-registration03.json').read_text())
    budget={k:future[k] for k in ['kind','updates','labeled_center_exposures','spectral_pixel_exposures_with_repetition',
                                'selection_forward_calls','state_bytes_upper_bound','ledger','registration_seconds']}
    budget['registration_sha256']=sha(run/'GENERATED-future-registration03.json')
    budget['unique_shapes']=[{k:c['contract'][k] for k in ['scene','method','trainable_parameters','state_bytes_upper_bound']}
                             for c in future['cases'] if c['contract']['seed']==64]
    budget['real_fits_started']=0;budget['note']='GENERATED input schema only; blocked from fitting; real reviewed role manifest unavailable'
    new_json(public/'future-budget-shapes.json',budget)
    receipt={'schema':'c13_fresh_restore_receipt_v1','source':a.expected_commit,'archive_sha256':sha(archive),
             'archive_bytes':archive.stat().st_size,'files':len(files),'logical_bytes':sum(r['bytes'] for r in files),
             'fresh_restore_differences':differences,'manifest_sha256':sha(capsule/'manifest.json'),
             'same_LXC_copy_not_offhost_backup':True,'seconds':perf_counter()-start}
    new_json(out/'preservation-receipt.json',receipt);new_json(public/'preservation-receipt.json',receipt)
    index=[{'path':str(f.relative_to(public)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(public.rglob('*')) if f.is_file()]
    new_json(out/'public-index.json',index)
    with zipfile.ZipFile(out/'public01.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(public.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(public)))
    print(json.dumps(receipt))


if __name__=='__main__':main()
