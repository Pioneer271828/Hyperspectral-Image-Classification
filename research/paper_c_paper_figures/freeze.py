"""Freeze restored aggregate-only evidence and audit D's existing figure inventory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import tarfile

BASE=Path('/home/lbx/hsi-pioneer271828')
REC=BASE/'backups/paper-c-paper-recovery-20261127-03'
SEEDS=(64,512,2048)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def csvread(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))


def clean(expected):
    root=Path(__file__).resolve().parents[2]
    git=lambda *a:subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
    require(socket.gethostname()=='lbx' and git('rev-parse','HEAD')==expected and not git('status','--porcelain'),'Clean exact source on authorized LXC required')
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=clean(a.expected_commit);run=a.run.resolve()
    require(run.is_relative_to(BASE/'runs'),'Authorized external run required')
    inputs=run/'inputs';frozen=run/'frozen';frozen.mkdir(mode=0o700)
    registration=read(root/'research/paper_c_paper_figures/registration.json')
    require(sha(inputs/'manifest.json')==registration['input_manifest_sha256']
            and sha(inputs/'complete-paper-source.tar')==registration['D_source_archive_sha256'],'Registered aggregate input identity differs')
    input_manifest=read(inputs/'manifest.json')
    for row in input_manifest:
        path=inputs/row['name'];require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Transferred deidentified input differs')
    droot=run/'D-source';droot.mkdir(mode=0o700)
    with tarfile.open(inputs/'complete-paper-source.tar') as tar:
        for member in tar:
            rel=PurePosixPath(member.name)
            require(not rel.is_absolute() and '..' not in rel.parts and not member.issym() and not member.islnk(),'Unsafe D archive member')
            path=droot/member.name
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            require(member.isfile(),'Unsupported D archive member')
            path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,path.open('xb') as dst:shutil.copyfileobj(src,dst)
    inventory=[]
    for tex in sorted((droot/'paper').rglob('*.tex')):
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex.read_text()):
            image=(tex.parent/name).resolve()
            resolution='relative_to_fragment'
            if not image.is_file() and image.suffix=='.pdf' and image.with_suffix('.svg').is_file():
                image=image.with_suffix('.svg');resolution='existing_SVG_PDF_generated_at_build'
            if not image.is_file():
                candidates=list((droot/'paper').rglob(Path(name).name))
                if not candidates:candidates=list((droot/'paper').rglob(Path(name).with_suffix('.svg').name))
                require(not candidates or len({sha(p) for p in candidates})==1,'Ambiguous included-fragment figure source')
                if candidates:image=candidates[0];resolution='including_document_vector_source'
            inventory.append({'tex':str(tex.relative_to(droot)),'reference':name,'exists':image.is_file(),
                              'resolution':resolution,'source_file':str(image.relative_to(droot)),
                              'figure_sha256':sha(image) if image.is_file() else None})
    require(inventory and all(r['exists'] for r in inventory),'Frozen D figure reference cannot be resolved')
    write(frozen/'D-figure-inventory.json',{'D_revision':'79f34c681d02e7849305e850befb2beca6fed97a',
          'archive_sha256':sha(inputs/'complete-paper-source.tar'),'references':inventory,
          'new_gaps':['S2 four-cell seed-linked view','WP04 failed per-class adoption protection','C7/C9 operational role support','C10 prospective admission separated from current support'],
          'reuse_without_redraw':['submission-prep/figures/scene-effects.pdf','submission-prep/figures/class-effects.pdf'],
          'D_source_modified':False})
    # C11 already restored and verified these aggregate JSON files.
    manifests=read(REC/'increment/manifest.json')['files'];source=[]
    for name in ('results','summary'):
        row=next(r for r in manifests if r['path'].endswith('/m4s2-issue135-analysis02/'+name+'.json'))
        path=REC/'increment-restored'/row['sha256'];require(sha(path)==row['sha256'],'C11 restored S2 aggregate differs')
        shutil.copyfile(path,frozen/('s2-'+name+'.json'));source.append({'name':'s2-'+name+'.json','sha256':row['sha256'],'restored_from_C11':True})
    # Reduce twelve backed-up WP04 result records to deidentified selected metrics.
    available=read(REC/'available-private.json');a_root=BASE/'backups/wp04-matched-joint-20261127';am=read(a_root/'manifest.json');cases={}
    def include(seed,arm,path,checksum):
        require(sha(path)==checksum,'Frozen factorial aggregate differs')
        record=read(path)
        require(record['selected_step']==12 and record['optimizer_steps']==12 and not record['test_scoring'],'Only accepted fixed-step development result admitted')
        metrics=record['selected_validation']['metrics']['xuzhou']['val']
        cases[f'{seed}-{arm}']={'seed':seed,'arm':arm,'selected_step':12,'metrics':metrics,'original_result_sha256':checksum}
    for row in am['old_cases_by_reference']:
        pin=row['result'];include(row['seed'],row['arm'],Path(available[pin['sha256']]['path']),pin['sha256'])
    for name,pin in am['entries'].items():
        if name.endswith('/split/result.json'):
            case=name.split('/')[3];seed,arm=case.split('-');include(int(seed),arm,a_root/'objects'/pin['sha256'],pin['sha256'])
    require(set(cases)=={f'{s}-{c}' for s in SEEDS for c in ('M00','M10','M01','M11')},'Complete original four-cell roster required')
    write(frozen/'wp04-cases.json',{'source_manifest_sha256':sha(a_root/'manifest.json'),'cases':cases,'scope':'Xuzhou K1/V5, twelve updates, common initialization and fixed topology; separate from historical S2'})
    for name in ('a-matched-effects.snapshot.csv','a-matched-classes.snapshot.csv','spatial-unit-evidence.csv','honghu-class-parent-evidence.csv','prospective-cells.csv'):
        shutil.copyfile(inputs/name,frozen/name)
    hypothesis=root/'research/paper_c_hyrank_protocol/hypotheses.json';shutil.copyfile(hypothesis,frozen/'C10-hypotheses.json')
    write(frozen/'frozen-receipt.json',{'producer':a.expected_commit,'input_manifest_sha256':sha(inputs/'manifest.json'),
          'source_C11_producer':'75671680fb92c6392c50c993440a051dd375c3c3','source_C9':'11d7a8b398c03c421fb6fb50ca881e182ac1f360','source_C10':'ad663e9af23356568a0c28a2f508353887d8286a',
          'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(frozen.iterdir())},'new_neural_calls':0,'optimizer_updates':0,'query_arrays_read':0})
    print(json.dumps({'status':'FROZEN_AGGREGATE_ONLY','D_figure_references':len(inventory),'D_unique_figures':len({r['figure_sha256'] for r in inventory}),'WP04_cases':len(cases)}))


if __name__=='__main__':main()
