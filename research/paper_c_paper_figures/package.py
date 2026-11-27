"""Package D-ready vectors, freeze provenance, and regenerate from a fresh backup."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pymupdf

from freeze import BASE, REC, clean, read, require, sha, write

STEMS=('R1-S2-four-cells','R2-WP04-class-protection','D1-admission-flow','D2-support-units')
SHORT={
 'R1-S2-four-cells':'Historical S2 residual-by-supervision cells. Marker shapes link three model seeds on fixed scene splits; heavy paths show means. The macro panel weights the three scenes equally. NUM-2 amended readout; cropped OA axes. No independent spatial confidence interval is implied.',
 'R2-WP04-class-protection':'Separate WP04 Xuzhou K1/V5 development factorial, twelve steps. Each class has five points, shared across all models. Open markers show three seeds; diamonds are means. Both positive-mean main effects fail the preregistered class floor: message classes4,5,8; joint-edge class8. This is not the S2 protocol.',
 'D1-admission-flow':'Solid boxes show established C7/C9 input support; dashed boxes show the unadmitted C10 plan. The ground/tree and water/woody tasks are different. Dioni and Loukia share Hyperion. No new coarse-pair classifier outcome or prospective performance value is available.',
 'D2-support-units':'Ground/tree support by scene and role. Both directions reuse40 unique points. One original image per scene is reused across rows; parent tiles are not validated independent parcels. HongHu development has one parent per class. Xuzhou per-class parents remain unverified. Confirmation units remain0; prospective requirements are not released points.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--backup',type=Path,required=True);a=p.parse_args()
    root=clean(a.expected_commit);run=a.run.resolve();output=a.output.resolve();backup=a.backup.resolve()
    require(backup.is_relative_to(BASE/'backups') and not backup.exists(),'Fresh authorized backup root required');backup.mkdir(mode=0o700)
    audit=read(a.audit);visual=read(root/'research/paper_c_paper_figures/visual-review.json')
    require(audit['status']=='PASSED' and visual['all_eight_inspected'] and not visual['remaining_visual_defects'],'Completed independent and visual audit required')
    for name,checksum in visual['preview_sha256'].items():require(sha(output/'figures'/name)==checksum,'Visually reviewed image differs')
    delivery=backup/'for-D';delivery.mkdir();(delivery/'figures').mkdir();(delivery/'print-previews').mkdir();(delivery/'receipts').mkdir()
    shutil.copytree(output/'data',delivery/'data');shutil.copytree(run/'frozen',delivery/'frozen');shutil.copytree(output/'reused-D',delivery/'reused-D')
    reuse=delivery/'D-source/paper/submission-prep/figures';reuse.mkdir(parents=True)
    for path in (delivery/'reused-D').iterdir():shutil.copyfile(path,reuse/path.name)
    for path in (output/'figures').iterdir():shutil.copyfile(path,delivery/('print-previews' if path.suffix=='.png' else 'figures')/path.name)
    own=root/'research/paper_c_paper_figures';shutil.copytree(own,delivery/'source',ignore=shutil.ignore_patterns('__pycache__'))
    origins=delivery/'original-receipts';origins.mkdir()
    for path,name in [(run/'inputs/manifest.json','registered-input-manifest.json'),
                      (run/'inputs/table-receipt.json','C9-original-table-receipt.json'),
                      (run/'inputs/paper-c-hsi-pair-review09.zip','C7-original-deidentified-review.zip'),
                      (root/'research/paper_c_hyrank_protocol/independent-audit-receipt.json','C10-original-independent-audit.json')]:
        shutil.copyfile(path,origins/name)
    for name in ('reconciliation','rebuild-receipt','preservation'):
        shutil.copyfile(REC/'seal03/review'/f'{name}.json',origins/f'C11-{name}.json')
    write(origins/'origin-manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in origins.iterdir()})
    shutil.copyfile(own/'captions.md',delivery/'captions.md');shutil.copyfile(a.audit,delivery/'receipts/independent-numeric-audit.json')
    shutil.copyfile(own/'visual-review.json',delivery/'receipts/visual-review.json');shutil.copyfile(output/'figure-receipt.json',delivery/'receipts/figure-receipt.json')
    write(delivery/'receipts/environment.json',{'python':sys.version.split()[0],
          'numpy':importlib.metadata.version('numpy'),'matplotlib':importlib.metadata.version('matplotlib'),
          'PyMuPDF':importlib.metadata.version('PyMuPDF'),'additional_install':'PyMuPDF==1.26.7 on authorized LXC',
          'fresh_environment_install_verified':False,'private_data_or_model_dependencies_required_for_plot_rebuild':False})
    for width in (85,175):
        texts=['% Opt-in inserts only. Choose one width; do not include both.','% Requires graphicx. Paths are relative to this independent delivery.']
        for stem in STEMS:
            env='figure*' if width==175 else 'figure'
            texts += ['\\begin{'+env+'}[t]','\\centering','\\includegraphics[width='+str(width)+'mm]{figures/'+stem+'-'+str(width)+'mm.pdf}',
                      '\\caption{'+SHORT[stem]+'}','\\label{fig:c12-'+stem.lower()+'}','\\end{'+env+'}','']
        (delivery/f'figure-inserts-{width}mm.tex').write_text('\n'.join(texts),encoding='utf8')
    # One review artifact contains every actual fixed-width vector page.
    proof=pymupdf.open()
    for stem in STEMS:
        for width in (175,85):
            with pymupdf.open(delivery/'figures'/f'{stem}-{width}mm.pdf') as doc:proof.insert_pdf(doc)
    proof.save(delivery/'print-width-proof.pdf');proof.close()
    ledger=[{'figure':'R1','unit':'model seed','count':'3','reuse':'Same64/512/2048 across all four cells; fixed points per scene'},
            {'figure':'R1','unit':'fixed evaluation points','count':json.dumps(audit['S2_fixed_evaluation_points'],sort_keys=True),'reuse':'Same scene points across models; no new acquisitions'},
            {'figure':'R2','unit':'development points','count':'45','reuse':'Same5 points/class,3 seeds,4 cases; separate K1/V5 protocol'},
            {'figure':'D1/D2','unit':'unique coarse-task points','count':'40','reuse':'Two directions give80 presentations, not80 unique points'},
            {'figure':'D1/D2','unit':'original images','count':'1 per Xuzhou/HongHu scene','reuse':'Repeated across roles and historical analyses; not instrument replicas'},
            {'figure':'D1/D2','unit':'untouched confirmation units','count':'0','reuse':'Parent/parcel independence unverified'},
            {'figure':'C10 prospective','unit':'role points acquired / fits executed','count':'0 / 0','reuse':'40 new Hyperion points required only conditionally; no performance'},
            {'figure':'all','unit':'pooled unique pixels across protocols','count':'NA','reuse':'Overlap unresolved by aggregate files; no sum across S2/WP04/C7'}]
    with (delivery/'denominator-ledger.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    text='''# D-ready independent figure package

Four missing figures are supplied as embedded-font vector PDF/SVG at85 and175mm.
Choose one layout per figure; use the opt-in TeX fragments or ready English
captions. D's manuscript has not been edited. Reused D scene/class figures are
byte-identical copies; their original mean-only limitations still apply.

`receipts/` binds the independent count-to-plot checks, visual review and actual
producer SHA. `frozen/` contains only aggregate counts/metadata, never image
arrays, trained tensors, raw labels or point identities. See the denominator
ledger before quoting sample counts. S2, WP04 and C7 are different releases.
`original-receipts/` includes the original C7 deidentified review, C9 table receipt,
C10 independent audit and C11 preservation/reconstruction receipts, with exact
source identities and a separate file-hash manifest.

Restore the complete source bundle on the authorized LXC:

```
git init SOURCE
git -C SOURCE fetch /ABS/for-D/source-code.bundle HEAD:refs/heads/codex/rebuild
git -C SOURCE switch codex/rebuild
```

With the existing recorded environment and a clean source checkout, reconstruct
using the committed figures.py and audit.py commands in source/README.md. Point
--run at this for-D directory and use a fresh child --out. No private C11 input
store is needed for reconstruction from these frozen deidentified aggregates.
No training or query scoring is authorized. A fresh-environment install and
independent-host disaster recovery are not claimed.
'''
    (delivery/'README.md').write_text(text,encoding='utf8')
    bundle=delivery/'source-code.bundle';subprocess.run(['git','-C',str(root),'bundle','create',str(bundle),'HEAD'],check=True)
    subprocess.run(['git','-C',str(root),'bundle','verify',str(bundle)],check=True)
    write(delivery/'receipts/source-and-files.json',{'package_commit':a.expected_commit,'plot_commit':read(output/'figure-receipt.json')['producer'],
          'independent_verifier':audit['verifier'],'source_bundle_sha256':sha(bundle),'D_manuscript_modified':False,
          'new_predictions':0,'optimizer_updates':0,'query_arrays_read':0})
    files={str(p.relative_to(backup)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(delivery.rglob('*')) if p.is_file()}
    write(backup/'preservation-manifest.json',files);archive=backup/'preservation01.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for name in files:tar.add(backup/name,arcname=name,recursive=False)
    restore=backup/'restored01';restore.mkdir(mode=0o700);pending=set(files)
    with tarfile.open(archive) as tar:
        for member in tar:
            name=PurePosixPath(member.name);require(not name.is_absolute() and '..' not in name.parts and member.isfile() and member.name in pending,'Unexpected backup member')
            pending.remove(member.name);dest=restore/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,dest.open('xb') as target:shutil.copyfileobj(src,target)
            require(dest.stat().st_size==files[member.name]['bytes'] and sha(dest)==files[member.name]['sha256'],'Restored publication bytes differ')
    require(not pending,'Incomplete publication restore')
    checkout=backup/'source-restored01';subprocess.run(['git','init','-q',str(checkout)],check=True)
    subprocess.run(['git','-C',str(checkout),'fetch','-q',str(restore/'for-D/source-code.bundle'),'HEAD:refs/heads/codex/rebuild'],check=True)
    subprocess.run(['git','-C',str(checkout),'switch','-q','codex/rebuild'],check=True)
    require(subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()==a.expected_commit,'Restored source revision differs')
    script=checkout/'research/paper_c_paper_figures';rebuilt=restore/'for-D/rebuilt01'
    subprocess.run([sys.executable,str(script/'figures.py'),'--expected-commit',a.expected_commit,'--run',str(restore/'for-D'),'--out',str(rebuilt)],check=True)
    subprocess.run([sys.executable,str(script/'audit.py'),'--expected-commit',a.expected_commit,'--run',str(restore/'for-D'),'--output',str(rebuilt),'--out',str(backup/'restored-independent-audit.json')],check=True)
    regenerated=[]
    for path in (output/'figures').iterdir():
        require(sha(path)==sha(rebuilt/'figures'/path.name),'Restored-input figure regeneration differs');regenerated.append(path.name)
    receipt={'status':'PASSED','package_commit':a.expected_commit,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
             'files_restored':len(files),'file_differences':0,'regenerated_asset_files':len(regenerated),'regenerated_differences':0,
             'restored_independent_audit':'PASSED','same_host_backup_only':True,'private_payload_or_point_IDs_exported':False}
    write(backup/'recovery-receipt.json',receipt)
    export=backup/'D-ready-figures.zip'
    with zipfile.ZipFile(export,'x',compression=zipfile.ZIP_DEFLATED) as zip:
        for name in files:zip.write(backup/name,name)
        zip.write(backup/'recovery-receipt.json','recovery-receipt.json')
    write(backup/'D-ready-figures.receipt.json',{'zip_bytes':export.stat().st_size,'zip_sha256':sha(export),'files':files,
          'recovery_receipt_sha256':sha(backup/'recovery-receipt.json'),'files_with_private_weights_arrays_or_point_IDs':0})
    print(json.dumps(receipt))


if __name__=='__main__':main()
