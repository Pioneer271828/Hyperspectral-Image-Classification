"""Seal a redacted review ZIP; never package tensors, weights, labels or PDFs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--ended-at',required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];suite=Path(a.suite)
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args])
    if git('rev-parse','HEAD').decode().strip()!=a.expected_commit or git('status','--porcelain'):
        raise ValueError('Clean explicit review commit required')
    entries={};manifest=[];cases=[];counts={'light_calls':0,'full_calls':0,'graph_builds':0}
    measured={'pipeline_seconds':0.,'setup_reference_seconds':0.,'first_setup_loading_seconds':0.}
    project=str(suite.parents[1]);source_project='/home/train/hsi-pioneer271828'
    def redact(raw):
        return raw.decode('utf-8').replace(project,'$PROJECT').replace(source_project,'$SOURCE_PROJECT').replace('/home/lbx','$PRIVATE_HOME').encode('utf-8')
    def add(name,raw,*,text=True,kind):
        if name in entries:raise ValueError('Duplicate review entry')
        if Path(name).suffix.lower() in ('.npy','.npz','.pt','.pth','.mat','.pdf','.bundle'):
            raise ValueError('Forbidden raw artifact extension')
        exported=redact(raw) if text else raw
        entries[name]=exported
        manifest.append({'entry':name,'source_sha256':sha(raw),'review_sha256':sha(exported),
                         'review_bytes':len(exported),'kind':kind,'redaction_applied':exported!=raw})
    for name in git('ls-files','research/paper_c_routing').decode().splitlines():
        add('source/'+name,git('show',a.expected_commit+':'+name),kind='committed_source')
    for folder in sorted(suite.iterdir()):
        if not folder.is_dir() or not (folder/'summary.json').is_file():continue
        summary=json.loads((folder/'summary.json').read_text());setup=json.loads((folder/'setup.json').read_text())
        contract=json.loads((folder/'contract.json').read_text())
        if summary['optimizer_updates']!=0 or summary['confirmation']:
            raise ValueError('Unexpected scientific scope')
        for name,h in summary['files'].items():
            if sha((folder/name).read_bytes())!=h:raise ValueError('Original producer evidence changed')
        for name,h in setup['protected'].items():
            if sha(Path(name).read_bytes())!=h:raise ValueError('Original protected input changed')
        for row in summary['results']:
            measured['pipeline_seconds']+=row['pipeline_ms']/1000
            for key in counts:counts[key]+=row['counts'][key]
        measured['setup_reference_seconds']+=setup['reference_ms']/1000
        measured['first_setup_loading_seconds']+=setup['loading_ms']/1000
        for key in counts:counts[key]+=setup['reference_counts'][key]
        cases.append({'case':folder.name,'producer':contract['code'],'stages':len(summary['results']),
                      'normalization':contract.get('normalization','released_development'),
                      'scope':contract['qualification'],'contract_sha256':sha((folder/'contract.json').read_bytes())})
        for path in sorted(folder.glob('*.json')):
            add('evidence/'+folder.name+'/'+path.name,path.read_bytes(),kind='redacted_saved_numeric_evidence')
    primary=[x for x in cases if x['normalization']=='checkpoint_original']
    if len(primary)!=9 or sum(x['stages'] for x in primary)!=324:
        raise ValueError('Nine complete original-normalization model cases required')
    for path in sorted(suite.glob('*')):
        if path.is_file() and path.suffix in ('.json','.csv','.log'):
            add('evidence/'+path.name,path.read_bytes(),kind='redacted_audit_analysis_or_log')
    figures=suite/'figures03'
    if not figures.is_dir():raise ValueError('Visually reviewed final figures required')
    for path in sorted(figures.iterdir()):
        if path.suffix not in ('.png','.svg','.csv','.json'):raise ValueError('Unexpected final figure artifact')
        add('figures/'+path.name,path.read_bytes(),text=path.suffix!='.png',kind='generated_figure_or_source_data')
    timing=json.loads((root/'research/paper_c_routing/timing.json').read_text())
    seconds=(datetime.fromisoformat(a.ended_at)-datetime.fromisoformat(timing['started_at'])).total_seconds()
    receipt={'schema':'paper_c_review_seal_v1','review_commit':a.expected_commit,'base_commit':'6b076052da3797c367f262c1ec96bd8c7f4a7328',
             'started_at':timing['started_at'],'ended_at':a.ended_at,'wall_seconds':seconds,
             'effective_work_seconds':None,'pure_wait_seconds':None,'recorded_compute_components':measured,
             'timing_limit':'Pipeline/setup/loading are measured components only, not an effective-work clock. Pure waiting/overlap was not reliably segmented. Do not claim four hours.',
             'cases':cases,'primary_cases':len(primary),'real_development_top_level_calls_including_setup':counts,
             'new_optimizer_updates':0,'new_query_reads':0,'new_raw_GT_arrays_decoded':0,'formal_confirmation_cases':0,
             'repository_acceptance':'FAILED_INHERITED_STALE_CURRENT_STATUS_PROJECTION; no shared ledger modified; no CI claim',
             'contents_policy':'No raw observation/point-label arrays, dense probabilities, real weights, normalizer values, original PDFs, external implementation or credentials included. Aggregate numeric results are included.',
             'source_identity_note':'Source hashes pin originals; review hashes pin redacted exports. These are different when paths were anonymized.',
             'entries':manifest}
    entries['review-receipt.json']=json.dumps(receipt,indent=2,allow_nan=False).encode()
    target=Path(a.out)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in entries.items():z.writestr(name,raw)
    receipt.update(zip_name=target.name,zip_sha256=sha(target.read_bytes()),zip_members=len(entries))
    with target.with_suffix('.receipt.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('zip_name','zip_sha256','zip_members','primary_cases','wall_seconds','new_optimizer_updates')}))


if __name__=='__main__':main()
