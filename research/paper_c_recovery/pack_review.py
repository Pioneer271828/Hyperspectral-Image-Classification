"""Export only source, aggregate evidence and redacted receipts for review."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('expected-commit','baseline','transfer','routes','backups','out','ended-at'):p.add_argument('--'+n,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];out=Path(a.out).resolve()
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if actual!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():
        raise ValueError('Clean expected complete source SHA required')
    if out.exists() or out.is_relative_to(repo):raise ValueError('New external review archive required')
    sources={}
    tracked=subprocess.check_output(['git','ls-files','research/paper_c_recovery'],cwd=repo,text=True).splitlines()
    for name in tracked:
        f=repo/name
        if f.is_symlink() or f.suffix not in ('.py','.ps1','.json','.md'):raise ValueError('Unexpected source export')
        sources['source/'+name]=f
    baseline=Path(a.baseline);transfer=Path(a.transfer);routes=Path(a.routes);backups=Path(a.backups)
    base=json.loads((baseline/'engineering3-independent-audit01.json').read_text())
    adapted=json.loads((transfer/'lineage-audit01.json').read_text())
    routed=json.loads((routes/'five-direction-collection01.json').read_text())
    if (len(base['cases'])!=18 or base['committed_optimizer_updates']!=2286 or
        len(adapted['cases'])!=15 or adapted['status']!='PASSED' or adapted['new_target_optimizer_updates']!=2007 or
        any(r['status']!='PASSED' for r in base['cases']+adapted['cases']) or routed['cases']!=15 or routed['stages']!=540):
        raise ValueError('Actual independent evidence does not match the declared completed matrix')
    # No registration/block roster, model state, probability arrays, raw bank,
    # normalizer vectors or external implementation is exported.
    keep={
      'baseline':['engineering3-independent-audit01.json','engineering3-independent-audit01.csv',
        'actual-cli-reentry01.json','selected-reload01.json','lxc-model-inventory01.json','rmvision-model-inventory01.json'],
      'transfer':['lineage-audit01.json'],
      'routes':['five-direction-collection01.json','five-direction-collection01.csv','actual-route-reentry01.json',
        'direction-eligibility-v3-01.json','method-pair-cost-audit02.json','method-pair-cost-audit02.csv']}
    for group,root in (('baseline',baseline),('transfer',transfer),('routes',routes)):
        for name in keep[group]:
            f=root/name
            if not f.is_file():raise ValueError('Required aggregate evidence missing: '+group+'/'+name)
            sources['evidence/'+group+'/'+name]=f
        for f in sorted(root.glob('*.log')):sources['evidence/'+group+'/'+f.name]=f
    for f in sorted((routes/'audits01').glob('*.json')):sources['evidence/routes/audits01/'+f.name]=f
    for f in sorted((routes/'figures02').iterdir()):
        if f.suffix not in ('.png','.svg','.csv'):raise ValueError('Unexpected figure output')
        sources['figures/'+f.name]=f
    for f in sorted(backups.glob('*.receipt.json')):sources['receipts/source/'+f.name]=f
    for f in sorted(backups.glob('source-parse*.log')):sources['evidence/integration/'+f.name]=f
    receipts=backups/'offhost-receipts01'
    if not receipts.is_dir():raise ValueError('Off-host verification receipts required')
    for name,count in (('engineering3-state-verification02.json',18),('transfer3-state-verification01.json',15)):
        value=json.loads((receipts/name).read_text())
        if (value['status']!='VERIFIED_PRIORITY_RECOVERY_STATES' or value['cases']!=count or
            any(value[k]!=count for k in ('initial_states','selected_states','current_states')) or
            not value['all_physical_file_bytes_and_SHA_verified'] or value['new_optimizer_updates']!=0):
            raise ValueError('Critical off-host recovery-state verification incomplete')
    routing_receipt=json.loads((receipts/'routing-bytes-verification01.json').read_text())
    if (routing_receipt['status']!='VERIFIED_ALL_UNIQUE_OBJECTS_AND_RESTORED_LOGICAL_FILES' or
        routing_receipt['logical_files']!=1173 or routing_receipt['archive_sha256']!='4684a7ee159b400aa56322614b77089bfcf1e4ddfee15fca16365b4d4923d6d6'):
        raise ValueError('All-output off-host routing verification incomplete')
    for f in sorted(receipts.glob('*.json')):sources['receipts/offhost/'+f.name]=f
    def redact(raw):
        value=raw.decode('utf-8-sig')
        value=re.sub(r'/home/(?:lbx|train|rmvision)/[^\s\"\'<>;,\)]+','<PRIVATE_PATH>',value)
        value=re.sub(r'(?:10\.69\.37\.243|101\.43\.25\.120)','<PRIVATE_HOST>',value)
        value=re.sub(r'(?i)[A-Z]:[/\\][^\s\"\'<>;,\)]+','<PRIVATE_PATH>',value)
        value=re.sub(r'("(?:host|hostname|user|username|login)"\s*:\s*")[^"]*(")',r'\1<PRIVATE_IDENTITY>\2',value)
        return value.encode('utf-8')
    rows=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,f in sorted(sources.items()):
            if f.is_symlink() or f.suffix not in ('.py','.ps1','.json','.md','.csv','.log','.png','.svg'):
                raise ValueError('Unapproved review artifact')
            raw=f.read_bytes();review=raw if f.suffix=='.png' else redact(raw)
            z.writestr(name,review)
            rows.append({'name':name,'original_sha256':sha(raw),'review_sha256':sha(review),'bytes':len(review),
                         'redacted':review!=raw})
        manifest={'schema':'paper_c_redacted_review_v1','source_complete_commit':actual,'ended_at_clock_tool':a.ended_at,
          'files':rows,'new_real_optimizer_updates':4293,'matched_scratch_models':18,'new_transfer_models':15,
          'routing_cases':15,'routing_stages':540,'new_query_or_label_release':0,'independent_confirmation_directions':0,
          'critical_recovery_models_verified_offhost':33,'all_routing_outputs_verified_offhost':True,
          'all_historical_intermediate_states_backed_up_offhost':False,
          'shared_repository_acceptance':False,'CI_run':False,
          'limits':['Review copies are redacted; hashes retain original/review distinction.',
            'Source code in this ZIP is a redacted review copy. Exact reproducible source is the named Git commit/bundle.',
            'Three-sweep development engineering does not complete the pending forty-sweep efficacy study.',
            'Critical recovery states and historical intermediate-state backup have separate receipts; consult the delivery review.']}
        z.writestr('manifest.json',json.dumps(manifest,indent=2,allow_nan=False))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
        for row in rows:
            if sha(z.read(row['name']))!=row['review_sha256']:raise ValueError('Review payload hash differs')
            if row['name'].endswith('.json'):json.loads(z.read(row['name']))
    result={'status':'VERIFIED_REDACTED_REVIEW_ZIP','archive':out.name,'archive_bytes':out.stat().st_size,
            'archive_sha256':sha(out.read_bytes()),'source_complete_commit':actual,'files':len(rows),
            'model_or_observation_payloads':0,'raw_probability_payloads':0,'credentials_exported':0}
    with out.with_suffix('.receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
