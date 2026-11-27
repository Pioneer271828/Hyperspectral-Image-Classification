"""Audit existing HSI role metadata first; this command opens no spectral payload."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess

from bank_reader import ReleasedBank


def guard(expected,out):
    repo=Path(__file__).resolve().parents[2];out=Path(out).resolve()
    if os.name!='posix' or out.is_relative_to(repo) or out.exists():raise ValueError('New external authorized Linux output required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=expected or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected complete source commit required')
    return out


def inspect(bank):
    roles={};shared=set(r['parent_tile'] for r in bank.role_rows('devtrain'))&set(r['parent_tile'] for r in bank.role_rows('devval'))
    for role in ('devtrain','devval'):
        rows=bank.role_rows(role);counts=Counter();strict=Counter();parents=defaultdict(set)
        for row in rows:
            for _,label in bank.samples[row['id']]:
                counts[label]+=1;parents[label].add(row['parent_tile'])
                if row['parent_tile'] not in shared:strict[label]+=1
        roles[role]={'blocks':len(rows),'labeled_blocks':sum(bool(bank.samples[r['id']]) for r in rows),'labeled_points':sum(counts.values()),'per_class':dict(counts),'unshared_parent_per_class':dict(strict),'parents_per_class':{k:len(v) for k,v in parents.items()}}
    train=bank.role_rows('devtrain');val=bank.role_rows('devval');gaps=[];overlap=0
    for a in train:
        ay,ae,ax,az=a['box']
        for b in val:
            by,be,bx,bz=b['box'];overlap+=max(0,min(ae,be)-max(ay,by))*max(0,min(az,bz)-max(ax,bx));gaps.append(max(by-ae,ay-be,bx-az,ax-bz,0))
    if overlap:raise ValueError('Original development observation footprints overlap')
    return {'scene':bank.scene,'original_bank_pin':{k:v for k,v in bank.pin.items() if k in ('bank','registration')},'shape':bank.pin['shape'],'original_numeric_classes':bank.pin['classes'],'roles':roles,'shared_parent_count':len(shared),'minimum_role_box_gap_pixels':min(gaps),'cross_role_box_overlap_pixels':overlap,'normalization':'Original frozen training statistics, no C6 refit','spectral_response_authenticated':False,'spectral_payload_reads':bank.payload_reads,'independent_confirmation':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--scene',required=True);p.add_argument('--bank',required=True);p.add_argument('--labels');p.add_argument('--portable-index');p.add_argument('--out',required=True);a=p.parse_args();out=guard(a.expected_commit,a.out)
    bank=ReleasedBank(a.bank,a.scene,a.labels,portable_index=a.portable_index);result={'schema':'paper_c_existing_hsi_role_audit_v1','producer':a.expected_commit,'metadata':inspect(bank),'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'raw_GT_or_query_reads':0}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
