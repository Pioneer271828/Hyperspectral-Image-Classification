"""Independently reconstruct membership ranks, raw labels and actual patch values."""
import argparse
from collections import Counter
import hashlib
import heapq
from html import unescape
import json
from pathlib import Path
import re

import numpy as np

from c6_bridge import ReleasedBank,frozen_source,check_source
from audit_inputs import independent_npy


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def checksum(raw):return hashlib.sha256(raw).hexdigest()


def independent_memberships(bank):
    raw_mapping={1:0,7:0,6:1} if bank.scene=='xuzhou' else {3:0,22:1}
    parents={role:{r['parent_tile'] for r in bank.role_rows(role)} for role in ('devtrain','devval')};blocked=parents['devtrain'].intersection(parents['devval']);result={}
    for role in parents:
        result[role]={}
        for common in (0,1):
            candidates=[]
            for row in bank.role_rows(role):
                if row['parent_tile'] in blocked:continue
                for identity,raw in bank.samples[row['id']]:
                    if raw_mapping.get(raw)!=common:continue
                    token={'salt':'paper-c-C7-common2-K5V5-parent-disjoint-v1','scene':bank.scene,'original_role':role,'common_id':common,'point_id':identity,'original_label':raw}
                    candidates.append((checksum(encoded(token)),identity,row['id'],raw,common))
            if len(candidates)<5:raise ValueError('Independent common budget infeasible')
            for _,identity,tile,raw,common in heapq.nsmallest(5,candidates):result[role][identity]=(tile,raw,common)
    if set(result['devtrain'])&set(result['devval']):raise ValueError('Independent role identity overlap')
    return result


def independent_honghu_legend(path):
    text=path.read_text();tables=re.findall(r'<table\b[^>]*>(.*?)</table\s*>',text,flags=re.S|re.I);matches=[]
    for table in tables:
        if not re.search(r'>\s*C22\s*<',table):continue
        classes={}
        for row in re.findall(r'<tr\b[^>]*>(.*?)</tr\s*>',table,flags=re.S|re.I):
            cells=[' '.join(unescape(re.sub(r'<[^>]+>',' ',c)).split()) for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]\s*>',row,flags=re.S|re.I)]
            for i,c in enumerate(cells[:-2]):
                if re.fullmatch(r'C\d+',c):classes[int(c[1:])]=(cells[i+1],int(cells[i+2]))
        if len(classes)==22:matches.append(classes)
    if len(matches)!=1 or matches[0][3]!=('Bare soil',21821) or matches[0][22]!=('Tree',4040):raise ValueError('Independent primary common-label legend differs')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--producer-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--probe',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);root=Path(a.run_root);inputs=Path(a.inputs)
    probe=json.loads(Path(a.probe).read_text());envelope=json.loads(Path(a.plan).read_text());plan=envelope['plan'];admission=json.loads((Path(__file__).parent/'admission.json').read_text())
    if probe['producer']!=a.producer_commit or not probe['passed'] or checksum(encoded(plan))!=admission['plan_sha256'] or probe['plan_sha256']!=admission['plan_sha256']:raise ValueError('Producer/plan identity differs')
    if any(probe[k] for k in ('new_model_calls','new_optimizer_updates','new_label_memberships','new_raw_GT_or_query_reads','torch_imported')) or any(r['spectral_arrays_opened'] for r in probe['rejected_requests_before_arrays']):raise ValueError('Producer scope/rejection evidence differs')
    primary=json.loads((root/'metadata01/xuzhou-datacite01.json').read_text())['data']['attributes']
    if primary['doi']!='10.21227/t3c9-h862' or not any(r.get('rightsUri','').rstrip('/')=='https://creativecommons.org/licenses/by/4.0' for r in primary['rightsList']):raise ValueError('Independent Xuzhou DOI/rights differs')
    independent_honghu_legend(root/'metadata01/honghu-provider-page01.html')
    banks={s:ReleasedBank(inputs/s,s,portable_index=inputs/'index.json') for s in ('xuzhou','whu-honghu')};selected={s:independent_memberships(b) for s,b in banks.items()}
    for scene in plan['scenes']:
        for role,points in scene['roles'].items():
            if {p['point_id']:(p['tile_id'],p['original_label'],p['common_id']) for p in points}!=selected[scene['scene']][role]:raise ValueError('Independent deterministic membership differs from frozen plan')
    spatial={}
    for scene,bank in banks.items():
        tiles={role:{p[0] for p in points.values()} for role,points in selected[scene].items()};parents={role:{bank.rows[i]['parent_tile'] for i in ids} for role,ids in tiles.items()}
        if parents['devtrain']&parents['devval']:raise ValueError('Independent selected parent overlap')
        gaps=[]
        for i in tiles['devtrain']:
            a_box=bank.rows[i]['box']
            for j in tiles['devval']:
                b_box=bank.rows[j]['box'];dy=max(0,b_box[0]-a_box[1],a_box[0]-b_box[1]);dx=max(0,b_box[2]-a_box[3],a_box[2]-b_box[3]);gaps.append(max(dx,dy))
                if min(a_box[1],b_box[1])>max(a_box[0],b_box[0]) and min(a_box[3],b_box[3])>max(a_box[2],b_box[2]):raise ValueError('Independent selected original footprint overlap')
        if bank.registration['statistics']['fit_role']!='devtrain':raise ValueError('Original normalizer does not declare development-training-only fit')
        spatial[scene]={'selected_parent_overlap':0,'selected_box_overlap_pixels':0,'minimum_box_gap_pixels':min(gaps),'original_normalizer_fit_role':'devtrain'}
    cache={};unique=set();verified_batches=0;verified_presentations=0;directions=[];seen_directions=set()
    for direction in probe['directions']:
        if (direction['source'],direction['target']) not in (('xuzhou','whu-honghu'),('whu-honghu','xuzhou')):raise ValueError('Unregistered ordered device pair')
        if (direction['source'],direction['target']) in seen_directions:raise ValueError('Repeated direction cannot replace reverse-direction coverage')
        seen_directions.add((direction['source'],direction['target']))
        role_bindings={'source_fit':(direction['source'],'devtrain'),'source_development':(direction['source'],'devval'),'target_adapt':(direction['target'],'devtrain'),'target_development':(direction['target'],'devval')};actual_sets={phase:set() for phase in role_bindings};class_counts={phase:Counter() for phase in role_bindings}
        for batch in direction['batches']:
            phase=batch['phase'];scene,role=role_bindings[phase]
            if (batch['scene'],batch['original_role'])!=(scene,role):raise ValueError('Source fitting/adaptation phase swapped roles')
            bank=banks[scene];tile=batch['tile_id'];row=bank.rows[tile];key=(scene,tile)
            if row['role']!=role:raise ValueError('Original observation role differs')
            if batch['original_npy_sha256']!=bank.members[tile]['content']['npy_sha256']:raise ValueError('Producer original array pin differs')
            if key not in cache:
                box=row['box'];shape=(box[1]-box[0],box[3]-box[2],bank.pin['shape'][2]);path=inputs/scene/f'tile-{tile}.npy';blob=path.read_bytes()
                if checksum(blob)!=batch['original_npy_sha256']:raise ValueError('Original tile digest differs')
                raw=independent_npy(path,shape);ref=np.empty_like(raw);stats=bank.registration['statistics']
                if row['representation']!='raw_fp32':raise ValueError('Independent original raw representation required')
                for c in range(shape[2]):
                    lo=np.float32(stats['minimum'][c]);hi=np.float32(stats['maximum'][c]);ref[:,:,c]=(raw[:,:,c]-lo)/(hi-lo if hi>lo else np.float32(1))
                cache[key]=(ref,len(blob))
            normalized=cache[key][0];expected_ids=sorted(identity for identity,(tid,_,_) in selected[scene][role].items() if tid==tile)
            if batch['point_ids']!=expected_ids or set(expected_ids)&actual_sets[phase]:raise ValueError('Missing/repeated/unselected point identity')
            refs=[];targets=[];y0,_,x0,_=row['box'];width=bank.pin['shape'][1]
            for identity in expected_ids:
                _,original_label,common=selected[scene][role][identity]
                if original_label==0:raise ValueError('Raw unannotated zero used as common ground')
                yy,xx=divmod(identity,width);yy-=y0;xx-=x0;patch=np.empty((normalized.shape[2],7,7),np.float32)
                for dy in range(7):
                    for dx in range(7):patch[:,dy,dx]=normalized[min(max(yy+dy-3,0),normalized.shape[0]-1),min(max(xx+dx-3,0),normalized.shape[1]-1)]
                refs.append(patch);targets.append(common);unique.add((scene,role,identity))
            value=np.stack(refs)
            if list(value.shape)!=batch['patch_shape'] or checksum(value.tobytes())!=batch['patch_sha256'] or targets!=batch['common_targets']:raise ValueError('Independent normalized patches or common labels differ')
            actual_sets[phase].update(expected_ids);class_counts[phase].update(targets);verified_batches+=1;verified_presentations+=len(targets)
        phases=[]
        for phase,(scene,role) in role_bindings.items():
            if actual_sets[phase]!=set(selected[scene][role]) or dict(class_counts[phase])!={0:5,1:5}:raise ValueError('Independent label coverage/budget differs')
            phases.append({'phase':phase,'scene':scene,'original_role':role,'original_unique_points':10,'common_class_counts':dict(class_counts[phase]),'processed_columns':banks[scene].pin['shape'][2]})
        directions.append({'source':direction['source'],'target':direction['target'],'phases':phases})
    if len(directions)!=2 or len(unique)!=40 or verified_presentations!=80 or len(cache)!=17:raise ValueError('Actual direction/point/block accounting differs')
    check_source(a.expected_commit);result={'schema':'paper_c_C7_independent_common_pair_audit_v1','auditor':a.expected_commit,'producer':a.producer_commit,'passed':True,'plan_sha256':admission['plan_sha256'],'directions':directions,'spatial':spatial,'unique_original_points':len(unique),'actual_presentations_two_directions':verified_presentations,'verified_batch_presentations':verified_batches,'independently_decoded_unique_tiles':len(cache),'unique_tile_payload_bytes':sum(v[1] for v in cache.values()),'production_tile_reads':sum(d['payload_reads'] for d in probe['directions']),'production_tile_payload_bytes':sum(d['payload_bytes'] for d in probe['directions']),'independent_primary_HongHu_legend_and_Xuzhou_CC_BY_verified':True,'membership_rank_raw_label_and_patch_checks_passed':True,'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'new_raw_GT_or_query_reads':0,'classification_scope':'Supported coarse cross-scene/device development input pair with disclosed derived-MAT mapping inference; no accuracy, canonical byte reproduction, independent confirmation or causal sensor-effect claim.'};out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('passed','auditor','producer','unique_original_points','independently_decoded_unique_tiles','verified_batch_presentations','unique_tile_payload_bytes','new_model_calls','new_optimizer_updates')}))


if __name__=='__main__':main()
