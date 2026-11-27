"""Freeze common K5/V5 subsets of original roles, without opening spectral arrays."""
import argparse
from collections import Counter
import json
from pathlib import Path

from c6_bridge import ReleasedBank,canonical,sha,frozen_source
from review_metadata import review,validate_ontology

SALT='paper-c-C7-common2-K5V5-parent-disjoint-v1'
ROLES=('devtrain','devval')
SCENES=('xuzhou','whu-honghu')


def rank(scene,role,common_id,point):
    return sha(canonical({'salt':SALT,'scene':scene,'original_role':role,'common_id':common_id,'point_id':point['point_id'],'original_label':point['original_label']}))


def select_scene(bank,ontology):
    validate_ontology(ontology);key='xuzhou_original_ids' if bank.scene=='xuzhou' else 'honghu_original_ids'
    shared={r['parent_tile'] for r in bank.role_rows('devtrain')}&{r['parent_tile'] for r in bank.role_rows('devval')}
    groups={};counts={}
    for role in ROLES:
        selected=[];group_counts={}
        for definition in ontology['classes']:
            common=definition['common_id'];original=set(definition[key]);candidates=[];all_count=0
            for row in bank.role_rows(role):
                for identity,label in bank.samples[row['id']]:
                    if label not in original:continue
                    all_count+=1
                    if row['parent_tile'] not in shared:candidates.append({'tile_id':row['id'],'parent_id':row['parent_tile'],'point_id':identity,'original_label':label,'common_id':common})
            if len(candidates)<5:raise ValueError('Common class lacks original K5/V5 after parent exclusion; no role reassignment/fallback')
            chosen=sorted(candidates,key=lambda p:(rank(bank.scene,role,common,p),p['point_id']))[:5];selected.extend(chosen)
            group_counts[str(common)]={'original_candidates':all_count,'retained_candidates':len(candidates),'selected':5,'selected_original_subclass_counts':dict(Counter(p['original_label'] for p in chosen))}
        if len(selected)!=10 or len({p['point_id'] for p in selected})!=10:raise ValueError('Common class membership duplicated')
        groups[role]=sorted(selected,key=lambda p:(p['tile_id'],p['point_id']));counts[role]=group_counts
    train_parents={p['parent_id'] for p in groups['devtrain']};val_parents={p['parent_id'] for p in groups['devval']}
    if train_parents&val_parents or {p['point_id'] for p in groups['devtrain']}&{p['point_id'] for p in groups['devval']}:raise ValueError('Original fitting/validation separation differs')
    boxes={role:[bank.rows[i]['box'] for i in sorted({p['tile_id'] for p in groups[role]})] for role in ROLES};overlap=0;gaps=[]
    for a in boxes['devtrain']:
        for b in boxes['devval']:
            overlap+=max(0,min(a[1],b[1])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[2],b[2]));gaps.append(max(b[0]-a[1],a[0]-b[1],b[2]-a[3],a[2]-b[3],0))
    if overlap:raise ValueError('Selected original observation boxes overlap')
    return {'scene':bank.scene,'original_bank_pin':{k:bank.pin[k] for k in ('bank','registration')},'shape':list(bank.pin['shape']),'roles':groups,'counts':counts,'excluded_shared_parent_count':len(shared),'selected_parent_overlap':0,'cross_role_observation_box_overlap_pixels':0,'minimum_role_box_gap_pixels':min(gaps),'normalization_statistics_sha256':sha(canonical(bank.registration['statistics'])),'processed_band_metadata_sha256':sha(canonical(bank.registration['bands'])),'new_array_reads':bank.payload_reads}


def identity(root,inputs):
    facts=review(root,inputs)
    if not facts['passed'] or facts['new_array_reads']:raise ValueError('Pinned metadata-only admission required')
    ontology=json.loads((Path(__file__).parent/'ontology.json').read_text());scenes=[]
    for scene in SCENES:scenes.append(select_scene(ReleasedBank(inputs/scene,scene,portable_index=inputs/'index.json'),ontology))
    return {'schema':'paper_c_C7_common2_private_plan_v1','selection_salt':SALT,'K_per_common_class':5,'V_per_common_class':5,'common_classes':['exposed bare ground','tree canopy'],'ontology_sha256':sha(canonical(ontology)),'metadata_review_sha256':sha(canonical(facts)),'scenes':scenes,'classification_scope':facts['scope'],'consumer_purpose':'development_input_probe_only','directions':[['xuzhou','whu-honghu'],['whu-honghu','xuzhou']],'context':'Each selected original labeled center receives a7x7 normalized patch, clipped to its original authorized tile with replicated edges. No new halo, whole-scene graph or Q. File I/O reads the complete original authorized tile; models would receive only the returned patches.','normalization':'Original frozen per-scene devtrain statistics, including their historical broader unlabeled training footprint; no common-class/validation refit. All future arms must use identical preprocessing/exposure accounting.','scale_limit':'Same7x7 pixel context is not equal ground area: published nominal GSD0.73m versus approximately0.043m. No spatial-scale or causal-device isolation claim.','heldout_confirmation_admitted':False,'new_label_memberships':0,'new_model_calls_authorized':0,'new_optimizer_updates_authorized':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);body=identity(Path(a.run_root),Path(a.inputs));value={'producer':a.expected_commit,'plan_sha256':sha(canonical(body)),'plan':body};out.write_text(json.dumps(value,indent=2));print(json.dumps({'producer':a.expected_commit,'plan_sha256':value['plan_sha256'],'common_classes':body['common_classes'],'scenes':[{'scene':s['scene'],'counts':s['counts'],'cross_role_parent_overlap':s['selected_parent_overlap'],'minimum_role_box_gap_pixels':s['minimum_role_box_gap_pixels']} for s in body['scenes']],'new_array_reads':0,'new_label_memberships':0,'new_optimizer_updates':0}))


if __name__=='__main__':main()
