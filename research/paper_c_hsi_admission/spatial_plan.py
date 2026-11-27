"""Freeze a stricter subset of existing development roles without reading arrays."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from bank_reader import ReleasedBank,canonical
from inspect_bank import guard


def strict_plan(bank):
    shared=set(r['parent_tile'] for r in bank.role_rows('devtrain'))&set(r['parent_tile'] for r in bank.role_rows('devval'))
    roles={};original=[]
    for role in ('devtrain','devval'):
        accepted=[];counts=defaultdict(int);parents=defaultdict(set);dropped=defaultdict(int)
        for row in bank.role_rows(role):
            points=bank.samples[row['id']]
            if row['parent_tile'] in shared:
                for _,c in points:dropped[c]+=1
                continue
            for _,c in points:counts[c]+=1;parents[c].add(row['parent_tile'])
            if points:accepted.append({'original_tile_id':row['id'],'original_role':role,'original_parent_id':row['parent_tile'],'original_box':row['box'],'samples':points,'samples_sha256':hashlib.sha256(canonical(points)).hexdigest()})
        roles[role]={'tiles':accepted,'per_original_class':dict(counts),'parents_per_original_class':{c:len(p) for c,p in parents.items()},'excluded_shared_parent_points_per_class':dict(dropped)}
    complete=[c for c in range(1,bank.pin['classes']+1) if all(roles[r]['per_original_class'].get(c,0)>=5 for r in roles)]
    return {'scene':bank.scene,'bank_pin':{k:bank.pin[k] for k in ('bank','registration')},'roles':roles,'common_K5_V5_feasible_original_classes':complete,'excluded_shared_parents':sorted(shared),'context':'Original retained tile only, clipped7x7 crops with replicated edges. No cross-tile halo, whole-scene Q, or external graph.','normalizer':'Original frozen bank statistics; no subset or validation refit.','membership_limit':'Strict subset of original devtrain/devval, no reassignment or new label IDs.','exposed_development_validation':True,'independent_confirmation':False,'new_array_reads':bank.payload_reads}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=guard(a.expected_commit,a.out);root=Path(a.root)
    scenes=[strict_plan(ReleasedBank(root/scene,scene,portable_index=root/'index.json')) for scene in ('xuzhou','whu-honghu')]
    result={'schema':'paper_c_original_HSI_strict_role_subset_v1','producer':a.expected_commit,'scenes':scenes,'shared_taxonomy_admission':'pending original-to-local semantic/version evidence; no shared-head classifier admission','new_optimizer_updates':0,'new_model_calls':0,'new_label_memberships':0,'raw_GT_or_query_reads':0}
    out.write_text(json.dumps(result,indent=2));print(json.dumps({'source':a.expected_commit,'scenes':[{'scene':x['scene'],'K5_V5_complete_classes':x['common_K5_V5_feasible_original_classes'],'excluded_shared_parent_count':len(x['excluded_shared_parents']),'new_array_reads':x['new_array_reads']} for x in scenes],'new_optimizer_updates':0}))


if __name__=='__main__':main()
