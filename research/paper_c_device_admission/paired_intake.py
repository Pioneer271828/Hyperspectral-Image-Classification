"""Add two real Sentinel-2 acquisition sequences paired to preselected train aerials."""
import argparse
import json
from pathlib import Path
import time

from catalog import guard
from intake import retrieve,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--intake',required=True);p.add_argument('--metadata',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out);out.mkdir(mode=0o700);(out/'objects').mkdir(mode=0o700)
    root=Path(a.intake);metadata=Path(a.metadata);catalog=Path(a.catalog)
    parent=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'plan.json').read_text());cm=json.loads((catalog/'manifest.json').read_text())
    if sha(root/'plan.json')!=parent['plan_sha256']:raise ValueError('Parent frozen sample plan differs')
    if sha(catalog/'sentinel.json')!=cm['archives']['sentinel']['catalog_sha256']:raise ValueError('Satellite catalog differs')
    matching=json.loads((metadata/'centroid-matching.json').read_text());mdm=json.loads((metadata/'manifest.json').read_text())
    if sha(metadata/'centroid-matching.json')!=mdm['members']['centroid-matching']['sha256']:raise ValueError('Released matching dictionary differs')
    cat=json.loads((catalog/'sentinel.json').read_text());pairs=[];members=[]
    for group in ('CAMERA#','UCE-M3-'):
        candidates=[]
        for s in plan['samples']:
            if s['camera_group_prefix']!=group:continue
            rows=[r for r in cat if '/'+s['domain']+'/'+s['zone']+'/' in r['name']]
            data=[r for r in rows if r['name'].endswith('_data.npy')]
            if len(data)!=1 or len(rows)!=3:raise ValueError('Exactly one data/masks/products triple required')
            candidates.append((data[0]['compressed_bytes'],s['image_id'],s,rows))
        _,_,s,rows=min(candidates,key=lambda r:r[:2]);center=matching[s['image_id']+'.tif']
        pairs.append({'image_id':s['image_id'],'domain':s['domain'],'zone':s['zone'],'camera_raw':s['camera_raw'],'camera_group_prefix':group,'center_row_column':center,'aerial_date':s['acquisition_date'],'sentinel_members':rows});members+=rows
    selection={'schema':'paper_c_paired_public_train_plan_v1','producer':a.expected_commit,'created_unix':time.time(),'parent_plan_sha256':parent['plan_sha256'],'matching_sha256':sha(metadata/'centroid-matching.json'),'rule':'For each preselected raw camera-prefix group choose the already selected aerial with smallest catalog compressed Sentinel data member (image ID tie break); never inspect label values for selection.','role':'public_train_intake_only','pairs':pairs,'crop_shape':[40,40],'pixel_budget_shared_with_parent_bytes':268435456,'parent_network_bytes':parent['network_bytes'],'no_cloud_filter_or_temporal_average':True}
    (out/'plan.json').write_text(json.dumps(selection,indent=2));plan_sha=sha(out/'plan.json')
    rows,used=retrieve('sentinel',members,cm,out,268435456-parent['network_bytes'],request_cap=32*1024**2)
    (out/'manifest.json').write_text(json.dumps({'schema':'paper_c_paired_public_train_intake_v1','producer':a.expected_commit,'plan_sha256':plan_sha,'members':rows,'network_bytes':used,'combined_parent_network_bytes':used+parent['network_bytes'],'shared_budget_bytes':268435456,'new_training_updates':0,'new_model_calls':0,'test_or_query_payload_reads':0,'scope':'Paired spatial context and product metadata intake, no classification fit or evaluation.'},indent=2))
    print(json.dumps({'status':'PAIRED_TRAIN_MEMBERS_FETCHED','pairs':len(pairs),'members':len(rows),'network_bytes':used,'plan_sha256':plan_sha}))


if __name__=='__main__':main()
