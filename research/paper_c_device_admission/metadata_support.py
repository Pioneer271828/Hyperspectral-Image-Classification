"""Describe official public-train camera/domain support without reading pixel labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from catalog import TRAIN_DOMAINS,domain_of,guard


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--metadata',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out)
    if out.exists():raise ValueError('New aggregate support receipt required')
    mp=Path(a.metadata)/'camera-metadata.json';cp=Path(a.catalog)/'aerial.json';md=json.loads(mp.read_text());cat=json.loads(cp.read_text());records=[];mismatch=0
    for row in cat:
        key=Path(row['name']).stem;domain=domain_of(row['name'])
        if domain not in TRAIN_DOMAINS:continue
        m=md[key]
        if m['domain']!=domain:raise ValueError('Catalog-to-metadata training domain join differs')
        camera=m['camera'];group='UCE' if 'UCE' in camera else 'non-UCE'
        records.append((domain,group,camera,m['date'][:4],m['date'][5:7]))
    domain_group=collections.defaultdict(collections.Counter);group=collections.Counter();camera=collections.Counter();year_group=collections.defaultdict(collections.Counter);month_group=collections.defaultdict(collections.Counter)
    for domain,g,cam,y,month in records:
        domain_group[domain][g]+=1;group[g]+=1;camera[cam]+=1;year_group[y][g]+=1;month_group[month][g]+=1
    mixed=sorted(d for d,c in domain_group.items() if len(c)>1)
    result={'schema':'paper_c_public_train_camera_domain_support_v1','producer':a.expected_commit,'camera_metadata_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'aerial_catalog_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'public_train_records':len(records),'public_train_domains':len(domain_group),'camera_groups':dict(group),'raw_camera_ids':dict(camera),'domains_by_group_count':{g:sum(g in c for c in domain_group.values()) for g in group},'mixed_group_domain_count':len(mixed),'mixed_group_domains':mixed,'year_group_counts':dict(year_group),'month_group_counts':dict(month_group),'group_rule':'Author UCE-substring vs non-UCE operational acquisition groups; no camera-model or SRF authentication.','classification_implication':'Composite cross-acquisition/domain transfer can be preregistered separately. Metadata support is not a fitted classifier result.','causal_implication':'If groups never coexist within a domain, device group is confounded with geography under this roster. Sharing a year/month does not remove this.','new_pixel_or_label_payload_reads':0,'test_or_query_payload_reads':0,'new_nn_or_optimizer_updates':0}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
