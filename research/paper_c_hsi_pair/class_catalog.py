"""Export every original class decision from authenticated public-level facts."""
import argparse
import csv
import json
from pathlib import Path

from c6_bridge import canonical,sha,frozen_source
from review_metadata import validate_ontology


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);folder=Path(__file__).parent;admission=json.loads((folder/'admission.json').read_text());ontology=json.loads((folder/'ontology.json').read_text());validate_ontology(ontology)
    facts=json.loads(Path(a.review).read_text());facts.pop('producer',None)
    if not facts['passed'] or sha(canonical(facts))!=admission['metadata_review_sha256']:raise ValueError('Original metadata review commitment differs')
    rows=[]
    for scene,key in [('xuzhou','xuzhou_original_ids'),('whu-honghu','honghu_original_ids')]:
        definitions=ontology['xuzhou_original_TableIX'] if scene=='xuzhou' else facts['honghu']['original22_class_legend'];mapped={raw:(c['common_id'],c['name'],c['interpretation']) for c in ontology['classes'] for raw in c[key]}
        for raw,definition in sorted(definitions.items(),key=lambda row:int(row[0])):
            raw=int(raw)
            if raw in mapped:common,name,reason=mapped[raw];decision='coarse_development_mapping'
            else:
                common='';name='';decision='excluded'
                if scene=='xuzhou':reason=ontology['excluded_xuzhou_ids'][str(raw)]
                elif raw==1:reason='Object-level Red roof is not automatically Xuzhou Red-tiles.'
                elif raw==2:reason='Road is not automatically Xuzhou Cement.'
                else:reason='No authoritative cultivar/material alignment to Xuzhou Crops-1/2; excluded rather than guessed.'
            rows.append({'scene':scene,'original_id':raw,'published_original_name':definition['name'],'published_full_scene_count':definition['count'],'common_id':common,'common_name':name,'decision':decision,'reason':reason})
    if len(rows)!=31:raise ValueError('Expected all9+22 original classes')
    with out.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'source':a.expected_commit,'rows':len(rows),'sha256':sha(out.read_bytes()),'individual_point_or_coordinate_rows':0}))


if __name__=='__main__':main()
