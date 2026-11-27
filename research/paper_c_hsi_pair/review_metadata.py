"""Verify pinned primary facts and original numeric-label custody, without GT reads."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from c6_bridge import ReleasedBank,canonical,sha,pinned_json,frozen_source

TASK_PIN='d02dc5415e945ed0df4d14839b16856b5dec013409b369ec244cae78a4b5a02c'
COVERAGE_PIN='78624b5f08fef0f7b9beb5e3c746063cf9008a64334a91c9532992c564b7c2ea'
CURATOR_PIN='e012bce1197154f6b6e81978758ae4d8ee3f11fdc374813dd7d317b74b0e789e'
DEVELOPMENT_PIN='313c37a0e1679c93d1d8344fee4ba5a50c3a69f60c1d0468cfdc56fd5d777ed5'


class Page(HTMLParser):
    def __init__(self):super().__init__();self.text=[];self.skip=0;self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
        if tag=='table':self.table=[]
        if self.table is not None and tag=='tr':self.row=[]
        if self.row is not None and tag in ('td','th'):self.cell=''
    def handle_data(self,data):
        if not self.skip and data.strip():self.text.append(data.strip())
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
        if tag in ('td','th') and self.cell is not None:self.row.append(' '.join(self.cell.split()));self.cell=None
        if tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        if tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None


def original_points(group):return sorted([list(p) for p in zip(group['indices'],group['labels'],strict=True)])


def validate_ontology(value):
    rows=value['classes']
    expected=[(0,[1,7],[3]),(1,[6],[22])]
    if [(r['common_id'],r['xuzhou_original_ids'],r['honghu_original_ids']) for r in rows]!=expected:raise ValueError('Unreviewed common-class mapping')
    table=value['xuzhou_original_TableIX']
    if [table[str(i)]['name'] for i in (1,6,7)]!=['Bareland-1','Trees','Bareland-2']:raise ValueError('Original published ordinal names differ')


def review(root,inputs):
    source=Path(__file__).parent;pins=json.loads((source/'metadata-pins.json').read_text());payload={}
    for pin in pins['files']:
        path=root/'metadata01'/pin['file'];raw=path.read_bytes()
        if path.is_symlink() or len(raw)!=pin['bytes'] or sha(raw)!=pin['sha256']:raise ValueError('Primary metadata pin differs')
        payload[pin['file']]=raw
    deposited=json.loads(payload['xuzhou-datacite01.json'])['data']['attributes']
    if deposited['doi'].lower()!='10.21227/t3c9-h862' or not any(r.get('rightsUri','').rstrip('/')=='https://creativecommons.org/licenses/by/4.0' for r in deposited['rightsList']):raise ValueError('Original nine-class DOI/license differs')
    description=deposited['descriptions'][0]['description'];provider=Page();provider.feed(payload['ieee-xuzhou-page01.html'].decode());provider_text=' '.join(provider.text)
    for token in ('500','260','436','2014','0.73','nine'):
        if token not in description or token not in provider_text:raise ValueError('Deposited and publisher product descriptors differ')
    if b'https://creativecommons.org/licenses/by/4.0/' not in payload['ieee-xuzhou-page01.html']:raise ValueError('Publisher license corroboration absent')
    readme=payload['local-Xuzhou-Readme01.txt'].decode();author=Page();author.feed(payload['xuzhou-author-page01.html'].decode());author_text=' '.join(author.text)
    for token in ('image.hdr','image.img','all_x','all_y','data.mat','68877','13000'):
        if any(token not in text for text in (readme,provider_text,author_text)):raise ValueError('Local/provider release instructions differ')
    task=pinned_json(root/'legacy01/xuzhou-original-task01.json',TASK_PIN)['target_identity'];coverage=pinned_json(root/'legacy01/xuzhou-original-coverage01.json',COVERAGE_PIN)
    custody=root/'custody01/runs';dev=pinned_json(custody/'wp06-xuzhou-route-preflight01/development/development.json',DEVELOPMENT_PIN)
    receipt=pinned_json(custody/'p4-spatial-dev-curator01/receipt.json',CURATOR_PIN);registration=pinned_json(custody/'p4-spatial-dev-curator01/registration.json',receipt['registration_sha256'])
    original_label_pin=next(v for k,v in registration['inputs'].items() if k.endswith('/Xuzhou/xuzhou_gt.mat'))
    if task['label_sha256']!=coverage['label_sha256'] or task['label_sha256']!=original_label_pin or task['image_sha256']!=dev['record']['wp02']['image']['sha256']:raise ValueError('Historical label-count identity does not reach released image/GT custody')
    ontology=json.loads((source/'ontology.json').read_text());validate_ontology(ontology);expected={k:v['count'] for k,v in ontology['xuzhou_original_TableIX'].items()}
    if {str(i):task['label_counts'][str(i)] for i in range(1,10)}!=expected or {str(r['class']):r['full'] for r in coverage['rows']}!=expected or len(set(expected.values()))!=9:raise ValueError('Published per-ID class counts disagree; inferred ordinal mapping is not admitted')
    banks={s:ReleasedBank(inputs/s,s,portable_index=inputs/'index.json') for s in ('xuzhou','whu-honghu')};x=banks['xuzhou'];record=dev['record'];original_release=pinned_json(custody/'p4-spatial-dev-curator01/xuzhou-support512-k5.json',x.registration['custody']['label_release']['sha256'])
    if x.registration['custody']['development']['sha256']!=DEVELOPMENT_PIN or x.registration['plan_hash']!=dev['identity']['plan_hash'] or tuple(task['shape'])!=x.pin['shape'] or task['class_count']!=x.pin['classes']:raise ValueError('Prepared image/shape/plan does not reach the original bank')
    for role,key,record_key in [('devtrain','support','train_support'),('devval','validation','validation_support')]:
        actual=sorted(p for row in x.role_rows(role) for p in x.samples[row['id']])
        if actual!=original_points(original_release[key]) or actual!=original_points(record[record_key]):raise ValueError('Numeric labels or point membership changed through release/bank conversion')
    independent=pinned_json(custody/'p4-spatial-dev-curator01/independent01.json',record['wp02']['curator_audit']['sha256'])
    if independent['status']!='passed' or independent['raw_gt_loaded'] or independent['query_accessed'] or receipt['original_val_test_values_examined'] or receipt['original_test_geometry_changed']:raise ValueError('Original development authority/access disclosure differs')
    hh=Page();hh.feed(payload['honghu-provider-page01.html'].decode());hh_text=' '.join(hh.text);tables=[]
    for table in hh.tables:
        classes={}
        for row in table:
            for start in (0,4):
                if len(row)>start+2 and re.fullmatch(r'C\d+',row[start]):classes[int(row[start][1:])]={'name':row[start+1],'count':int(row[start+2])}
        if len(classes)==22:tables.append(classes)
    if len(tables)!=1 or tables[0][3]['name']!='Bare soil' or tables[0][22]['name']!='Tree':raise ValueError('Unique primary HongHu22-class legend not established')
    for token in ('Nano-Hyperspec','2017','270','940','475','academic','commercial'):
        if token.lower() not in hh_text.lower():raise ValueError('Primary HongHu acquisition/terms incomplete')
    h=banks['whu-honghu']
    if h.registration['custody']['original_image']['sha256']!='394eebb36e70931711b02d127f73c641ad67a83a5bbfc5294b695b2f7336dfce' or h.pin['shape']!=(940,475,270):raise ValueError('Existing HongHu image identity/processed product differs')
    return {'schema':'paper_c_C7_metadata_corroborated_pair_v1','passed':True,'scope':'Development coarse class ontology supported by converging primary metadata and exact project custody. Local derived MAT is not publisher byte-certified; original all_x conversion script/header was not recovered.','xuzhou':{'DOI':'10.21227/t3c9-h862','license':'CC-BY4.0','retained_product':'2014 nine-class500x260x436','image_sha256':task['image_sha256'],'original_GT_file_sha256_from_existing_metadata':task['label_sha256'],'per_original_ID_counts':expected,'counts_reach_original_curator_GT_pin':True,'release_point_ID_and_raw_label_values_preserved':True,'publisher_byte_identical_certification':False,'original_curator_used_train_class_oracle':receipt['raw_gt_decoded_in_curator'],'original_curator_examined_val_test_values':False,'prior_full_label_count_audit_disclosed':coverage['disclosure']},'honghu':{'license':'Academic research with prescribed citation; commercial use prohibited','instrument':'Headwall Nano-Hyperspec17mm','acquisition':'2017-11-20','product':'940x475x270','original22_class_legend':tables[0]},'shared_ontology':ontology['classes'],'classification_scope':'Coarse cross-scene/device development; no source-class histogram or label names learned from model scores.','independent_confirmation':False,'causal_sensor_effect_identified':False,'new_array_reads':sum(b.payload_reads for b in banks.values()),'new_raw_GT_or_query_reads':0,'new_model_calls':0,'new_optimizer_updates':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out);r=review(Path(a.run_root),Path(a.inputs));r['producer']=a.expected_commit;out.write_text(json.dumps(r,indent=2));print(json.dumps({'passed':r['passed'],'producer':a.expected_commit,'new_array_reads':r['new_array_reads'],'classification_scope':r['classification_scope']}))


if __name__=='__main__':main()
