"""Freeze metadata-selected development examples, then fetch bounded official train members."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

from catalog import ARCHIVES,TRAIN_DOMAINS,domain_of,guard,train_member
from ovh_metadata import BASE
from range_zip import RangeFile,validate_member


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(metadata,catalogs):
    images={Path(r['name']).stem:r for r in catalogs['aerial']}
    labels={Path(r['name']).stem:r for r in catalogs['labels']}
    selected=[]
    # Fixed before the first label read: two lexicographically first distinct
    # training domains per raw camera-prefix group, restricted to year 2020.
    # Prefix groups are acquisition identifiers, not authenticated model/SRF IDs.
    for prefix in ('CAMERA#','UCE-M3-'):
        candidates=sorted((m['domain'],key) for key,m in metadata.items() if m['domain'] in TRAIN_DOMAINS and m['domain'].endswith('_2020') and m['camera'].startswith(prefix) and key in images)
        domains=[]
        for domain,key in candidates:
            if domain in domains:continue
            domains.append(domain);m=metadata[key];im=images[key];lab=labels['MSK_'+key.split('_',1)[1]]
            for row in (im,lab):
                train_member(row['name'])
                if domain_of(row['name'])!=domain:raise ValueError('Metadata/archive domain mismatch')
            selected.append({'image_id':key,'domain':domain,'zone':m['zone'],'camera_raw':m['camera'],'camera_group_prefix':prefix,'acquisition_date':m['date'],'acquisition_time':m['time'],'aerial':im,'labels':lab})
            if len(domains)==2:break
        if len(domains)!=2:raise ValueError('Two training domains per prefix group required')
    return selected


def retrieve(kind,members,catalog_manifest,out,remaining):
    expected=catalog_manifest['archives'][kind]
    rf=RangeFile(BASE+ARCHIVES[kind],budget=remaining,request_cap=16*1024**2)
    if rf.etag!=expected['etag'] or rf.size!=expected['archive_bytes']:raise ValueError('Archive changed since the catalog selection')
    receipts=[]
    try:
        with zipfile.ZipFile(rf) as z:
            for row in members:
                train_member(row['name']);info=z.getinfo(row['name']);validate_member(info,cap=64*1024**2)
                if (info.CRC,info.file_size,info.compress_size,info.header_offset)!=(row['crc32'],row['bytes'],row['compressed_bytes'],row['header_offset']):raise ValueError('Selected ZIP member differs from frozen catalog')
                if info.compress_size>16*1024**2:raise ValueError('Selected member exceeds per-request intake cap')
                raw=z.read(info) # zipfile verifies the complete decoded member CRC.
                digest=hashlib.sha256(raw).hexdigest();dest=out/'objects'/digest
                if dest.exists():
                    if sha(dest)!=digest:raise ValueError('Existing content-addressed object differs')
                else:dest.write_bytes(raw)
                receipts.append({'archive':kind,'member':info.filename,'bytes':len(raw),'sha256':digest,'crc32':info.CRC,'object':'objects/'+digest})
                print(json.dumps({'member':info.filename,'bytes':len(raw),'sha256':digest}),flush=True)
    finally:
        (out/(kind+'-network.json')).write_text(json.dumps({'url':BASE+ARCHIVES[kind],'etag':rf.etag,'archive_bytes':rf.size,'received_bytes':rf.received,'ranges':rf.ranges,'members':receipts},indent=2))
    return receipts,rf.received


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--metadata',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out);out.mkdir(mode=0o700);(out/'objects').mkdir(mode=0o700)
    metadata=Path(a.metadata);catalog=Path(a.catalog);cm=json.loads((catalog/'manifest.json').read_text());catalogs={k:json.loads((catalog/(k+'.json')).read_text()) for k in ARCHIVES}
    for k in ARCHIVES:
        if sha(catalog/(k+'.json'))!=cm['archives'][k]['catalog_sha256']:raise ValueError('Catalog byte identity differs')
    md=json.loads((metadata/'camera-metadata.json').read_text());mm=json.loads((metadata/'manifest.json').read_text())
    if sha(metadata/'camera-metadata.json')!=mm['members']['camera-metadata']['sha256']:raise ValueError('Camera metadata identity differs')
    samples=select(md,catalogs)
    plan={'schema':'paper_c_metadata_selected_public_train_plan_v1','producer':a.expected_commit,'created_unix':time.time(),'camera_metadata_sha256':sha(metadata/'camera-metadata.json'),'catalog_sha256':{k:sha(catalog/(k+'.json')) for k in ARCHIVES},'rule':'2020, two lexicographically first distinct official training domains per CAMERA#/UCE-M3- raw prefix; first image ID in each; no label-content-dependent selection','role':'public_train_intake_only','samples':samples,'new_optimizer_updates':0,'new_model_calls':0,'sealed_test_query_reads':0,'native_camera_model_binding_authenticated':False}
    plan_path=out/'plan.json';plan_path.write_text(json.dumps(plan,indent=2));plan_digest=sha(plan_path)
    remaining=268435456;all_members=[];net=0
    for kind in ('aerial','labels'):
        rows,used=retrieve(kind,[s[kind] for s in samples],cm,out,remaining);remaining-=used;net+=used;all_members+=rows
    (out/'manifest.json').write_text(json.dumps({'schema':'paper_c_public_train_intake_v1','producer':a.expected_commit,'plan_sha256':plan_digest,'members':all_members,'network_bytes':net,'budget_bytes':268435456,'new_optimizer_updates':0,'new_model_calls':0,'test_or_query_payload_reads':0,'classification_gate':'pending independent payload validation','internal_passband_gate':'not admitted; native SRF/binding and matched causal controls absent'},indent=2))
    print(json.dumps({'status':'PUBLIC_TRAIN_MEMBERS_FETCHED','samples':len(samples),'members':len(all_members),'network_bytes':net,'plan_sha256':plan_digest}))


if __name__=='__main__':main()
