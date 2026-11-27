"""Independent NPY header, product, mask and released-pair audit (no producer imports)."""
import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import zlib

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manual_npy(path):
    with Path(path).open('rb') as f:
        first=f.read(8)
        if first[:6]!=b'\x93NUMPY' or first[6] not in (1,2):raise ValueError('Unsupported NPY magic/version')
        width=2 if first[6]==1 else 4;n=struct.unpack('<H' if width==2 else '<I',f.read(width))[0]
        if n>65536:raise ValueError('NPY header exceeds audit cap')
        header=ast.literal_eval(f.read(n).decode('latin1'));offset=f.tell()
    dtype=np.dtype(header['descr']);shape=header['shape']
    if dtype.hasobject or header['fortran_order'] or len(shape)!=4 or any(type(x)!=int or x<=0 for x in shape):raise ValueError('Unsafe/unsupported NPY axes or object dtype')
    if Path(path).stat().st_size!=offset+math.prod(shape)*dtype.itemsize:raise ValueError('NPY byte count and shape disagree')
    return np.memmap(path,mode='r',dtype=dtype,offset=offset,shape=shape,order='C')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--paired',required=True);p.add_argument('--intake',required=True);p.add_argument('--metadata',required=True);p.add_argument('--aerial-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];out=Path(a.out).resolve()
    if os.name!='posix' or out.is_relative_to(repo) or out.exists():raise ValueError('New external Linux receipt required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected audit commit required')
    root=Path(a.paired);parent=Path(a.intake);meta=Path(a.metadata);m=json.loads((root/'manifest.json').read_text());p=json.loads((root/'plan.json').read_text());original=json.loads((parent/'plan.json').read_text());matching=json.loads((meta/'centroid-matching.json').read_text())
    if sha(root/'plan.json')!=m['plan_sha256'] or sha(parent/'plan.json')!=p['parent_plan_sha256'] or sha(meta/'centroid-matching.json')!=p['matching_sha256']:raise ValueError('Paired plan or source identities differ')
    aa=json.loads(Path(a.aerial_audit).read_text())
    if not aa['passed'] or aa['plan_sha256']!=sha(parent/'plan.json'):raise ValueError('Independent aerial admission receipt required')
    admission={x['sample_opaque_id']:x['metadata_center_binding_admitted'] for x in aa['samples']}
    if m['combined_parent_network_bytes']>m['shared_budget_bytes'] or len(p['pairs'])!=2:raise ValueError('Shared byte/pair budget differs')
    members={x['member']:x for x in m['members']};results=[]
    for s in p['pairs']:
        source=[x for x in original['samples'] if x['image_id']==s['image_id']]
        if len(source)!=1 or source[0]['domain']!=s['domain'] or source[0]['zone']!=s['zone'] or not s['domain'].endswith('_2020'):raise ValueError('Pair is not one of the frozen public train aerials')
        if matching[s['image_id']+'.tif']!=s['center_row_column']:raise ValueError('Released matching center differs')
        files={}
        for x in s['sentinel_members']:
            if '/'+s['domain']+'/'+s['zone']+'/' not in x['name']:raise ValueError('Satellite member domain/area differs')
            row=members[x['name']];file=root/row['object'];raw=file.read_bytes()
            if len(raw)!=x['bytes'] or sha(file)!=row['sha256'] or zlib.crc32(raw)!=x['crc32'] or x['crc32']!=row['crc32']:raise ValueError('Independent satellite size/hash/ZIP CRC failed')
            files[x['name'].rsplit('_',1)[1]]=file
        data=manual_npy(files['data.npy']);masks=manual_npy(files['masks.npy']);products=files['products.txt'].read_text().splitlines()
        if data.shape[1]!=10 or masks.shape!=(len(data),2,*data.shape[2:]) or len(products)!=len(data):raise ValueError('Satellite axes/products disagree')
        r,c=s['center_row_column'];crop=data[:,:,r-20:r+20,c-20:c+20];mask=masks[:,:,r-20:r+20,c-20:c+20]
        if r<20 or c<20 or crop.shape!=(len(data),10,40,40) or mask.shape!=(len(data),2,40,40):raise ValueError('Released center cannot support crop')
        decoder=np.load(files['data.npy'],allow_pickle=False,mmap_mode='r')
        if not np.array_equal(crop,decoder[:,:,r-20:r+20,c-20:c+20]) or not np.isfinite(crop).all():raise ValueError('Independent manual/NumPy crop differs')
        dates=[];platforms=set()
        for line in products:
            platform=re.search(r'(SENTINEL2[AB]|S2[AB])',line,re.I);date=re.search(r'20\d{6}',line)
            if not platform or not date or 'L2A' not in line.upper():raise ValueError('Expected real Sentinel-2 L2A product identity')
            value=dt.datetime.strptime(date[0],'%Y%m%d').date()
            if value.year!=2020:raise ValueError('Product year and paired acquisition year differ')
            dates.append(value);platforms.add(platform[0])
        aerial_date=dt.date.fromisoformat(s['aerial_date']);delta=min(abs((d-aerial_date).days) for d in dates)
        results.append({'pair_opaque_id':hashlib.sha256(s['image_id'].encode()).hexdigest()[:16],'camera_group_prefix':s['camera_group_prefix'],'camera_raw':s['camera_raw'],'sentinel_platform_ids':sorted(platforms),'sequence_shape':list(data.shape),'crop_shape':list(crop.shape),'dtype':str(data.dtype),'native_released_spectral_columns':10,'acquisitions':len(data),'earliest_product':min(dates).isoformat(),'latest_product':max(dates).isoformat(),'nearest_satellite_to_aerial_days':delta,'mask_range':[int(mask.min()),int(mask.max())],'manual_npy_vs_numpy_exact':True,'released_matching_center_verified':True,'same_instant_acquisition':False,'raw_product_files_sha256':sha(files['products.txt']),'role':'public_train_intake_only'})
    result={'schema':'paper_c_independent_real_paired_intake_audit_v1','auditor':a.expected_commit,'producer':m['producer'],'passed':True,'plan_sha256':m['plan_sha256'],'pairs':results,'classification_intake_gate':'passed; real aerial/Sentinel-2 multispectral development pairs available through an isolated reader','internal_passband_gate':'closed; no native per-camera response binding or same-material controlled internal-band intervention','spatial_correspondence_limit':'Author matching-center dictionary verified. NPY products omit a geotransform, so independent physical registration accuracy and pixelwise label resampling are not certified.','shared_network_bytes':m['combined_parent_network_bytes'],'new_model_calls':0,'new_optimizer_updates':0,'test_or_query_payload_reads':0,'independent_confirmation':False}
    for row in results:
        row['metadata_center_binding_admitted']=admission[row['pair_opaque_id']]
        row['quarantine_reason']=None if row['metadata_center_binding_admitted'] else 'Paired aerial metadata center failed the frozen one-pixel geometric gate; no independent same-location claim.'
    result['qualified_pairs']=sum(x['metadata_center_binding_admitted'] for x in results)
    result['quarantined_pairs']=len(results)-result['qualified_pairs']
    result['classification_intake_gate']='partial; only pairs whose corresponding aerial passed the independent metadata-center gate are admitted'
    result['aerial_audit_sha256']=sha(a.aerial_audit)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
