"""Independent byte/role/selection/TIFF checks; imports no intake producer or model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import zlib

import numpy as np
import tifffile
from PIL import Image

TRAIN=set('D006_2020 D007_2020 D008_2019 D009_2019 D013_2020 D016_2020 D017_2018 D021_2020 D023_2020 D030_2021 D032_2019 D033_2021 D034_2021 D035_2020 D038_2021 D041_2021 D044_2020 D046_2019 D049_2020 D051_2019 D052_2019 D055_2018 D060_2021 D063_2019 D070_2020 D072_2019 D074_2020 D078_2021 D080_2021 D081_2020 D086_2020 D091_2021'.split())


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lzw_strip(raw,cap):
    """Independent TIFF6 section13 decoder, MSB-first codes and early width change.

    Specification: https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf
    No external source code copied. Bounded to the declared image sample bytes.
    """
    table={i:bytes([i]) for i in range(256)};width=9;next_code=258;pos=0;previous=None;output=bytearray();cleared=False
    while pos+width<=8*len(raw):
        start=pos//8;shift=pos%8;bits=int.from_bytes(raw[start:start+3],'big');available=8*len(raw[start:start+3]);code=(bits>>(available-shift-width))&((1<<width)-1);pos+=width
        if code==256:
            table={i:bytes([i]) for i in range(256)};width=9;next_code=258;previous=None;cleared=True;continue
        if not cleared:raise ValueError('TIFF LZW strip must begin with clear code')
        if code==257:return bytes(output)
        if code in table:entry=table[code]
        elif code==next_code and previous is not None:entry=previous+previous[:1]
        else:raise ValueError('Invalid TIFF LZW dictionary code')
        output.extend(entry)
        if len(output)>cap:raise ValueError('TIFF LZW exceeds declared decoded sample budget')
        if previous is not None:
            if next_code>4095:raise ValueError('TIFF LZW table requires clear code')
            table[next_code]=previous+entry[:1];next_code+=1
            if width<12 and next_code==(1<<width)-1:width+=1
        previous=entry
    raise ValueError('TIFF LZW strip has no end code')


def manual_tiff(raw):
    """Small classic, uncompressed integer TIFF parser independent of tifffile."""
    endian={'II':'<','MM':'>'}.get(raw[:2].decode('ascii'))
    if not endian or struct.unpack_from(endian+'H',raw,2)[0]!=42:raise ValueError('Auditor requires classic TIFF')
    offset=struct.unpack_from(endian+'I',raw,4)[0];n=struct.unpack_from(endian+'H',raw,offset)[0];tags={}
    formats={1:'B',2:'c',3:'H',4:'I',12:'d'}
    for i in range(n):
        pos=offset+2+12*i;tag,kind,count=struct.unpack_from(endian+'HHI',raw,pos)
        if kind not in formats:continue
        fmt=formats[kind];size=struct.calcsize(fmt)*count
        start=pos+8 if size<=4 else struct.unpack_from(endian+'I',raw,pos+8)[0]
        if start+size>len(raw):raise ValueError('TIFF tag extends beyond bytes')
        tags[tag]=struct.unpack_from(endian+fmt*count,raw,start)
    w=tags[256][0];h=tags[257][0];c=tags.get(277,(1,))[0];bits=tags[258]
    compression=tags.get(259,(1,))[0]
    if compression not in (1,5) or tags.get(317,(1,))!=(1,) or len(set(bits))!=1 or bits[0] not in (8,16) or tags.get(339,(1,)) not in ((1,),tuple([1]*c)):raise ValueError('Auditor supports unsigned8/16 uncompressed or TIFF LZW without predictor only')
    strips=tags[273];sizes=tags[279]
    if len(strips)!=len(sizes):raise ValueError('Strip offset/size mismatch')
    chunks=[]
    for start,size in zip(strips,sizes):
        if start+size>len(raw):raise ValueError('TIFF strip exceeds member bytes')
        chunk=raw[start:start+size];chunks.append(lzw_strip(chunk,w*h*c*(bits[0]//8)) if compression==5 else chunk)
    payload=b''.join(chunks)
    if len(payload)!=w*h*c*(bits[0]//8):raise ValueError('Uncompressed TIFF sample bytes differ')
    dtype=np.dtype(endian+('u1' if bits[0]==8 else 'u2'));a=np.frombuffer(payload,dtype=dtype)
    if tags.get(284,(1,))==(1,):a=a.reshape(h,w,c)
    elif tags[284]==(2,):a=a.reshape(c,h,w).transpose(1,2,0)
    else:raise ValueError('Unknown TIFF planar layout')
    return tags,a


def check_roles(samples):
    if len(samples)!=4:raise ValueError('Exactly the four preselected intake examples required')
    for s in samples:
        if s['domain'] not in TRAIN or not s['domain'].endswith('_2020'):raise ValueError('Sample is not in registered public training year/domain')
        for k in ('aerial','labels'):
            domains=re.findall(r'D\d{3}_\d{4}',s[k]['name'])
            if domains!=[s['domain']] or '..' in Path(s[k]['name']).parts:raise ValueError('Archive path/role inconsistent')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--intake',required=True);p.add_argument('--metadata',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];out=Path(a.out).resolve()
    if os.name!='posix' or out.is_relative_to(repo) or out.exists():raise ValueError('New external Linux audit receipt required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected auditor commit required')
    root=Path(a.intake);metadata=Path(a.metadata);catalog=Path(a.catalog)
    manifest=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'plan.json').read_text());md=json.loads((metadata/'camera-metadata.json').read_text());cam_sha=digest(metadata/'camera-metadata.json')
    if digest(root/'plan.json')!=manifest['plan_sha256'] or cam_sha!=plan['camera_metadata_sha256']:raise ValueError('Frozen plan/metadata hash differs')
    available=set(Path(x['name']).stem for x in json.loads((catalog/'aerial.json').read_text()))
    expected=[]
    for prefix in ('CAMERA#','UCE-M3-'):
        domain_map={}
        for key,m in md.items():
            if m['domain'] in TRAIN and m['domain'].endswith('_2020') and m['camera'].startswith(prefix) and key in available:domain_map.setdefault(m['domain'],[]).append(key)
        expected.extend(min(domain_map[d]) for d in sorted(domain_map)[:2])
    if [s['image_id'] for s in plan['samples']]!=expected:raise ValueError('Examples differ from label-blind independent selection')
    check_roles(plan['samples']);members={r['member']:r for r in manifest['members']};results=[]
    for s in plan['samples']:
        arrays={};headers={}
        for kind in ('aerial','labels'):
            row=members[s[kind]['name']];path=root/row['object'];raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256'] or zlib.crc32(raw)!=row['crc32'] or row['crc32']!=s[kind]['crc32'] or len(raw)!=s[kind]['bytes']:raise ValueError('Independent member SHA/ZIP CRC/size mismatch')
            tags,manual=manual_tiff(raw)
            with tifffile.TiffFile(path) as tf:
                if kind=='labels':
                    with Image.open(path) as im:decoded=np.array(im)
                else:decoded=tf.asarray()
                page=tf.pages[0]
                if decoded.ndim==2:decoded=decoded[:,:,None]
                elif decoded.shape[0] in (4,5) and decoded.shape[1:]==(512,512):decoded=decoded.transpose(1,2,0)
                if not np.array_equal(decoded,manual):raise ValueError('Independent manual TIFF pixels disagree with external decoder')
                if kind=='aerial' and (tf.geotiff_metadata or {}).get('ProjectedCSTypeGeoKey')!=2154:raise ValueError('Expected aerial EPSG2154 not authenticated by GeoTIFF')
            if manual.shape!=(512,512,5 if kind=='aerial' else 1):raise ValueError('Unexpected channel count or dimensions')
            if kind=='aerial' and not np.allclose(tags[33550][:2],(0.2,0.2),atol=1e-9,rtol=0):raise ValueError('Unexpected native released pixel scale')
            arrays[kind]=manual;headers[kind]=tags
        if 33922 in headers['labels'] and headers['aerial'][33922]!=headers['labels'][33922]:raise ValueError('Image/label georeferencing differs')
        raw_classes=sorted(map(int,np.unique(arrays['labels'])))
        if not raw_classes or min(raw_classes)<1 or max(raw_classes)>19:raise ValueError('Unexpected raw label taxonomy')
        m=md[s['image_id']]
        if m['domain']!=s['domain'] or m['camera']!=s['camera_raw'] or m['date']!=s['acquisition_date']:raise ValueError('Acquisition metadata identity differs')
        tie=headers['aerial'][33922];cx=tie[3]+51.2;cy=tie[4]-51.2
        if abs(cx-m['patch_centroid_x'])>0.21 or abs(cy-m['patch_centroid_y'])>0.21:raise ValueError('TIFF extent and released metadata centroid differ')
        results.append({'sample_opaque_id':hashlib.sha256(s['image_id'].encode()).hexdigest()[:16],'camera_raw':m['camera'],'camera_prefix_group':s['camera_group_prefix'],'year':2020,'aerial_shape':[512,512,5],'spectral_channels':4,'auxiliary_height_channels':1,'dtype':str(arrays['aerial'].dtype),'raw_class_ids':raw_classes,'manual_pixels_vs_external_decoder_exact':True,'aerial_external_decoder':'tifffile','label_external_decoder':'Pillow/libtiff','image_label_dimensions_and_released_id_match':True,'label_georeferencing_present':33922 in headers['labels'],'metadata_centroid_within_one_pixel':True,'ignored_raw13plus_pixels':int(np.count_nonzero(arrays['labels']>=13)),'role':'public_train_intake_only'})
    result={'schema':'paper_c_independent_device_intake_audit_v1','auditor':a.expected_commit,'producer':manifest['producer'],'plan_sha256':manifest['plan_sha256'],'passed':True,'samples':results,'classification_intake_gate':'passed for released acquisition-ID groups, optical multispectral development only; no authenticated camera-model binding','native_camera_model_binding_authenticated':False,'internal_passband_gate':'closed; missing native spectral-response binding and controlled same-material internal-mask measurements','new_model_calls':0,'new_optimizer_updates':0,'test_or_query_payload_reads':0,'independent_confirmation':False,'limits':['Four metadata-selected examples do not establish class coverage or a representative train/validation experiment.','No causal camera-only attribution: region/date/radiometric harmonization remain confounded.','Released-ID/dimensions and available affine tags are compared; coordinate-system code may be omitted in masks. Independent ground-truth geometric registration accuracy is not certified.','Shared labels are released numerical IDs; class16/17 naming conflicts remain unresolved.']}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
