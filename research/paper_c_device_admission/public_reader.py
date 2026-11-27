"""Isolated public-development reader. No training or existing model entry is changed."""
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import tifffile
from PIL import Image

from catalog import train_member


def verified_object(root,row):
    train_member(row['member'])
    if row['object']!='objects/'+row['sha256'] or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Content-addressed sample path required')
    raw=(Path(root)/row['object']).read_bytes()
    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256'] or zlib.crc32(raw)!=row['crc32']:raise ValueError('Public sample object identity differs')
    return Path(root)/row['object']


def read_aerial(root,index,*,include_public_train_labels=False):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()!=manifest['plan_sha256'] or plan['role']!='public_train_intake_only':raise ValueError('Frozen public development plan required')
    s=plan['samples'][index];lookup={r['member']:r for r in manifest['members']}
    arr=tifffile.imread(verified_object(root,lookup[s['aerial']['name']]))
    if arr.shape==(5,512,512):arr=arr.transpose(1,2,0)
    if arr.shape!=(512,512,5):raise ValueError('Released optical+height layout differs')
    value={'optical_hwc':arr[:,:,:4],'height_hw':arr[:,:,4],'released_band_ids':('red','green','blue','nir'),'wavelength_nm':None,'spectral_response_authenticated':False,'camera_group_prefix':s['camera_group_prefix'],'camera_raw':s['camera_raw'],'role':'public_train_intake_only','radiometry':'released harmonized integer values; no physical reflectance conversion','native_spatial_resolution_m':0.2}
    if include_public_train_labels:
        with Image.open(verified_object(root,lookup[s['labels']['name']])) as im:raw=np.array(im).squeeze()
        if raw.shape!=(512,512) or not np.all((raw>=1)&(raw<=19)):raise ValueError('Released raw taxonomy differs')
        value['labels_hw_raw']=raw
        # A separate, explicit future protocol track; never modifies the raw mask.
        value['labels_hw_unambiguous12']=np.where(raw<=12,raw.astype(np.int16)-1,-1)
        value['ignored_raw13plus_pixels']=int(np.count_nonzero(raw>=13))
    return value


def read_satellite(root,index):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());p=json.loads((root/'plan.json').read_text())
    if hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()!=m['plan_sha256'] or p['role']!='public_train_intake_only':raise ValueError('Frozen public development satellite plan required')
    s=p['pairs'][index];lookup={r['member']:r for r in m['members']};rows={r['name'].rsplit('_',1)[1]:r for r in s['sentinel_members']}
    data=np.load(verified_object(root,lookup[rows['data.npy']['name']]),allow_pickle=False,mmap_mode='r')
    masks=np.load(verified_object(root,lookup[rows['masks.npy']['name']]),allow_pickle=False,mmap_mode='r')
    products=verified_object(root,lookup[rows['products.txt']['name']]).read_text().splitlines()
    if data.ndim!=4 or data.shape[1]!=10 or masks.shape!=(data.shape[0],2,*data.shape[2:]) or len(products)!=len(data):raise ValueError('Released data/mask/product axes differ')
    r,c=s['center_row_column']
    if r<20 or c<20 or r+20>data.shape[2] or c+20>data.shape[3]:raise ValueError('Released match cannot support the fixed context crop')
    return {'optical_tchw':data[:,:,r-20:r+20,c-20:c+20],'cloud_snow_tchw':masks[:,:,r-20:r+20,c-20:c+20],'products':products,'released_band_ids':('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'),'wavelength_nm':None,'spectral_response_authenticated':False,'role':'public_train_intake_only','released_resolution_m':10,'context_extent_m':400,'native_20m_bands_resampled_to10m':('B5','B6','B7','B8A','B11','B12'),'spatial_limit':'Corresponding larger context centered on the aerial tile; no one-to-one pixel label correspondence is asserted.'}
