"""Explicit synthetic-role generator; never reads an observation or label file."""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import sha, digest, CLASSES, BANDS


def make(root, *, bands=8):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7713); groups = {}; roles = {}
    for role, offset in [('train', 0), ('devval', 100)]:
        x = rng.uniform(.2, 1.2, (2, 7, 7, bands)).astype('float32')
        path = root/f'{role}.npz'; np.savez(path, spectra=x, labels=np.array([0, 1], dtype='int64'))
        samples = []
        for i in range(2):
            gid = f'{role}-{i}'; y = offset+i*20
            groups[gid] = {'box': [y, y+10, 0, 10], 'acquisition': 'synthetic-image',
                           'original_parent': gid, 'grid': 'synthetic-grid', 'provenance': 'verified', 'source_sha256': '1'*64}
            samples.append({'index': i, 'center': [y+4, 4], 'box': [y+1, y+8, 1, 8], 'group': gid})
        roles[role] = {'path': path.name, 'sha256': sha(path), 'samples': samples,
                       'label_access': 'released_development', 'used_for_selection': role == 'devval'}
    value = {'schema': 'c13_common_patch_roles_v1', 'kind': 'synthetic', 'origin': 'generated_synthetic', 'confirmation_access': False,
             'query_access': False, 'scenes': {'toy': {'classes': 2, 'bands': bands,
             'wavelengths_nm': np.linspace(420, 900, bands).tolist(), 'grid': 'synthetic-grid',
             'effective_psf_radius_pixels': 1., 'geolocation_error_pixels': 1.,
             'normalizer': 'train_patch_minmax_no_clamp_v1', 'groups': groups, 'roles': roles}}}
    path = root/'manifest.json'; path.write_text(json.dumps(value, indent=2))
    return path


def future_shape(root):
    """Generated real-DIMENSION fixture; explicitly blocked from execution.

    Its fabricated geography tests schema behavior, never real eligibility.
    """
    root=Path(root);root.mkdir(exist_ok=False);rng=np.random.default_rng(3113);scenes={}
    for scene,c in CLASSES.items():
        bands=BANDS[scene];n=5*c;roles={};groups={}
        for role,offset in [('train',0),('devval',100000)]:
            p=root/f'{scene}-{role}.npz'
            np.savez(p,spectra=rng.uniform(.2,1.2,(n,7,7,bands)).astype('float32'),labels=np.repeat(np.arange(c,dtype='int64'),5))
            samples=[]
            for i in range(n):
                gid=f'{role}-{i}';y=offset+i*20
                groups[gid]={'box':[y,y+10,0,10],'acquisition':f'GENERATED-{scene}', 'original_parent':gid,
                             'grid':f'GENERATED-{scene}','provenance':'verified','source_sha256':sha(p)}
                samples.append({'index':i,'center':[y+4,4],'box':[y+1,y+8,1,8],'group':gid})
            roles[role]={'path':p.name,'sha256':sha(p),'samples':samples,'label_access':'released_development','used_for_selection':role=='devval'}
        scenes[scene]={'classes':c,'bands':bands,'wavelengths_nm':np.linspace(420,900,bands).tolist(),
                       'grid':f'GENERATED-{scene}','effective_psf_radius_pixels':1.,'geolocation_error_pixels':1.,
                       'normalizer':'train_patch_minmax_no_clamp_v1','groups':groups,'roles':roles}
    receipt={'schema':'c13_external_spatial_review_v1','scene_contracts_sha256':digest(scenes),
             'decision':'parent_footprints_uncertainty_and_development_release_verified',
             'reviewer':'GENERATED TEST FIXTURE; NO REAL CUSTODIAN REVIEW','confirmation_issued':False}
    rp=root/'GENERATED-review.json';rp.write_text(json.dumps(receipt,indent=2))
    value={'schema':'c13_common_patch_roles_v1','kind':'future_development','origin':'generated_synthetic',
           'confirmation_access':False,'query_access':False,'scenes':scenes,'custodian_receipt':{'path':rp.name,'sha256':sha(rp)}}
    path=root/'manifest.json';path.write_text(json.dumps(value,indent=2));return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--bands', type=int, default=8)
    p.add_argument('--future-shape',action='store_true')
    a = p.parse_args(); path = future_shape(a.out) if a.future_shape else make(a.out, bands=a.bands)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path)}))
