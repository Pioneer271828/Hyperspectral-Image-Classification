"""C13 common-footprint development admission. No scene or query reader."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

METHODS = ('none', 'x_residual', 'capacity_control', 'dsnet')
SEEDS = (64, 512, 2048)
CLASSES = {'paviaU': 9, 'xuzhou': 9, 'whu-honghu': 22}
BANDS = {'paviaU': 103, 'xuzhou': 436, 'whu-honghu': 270}
NATIVE = '395caef57fba8ab6a09223538f748179dab552f2'
AUTHOR = 'e910795073b52373ce0771ac85139a17f250b236'
LICENSE = '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'
OPTIMIZER = {'name': 'Adam', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
             'weight_decay': 0., 'amsgrad': False}
MODEL = {'latent_channels': 64, 'adapter_embed_dim': 32, 'norm_type': 'group_norm',
         'graph_mode': 'learned', 'alph': .8, 'coverage_mode': 'none', 'interaction_mode': 'none'}
LOSSES = {'class_balance': True, 'graph_loss_weights': {'compactness': .01, 'entropy': .001, 'balance': .01},
          'cnn_loss_weight': 1., 'graph_branch_loss_weight': 1.}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def box(value):
    require(isinstance(value, list) and len(value) == 4 and all(type(x) is int for x in value), 'Integer half-open box required')
    y0, y1, x0, x1 = value
    require(0 <= y0 < y1 and 0 <= x0 < x1, 'Empty or negative footprint')
    return y0, y1, x0, x1


def overlaps(a, b, margin):
    ay, az, ax, aw = a; by, bz, bx, bw = b
    # Expand BOTH footprints by the declared effective PSF/geolocation bound.
    return ay-margin < bz+margin and by-margin < az+margin and ax-margin < bw+margin and bx-margin < aw+margin


def admit(path, *, synthetic=False):
    """Inspect metadata first, then only separately released train/devval NPZs.

    Geometry assertions require external provenance; this code cannot establish
    that a custodian's geographic groups are true fields or unseen acquisitions.
    No confirmation role is accepted or opened here.
    """
    path = Path(path).resolve(); value = json.loads(path.read_text())
    require(value.get('schema') == 'c13_common_patch_roles_v1', 'Unknown role schema')
    require(value.get('kind') == ('synthetic' if synthetic else 'future_development'), 'Synthetic/real role mismatch')
    require(value.get('origin') in ['generated_synthetic','real_observation'], 'Input origin must be explicit')
    require(not synthetic or value['origin']=='generated_synthetic', 'Real observation cannot use synthetic authorization')
    require(value.get('confirmation_access') is False and value.get('query_access') is False, 'Query/confirmation access forbidden')
    require(set(value.get('scenes', {})) == ({'toy'} if synthetic else set(CLASSES)), 'Frozen scene set differs')
    protected = {str(path): sha(path)}
    if not synthetic:
        receipt = value.get('custodian_receipt', {})
        require(isinstance(receipt, dict) and hex64(receipt.get('sha256')) and isinstance(receipt.get('path'), str), 'External spatial-role provenance not admitted')
        rp = (path.parent/receipt['path']).resolve()
        require(rp.is_relative_to(path.parent) and rp.is_file() and sha(rp) == receipt['sha256'], 'Custodian receipt bytes differ')
        review = json.loads(rp.read_text())
        require(review.get('schema') == 'c13_external_spatial_review_v1'
                and review.get('scene_contracts_sha256') == digest(value['scenes'])
                and review.get('decision') == 'parent_footprints_uncertainty_and_development_release_verified'
                and isinstance(review.get('reviewer'), str) and bool(review['reviewer'])
                and review.get('confirmation_issued') is False, 'External review scope differs')
        protected[str(rp)] = receipt['sha256']
    result = {}; ledger = []
    for scene, s in value['scenes'].items():
        c = s['classes']; bands = s['bands']; groups = s['groups']
        require(type(c) is int and c >= 2 and type(bands) is int and bands >= 4, 'Invalid dimensions')
        require(synthetic or (c == CLASSES[scene] and bands == BANDS[scene]), 'Native taxonomy/band count differs')
        require(set(s['roles']) == {'train', 'devval'}, 'Only released train/devval roles accepted')
        require(s['normalizer'] == 'train_patch_minmax_no_clamp_v1', 'Unregistered normalization')
        wl = np.asarray(s['wavelengths_nm'], dtype=np.float32)
        require(wl.shape == (bands,) and np.isfinite(wl).all() and (np.diff(wl) > 0).all() and wl[0] > 0, 'Invalid wavelength metadata')
        require(isinstance(s.get('grid'), str) and bool(s['grid']), 'Common spatial grid not bound')
        psf, geo = s['effective_psf_radius_pixels'], s['geolocation_error_pixels']
        require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (psf, geo)), 'Unknown spatial uncertainty')
        require(synthetic or psf + geo > 0, 'Real uncertainty bound missing')
        margin = math.ceil(psf + geo); rows = {}; seen = set(); memberships = {}
        for role, r in s['roles'].items():
            require(r['label_access'] == 'released_development' and r['used_for_selection'] is (role == 'devval'), 'Role/selection access differs')
            require(len(r['samples']) > 0, 'No released samples')
            for i, sample in enumerate(r['samples']):
                gid = sample['group']; require(gid in groups, 'Unknown geographic parent')
                require(gid not in memberships or memberships[gid] == role, 'Parent shared across roles')
                memberships[gid] = role
                group = groups[gid]
                require(group['grid'] == s['grid'] and bool(group['acquisition']) and bool(group['original_parent']), 'Unbound group provenance')
                require(group['provenance'] == 'verified' and hex64(group['source_sha256']), 'Unverified original product')
                p = box(group['box']); q = box(sample['box'])
                require(q[1]-q[0] == q[3]-q[2] == 7 and p[0] <= q[0] < q[1] <= p[1] and p[2] <= q[2] < q[3] <= p[3], 'Patch leaves its original parent')
                require(sample['center'] == [q[0]+3, q[2]+3] and sample['index'] == i, 'Center/index order differs')
                identity = (group['acquisition'], *sample['center'])
                require(identity not in seen, 'Duplicate physical center')
                seen.add(identity)
            # All parent footprints, rather than only sampled centers, must separate.
            rows[role] = r
        train_groups = {x['group'] for x in rows['train']['samples']}
        val_groups = {x['group'] for x in rows['devval']['samples']}
        for a in train_groups:
            for b in val_groups:
                ga, gb = groups[a], groups[b]
                require((ga['acquisition'], ga['original_parent']) != (gb['acquisition'], gb['original_parent']), 'Aliased original parent across roles')
                require(not overlaps(box(ga['box']), box(gb['box']), margin), 'Expanded parent footprints overlap across roles')
        data = {}
        for role, r in rows.items():
            f = (path.parent / r['path']).resolve()
            require(f.is_relative_to(path.parent) and f.is_file() and hex64(r['sha256']) and sha(f) == r['sha256'], 'Role file escapes root or content hash differs')
            protected[str(f)] = r['sha256']
            with np.load(f, allow_pickle=False) as z:
                require(set(z.files) == {'spectra', 'labels'}, 'Unexpected or hidden role arrays')
                x, y = z['spectra'].copy(), z['labels'].copy()
            require(x.dtype == np.float32 and x.shape == (len(r['samples']), 7, 7, bands) and np.isfinite(x).all(), 'Role spectral tensor differs')
            require(y.dtype == np.int64 and y.shape == (len(x),) and (y >= 0).all() and (y < c).all(), 'Role labels differ')
            counts = np.bincount(y, minlength=c).tolist()
            k = 1 if synthetic else 5
            require(counts == [k] * c, 'Frozen per-class K5/V5 support differs')
            data[role] = {'raw': x, 'labels': y, 'samples': r['samples']}
            ledger.append({'scene': scene, 'role': role, 'points': len(y), 'per_class': counts,
                           'parents': len({p['group'] for p in r['samples']}),
                           'acquisitions': len({groups[p['group']]['acquisition'] for p in r['samples']}),
                           'margin_pixels_each_side': margin, 'array_sha256': r['sha256']})
        lo = data['train']['raw'].min(axis=(0, 1, 2)); hi = data['train']['raw'].max(axis=(0, 1, 2))
        require((hi > lo).all(), 'Training normalization has constant bands')
        for r in data.values():
            r['spectra'] = np.ascontiguousarray((r['raw']-lo)/(hi-lo), dtype=np.float32)
        result[scene] = {'classes': c, 'bands': bands, 'wavelengths': wl, 'roles': data,
                         'normalizer_sha256': hashlib.sha256(lo.tobytes()+hi.tobytes()).hexdigest()}
    return {'data': result, 'protected': protected, 'ledger': ledger,
            'manifest_sha256': protected[str(path)],
            'kind': 'synthetic_future_shape' if not synthetic and value['origin']=='generated_synthetic' else value['kind']}
