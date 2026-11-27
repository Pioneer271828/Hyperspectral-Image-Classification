"""Explicit synthetic-role generator; never reads an observation or label file."""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import sha


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
    value = {'schema': 'c13_common_patch_roles_v1', 'kind': 'synthetic', 'confirmation_access': False,
             'query_access': False, 'scenes': {'toy': {'classes': 2, 'bands': bands,
             'wavelengths_nm': np.linspace(420, 900, bands).tolist(), 'grid': 'synthetic-grid',
             'effective_psf_radius_pixels': 1., 'geolocation_error_pixels': 1.,
             'normalizer': 'train_patch_minmax_no_clamp_v1', 'groups': groups, 'roles': roles}}}
    path = root/'manifest.json'; path.write_text(json.dumps(value, indent=2))
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--bands', type=int, default=8)
    a = p.parse_args(); path = make(a.out, bands=a.bands)
    print(json.dumps({'manifest': str(path), 'sha256': sha(path)}))
