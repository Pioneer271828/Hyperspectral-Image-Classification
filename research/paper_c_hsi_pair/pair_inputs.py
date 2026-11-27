"""Fixed common-label patches, with original authority checked before every read."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from c6_bridge import ReleasedBank,canonical,sha,patch_batch,check_source
from build_plan import identity

PURPOSE='development_input_probe_only'
DIRECTIONS=(('xuzhou','whu-honghu'),('whu-honghu','xuzhou'))


def phase_binding(direction,phase):
    if tuple(direction) not in DIRECTIONS:raise ValueError('Two registered independent acquisitions required')
    allowed={'source_fit':(direction[0],'devtrain'),'source_development':(direction[0],'devval'),'target_adapt':(direction[1],'devtrain'),'target_development':(direction[1],'devval')}
    if phase not in allowed:raise ValueError('No query, confirmation, causal-sensor or extra role admitted')
    return allowed[phase]


class PairInputs:
    """Data-only development adapter. It imports no neural model or optimizer.

    Protocol boundaries prevent accidental role/budget expansion. Python object
    access is not a security sandbox against an intentionally malicious caller.
    """
    def __init__(self,*,expected_commit,run_root,inputs,plan_file,direction,purpose=PURPOSE):
        if purpose!=PURPOSE:raise ValueError('Only registered data-input development probes; training and confirmation are not authorized')
        if tuple(direction) not in DIRECTIONS:raise ValueError('Two registered independent acquisitions required')
        check_source(expected_commit);self.direction=tuple(direction);self.expected_commit=expected_commit;self.inputs=Path(inputs);self.root=Path(run_root)
        self.admission=json.loads((Path(__file__).parent/'admission.json').read_text());self._plan_file=Path(plan_file)
        if self._plan_file.is_symlink() or self._plan_file.stat().st_size>1024**2:raise ValueError('Bounded original private plan required')
        envelope=json.loads(self._plan_file.read_text());body=envelope['plan']
        if envelope['producer']!=self.admission['plan_producer'] or envelope['plan_sha256']!=self.admission['plan_sha256'] or sha(canonical(body))!=self.admission['plan_sha256']:raise ValueError('Frozen common membership plan differs')
        if sha(canonical(identity(self.root,self.inputs)))!=self.admission['plan_sha256']:raise ValueError('Original metadata/custody/selection no longer reconstruct frozen plan')
        self._plan=body;self.banks={s:ReleasedBank(self.inputs/s,s,portable_index=self.inputs/'index.json') for s in DIRECTIONS[0]};self._selected={}
        for scene in body['scenes']:
            roles={}
            for role,points in scene['roles'].items():
                grouped=defaultdict(list)
                for point in points:grouped[point['tile_id']].append(point)
                roles[role]=dict(grouped)
            self._selected[scene['scene']]=roles
        if sum(b.payload_reads for b in self.banks.values()):raise AssertionError('Admission unexpectedly opened an array')

    def tile_ids(self,phase):
        scene,role=phase_binding(self.direction,phase);return tuple(sorted(self._selected[scene][role]))

    def batch(self,phase,tile_id):
        scene,role=phase_binding(self.direction,phase)
        if sha(canonical(self._plan))!=self.admission['plan_sha256']:raise ValueError('Frozen in-memory membership changed')
        if type(tile_id)!=int or tile_id not in self._selected[scene][role]:raise ValueError('Tile is outside this original phase and common-label budget')
        bank=self.banks[scene];points=self._selected[scene][role][tile_id];row=bank.rows[tile_id];original=bank.samples[tile_id]
        for point in points:
            if row['role']!=role or row['parent_tile']!=point['parent_id'] or [point['point_id'],point['original_label']] not in original:raise ValueError('Original numeric label/role/parent changed')
        data=bank.read_tile(role,tile_id);ids=tuple(p['point_id'] for p in points);patches,centers=patch_batch(data['spectral_normalized_hwc'],ids,data['original_box'],bank.pin['shape'][1]);targets=np.asarray([p['common_id'] for p in points],dtype=np.int64)
        if not set(targets.tolist())<={0,1}:raise ValueError('Unknown common class')
        patches.setflags(write=False);targets.setflags(write=False)
        return {'scene':scene,'phase':phase,'original_role':role,'tile_id':tile_id,'point_ids':ids,'local_centers':centers,'patches_nchw':patches,'common_targets':targets,'processed_column_ids':data['processed_column_ids'],'original_npy_sha256':data['npy_sha256'],'plan_sha256':self.admission['plan_sha256'],'metadata_corroborated_development_mapping':True,'native_physical_band_mapping_authenticated':False,'independent_confirmation':False}
