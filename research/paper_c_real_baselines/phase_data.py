"""C8 role allowlist around the unchanged C7/C6 readers, with no new decoder."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from c8_bridge import PairInputs,phase_binding,patch_batch,write,digest
from features import transform,REPRESENTATIONS


class PhaseData:
    def __init__(self,*,expected_commit,c7_root,inputs,plan,direction,allowed):
        self.direction=tuple(direction);self.allowed=frozenset(allowed);self.inputs=Path(inputs).resolve();self.adapter=PairInputs(expected_commit=expected_commit,run_root=c7_root,inputs=inputs,plan_file=plan,direction=direction);self.opens=[];self.denied=[]
        paths=set()
        for phase in self.allowed:
            scene,_=phase_binding(self.direction,phase)
            for tile in self.adapter.tile_ids(phase):paths.add((self.inputs/scene/f'tile-{tile}.npy').resolve())
        self.allowed_paths=frozenset(paths)
        def audit(event,args):
            if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
            path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
            if path.suffix.lower() in ('.pt','.pth','.ckpt','.mat'):
                self.denied.append(path.suffix);raise PermissionError('No raw-MAT or inherited checkpoint opens in C8')
            if path.suffix.lower()=='.npy':
                resolved=path.resolve()
                if resolved not in self.allowed_paths:self.denied.append('unreleased_NPY');raise PermissionError('NPY outside this registered stage/role')
                self.opens.append(str(resolved))
        sys.addaudithook(audit)

    def load(self,phase,out_prefix):
        if phase not in self.allowed:raise ValueError('Phase not available in this fitting/evaluation stage')
        began=perf_counter();features={rep:[] for rep in REPRESENTATIONS};targets=[];ids=[];tiles=[];parents=[];batches=[];scene,role=phase_binding(self.direction,phase);bank=self.adapter.banks[scene]
        for tile in self.adapter.tile_ids(phase):
            admitted=self.adapter.batch(phase,tile);original=bank.read_tile(role,tile)
            raw,_=patch_batch(original['spectral_raw_hwc'],admitted['point_ids'],original['original_box'],bank.pin['shape'][1])
            features['raw_distribution'].append(transform(raw));features['legacy_scene_distribution'].append(transform(admitted['patches_nchw']))
            targets.extend(admitted['common_targets'].tolist());ids.extend(admitted['point_ids']);tiles.extend([tile]*len(admitted['point_ids']));parents.extend([bank.rows[tile]['parent_tile']]*len(admitted['point_ids']));batches.append({'tile':tile,'original_npy_sha256':admitted['original_npy_sha256'],'points':list(admitted['point_ids'])})
        X={rep:np.concatenate(parts) for rep,parts in features.items()};y=np.asarray(targets,np.int64)
        if len(y)!=10 or [int((y==c).sum()) for c in (0,1)]!=[5,5] or len(set(ids))!=10:raise ValueError('Original K5/V5 common budget differs')
        cache=Path(str(out_prefix)+'.npz');meta=Path(str(out_prefix)+'.json')
        with cache.open('xb') as stream:np.savez(stream,**X,y=y,point_ids=np.asarray(ids,np.int64),tile_ids=np.asarray(tiles,np.int64),parent_ids=np.asarray(parents,np.int64))
        record={'scene':scene,'phase':phase,'original_role':role,'points':10,'class_counts':[5,5],'processed_columns':bank.pin['shape'][2],'batches':batches,'cache_name':cache.name,'cache_sha256':digest(cache),'feature_extraction_seconds':perf_counter()-began,'data_array_opens':len(self.opens),'declared_allowed_phase':sorted(self.allowed),'raw_descriptor_uses_fitted_population_statistics':False,'legacy_descriptor_uses_original_scene_devtrain_statistics':True,'original_bank_NPY_logical_reads':sum(b.payload_reads for b in self.adapter.banks.values()),'original_bank_NPY_logical_bytes':sum(b.payload_bytes for b in self.adapter.banks.values())}
        write(meta,record);return X,y,record


def load_cache(prefix):
    from c8_bridge import read
    prefix=Path(prefix);record=read(Path(str(prefix)+'.json'));path=prefix.parent/record['cache_name']
    if digest(path)!=record['cache_sha256']:raise ValueError('Frozen feature cache differs')
    with np.load(path,allow_pickle=False) as z:data={key:z[key].copy() for key in z.files}
    return data,record
