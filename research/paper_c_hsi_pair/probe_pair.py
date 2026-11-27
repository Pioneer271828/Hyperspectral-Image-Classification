"""Exercise both registered directions with actual common-label data, zero models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from c6_bridge import canonical,sha,frozen_source,check_source
from pair_inputs import PairInputs,DIRECTIONS

PHASES=('source_fit','source_development','target_adapt','target_development')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=frozen_source(a.expected_commit,a.out)
    kw={'expected_commit':a.expected_commit,'run_root':a.run_root,'inputs':a.inputs,'plan_file':a.plan};directions=[];unique=set();presentations=0;rejected=[]
    for request in ({'direction':DIRECTIONS[0],'purpose':'training'},{'direction':('xuzhou','xuzhou')},{'direction':DIRECTIONS[0],'purpose':'independent_confirmation'}):
        try:PairInputs(**kw,**request)
        except ValueError as e:rejected.append({'kind':'constructor','reason':str(e),'spectral_arrays_opened':0})
        else:raise AssertionError('Unregistered purpose/direction admitted')
    for direction in DIRECTIONS:
        adapter=PairInputs(**kw,direction=direction);batches=[];phases=[]
        before=sum(b.payload_reads for b in adapter.banks.values())
        wrong_tile=adapter.tile_ids('source_development')[0]
        for phase,tile in [('query',adapter.tile_ids('source_fit')[0]),('source_fit',wrong_tile),('confirmation',wrong_tile)]:
            try:adapter.batch(phase,tile)
            except ValueError as e:rejected.append({'kind':'phase_or_tile','reason':str(e),'spectral_arrays_opened':sum(b.payload_reads for b in adapter.banks.values())-before})
            else:raise AssertionError('Unreleased phase/tile admitted')
        scene=adapter._plan['scenes'][0];old=scene['roles']['devtrain'][0]['common_id'];scene['roles']['devtrain'][0]['common_id']=1-old
        try:
            try:adapter.batch('source_fit',adapter.tile_ids('source_fit')[0])
            except ValueError as e:rejected.append({'kind':'plan_tamper','reason':str(e),'spectral_arrays_opened':sum(b.payload_reads for b in adapter.banks.values())-before})
            else:raise AssertionError('In-memory plan mutation admitted')
        finally:scene['roles']['devtrain'][0]['common_id']=old
        for phase in PHASES:
            counts=Counter();points=0
            for tile in adapter.tile_ids(phase):
                b=adapter.batch(phase,tile);array=b['patches_nchw'];targets=b['common_targets'];bands=adapter.banks[b['scene']].pin['shape'][2]
                if array.shape!=(len(targets),bands,7,7) or not np.isfinite(array).all() or b['processed_column_ids']!=tuple(range(bands)):raise AssertionError('Actual common input shape/columns differ')
                if array.flags.writeable or targets.flags.writeable:raise AssertionError('Mutable returned input')
                counts.update(int(c) for c in targets);points+=len(targets);presentations+=len(targets)
                for identity in b['point_ids']:unique.add((b['scene'],b['original_role'],identity))
                batches.append({'scene':b['scene'],'phase':phase,'original_role':b['original_role'],'tile_id':tile,'point_ids':b['point_ids'],'common_targets':targets.tolist(),'patch_shape':list(array.shape),'patch_sha256':sha(array.tobytes()),'original_npy_sha256':b['original_npy_sha256']})
            if dict(counts)!={0:5,1:5} or points!=10:raise AssertionError('Original common-class label budget changed')
            phases.append({'phase':phase,'scene':b['scene'],'original_role':b['original_role'],'common_class_counts':dict(counts),'points':points,'blocks':len(adapter.tile_ids(phase))})
        directions.append({'source':direction[0],'target':direction[1],'phases':phases,'batches':batches,'payload_reads':sum(b.payload_reads for b in adapter.banks.values()),'payload_bytes':sum(b.payload_bytes for b in adapter.banks.values())})
    if any(r['spectral_arrays_opened'] for r in rejected) or len(unique)!=40 or presentations!=80 or 'torch' in sys.modules:raise AssertionError('Unexpected payload/label/model scope')
    check_source(a.expected_commit);result={'schema':'paper_c_C7_private_pair_probe_v1','producer':a.expected_commit,'passed':True,'plan_sha256':adapter.admission['plan_sha256'],'directions':directions,'unique_original_points':len(unique),'point_presentations_two_directions':presentations,'rejected_requests_before_arrays':rejected,'torch_imported':False,'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'new_raw_GT_or_query_reads':0,'independent_confirmation':False};out.write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'producer':a.expected_commit,'plan_sha256':result['plan_sha256'],'directions':2,'unique_original_points':len(unique),'point_presentations':presentations,'rejected_requests_before_arrays':len(rejected),'payload_reads':sum(d['payload_reads'] for d in directions),'new_model_calls':0,'new_optimizer_updates':0}))


if __name__=='__main__':main()
