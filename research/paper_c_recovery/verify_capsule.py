"""Inspect all priority recovery states on a CPU without model construction."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--unpacked',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    unpacked=Path(a.unpacked);index=json.loads((unpacked/'object-index01.json').read_text());root=unpacked/'restored'
    if index['schema']!='paper_c_priority_recovery_capsule_v1':raise ValueError('Explicit priority capsule required')
    for row in index['files']:
        f=root/row['name']
        if f.stat().st_size!=row['bytes'] or sha(f)!=row['sha256']:raise ValueError('Physical capsule file differs')
    current=selected=initial=0
    for case in index['recovery']:
        for kind,row in case['checkpoints'].items():
            f=root/row['name']
            if sha(f)!=row['sha256']:raise ValueError('Recovery pointer differs')
            state=torch.load(f,weights_only=True,map_location='cpu')
            step=0 if kind=='initial' else case['selected_step'] if kind=='selected' else case['updates']
            if state['step']!=step or any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('Recovery state cursor/model differs')
            adam=state['optimizer']['state']
            if (step==0 and adam) or (step>0 and (not adam or any(float(s['step'])!=step for s in adam.values()))):raise ValueError('Recovery Adam cursor differs')
            if set(state['rng'])!={'python','numpy','torch_cpu'} or state['rng']['torch_cpu'].dtype!=torch.uint8:raise ValueError('Recovery RNG envelope incomplete')
            if not state['module_modes'] or any(type(v) is not bool for v in state['module_modes'].values()):raise ValueError('Module modes incomplete')
            if kind=='current':
                current+=1
                if state['best']['step']!=case['selected_step']:raise ValueError('Current selection pointer differs')
            elif kind=='selected':selected+=1
            else:initial+=1
    receipt={'status':'VERIFIED_PRIORITY_RECOVERY_STATES','cases':len(index['recovery']),'current_states':current,'selected_states':selected,'initial_states':initial,
        'all_physical_file_bytes_and_SHA_verified':True,'new_model_constructions':0,'new_neural_calls':0,'new_optimizer_updates':0,
        'omitted_intermediate_checkpoint_files':index['omitted_intermediate_checkpoint_files'],
        'omitted_intermediate_checkpoint_bytes':index['omitted_intermediate_checkpoint_bytes'],
        'limit':index['limit'],'exact_cross_host_optimizer_continuation_exercised':False}
    with Path(a.out).open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
