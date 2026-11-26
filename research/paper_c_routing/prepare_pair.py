"""Emit an input/code-bound DSNet/native proposal without loading any tensor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PILOT='bddd9f59daf2a755c719de647a429d08d0932337547e65eb2e91a8070fea58b9'
NATIVE='395caef57fba8ab6a09223538f748179dab552f2'
FILES=('configs/experiments/m5_spatial_plain.json','model/VCASAHGCN.py','model/SAHGCN.py',
       'model/task_losses.py','model/utils.py','model/learnable_superpixel_graph.py',
       'model/research_config.py','model/checkpoint_io.py','inference/light_cnn.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot-registration',required=True)
    p.add_argument('--native-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if sha(a.pilot_registration)!=PILOT:raise ValueError('Original bank registration differs')
    pilot=json.loads(Path(a.pilot_registration).read_text());root=Path(a.native_root)
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args])
    if git('rev-parse','HEAD').decode().strip()!=NATIVE or git('status','--porcelain'):
        raise ValueError('Explicit clean native dependency required')
    draft=Path(__file__).parent/'dsnet-paired-draft.json';spec=json.loads(draft.read_text())
    cases=[]
    for scene,row in pilot['scenes'].items():
        roster=row['train_roster'];n=spec['labeled_blocks_per_sweep'][scene]
        if len(roster)!=n or len(set(roster))!=n:raise ValueError('Frozen labeled-block roster differs')
        for seed in spec['seeds']:
            for model in spec['models']:
                cases.append({'scene':scene,'seed':seed,'model':model,'train_block_ids':roster,
                              'updates':n*spec['proposed_sweeps'],
                              'selection_steps':[i*n for i in range(0,spec['proposed_sweeps']+1,5)],
                              'bank_receipt':row['receipt'],'bank_pin':row['pin'],'geometry':row['geometry']})
    if sum(c['updates'] for c in cases)!=spec['maximum_proposed_updates']:raise ValueError('Proposal budget differs')
    for name,h in pilot['protected_inputs'].items():
        if sha(name)!=h:raise ValueError('Previously released bank input changed')
    value={'schema':'paper_c_bound_external_pair_proposal_v1','status':'DRAFT_NOT_AUTHORIZED_TO_EXECUTE',
           'spec_sha256':sha(draft),'prior_bank_registration_sha256':PILOT,'native_code':NATIVE,
           'native_file_sha256':{name:hashlib.sha256(git('show',NATIVE+':'+name)).hexdigest() for name in FILES},
           'cases':cases,'maximum_proposed_updates':sum(c['updates'] for c in cases),'actual_updates':0,
           'new_tensor_or_label_arrays_loaded':0,'new_query_or_raw_gt_reads':0,
           'original_bank_inputs_unchanged':True,'protected_original_files':len(pilot['protected_inputs']),
           'blocked_before_training':['native backward/cost profile and elapsed budget estimate',
                                      'review of explicit method-specific recipe and loss confounds',
                                      'new executor recovery implementation and user training authorization']}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':value['status'],'cases':len(cases),'proposed_updates':value['maximum_proposed_updates'],'actual_updates':0}))


if __name__=='__main__':main()
