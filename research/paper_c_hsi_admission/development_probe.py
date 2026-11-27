"""Zero-model source/target development intake; blocked shared heads read no arrays."""
import argparse
import hashlib
import json
from pathlib import Path

from admission import pair_plan,require_role
from bank_reader import ReleasedBank
from batch import tile_batch
from inspect_bank import guard


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=guard(a.expected_commit,a.out);root=Path(a.root)
    banks={s:ReleasedBank(root/s,s,portable_index=root/'index.json') for s in ('xuzhou','whu-honghu')}
    x,h=banks.values();plans=[pair_plan(x,h),pair_plan(h,x)]
    rejections=[]
    for options in ({'mode':'shared_head'},{'heldout_scenes':('new_confirmation',)}):
        try:pair_plan(x,h,**options)
        except ValueError as e:rejections.append({'request':options,'reason':str(e),'payload_reads_before':sum(b.payload_reads for b in banks.values())})
        else:raise AssertionError('Unreleased semantic/held-out request admitted')
    batches=[]
    for bank in banks.values():
        for role,purpose in [('devtrain','source_fit'),('devval','target_development')]:
            row=next(r for r in bank.role_rows(role) if bank.samples[r['id']]);require_role(bank,role,row['id'],purpose=purpose)
            batch=tile_batch(bank,role,row['id'])
            if batch['cross_dataset_shared_head_admitted'] or batch['patches_nchw'].shape[1:]!=(bank.pin['shape'][2],7,7):raise AssertionError('Native input semantics/dimensions differ')
            batches.append({'scene':bank.scene,'role':role,'purpose':purpose,'single_original_block_probe':True,'points':len(batch['labels_original']),'original_class_counts':{str(c):int((batch['labels_original']==c).sum()) for c in sorted(set(batch['labels_original']))},'patch_shape':list(batch['patches_nchw'].shape),'patches_sha256':hashlib.sha256(batch['patches_nchw'].tobytes()).hexdigest(),'normalization':batch['normalization']})
    result={'schema':'paper_c_numeric_HSI_development_probe_v1','source':a.expected_commit,'passed':True,'plans':plans,'blocked_requests':rejections,'batches':batches,'payload_tile_reads':sum(b.payload_reads for b in banks.values()),'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'raw_GT_or_query_reads':0,'scope':'Four single-block input probes; no classifier forward, learned transfer, balanced minibatch or accuracy claim.'};out.write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'source':a.expected_commit,'directions':len(plans),'input_probe_blocks':len(batches),'rejected_shared_or_confirmation_before_payload':len(rejections),'new_model_calls':0,'new_optimizer_updates':0}))


if __name__=='__main__':main()
