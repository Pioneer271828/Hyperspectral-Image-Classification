"""Independently re-read original role-authorized NPY bytes and compare reader batches."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from bank_reader import ReleasedBank
from batch import tile_batch
from inspect_bank import guard


def independent_npy(path,expected_shape):
    raw=path.read_bytes()
    if raw[:8]!=b'\x93NUMPY\x01\x00':raise ValueError('Independent reader requires original NPYv1')
    size=struct.unpack_from('<H',raw,8)[0]
    if size>4096:raise ValueError('Independent header exceeds cap')
    header=ast.literal_eval(raw[10:10+size].decode('latin1'))
    if header!={'descr':'<f4','fortran_order':False,'shape':expected_shape}:raise ValueError('Independent header differs from original tile shape')
    if len(raw)-10-size!=np.prod(expected_shape)*4:raise ValueError('Independent sample byte count differs')
    return np.frombuffer(raw,dtype='<f4',offset=10+size).reshape(expected_shape)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=guard(a.expected_commit,a.out);root=Path(a.root);report=[]
    for scene in ('xuzhou','whu-honghu'):
        bank=ReleasedBank(root/scene,scene,portable_index=root/'index.json');roles=[]
        for role in ('devtrain','devval'):
            examples=0;tiles=0;counts={};patch_hash=hashlib.sha256()
            for row in bank.role_rows(role):
                if not bank.samples[row['id']]:continue
                x=tile_batch(bank,role,row['id']);box=row['box'];shape=(box[1]-box[0],box[3]-box[2],bank.pin['shape'][2]);raw=independent_npy(root/scene/f"tile-{row['id']}.npy",shape)
                if not np.array_equal(raw,x['spectral_raw_hwc']):raise ValueError('Independent NPY and production values differ')
                stats=bank.registration['statistics'];low=np.array(stats['minimum'],dtype=np.float32);high=np.array(stats['maximum'],dtype=np.float32);ref=np.empty_like(raw)
                for c in range(raw.shape[2]):ref[:,:,c]=(raw[:,:,c]-low[c])/(high[c]-low[c] if high[c]>low[c] else 1)
                if not np.array_equal(ref,x['spectral_normalized_hwc']):raise ValueError('Frozen normalization differs independently')
                for i,(yy,xx) in enumerate(x['selected_local_centers']):
                    expected=np.empty((shape[2],7,7),np.float32)
                    for dy in range(-3,4):
                        for dx in range(-3,4):expected[:,dy+3,dx+3]=ref[min(max(yy+dy,0),shape[0]-1),min(max(xx+dx,0),shape[1]-1)]
                    if not np.array_equal(expected,x['patches_nchw'][i]):raise ValueError('Independent clipped patches differ')
                for c in x['labels_original']:counts[int(c)]=counts.get(int(c),0)+1
                patch_hash.update(x['patches_nchw'].tobytes());tiles+=1;examples+=len(x['labels_original'])
            if examples!=5*bank.pin['classes'] or set(counts.values())!={5}:raise ValueError('Original K5/V5 role count changed')
            roles.append({'original_role':role,'labeled_tiles':tiles,'original_labeled_points':examples,'every_original_class_count':5,'patch_shape':['N',bank.pin['shape'][2],7,7],'patches_sha256':patch_hash.hexdigest(),'independent_NPY_normalization_and_patch_checks_passed':True})
        report.append({'scene':scene,'released_processed_spectral_columns':bank.pin['shape'][2],'original_numeric_classes':bank.pin['classes'],'roles':roles,'payload_tile_reads':bank.payload_reads,'payload_bytes':bank.payload_bytes,'shared_class_semantics_admitted':False,'independent_confirmation':False})
    result={'schema':'paper_c_original_HSI_inputs_independent_audit_v1','source':a.expected_commit,'passed':True,'scenes':report,'new_model_calls':0,'new_optimizer_updates':0,'new_label_memberships':0,'raw_GT_or_query_reads':0,'scope':'Role/bytes/header/numeric-label batch intake only; a shared semantic head remains separately gated.'};out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
