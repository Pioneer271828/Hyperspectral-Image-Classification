"""Bind existing released role identities and planning costs; never decode arrays."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from tasks import NATIVE,prior_module

PILOT='bddd9f59daf2a755c719de647a429d08d0932337547e65eb2e91a8070fea58b9'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect(pilot,profile,output,dependency):
    if sha(pilot)!=PILOT:raise ValueError('Prior released supervision registration differs')
    original=json.loads(Path(pilot).read_text());cost=json.loads(Path(profile).read_text())
    git=lambda *args:subprocess.check_output(['git','-C',str(dependency),*args],text=True).strip()
    if git('rev-parse','HEAD')!=NATIVE or git('status','--porcelain'):raise ValueError('Clean pinned bank/native dependency required')
    sys.path.insert(0,str(Path(dependency).resolve()))
    previous=prior_module(Path(__file__).resolve().parents[2],'pilot')
    rows=[];protected={str(Path(pilot).resolve()):PILOT,str(Path(profile).resolve()):sha(profile)}
    for scene,old in original['scenes'].items():
        receipt=Path(old['receipt']['path'])
        if sha(receipt)!=old['receipt']['sha256']:raise ValueError('Original bank receipt differs')
        r=json.loads(receipt.read_text());registration=Path(r['root'])/'registration.json'
        release=json.loads(registration.read_text());protected[str(receipt)]=sha(receipt);protected[str(registration)]=sha(registration)
        bank,actual_receipt=previous.load_bank(SimpleNamespace(runs_root=Path(pilot).parents[1]),scene)
        if actual_receipt!=old['receipt'] or bank.pin!=old['pin'] or bank.reads!=0:raise ValueError('Bank metadata admission differs')
        if release['scene']!=scene:raise ValueError('Scene binding differs')
        roles={};sample_points={};maxh=maxw=0
        for role in ('devtrain','devval'):
            tiles=[x for x in bank.role_rows(role) if x['samples']]
            count=Counter();parents=defaultdict(set);points=set()
            for tile in tiles:
                y0,y1,x0,x1=tile['box'];maxh=max(maxh,y1-y0);maxw=max(maxw,x1-x0)
                for idx,label in tile['samples']:
                    if idx in points:raise ValueError('Released point repeats within role')
                    points.add(idx);count[label]+=1;parents[label].add(tile['parent_tile'])
            expected={c:5 for c in range(1,release['classes']+1)}
            if dict(count)!=expected:raise ValueError('Original K5/V5 budget differs')
            roles[role]={'labeled_blocks':len(tiles),'points':len(points),'per_class':dict(sorted(count.items())),
                         'parents_per_class':{k:len(v) for k,v in sorted(parents.items())}}
            sample_points[role]=points
        if sample_points['devtrain']&sample_points['devval']:raise ValueError('Existing roles share label IDs')
        if [x['id'] for x in bank.role_rows('devtrain') if x['samples']]!=old['train_roster']:
            raise ValueError('Prior train roster changed')
        if maxh>28 or maxw>28:raise ValueError('Synthetic profile does not cover maximum admitted geometry')
        costs=[]
        for p in cost['rows']:
            if p['scene']!=scene:continue
            m=p['component_median_seconds'];n=roles['devtrain']['labeled_blocks'];v=roles['devval']['labeled_blocks']
            estimates={}
            for sweeps in (3,40):
                events=4 if sweeps==3 else 9
                update_seconds=sum(m[k] for k in ('forward_seconds','backward_seconds','optimizer_seconds','checkpoint_seconds'))
                estimates[str(sweeps)]={'updates_all_three_seeds':n*sweeps*3,
                    'checkpoint_bytes_all_three_seeds':p['checkpoint_max_bytes']*(n*sweeps+1)*3,
                    'shape_based_compute_seconds_all_three_seeds':(update_seconds*n*sweeps+m['selection_seconds']*v*events)*3,
                    'selection_events_per_case':events}
            costs.append({'method':p['method'],'estimates':estimates})
        rows.append({'scene':scene,'registration_sha256':sha(registration),'bank_receipt_sha256':sha(receipt),
            'original_bank_pin':old['pin'],'shape':release['shape'],'classes':release['classes'],
            'role_counts':roles,'max_released_tile_shape':[maxh,maxw],'label_id_overlap':0,
            'source_model_is_training_initialization_for_this_pair':False,
            'normalizer_is_original_bank_frozen_training_statistics':True,
            'statistics_hash':release['statistics']['statistics_hash'],'bands_fingerprint':release['band_order'],
            'wavelength_quality':release['bands']['wavelength_quality'],'sensor_band_mapping_authenticated':False,
            'independent_calibration':False,'independent_confirmation':False,'costs':costs})
    # Hash the existing whitelist, never load any protected tensor/label array.
    for p,h in original['protected_inputs'].items():
        if sha(p)!=h:raise ValueError('Previously released bank bytes changed')
    estimates={str(s):sum(c['estimates'][str(s)]['checkpoint_bytes_all_three_seeds'] for row in rows for c in row['costs']) for s in (3,40)}
    seconds={str(s):sum(c['estimates'][str(s)]['shape_based_compute_seconds_all_three_seeds'] for row in rows for c in row['costs']) for s in (3,40)}
    free=shutil.disk_usage(Path(output).parent).free
    value={'schema':'paper_c_existing_role_cost_binding_v1','scenes':rows,'metadata_and_hash_only':True,
        'protected':protected,'original_protected_bank_files':len(original['protected_inputs']),
        'new_GT_or_query_arrays_decoded':0,'new_labels_released':0,'new_real_optimizer_updates':0,
        'checkpoint_bytes_estimate':estimates,'shape_based_compute_seconds_estimate':seconds,'free_bytes_at_audit':free,
        'full40_fits_without_reserve':estimates['40']<free,
        'engineering3_fits_quarter_of_free_space':estimates['3']<free*.25,
        'estimate_limits':'Synthetic 28x28 steps and one-point-per-class patch batches; varying real labeled counts, cold load, hash checks, independent audit and backup are additional. Not a statistical convergence or hard elapsed bound.'}
    with Path(output).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({k:value[k] for k in ('checkpoint_bytes_estimate','shape_based_compute_seconds_estimate','free_bytes_at_audit','full40_fits_without_reserve','engineering3_fits_quarter_of_free_space')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True);p.add_argument('--profile',required=True);p.add_argument('--out',required=True);p.add_argument('--dependency',required=True);a=p.parse_args();inspect(a.pilot,a.profile,a.out,a.dependency)
