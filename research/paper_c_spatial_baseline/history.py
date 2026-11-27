"""Reconstruct ONLY frozen aggregates and protocol eligibility; no inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import sha, require


def main():
    p=argparse.ArgumentParser(); p.add_argument('--s2-launch',required=True); p.add_argument('--s2-results',required=True)
    p.add_argument('--a-registration',required=True); p.add_argument('--a-probabilities',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); sources={name:sha(getattr(a,name)) for name in ['s2_launch','s2_results','a_registration','a_probabilities']}
    launch=json.loads(Path(a.s2_launch).read_text()); s2=json.loads(Path(a.s2_results).read_text())
    ar=json.loads(Path(a.a_registration).read_text()); ap=json.loads(Path(a.a_probabilities).read_text())
    require(ap['registration_sha256']==sources['a_registration'],'Historical A registration binding differs')
    totals={'paviaU':42125,'xuzhou':67837,'whu-honghu':380865}; s2_heads=0
    for row in s2['rows']:
        heads={'fused','graph','cnn'} if row['supervision']=='legacy_three_heads' else {'fused'}
        require(row['precision'] in ['fp32','fp64'] and set(row['scores'])==heads,'S2 precision/head semantics differ')
        for name, metric in row['scores'].items():
            cm=np.asarray(metric['confusion'],dtype=np.int64)
            require((cm>=0).all() and cm.sum()==totals[row['dataset']],'S2 frozen denominator differs')
            require(abs(np.trace(cm)/cm.sum()-metric['oa'])<1e-12,'S2 confusion/OA differs')
            s2_heads+=1
    old=[]
    for r in ap['cases']:
        m=r['selected']; cm=np.asarray(m['confusion'],dtype=np.int64)
        require(cm.shape==(9,9) and (cm>=0).all() and cm.sum(1).tolist()==[5]*9,'A frozen support differs')
        oa=float(np.trace(cm)/cm.sum()); require(abs(oa-m['oa'])<1e-7,'A frozen OA differs')
        old.append({'method':r['method'],'seed':r['seed'],'selected_step':m['step'],
                    'oa':oa,'nll':m['nll'],'confusion':cm.tolist(),'support':cm.sum(1).tolist()})
    require(len(old)==6 and {r['method'] for r in old}=={'native','dsnet'},'A six historical cases differ')
    paired=[]
    for seed in [64,512,2048]:
        native=next(r for r in old if r['method']=='native' and r['seed']==seed)
        dsnet=next(r for r in old if r['method']=='dsnet' and r['seed']==seed)
        paired.append({'seed':seed,'native_minus_dsnet_oa_pp':100*(native['oa']-dsnet['oa']),
                       'dsnet_minus_native_nll':dsnet['nll']-native['nll']})
    cfg=launch['cases'][0]['settings']['config']
    require(cfg['data']['train_ratio']==.01 and cfg['data']['val_ratio']==.005
            and cfg['model']['graph_mode']=='static' and cfg['model']['norm_type']=='domain_bn','S2 protocol differs')
    require(all(c['contract']['updates']==1200 for c in ar['cases']) and ar['query_reads']==0,'A exposure protocol differs')
    # Histories establish no common held-out unit IDs/protocol binding between
    # these endpoints. Confusion marginals cannot recover paired point changes.
    value={'schema':'c13_frozen_history_scope_audit_v1','sources_sha256':sources,'historical_A_cases':old,'historical_A_pairs':paired,
           'historical_A_unique_devval_points':45,'historical_A_decisions_across_seeds':135,
           'historical_A_total_updates':sum(c['contract']['updates'] for c in ar['cases']),
           'S2_source':s2['source'],'S2_frozen_aggregate_rows':len(s2['rows']),'S2_head_confusions_verified':s2_heads,
           'S2_external_matched_endpoints':0,'S2_external_paired_accuracy':None,
           'failed_matching_axes':['S2 full-scene versus bank-tile/patch spatial context','S2 1%/.5% roles versus K5/V5 bank',
                                  'S2 joint1503 updates versus independent1200-update Pavia tasks',
                                  'static graph DomainBN128 versus learned graph GroupNorm64',
                                  'calibrated macro-validation-OA versus first-min development NLL',
                                  'different objectives/capacity and previously opened query roles'],
           'cannot_recover_point_pairing_from_confusion_marginals':True,'new_optimizer_updates':0,'new_neural_calls':0,
           'new_query_reads':0,'independent_confirmation_available':False}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:v for k,v in value.items() if k not in ['historical_A_cases','S2_source']}))


if __name__=='__main__':main()
