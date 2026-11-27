"""Development confusion/loss and exact small-parent resampling sensitivity."""
import itertools

import numpy as np


def measures(y,logits,probabilities,predictions,weights=None):
    if not np.isin(y,[0,1]).all() or not np.isin(predictions,[0,1]).all():raise ValueError('Binary labels/predictions without coercion required')
    y=np.asarray(y,np.int64);z=np.asarray(logits,np.float64);p=np.asarray(probabilities,np.float64);pred=np.asarray(predictions,np.int64);w=np.ones(len(y),np.float64) if weights is None else np.asarray(weights,np.float64)
    if y.shape!=z.shape or y.shape!=p.shape or y.shape!=pred.shape or y.shape!=w.shape or np.any(w<0) or not np.isfinite(w).all() or not np.equal(w,np.floor(w)).all() or w.sum()<=0 or not np.isfinite(z).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Aligned finite binary outcomes/integer cluster counts required')
    cm=np.bincount(y*2+pred,weights=w,minlength=4).reshape(2,2);den=cm.sum(axis=1);diag=cm.diagonal();recall=[float(diag[c]/den[c]) if den[c] else None for c in (0,1)];f1=[]
    for c in (0,1):
        bottom=2*cm[c,c]+cm[c,1-c]+cm[1-c,c];f1.append(float(2*cm[c,c]/bottom) if bottom else 0.)
    oa=float(diag.sum()/w.sum());pe=float(den@cm.sum(axis=0)/(w.sum()**2));kappa=(oa-pe)/(1-pe) if pe<1 else None
    return {'denominator':int(w.sum()),'class_denominators':den.astype(int).tolist(),'confusion':cm.astype(int).tolist(),'OA':oa,'AA':float(np.mean(recall)) if all(v is not None for v in recall) else None,'recall':recall,'macroF1':float(np.mean(f1)),'NLL':float(w@(np.logaddexp(0,z)-y*z)/w.sum()),'binary_Brier':float(w@((p-y)**2)/w.sum()),'kappa':float(kappa) if kappa is not None else None}


def cluster_weights(parents):
    parents=np.asarray(parents);clusters=sorted(set(parents.tolist()));m=len(clusters)
    if not 1<=m<=3:raise ValueError('Registered small-parent enumeration bound')
    result=[]
    for draws in itertools.product(range(m),repeat=m):
        counts=np.bincount(draws,minlength=m);result.append(np.array([counts[clusters.index(p)] for p in parents],np.float64))
    return result


def percentile(values):
    valid=[v for v in values if v is not None and np.isfinite(v)]
    return {'valid_draws':len(valid),'range_2_5_97_5':np.quantile(valid,[.025,.975],method='linear').tolist() if valid else [None,None]}


def sensitivity(row):
    draws=[measures(row['truth'],row['logits'],row['probabilities'],row['predictions'],w) for w in cluster_weights(row['parent_ids'])]
    return {'original_parent_clusters':len(set(row['parent_ids'])),'enumerated_draws':len(draws),'OA':percentile([d['OA'] for d in draws]),'AA':percentile([d['AA'] for d in draws]),'recall_ground':percentile([d['recall'][0] for d in draws]),'recall_tree':percentile([d['recall'][1] for d in draws]),'interpretation':'Descriptive resampling sensitivity on2/3 observed parents; no nominal coverage or new geography claim.'}


def paired(a,b):
    if a['scene']!=b['scene'] or a['point_ids']!=b['point_ids'] or a['truth']!=b['truth'] or a['parent_ids']!=b['parent_ids']:raise ValueError('Point pairing requires the same original development observations')
    y=np.asarray(a['truth']);pa=np.asarray(a['predictions']);pb=np.asarray(b['predictions']);ca=pa==y;cb=pb==y;delta=ca.astype(float)-cb.astype(float);boot=[float(w@delta/w.sum()) for w in cluster_weights(a['parent_ids'])]
    return {'A':a['id'],'B':b['id'],'scene':a['scene'],'points':len(y),'A_correct_B_wrong':int((ca&~cb).sum()),'A_wrong_B_correct':int((~ca&cb).sum()),'OA_delta':float(delta.mean()),'NLL_delta':a['metrics']['NLL']-b['metrics']['NLL'],'class_OA_delta':[float(delta[y==c].mean()) for c in (0,1)],'parent_resampling_OA_delta':percentile(boot),'pointwise_correctness_delta':delta.astype(int).tolist()}
