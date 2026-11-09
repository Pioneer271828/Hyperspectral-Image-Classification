"""Directed deletion controls with exact degree and declared distance-bin counts.

Independent experimental control, not a routing/training mechanism. Random
linear objectives are not uniform samples of feasible masks. No model is read.
"""
import math
import time
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
import torch
from evaluation.graph_edge_intervention import deletion_masks
from monitoring.graph_support import _matrix
from model.research_config import digest
from research_runtime import tensor_hash


def freeze(adjacency, coordinates, *, per_row, distance_bins, seeds, time_limit=30.):
    started=time.perf_counter(); before=tensor_hash(adjacency)
    ix, weights, _=_matrix(adjacency,'A'); n=adjacency.shape[0]
    if n>2048 or adjacency._nnz()>65536: raise ValueError('Bounded local graph required')
    if (not torch.is_tensor(coordinates) or coordinates.device.type!='cpu' or coordinates.shape!=(n,2)
            or not coordinates.is_floating_point() or not torch.isfinite(coordinates).all()):
        raise ValueError('Finite CPU pixel coordinates required')
    if (not isinstance(distance_bins,list) or not distance_bins or len(distance_bins)>32
            or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in distance_bins)
            or distance_bins!=sorted(set(distance_bins))):
        raise ValueError('Increasing finite positive distance boundaries required')
    if type(time_limit) not in (int,float) or not math.isfinite(time_limit) or not 0<time_limit<=60:
        raise ValueError('Explicit bounded solver time required')
    base=deletion_masks(adjacency,per_row=per_row,random_seeds=seeds,tile_id=0)
    nonself=ix[0]!=ix[1]; edges=ix[:,nonself]; weight=weights[nonself].astype(np.float64)
    ids=(edges[0]*n+edges[1]).tolist(); m=len(ids)
    xy=coordinates.detach().double().numpy().copy(); coordhash=tensor_hash(coordinates)
    lengths=np.linalg.norm(xy[edges[0]]-xy[edges[1]],axis=1)
    if not np.isfinite(lengths).all(): raise ValueError('Distance overflow')
    bins=np.searchsorted(distance_bins,lengths,side='right'); nb=len(distance_bins)+1
    chosen=set(base['removed']['important']); target=np.array([int(e in chosen) for e in ids],dtype=np.int64)
    # Rows and columns describe A@X dependency endpoints, not undirected degree.
    cols=np.tile(np.arange(m),3); rows=np.concatenate((edges[0],n+edges[1],2*n+bins))
    matrix=csc_matrix((np.ones(3*m),(rows,cols)),shape=(2*n+nb,m))
    counts=np.asarray(matrix@target,dtype=np.int64)
    controls={}; k=int(target.sum())
    objectives={'low_weight':weight}
    objectives.update({f'random:{s}':np.random.default_rng(s).random(m) for s in seeds})
    for name,cost in objectives.items():
        if k==0 or k==m:
            controls[name]={'status':'no_distinct_control','removed':None,'reason':'empty or all removable edges'}
            continue
        # Same cardinality plus overlap <= k-1 excludes the identical baseline.
        constraints=[LinearConstraint(matrix,counts,counts),LinearConstraint(target.reshape(1,-1),-np.inf,k-1)]
        result=milp(cost,integrality=np.ones(m),bounds=Bounds(0,1),constraints=constraints,
                    options={'time_limit':float(time_limit),'mip_rel_gap':0.})
        if result.status==2:
            controls[name]={'status':'no_distinct_control','removed':None,'reason':'solver proved infeasible under exact declared constraints'}
            continue
        if result.status!=0 or not result.success:
            controls[name]={'status':'unresolved','removed':None,'solver_status':int(result.status),'reason':str(result.message)}
            continue
        bits=np.rint(result.x).astype(np.int64)
        if (not np.isfinite(result.x).all() or np.max(np.abs(result.x-bits))>1e-7
                or not np.isin(bits,[0,1]).all() or not np.array_equal(matrix@bits,counts)
                or bits.sum()!=k or bits@target>=k):
            raise ValueError('Solver candidate violates exact discrete controls')
        controls[name]={'status':'optimal','removed':[e for e,v in zip(ids,bits) if v],
            'objective':float(cost@bits),'deleted_weight_sum':float(weight@bits),
            'deleted_distance_sum':float(lengths@bits),'counts':counts.tolist()}
    if tensor_hash(adjacency)!=before or tensor_hash(coordinates)!=coordhash: raise RuntimeError('Inputs changed')
    value={'version':1,'adjacency_sha256':before,'coordinates_sha256':coordhash,'node_count':n,
        'per_row':per_row,'distance_bins':distance_bins.copy(),'seeds':list(seeds),'time_limit':time_limit,
        'baseline':{'removed':base['removed']['important'],'counts':counts.tolist(),'deleted_weight_sum':float(weight@target),
                    'deleted_distance_sum':float(lengths@target),'degenerate_rows':base['degenerate_rows']},
        'controls':controls,'solver':'scipy.optimize.milp/HiGHS','scipy_version':scipy.__version__,
        'elapsed_seconds':time.perf_counter()-started,'input_unchanged':True,'optimizer_updates':0,'query':0,
        'limits':'Exact deleted row/column degree and global distance-bin counts only; distances, weight mass and connectivity may differ. Random objective is not uniform sampling.'}
    value['plan_hash']=digest(value)
    return value


def verify(value, adjacency, coordinates, *, expected_hash):
    """Validate saved discrete masks without rerunning or trusting solver output."""
    if (value.get('plan_hash')!=expected_hash or digest({k:v for k,v in value.items() if k!='plan_hash'})!=expected_hash
            or value.get('version')!=1 or value['adjacency_sha256']!=tensor_hash(adjacency)
            or value['coordinates_sha256']!=tensor_hash(coordinates)):
        raise ValueError('Frozen control identity differs')
    ix,weights,_=_matrix(adjacency,'A'); n=adjacency.shape[0]; ids=(ix[0]*n+ix[1]).tolist()
    if n!=value['node_count']: raise ValueError('Node count differs')
    xy=coordinates.detach().double().numpy()
    base=deletion_masks(adjacency,per_row=value['per_row'],random_seeds=value['seeds'],tile_id=0)
    if base['removed']['important']!=value['baseline']['removed']: raise ValueError('Baseline differs')
    def counts(removed):
        if removed!=sorted(set(removed)) or not set(removed)<=set(ids) or any(e//n==e%n for e in removed):
            raise ValueError('Invalid nonself deletion')
        rows=np.array([e//n for e in removed],dtype=int); cols=np.array([e%n for e in removed],dtype=int)
        lengths=np.linalg.norm(xy[rows]-xy[cols],axis=1)
        bins=np.searchsorted(value['distance_bins'],lengths,side='right')
        return np.concatenate((np.bincount(rows,minlength=n),np.bincount(cols,minlength=n),
                               np.bincount(bins,minlength=len(value['distance_bins'])+1))).tolist()
    expected=counts(value['baseline']['removed'])
    if expected!=value['baseline']['counts']: raise ValueError('Baseline counts differ')
    for control in value['controls'].values():
        if control['status']=='optimal':
            if (counts(control['removed'])!=expected or control['counts']!=expected
                    or control['removed']==value['baseline']['removed']): raise ValueError('Matched control differs')
        elif control['status'] not in ('no_distinct_control','unresolved') or control['removed'] is not None:
            raise ValueError('Uncertified control must not supply a mask')
    return {'discrete_constraints_verified':True,'optimality_independently_verified':False,
            'infeasibility_independently_verified':False,'plan_hash':expected_hash}
