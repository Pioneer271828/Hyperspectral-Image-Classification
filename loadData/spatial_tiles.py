"""Label-blind, seeded multi-block geometry; each block is a separate model input."""
import numpy as np
from model.research_config import digest
from loadData.spatial_protocol import PartitionObservation

def make_tile_plan(shape, tile_size=32, gap=2, seed=2026):
    if len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape):
        raise ValueError('Positive integer H/W required')
    if type(tile_size) is not int or type(gap) is not int or gap<0 or tile_size-2*gap<4:
        raise ValueError('Each tile must retain at least 4 pixels per side')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid split seed')
    boxes=[]
    for y in range(0,shape[0],tile_size):
        for x in range(0,shape[1],tile_size):
            box=[y+gap,min(y+tile_size,shape[0])-gap,x+gap,min(x+tile_size,shape[1])-gap]
            if min(box[1]-box[0],box[3]-box[2])>=4:boxes.append(box)
    if len(boxes)<5:raise ValueError('At least five usable blocks required')
    order=np.random.default_rng(seed).permutation(len(boxes))
    n_val=n_test=max(1,len(boxes)//5)
    assignment={int(i):('val' if k<n_val else 'test' if k<n_val+n_test else 'train') for k,i in enumerate(order)}
    plan={'schema_version':1,'protocol':'spatial_tiles_v1','shape':list(shape),'tile_size':tile_size,
          'gap':gap,'seed':seed,'allocation':'train remainder; val/test floor(n/5), min 1',
          'tiles':[{'id':i,'partition':assignment[i],'box':box} for i,box in enumerate(boxes)]}
    plan['plan_hash']=digest(plan)
    return plan

def validate_tile_plan(plan):
    expected=make_tile_plan(tuple(plan['shape']),plan['tile_size'],plan['gap'],plan['seed'])
    if plan!=expected:raise ValueError('Tile plan altered or unsupported')

def tile_slice(tile):
    y0,y1,x0,x1=tile['box']
    return slice(y0,y1),slice(x0,x1)

def fit_tile_statistics(cube,plan):
    validate_tile_plan(plan)
    if cube.ndim!=3 or list(cube.shape[:2])!=plan['shape']:raise ValueError('Cube shape mismatch')
    low=high=None
    for tile in plan['tiles']:
        if tile['partition']!='train':continue
        values=np.asarray(cube[tile_slice(tile)],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite training observation')
        a,b=values.min(axis=(0,1)),values.max(axis=(0,1))
        low=a if low is None else np.minimum(low,a)
        high=b if high is None else np.maximum(high,b)
    stats={'schema_version':1,'plan_hash':plan['plan_hash'],'fit_partition':'train',
           'minimum':low.tolist(),'maximum':high.tolist()}
    stats['statistics_hash']=digest(stats)
    return stats

def observe_tile(cube,plan,stats,tile):
    validate_tile_plan(plan)
    if type(tile.get('id')) is not int or tile['id']<0 or tile['id']>=len(plan['tiles']) or tile!=plan['tiles'][tile['id']]:
        raise ValueError('Unknown/modified tile')
    if cube.ndim!=3 or list(cube.shape[:2])!=plan['shape']:raise ValueError('Cube shape mismatch')
    identity={k:v for k,v in stats.items() if k!='statistics_hash'}
    if stats.get('plan_hash')!=plan['plan_hash'] or stats.get('fit_partition')!='train' or digest(identity)!=stats.get('statistics_hash'):
        raise ValueError('Statistics mismatch')
    lo,hi=np.asarray(stats['minimum'],dtype=np.float32),np.asarray(stats['maximum'],dtype=np.float32)
    if lo.shape!=(cube.shape[2],) or hi.shape!=lo.shape or not np.isfinite([lo,hi]).all() or (hi<lo).any():
        raise ValueError('Invalid statistics')
    values=np.array(cube[tile_slice(tile)],dtype=np.float32,copy=True)
    if not np.isfinite(values).all():raise ValueError('Nonfinite observation')
    values=(values-lo)/np.where(hi>lo,hi-lo,1)
    y0,y1,x0,x1=tile['box'];yy,xx=np.mgrid[y0:y1,x0:x1]
    coordinates=np.stack((yy/max(cube.shape[0]-1,1),xx/max(cube.shape[1]-1,1)),axis=-1).astype(np.float32)
    ids=yy*cube.shape[1]+xx
    for value in (values,coordinates,ids):value.setflags(write=False)
    return PartitionObservation(tile['partition'],values,coordinates,ids,plan['plan_hash'])

def tile_coverage(labels,plan,classes):
    validate_tile_plan(plan)
    if list(labels.shape)!=plan['shape']:raise ValueError('Label shape mismatch')
    counts={p:np.zeros(classes,dtype=np.int64) for p in ('train','val','test')}
    for tile in plan['tiles']:
        values=np.asarray(labels[tile_slice(tile)])
        if not np.isfinite(values).all() or (values<0).any() or (values>classes).any() or not np.equal(values,np.floor(values)).all():
            raise ValueError('Invalid labels')
        counts[tile['partition']]+=np.bincount(values.astype(np.int64).ravel(),minlength=classes+1)[1:]
    return {'counts':{p:v.tolist() for p,v in counts.items()},
            'missing_classes':{p:(np.flatnonzero(v==0)+1).tolist() for p,v in counts.items()}}
