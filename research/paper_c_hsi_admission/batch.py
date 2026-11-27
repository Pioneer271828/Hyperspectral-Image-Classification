"""Role-local native tile and spectral-patch tensors, without any model call."""
import numpy as np


def patch_batch(cube,global_ids,box,scene_width,*,radius=3):
    if radius!=3 or cube.ndim!=3:raise ValueError('Frozen7x7 role-local patch context required')
    y0,y1,x0,x1=box
    if cube.shape[:2]!=(y1-y0,x1-x0):raise ValueError('Tile context and original box differ')
    patches=[];centers=[]
    for identity in global_ids:
        y,x=divmod(int(identity),scene_width);y-=y0;x-=x0
        if not 0<=y<cube.shape[0] or not 0<=x<cube.shape[1]:raise ValueError('Released point escapes tile')
        rr=np.clip(np.arange(y-radius,y+radius+1),0,cube.shape[0]-1);cc=np.clip(np.arange(x-radius,x+radius+1),0,cube.shape[1]-1)
        patches.append(cube[rr[:,None],cc[None,:],:].transpose(2,0,1));centers.append((y,x))
    if not patches:raise ValueError('No released labels selected')
    return np.stack(patches),tuple(centers)


def tile_batch(bank,role,tile_id,*,allowed_original_classes=None):
    # Class subset membership is checked against original metadata before any
    # spectral-array read. It does not establish a shared semantic class map.
    if role not in ('devtrain','devval') or tile_id not in bank.rows or bank.rows[tile_id]['role']!=role:raise ValueError('Unreleased original role/tile')
    permitted=set(range(1,bank.pin['classes']+1)) if allowed_original_classes is None else set(allowed_original_classes)
    if not permitted or any(type(x)!=int or x<1 or x>bank.pin['classes'] for x in permitted):raise ValueError('Unregistered original numeric class subset')
    points=[(i,c) for i,c in bank.samples[tile_id] if c in permitted]
    if not points:raise ValueError('No admitted original labels on this tile')
    x=bank.read_tile(role,tile_id);ids,labels=zip(*points);patches,centers=patch_batch(x['spectral_normalized_hwc'],ids,x['original_box'],bank.pin['shape'][1])
    return {**x,'patches_nchw':patches,'selected_global_ids':ids,'selected_local_centers':centers,'labels_original':np.asarray(labels,dtype=np.int64),'labels_native_zero_based':np.asarray(labels,dtype=np.int64)-1,'cross_dataset_shared_head_admitted':False,'common_authorized_tile_boundary':True,'native_whole_tile_and_patch_observation_footprints_equal':False,'context_limit':'A whole-tile native forward and a patch-only forward have different effective footprints. A future same-context baseline must feed both the same clipped patches or explicitly account for tile observations.'}
