"""Fixed samplewise distribution descriptors; no band/wavelength correspondence."""
import numpy as np

NAMES=('center_lower_tail','center_upper_tail','center_quantile_skew','center_clipped_second_moment',
       'patch_mean_lower_tail','patch_mean_upper_tail','patch_mean_quantile_skew','patch_mean_clipped_second_moment')
REPRESENTATIONS=('raw_distribution','legacy_scene_distribution')


def transform(patches):
    x=np.asarray(patches,dtype=np.float64)
    if x.ndim!=4 or x.shape[2:]!=(7,7) or x.shape[1]<2 or not np.isfinite(x).all():raise ValueError('Finite native-column7x7 patches required')
    q=np.quantile(x,[.05,.25,.5,.75,.95],axis=1,method='linear');low,quarter,median,upper,high=q
    spread=upper-quarter;valid=spread>1e-12;denominator=np.where(valid,spread,1.0);z=(x-median[:,None,:,:])/denominator[:,None,:,:]
    values=np.stack((np.clip(-(low-median)/denominator,0,8)/8,np.clip((high-median)/denominator,0,8)/8,np.clip((upper+quarter-2*median)/denominator,-1,1),np.mean(np.clip(z,-4,4)**2,axis=1)/16),axis=-1)
    values=np.where(valid[:,:,:,None],values,0.0)
    result=np.concatenate((values[:,3,3,:],values.mean(axis=(1,2))),axis=1)
    if result.shape!=(len(x),8) or not np.isfinite(result).all():raise ValueError('Invalid fixed descriptor')
    result.setflags(write=False);return result
