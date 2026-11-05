"""Label-free target static hypergraph; source LDA graph is a distinct context."""
from contextlib import contextmanager
import json,random
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from createGraph import rdSLIC
from createGraph.CreatHStructure import createH
from joint_baseline import NAMES,_item_fingerprint
from loadData.dataset_registry import DATASETS,file_hash
from loadData.data_read import readWavelengths
from source_data_context import bounded
from train_vca_multidataset import normalize_maxmin,build_spatial_index


@contextmanager
def isolated_graph_rng(seed):
    if type(seed)is not int or not 0<=seed<2**32:raise ValueError('Graph seed must be uint32')
    python=random.getstate();numpy=np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed);yield
    finally:
        random.setstate(python);np.random.set_state(numpy)


def load_target_image(prepared,data_root,target,*,manifest_sha256):
    if target not in NAMES:raise ValueError('Unknown target')
    prepared=Path(prepared).resolve(strict=True);root=Path(data_root).resolve(strict=True)
    manifest_path=prepared/'manifest.json'
    if file_hash(manifest_path)!=manifest_sha256:raise ValueError('Pinned cohort metadata changed')
    manifest=json.loads(manifest_path.read_text())
    if manifest['protocol']!='acc1p_joint_prepared_v1' or manifest['schema_version']!=1:
        raise ValueError('Unaccepted cohort metadata')
    raw=manifest['datasets'][target]['raw'];path=bounded(root,raw['image_path'])
    if file_hash(path)!=raw['image_sha256']:raise ValueError('Target image identity changed')
    image=loadmat(path,variable_names=[DATASETS[target][1]])[DATASETS[target][1]]
    if list(image.shape)!=raw['shape'] or raw['class_count']!=DATASETS[target][2]:
        raise ValueError('Target image/class metadata mismatch')
    if file_hash(path)!=raw['image_sha256']:raise ValueError('Target image changed during load')
    return image,{'target':target,'manifest_sha256':manifest_sha256,'image_path':str(path),
        'image_sha256':raw['image_sha256'],'cohort_target_metadata_visible':True,
        'target_image_accessed':True,'target_label_payload_accessed':False,
        'target_split_or_prepared_payload_accessed':False}


def label_free_graph(image,scale,seed):
    image=np.asarray(image)
    if (image.ndim!=3 or min(image.shape[:2])<2 or image.shape[2]<1
            or not np.isfinite(image).all() or float(image.max())==float(image.min())
            or type(scale)is not int or scale<=0):raise ValueError('Nonconstant finite HWC image/positive scale required')
    height,width,bands=image.shape
    with isolated_graph_rng(seed):
        # Same existing no-LDA standardization/SLIC parameters, no label object.
        old=rdSLIC.SLIC(image,None,n_segments=height*width/scale,compactness=.5,
            sigma=1,min_size_factor=.1,max_size_factor=2)
        segments=rdSLIC.slic(old.data,n_segments=old.n_segments,compactness=old.compactness,
            max_num_iter=old.max_iter,convert2lab=False,sigma=old.sigma,enforce_connectivity=True,
            min_size_factor=old.min_size_factor,max_size_factor=old.max_size_factor,slic_zero=False,start_label=0)
        if segments.max()+1!=len(np.unique(segments)):segments=rdSLIC.SegmentsLabelProcess(segments)
        flat=segments.reshape(-1);regions=int(flat.max())+1;counts=np.bincount(flat,minlength=regions)
        if not counts.all() or regions<2:raise ValueError('Graph needs at least two nonempty regions')
        order=np.argsort(flat,kind='stable');ends=np.cumsum(counts);starts=ends-counts
        features=np.zeros((regions,bands),np.float32);centers=np.zeros((regions,2),np.float32)
        values=old.data.reshape(-1,bands)
        for r,(start,end) in enumerate(zip(starts,ends)):
            ids=order[start:end]
            features[r]=np.sum(values[ids],axis=0)/len(ids)
            centers[r]=[np.mean(ids//width),np.mean(ids%width)]
        adjacency=np.asarray(createH(features,centers,segments),dtype=np.float32)
        if adjacency.shape!=(regions,regions) or not np.isfinite(adjacency).all():raise ValueError('Nonfinite target hypergraph')
    indices=torch.stack((torch.arange(height*width),torch.from_numpy(flat.astype(np.int64))))
    Q=torch.sparse_coo_tensor(indices,torch.ones(height*width),(height*width,regions)).coalesce()
    return Q,torch.from_numpy(adjacency),segments


def build_target_observation(image,target,seed):
    if target not in NAMES:raise ValueError('Unknown target')
    height,width,bands=np.shape(image);scale=50 if target=='paviaU' else 100
    wavelengths,fwhm=readWavelengths(target,bands)
    Q,A,segments=label_free_graph(image,scale,seed)
    normalized=normalize_maxmin(np.array(image,np.float32))
    item={'name':target,'height':height,'width':width,'bands':bands,'class_num':DATASETS[target][2],
        'Q':Q,'A':A,'superpixel_scale':scale,'net_input':torch.from_numpy(np.concatenate(
            (normalized,build_spatial_index(height,width)),axis=2).astype(np.float32)),
        'wavelengths':torch.from_numpy(wavelengths.astype(np.float32)),
        'fwhm':None if fwhm is None else torch.from_numpy(fwhm.astype(np.float32))}
    record={'policy':'g4_target_label_free_static_v1','target':target,'graph_seed':seed,
        'input_context':'full_target_scene_transductive','graph_fit_labels':0,
        'source_graph_context_differs':True,'source_graph_context':'source_train_label_lda_slic',
        'graph_method':'existing_no_lda_slic_same_hypergraph_sparse_pixel_assignment',
        'wavelength_quality':'approximate_provided_not_measured','fwhm_available':fwhm is not None,
        'regions':A.shape[0],'assignment_nnz':Q._nnz(),'prepared_fields':_item_fingerprint(item),
        'optimizer_updates':0,'model_test_scoring':False}
    return item,record
