"""Explicit branch-removal models and fixed region-propagation controls."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.region_pixel_alignment import pool_regions,broadcast_regions
from model.bn_calibration import CONTEXT,verify_calibration_binding
from joint_calibrated_selection import observations
from loadData.dataset_registry import file_hash
from research_runtime import tensor_hash

VARIANTS=('dual_legacy_hypergraph','dual_fused_hypergraph','cnn_fused_only',
          'graph_fused_hypergraph','dual_legacy_ordinary','dual_legacy_identity')


def variant_spec(name):
    if name not in VARIANTS: raise ValueError('Unknown branch/graph variant')
    return {'schema_version':1,'variant':name,
            'branches':'cnn' if name=='cnn_fused_only' else 'graph' if name=='graph_fused_hypergraph' else 'dual',
            'loss':'legacy_three_heads' if name.startswith('dual_legacy_') else 'primary_only',
            'propagation':'ordinary_centroid4nn_symmetric_v1' if name.endswith('ordinary') else
                          'identity' if name.endswith('identity') else 'accepted_hypergraph',
            'shared_front':'unchanged_vca_two_denoise_blocks','primary_coefficient':1.0}


def ordinary_graph(Q,height,width):
    """Geometry-only 4NN union, self loops, symmetric degree normalization."""
    if Q.requires_grad or Q.ndim!=2 or Q.shape[0]!=height*width or min(height,width)<=0 or Q.shape[1]<1:
        raise ValueError('Fixed nonempty assignment and image shape required')
    q=Q.detach().cpu().double();v=q.coalesce().values() if q.is_sparse else q
    if not torch.isfinite(v).all() or (v<0).any(): raise ValueError('Finite nonnegative assignment required')
    mass=torch.sparse.sum(q,0).to_dense() if q.is_sparse else q.sum(0)
    rows=torch.sparse.sum(q,1).to_dense() if q.is_sparse else q.sum(1)
    if (mass<=0).any() or not torch.allclose(rows,torch.ones_like(rows),atol=1e-6,rtol=1e-6):
        raise ValueError('Nonempty regions and unit row assignments required')
    yy,xx=torch.meshgrid(torch.arange(height,dtype=torch.float64),torch.arange(width,dtype=torch.float64),indexing='ij')
    centers=pool_regions(q,torch.stack((yy.flatten(),xx.flatten()),1)).numpy();n=len(centers)
    edges={(i,i) for i in range(n)}
    for i in range(n):
        distance=((centers-centers[i])**2).sum(1);distance[i]=np.inf
        for j in np.argsort(distance,kind='stable')[:min(4,n-1)]:edges.update(((i,int(j)),(int(j),i)))
    indices=torch.tensor(sorted(edges),dtype=torch.long).t();degree=torch.bincount(indices[0],minlength=n).double()
    values=(degree[indices[0]]*degree[indices[1]]).rsqrt().float()
    return torch.sparse_coo_tensor(indices,values,(n,n)).coalesce()


def controlled_items(items,variant):
    spec=variant_spec(variant);result={};binding={}
    for name,item in items.items():
        Q,A=item['Q'],item['A'];n=Q.shape[1]
        if A.shape!=(n,n): raise ValueError('Frozen Q/A region counts differ')
        if spec['propagation']=='ordinary_centroid4nn_symmetric_v1':control=ordinary_graph(Q,item['height'],item['width'])
        elif spec['propagation']=='identity':
            ids=torch.arange(n);control=torch.sparse_coo_tensor(torch.stack((ids,ids)),torch.ones(n),(n,n)).coalesce()
        else:control=A
        if control is not A:
            control=control.to(device=A.device,dtype=A.dtype)
            if not A.is_sparse:control=control.to_dense()
        result[name]={**item,'A':control}
        binding[name]={'Q_sha256':tensor_hash(Q),'source_A_sha256':tensor_hash(A),
                       'effective_A_sha256':tensor_hash(control),'propagation':spec['propagation'],
                       'regions':n,'height':item['height'],'width':item['width']}
    return result,binding


class BranchGraphAblation(MultiDatasetVCASAHGCN):
    """New mapping-based forward; inactive modules/heads are actually removed."""
    def __init__(self,class_counts,variant,**kwargs):
        self.ablation=variant_spec(variant)
        if (kwargs.get('graph_mode','static')!='static' or kwargs.get('interaction_mode','none')!='none'
            or kwargs.get('graph_update_mode','none')!='none' or kwargs.get('region_residual_mode','none')!='none'
            or kwargs.get('coverage_mode','none')!='none'):
            raise ValueError('Only the frozen unmodulated static baseline is supported')
        # Construct every original parameter first: removal must not change the
        # initialization/RNG sequence of any retained shared/head parameter.
        super().__init__(class_counts,**kwargs)
        self.register_buffer('_branch_graph_signature',torch.tensor([1,VARIANTS.index(variant)],dtype=torch.int64))
        if self.ablation['branches']=='cnn':self.backbone.HGNN_Branch=None
        elif self.ablation['branches']=='graph':self.backbone.CNN_Branch=nn.ModuleList()
        if self.ablation['loss']=='primary_only':
            self.hgnn_heads=nn.ModuleDict();self.cnn_heads=nn.ModuleDict()

    def export_config(self):
        return {**super().export_config(),'ablation':dict(self.ablation)}

    def add_dataset_head(self,*args,**kwargs):
        raise ValueError('Ablation task set is fixed by the registered plan')

    def forward(self,dataset_name,x,Q=None,A=None,wavelengths=None,fwhm=None,superpixel_scale=100):
        mode=self.ablation['branches'];b=self.backbone
        if mode=='dual':
            primary,graph,cnn=b.forward_features(x,Q,A,wavelengths=wavelengths,fwhm=fwhm,
                                               dataset_name=dataset_name,superpixel_scale=superpixel_scale)
        else:
            height,width,_=x.shape;spectral,extra=b._split_spectral_extra(x,wavelengths)
            latent=b.adapter(spectral,wavelengths=wavelengths,fwhm=fwhm)
            noise=torch.cat((latent,extra),2).permute(2,0,1).unsqueeze(0)
            for block in b.CNN_denoise:noise=block(noise,dataset_name)
            clean=noise.squeeze(0).permute(1,2,0)
            if mode=='cnn':primary=b.cnn_stage2(b.cnn_stage1(clean,dataset_name),(height,width),dataset_name)
            else:
                if Q is None or A is None:raise ValueError('Graph-only route requires fixed Q/A')
                regions=pool_regions(Q,clean.reshape(height*width,64))
                regions=b.graph_stage2(b.graph_stage1(regions,A,dataset_name),A,dataset_name)
                primary=broadcast_regions(Q,regions)
        result={'fused':F.softmax(self.heads[dataset_name](primary),-1)}
        if self.ablation['loss']=='legacy_three_heads':
            result['graph']=F.softmax(self.hgnn_heads[dataset_name](graph),-1)
            result['cnn']=F.softmax(self.cnn_heads[dataset_name](cnn),-1)
        return result


def load_ablation(path,device='cpu',expected_variant=None):
    path=Path(path);metadata=json.loads(Path(str(path)+'.json').read_text())
    spec=metadata.get('ablation',{});variant=spec.get('variant')
    if spec!=variant_spec(variant) or (expected_variant is not None and variant!=expected_variant):
        raise ValueError('Ablation checkpoint policy mismatch')
    state=torch.load(path,map_location=device,weights_only=True)
    signature=state.get('_branch_graph_signature')
    expected=torch.tensor([1,VARIANTS.index(variant)],dtype=torch.int64,device=device)
    if signature is None or signature.dtype!=expected.dtype or not torch.equal(signature,expected):
        raise ValueError('Ablation state/metadata signature mismatch')
    model=BranchGraphAblation(metadata['class_counts'],variant,**metadata['backbone_kwargs']).to(device)
    model.load_state_dict(state,strict=True);model.training_config=metadata.get('training_config',{})
    recorded=model.training_config.get('joint_runtime',{}).get('ablation')
    if recorded is not None and recorded!=spec:raise ValueError('Training metadata variant mismatch')
    return model.eval()


def load_ablation_selected(path,items,raw_source_path,*,device='cpu',expected_variant=None):
    model=load_ablation(path,device,expected_variant)
    verify_calibration_binding(model,observations(items),context=CONTEXT,source_checkpoint_sha256=file_hash(raw_source_path))
    expected=model.training_config.get('provenance',{}).get('ablation_graph_binding')
    if expected is None or set(expected)!=set(items):raise ValueError('Checkpoint graph binding missing')
    for name,item in items.items():
        value=expected[name]
        if value['Q_sha256']!=tensor_hash(item['Q']) or value['effective_A_sha256']!=tensor_hash(item['A']):
            raise ValueError('Checkpoint Q/A binding mismatch')
    return model
