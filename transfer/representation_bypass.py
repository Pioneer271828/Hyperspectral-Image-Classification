"""Opt-in equal-capacity residual repair on the existing region and sparse kernels."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn

from model.VCASAHGCN import SharedHGCN
from model.region_pixel_alignment import broadcast_regions
from model.region_residual_decomposition import RegionResidualDecomposition
from model.research_config import digest
from research_runtime import tensor_hash

TASKS = ("recoverable","removed","noise","context")
LAYOUTS = ("horizontal","vertical")
SEEDS = (64,512,2048)
ARMS = ("closed","aligned","matched_permutation")
STEPS, PREFIX = 72, 11
VALIDATE = (0,18,36,54,72)
BANDS = ("full","gap")
IDS = (103,107,113,127,129,131,137,149)
SELECTED = {"full":tuple(range(8)),"gap":(0,1,6,7)}
ROLE_SALT = {"train":14047,"validation":16069,"confirmation":19079}
ROLE_COUNT = {"train":12,"validation":6,"confirmation":8}


@dataclass(frozen=True)
class Window:
    observed: torch.Tensor
    band_ids: tuple
    q: torch.Tensor
    a: torch.Tensor
    shuffle: torch.Tensor
    focus: torch.Tensor
    label: torch.Tensor
    boundary: torch.Tensor
    identity: str
    witness: dict


def observation(task,layout,seed,role,tile,band):
    if (task not in TASKS or layout not in LAYOUTS or seed not in SEEDS or role not in ROLE_SALT
            or not 0<=tile<ROLE_COUNT[role] or band not in BANDS):
        raise ValueError("Unknown synthetic residual window")
    rng = np.random.default_rng(np.random.SeedSequence([seed,ROLE_SALT[role],tile,1 if layout=="horizontal" else 2]))
    yy,xx = np.mgrid[:4,:9]
    regions = yy//2*3+xx//3
    shape = regions.shape
    x = np.zeros((*shape,8),np.float32)
    y = np.zeros(shape,dtype=np.int64)
    mapping = np.arange(36).reshape(shape)
    donor = rng.permutation(3)
    if task=="context":
        order = rng.permutation(3)
        for dst,cls in enumerate(order):
            y[regions==dst] = cls
            source = 3+donor[dst]
            y[regions==source] = cls
            x[regions==source,(0,1,6)[cls]] = 1
        focus = regions<3
    else:
        for region in range(6):
            labels = rng.permutation([0,0,1,1,2,2])
            y[regions==region] = labels
            if task in ("recoverable","removed"):
                cols = (0,1,6) if task=="recoverable" else (2,3,4)
                for cls,col in enumerate(cols):
                    x[(regions==region)&(y==cls),col] = 1.
        focus = np.ones(shape,dtype=bool)
        if task=="noise":
            x[:,:,:] = rng.normal(0,.03,x.shape)
    perm = mapping.copy()
    for node in range(6):
        mask = regions==node
        perm[mask] = rng.permutation(mapping[mask])
    if layout=="vertical":
        x,y,regions,focus,perm,mapping = x.transpose(1,0,2),y.T,regions.T,focus.T,perm.T,mapping.T
    old_to_new = np.argsort(mapping.ravel())
    shuffle = old_to_new[perm.ravel()]
    region_order = regions.ravel()
    check_permutation = np.array_equal(region_order[shuffle],region_order)
    if not check_permutation:
        raise ValueError("Matched residual donor crossed a registered pooling region")
    edges = {(i,i) for i in range(6)}
    if task=="context":
        edges |= {(i,int(j+3)) for i,j in enumerate(donor)}
    ij = torch.tensor(sorted(edges),dtype=torch.long).T
    values = 1./torch.bincount(ij[0],minlength=6).float()[ij[0]]
    a = torch.sparse_coo_tensor(ij,values,(6,6)).coalesce()
    qi = torch.stack((torch.arange(36),torch.from_numpy(region_order.copy())))
    q = torch.sparse_coo_tensor(qi,torch.ones(36),(36,6)).coalesce()
    retained = SELECTED[band]
    visible = torch.from_numpy(x[...,list(retained)].copy().reshape(36,len(retained)))
    target = torch.from_numpy(y.copy().reshape(-1))
    boundary = np.zeros(y.shape,dtype=bool)
    for axis in (0,1):
        dif = np.diff(y,axis=axis)!=0
        if axis==0:
            boundary[:-1] |= dif
            boundary[1:] |= dif
        else:
            boundary[:,:-1] |= dif
            boundary[:,1:] |= dif
    identities = dict(task=task,layout=layout,seed=seed,stream=role,tile=tile,shape=list(y.shape),
                      band_ids=[IDS[j] for j in retained],observed=tensor_hash(visible),q=tensor_hash(q),a=tensor_hash(a),
                      target=tensor_hash(target),shuffle=shuffle.tolist())
    if task=="removed" and band=="gap" and torch.count_nonzero(visible):
        raise RuntimeError("A physically deleted class channel entered the observed view")
    return Window(visible,tuple(identities["band_ids"]),q,a,torch.from_numpy(shuffle.copy()),
                  torch.from_numpy(np.flatnonzero(focus)),target,torch.from_numpy(boundary.ravel()),
                  digest(identities),dict(focus=int(focus.sum()),boundary=int(boundary[focus].sum()),
                                         physically_present=task!="removed" or band=="full",graph_nnz=a._nnz()))


class ResidualBottleneckClassifier(nn.Module):
    """Identical learned layers across all arms, with one explicit residual source."""
    def __init__(self, seed, arm):
        super().__init__()
        if seed not in SEEDS or arm not in ARMS:
            raise ValueError("Explicit opt-in residual arm required")
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.adapter = nn.Linear(8,12,bias=False)
            self.graph = SharedHGCN(12,12,12).eval()
            self.decompose = RegionResidualDecomposition(12)
            self.local = nn.Sequential(nn.Linear(24,12),nn.GELU())
            self.fused_head = nn.Linear(24,3)
            self.graph_head = nn.Linear(12,3)
            self.cnn_head = nn.Linear(12,3)
        self.timing = {}

    def train(self, mode=True):
        super().train(mode)
        self.graph.eval()  # Fixed BN statistics avoid pooling evidence across windows.
        return self

    def split(self,window):
        known = [IDS.index(j) for j in window.band_ids]
        # Only authenticated originally retained columns are inserted. No imputation.
        full = window.observed.new_zeros((len(window.observed),len(IDS)))
        full[:,known] = window.observed
        adapted = self.adapter(full)
        return self.decompose(adapted,window.q,return_intermediates=True)

    def predictor(self,window,packet,residual):
        started = perf_counter()
        intermediate = self.graph.forward_stage1(packet.region_features,window.a)
        graph = broadcast_regions(window.q,self.graph.forward_stage2(intermediate,window.a))
        graph_seconds = perf_counter()-started
        local = self.local(torch.cat((packet.region_at_pixels,residual),dim=1))
        self.timing = dict(graph_seconds=graph_seconds,q_nnz=window.q._nnz(),a_nnz=window.a._nnz())
        return self.fused_head(torch.cat((local,graph),dim=-1)),self.graph_head(graph),self.cnn_head(local)

    def forward(self,window):
        packet = self.split(window)
        hint = packet.residual
        if self.arm=="closed":
            hint = torch.zeros_like(hint)
        elif self.arm=="matched_permutation":
            hint = hint[window.shuffle]
        return self.predictor(window,packet,hint)
