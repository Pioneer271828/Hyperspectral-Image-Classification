"""Analytic native-topology controls with frozen, planted weights and no optimization."""

from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, require
from evaluation.wp12_native_pathways import direction, measure, spatial_forward
from model.VCASAHGCN import MultiDatasetVCASAHGCN

SCENE = "xuzhou"
SEEDS = (64,512,2048)
DESIGN = dict(stage_width=[64,128,64],nodes=8,source=5,target=0,relays=4,epsilon=[.01,.005],
              activation="positive_branch",finite_atol=.00015,assembly="equal_degree_and_two_hop_distance")


@torch.no_grad()
def planted(seed,device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MultiDatasetVCASAHGCN({SCENE:9},graph_mode="learned",norm_type="batch_norm",
                   latent_channels=128,adapter_embed_dim=64).eval()
    b = model.backbone
    for stage in b.CNN_Branch:
        stage.point_conv.weight.zero_()
        stage.point_conv.weight[0,0,0,0] = 1.
        stage.depth_conv.weight.zero_()
        half = stage.depth_conv.kernel_size[0]//2
        stage.depth_conv.weight[0,0,half,half] = 1.
        stage.depth_conv.bias.zero_()
    for conv in (b.HGNN_Branch.hgc1,b.HGNN_Branch.hgc2):
        conv.weight.zero_()
        conv.weight[0,0] = 1.
        conv.bias.zero_()
    for family in (model.heads,model.hgnn_heads,model.cnn_heads):
        layer = family[SCENE]
        layer.weight.zero_()
        layer.weight[0,0] = 1.
        layer.weight[1,0] = -1.
        layer.bias.fill_(-5.)
        layer.bias[0] = -.35
        layer.bias[1] = .35
    return model.to(device).eval().requires_grad_(False)


def graph(signal_relay_count,device):
    """Swap sender endpoints with a nuisance route, preserving each node's in/out degree."""
    require(signal_relay_count in (0,1,2),"Unknown paired relay capacity")
    n = DESIGN["nodes"]
    edges = [(0,1),(0,2),(7,3),(7,4)]
    receivers = {2:(1,2),1:(1,3),0:(3,4)}[signal_relay_count]
    for relay in (1,2,3,4):
        edges.append((relay,5 if relay in receivers else 6))
    a = torch.eye(n,device=device)
    for target,source in edges:
        a[target,source] = 1.
    return (a/a.sum(1,keepdim=True)).to_sparse_coo().coalesce()


def assignment(device):
    ids = torch.arange(48,device=device)
    return torch.sparse_coo_tensor(torch.stack((ids,ids//6)),torch.ones(48,device=device),
                                   (48,8)).coalesce()


def controlled(model,clean,q,a,ids,channel,epsilon):
    original = model.backbone.graph_stage1
    actual = {}
    from unittest.mock import patch

    def read(regions,adjacency,dataset):
        value = original(regions,adjacency,dataset)
        actual["stage1"] = value
        return value

    d = direction(ids,clean,channel)
    leaf = clean.detach().clone().requires_grad_(True)
    with patch.object(model.backbone,"graph_stage1",new=read):
        out,_,_,_ = spatial_forward(model,SCENE,leaf,q,a)
    margins = {h:out[h][0,0]-out[h][0,1] for h in HEADS}
    gradients = {h:float((torch.autograd.grad(score,leaf,retain_graph=True)[0]*d).sum())
                 for h,score in margins.items()}
    # The same native stage1 at the target must not receive a two-hop source.
    gradients["stage1_target"] = float((torch.autograd.grad(actual["stage1"][0,0],leaf)[0]*d).sum())
    with torch.no_grad():
        plus = spatial_forward(model,SCENE,clean+epsilon*d,q,a)[0]
        minus = spatial_forward(model,SCENE,clean-epsilon*d,q,a)[0]
        fd = {h:float(((plus[h][0,0]-plus[h][0,1])-(minus[h][0,0]-minus[h][0,1]))/(2*epsilon))
              for h in HEADS}
    require(all(abs(fd[h]-gradients[h])<=DESIGN["finite_atol"] for h in HEADS),
            "Frozen native group derivative disagrees with its small finite perturbation")
    return dict(derivative=gradients,finite=fd,margins={h:float(v.detach()) for h,v in margins.items()})


def exercise(execution):
    started = perf_counter()
    device = execution.device
    q = assignment(device)
    controls = {}
    for seed in SEEDS:
        model = planted(seed,device)
        rng = np.random.default_rng([seed,902])
        values = rng.uniform(.23,.27,size=(6,8,64)).astype(np.float32)
        original = torch.from_numpy(values).to(device)
        cells = list(range(30,36))
        arms = {}
        adjacencies = {name:graph(count,device) for name,count in (("wide",2),("narrow",1),("cut",0))}
        fixed = adjacencies["wide"]
        for name,a in adjacencies.items():
            same = a.to_dense().gt(0)
            normal = fixed.to_dense().gt(0)
            require(torch.equal(same.sum(0),normal.sum(0)) and torch.equal(same.sum(1),normal.sum(1)),
                    "Capacity pair changed in/out degree")
            arms[name] = [controlled(model,original,q,a,cells,0,eps) for eps in DESIGN["epsilon"]]
        arms["missing_cue"] = [controlled(model,original,q,fixed,cells,1,eps) for eps in DESIGN["epsilon"]]
        arms["local_only"] = [controlled(model,original,q,fixed,[0],0,eps) for eps in DESIGN["epsilon"]]
        arms["one_hop_relay"] = [controlled(model,original,q,fixed,list(range(6,12)),0,eps)
                                  for eps in DESIGN["epsilon"]]
        narrowed = q.indices().clone()
        narrowed[1,33:36] = 6
        slim = torch.sparse_coo_tensor(narrowed,q.values(),q.shape,device=device).coalesce()
        arms["support_width"] = [[controlled(model,original,z,fixed,[30],0,eps) for z in (q,slim)]
                                  for eps in DESIGN["epsilon"]]
        for index in range(len(DESIGN["epsilon"])):
            selected = {k:rows[index]["derivative"] for k,rows in arms.items() if k!="support_width"}
            wide_pool,narrow_pool = (a["derivative"]["graph"] for a in arms["support_width"][index])
            require(abs(narrow_pool-2*wide_pool)<1e-6 and wide_pool>0
                    and selected["one_hop_relay"]["stage1_target"]>0,"Same signal pooling width or one-hop control failed")
            wide,narrow,cut,blank,local = (selected[k] for k in ("wide","narrow","cut","missing_cue","local_only"))
            require(wide["graph"]>0 and narrow["graph"]>0 and abs(wide["graph"]-2*narrow["graph"])<1e-6
                    and cut["graph"]==blank["graph"]==0 and all(item["stage1_target"]==0 for item in (wide,narrow,cut))
                    and wide["cnn"]==narrow["cnn"]==cut["cnn"]==blank["cnn"]==0 and local["cnn"]>0
                    and wide["fused"]>narrow["fused"]>cut["fused"] and cut["fused"]==0,
                    "The preregistered directed, bottleneck and local controls failed")
        controls[str(seed)] = arms
    model = planted(64,device)
    sample = torch.from_numpy(np.random.default_rng(431).uniform(.05,.65,size=(22,24,64)).astype(np.float32)).to(device)
    with torch.no_grad():
        _,_,q0,a0 = spatial_forward(model,SCENE,sample,None,None,fixed=False)
    observed = measure(model,SCENE,sample,q0,a0,[(143,1,2)],finite=True,rebuild=True)
    execution.sync()
    return dict(schema="wp12_native_pathway_controls_v1",predeclared=DESIGN,seeds=controls,
                native_graph={k:v for k,v in observed.items() if k!="features"},
                numeric=execution.numeric,optimizer_updates=0,seconds=perf_counter()-started)
