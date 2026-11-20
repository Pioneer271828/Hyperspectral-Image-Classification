"""Frozen native-head and spatial diagnostics on released point ordinals."""

from collections import Counter, defaultdict
from contextlib import ExitStack

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, require
from evaluation.wp12_mechanism_real import probe
from transfer.g3_training import _forward

FOCUS = {"xuzhou": (5, 6), "whu-honghu": (15, 21)}
RIVALS = {"xuzhou": {5: (8, 1), 6: (3, 1)}, "whu-honghu": {15: (1,), 21: (18,)}}


def heads(model, scene):
    modules = dict(zip(HEADS,(model.heads[scene],model.hgnn_heads[scene],model.cnn_heads[scene]),strict=True))
    require(all(isinstance(module,torch.nn.Linear) and module.bias is not None and module.in_features==64
                for module in modules.values()),"Expected the original three 64-channel linear heads")
    return modules


def weights(model, scene):
    return {name:(layer.weight.detach().double().cpu().clone(),layer.bias.detach().double().cpu().clone())
            for name,layer in heads(model,scene).items()}


def decompose(features, logits, targets, rivals, current, initial):
    """Reconstruct the actual logits with fixed true/rival pairs and one frozen head."""
    w,b = current
    w0,b0 = initial
    y,r = torch.as_tensor(targets)-1,torch.as_tensor(rivals)-1
    i = torch.arange(len(y))
    bias = b[y]-b[r]
    inner = ((w[y]-w[r])*features).sum(1)
    observed = logits[i,y]-logits[i,r]
    error = (bias+inner-observed).abs()
    require(torch.isfinite(error).all().item() and torch.allclose(bias+inner,observed,atol=2e-5,rtol=2e-6),
            "Recorded logits do not reconstruct from the actual native head feature and bias")
    reference = ((w0[y]-w0[r])*features).sum(1)
    return torch.column_stack((observed,bias,inner,reference,bias-(b0[y]-b0[r]),error)).tolist()


def spatial_stratum(tile, cells):
    y0,y1,x0,x1 = tile["box"]
    h,w = y1-y0,x1-x0
    ys,xs = np.divmod(cells,w)
    normalized_distance = np.minimum(np.minimum(ys,h-1-ys)/max(h-1,1),
                                     np.minimum(xs,w-1-xs)/max(w-1,1))
    return (normalized_distance<.125).tolist()


@torch.inference_mode()
def tile_rows(model, scene, tile, view, cells, truth, block, initial, *, layers=False, fixed_rivals=None):
    captured = {}
    modules = heads(model,scene.dataset)

    def hook(name):
        def collect(_module, args, out):
            require(name not in captured and out.ndim==2 and args[0].ndim==2,
                    "Expected one native linear readout per released block")
            ix = torch.as_tensor(cells,device=out.device)
            captured[name] = (args[0][ix].double().cpu(),out[ix].double().cpu())
        return collect

    with ExitStack() as hooks:
        for name,module in modules.items():
            hooks.callback(module.register_forward_hook(hook(name)).remove)
        if layers:
            from evaluation.wp12_mechanism_real import invoke_xuzhou
            probabilities, tensors, _ = probe(model,lambda action:invoke_xuzhou(model,scene,tile,view,action),collect=True)
        else:
            probabilities = _forward(model,scene,tile,observation_view=view)[:3]
            tensors = {}
    require(set(captured)==set(HEADS),"Three native head features required")
    predictions = {}
    components = {}
    second = {}
    fixed_rivals = {} if fixed_rivals is None else fixed_rivals
    rivals = [fixed_rivals.get(y,RIVALS[scene.dataset].get(y,(2 if y==1 else 1,))[0]) for y in truth]
    second_rivals = [RIVALS[scene.dataset].get(y,(r,))[-1] for y,r in zip(truth,rivals,strict=True)]
    for head,out in zip(HEADS,probabilities,strict=True):
        x,z = captured[head]
        p = out.detach().cpu().numpy()[cells]
        require(np.allclose(p,z.softmax(1).numpy(),atol=2e-6,rtol=2e-6),"Probe changed native three-head probability")
        predictions[head] = (p.argmax(1)+1).tolist()
        now = modules[head].weight.detach().double().cpu(),modules[head].bias.detach().double().cpu()
        components[head] = decompose(x,z,truth,rivals,now,initial[head])
        if scene.dataset=="xuzhou":
            second[head] = decompose(x,z,truth,second_rivals,now,initial[head])
    rows = []
    for i,(y,r,r2,edge) in enumerate(zip(truth,rivals,second_rivals,spatial_stratum(tile,cells),strict=True)):
        predicted = {head:predictions[head][i] for head in HEADS}
        rows.append(dict(block=block,ordinal=i+1,truth=y,rival=r,block_periphery=edge,
                         head_disagreement=len(set(predicted.values()))>1,predicted=predicted,
                         terms={head:parts[i] for head,parts in components.items()},
                         secondary=None if not second else dict(rival=r2,terms={h:x[i] for h,x in second.items()})))
    # Tensors remain in the private worker. Published summaries hold only blockwise norms.
    return rows,tensors


def class_rows(rows, classes):
    groups = defaultdict(list)
    for row in rows:
        groups[row["truth"]].append(row)
    result = {}
    for cls in range(1,classes+1):
        points = groups[cls]
        if not points:
            result[str(cls)] = dict(n=0,heads=None,blocks=None)
            continue
        score = {}
        for head in HEADS:
            components = np.asarray([row["terms"][head] for row in points])
            score[head] = dict(correct=sum(r["predicted"][head]==cls for r in points),
                guesses=dict(sorted(Counter(r["predicted"][head] for r in points).items())),
                mean_fixed_margin=float(components[:,0].mean()),mean_bias=float(components[:,1].mean()),
                mean_inner_product=float(components[:,2].mean()),
                without_bias_positive=int((components[:,2]>0).sum()),
                with_bias_positive=int((components[:,0]>0).sum()),
                max_reconstruction_error=float(components[:,5].max()))
        blocks = {}
        for group, predicate in (
            ("peripheral",lambda r:r["block_periphery"]),
            ("interior",lambda r:not r["block_periphery"]),
            ("heads_disagree",lambda r:r["head_disagreement"]),
            ("heads_agree",lambda r:not r["head_disagreement"]),
        ):
            subset = [p for p in points if predicate(p)]
            blocks[group] = dict(n=len(subset), correct={h:sum(r["predicted"][h]==cls for r in subset) for h in HEADS})
        result[str(cls)] = dict(n=len(points),heads=score,blocks=blocks)
    return result


def evolution(before,after):
    """Decompose changed margins using one point order, rival and initial head."""
    require(len(before)==len(after),"Endpoint comparison crosses a population")
    maximum = 0.
    for old,new in zip(before,after,strict=True):
        require(all(old[k]==new[k] for k in ("block","ordinal","truth","rival")),"Endpoint point order differs")
        for head in HEADS:
            a = old["terms"][head]
            b = new["terms"][head]
            delta = [b[4],b[2]-b[3],b[3]-a[2]]
            error = abs(sum(delta)-(b[0]-a[0]))
            maximum = max(maximum,error)
            new.setdefault("from_initial",{})[head] = delta
    require(maximum<5e-5,"Changed-margin reconstruction failed")
    return maximum


def parameter_deltas(reference,current):
    output = {}
    for name,(w,b) in current.items():
        prior_w,prior_b = reference[name]
        output[name] = dict(weights_l2=(w-prior_w).norm(dim=1).tolist(),bias_delta=(b-prior_b).tolist())
    return output


def paired_roles(labels,blocks,trace,cursors,classes):
    """Count realized labeled contributions and visits, independently of gradient magnitudes."""
    support = {tile["id"]:np.asarray(values).ravel() for tile,values in blocks}
    unique = defaultdict(set)
    exposure = np.zeros(classes+1,dtype=np.int64)
    visits = np.zeros(classes+1,dtype=np.int64)
    require(sum(int((values>0).sum()) for values in support.values())==len(labels),
            "Training exposure reference differs from released population")
    output = {}
    if 0 in cursors:
        output["0"] = dict(labeled_visits=[0]*classes,labeled_steps=[0]*classes,unique_labels=[0]*classes)
    for index,tile in enumerate(trace,1):
        require(tile in support,"Recorded optimizer exposure escaped released support blocks")
        values = support[tile]
        count = np.bincount(values,minlength=classes+1)
        exposure += count
        visits += count>0
        for point in np.flatnonzero(values):
            unique[int(values[point])].add((tile,int(point)))
        if index in cursors:
            output[str(index)] = dict(labeled_visits=exposure[1:].tolist(),labeled_steps=visits[1:].tolist(),
                                      unique_labels=[len(unique[c]) for c in range(1,classes+1)])
    require(set(output)=={str(cursor) for cursor in cursors},"Recorded trace misses a requested cursor")
    return output
