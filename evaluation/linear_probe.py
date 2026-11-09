"""Equal-capacity train/val probes on detached features, never a model loss.

Independent implementation of Alain & Bengio, arXiv:1610.01644v4 section3.2.
Fixed column subsampling, train-only standardization and finite SGD budget are
project controls. Linear separability is descriptive, not causal semantics.
"""
import copy
import math

import torch
import torch.nn.functional as F

from model.research_config import digest
from research_runtime import tensor_hash

POLICY = "equal_capacity_frozen_linear_probe_v1"
ROLES = ("train", "val")


def fingerprint(value):
    if torch.is_tensor(value): return tensor_hash(value)
    if isinstance(value, dict): return digest({k:fingerprint(v) for k,v in sorted(value.items())})
    if isinstance(value, (list, tuple)): return digest([fingerprint(v) for v in value])
    return digest(value)


def validate_plan(plan):
    if (set(plan) != {"policy", "classes", "width", "seed", "steps", "lr", "l2", "scale_floor"}
            or plan["policy"] != POLICY):
        raise ValueError("Exact declared linear-probe plan required")
    for k, low, high in (("classes",2,256), ("width",1,512), ("seed",0,2**32-1), ("steps",1,10000)):
        if type(plan[k]) is not int or not low <= plan[k] <= high:
            raise ValueError("Invalid probe integer budget: "+k)
    for k in ("lr", "l2", "scale_floor"):
        v = plan[k]
        if type(v) not in (int,float) or not math.isfinite(v) or v < 0 or (k != "l2" and v == 0):
            raise ValueError("Finite declared probe optimization/scale parameters required")
    return plan


def prepare(pack, plan):
    validate_plan(plan)
    if (set(pack) != {"features", "labels", "row_ids", "identity"}
            or set(pack["labels"]) != set(ROLES) or set(pack["row_ids"]) != set(ROLES)
            or not isinstance(pack["identity"], dict) or pack["identity"].get("query_labels_accessed") is not False
            or not pack["features"] or len(pack["features"]) > 32):
        raise ValueError("Only declared detached train/val feature roles allowed")
    labels, ids = {}, {}
    for role in ROLES:
        y, ix = pack["labels"][role], pack["row_ids"][role]
        if (not torch.is_tensor(y) or not torch.is_tensor(ix) or y.dtype != torch.int64 or ix.dtype != torch.int64
                or y.ndim != 1 or ix.shape != y.shape or not 1 <= len(y) <= 1000000
                or (ix < 0).any() or len(ix.unique()) != len(ix)
                or (y < 0).any() or (y >= plan["classes"]).any()):
            raise ValueError("Valid distinct row IDs and declared class labels required")
        labels[role], ids[role] = y.detach().cpu().clone(), ix.detach().cpu().clone()
    if len(torch.cat(tuple(ids.values())).unique()) != sum(len(v) for v in ids.values()):
        raise ValueError("Train and validation rows must not overlap")
    if set(labels["train"].tolist()) != set(range(plan["classes"])):
        raise ValueError("Training role must cover every declared class")
    layers, metadata = {}, {}
    for name, values in sorted(pack["features"].items()):
        if not isinstance(name,str) or not name or set(values) != set(ROLES):
            raise ValueError("Named paired train/val layers required")
        rows = {}
        for role in ROLES:
            x = values[role]
            if (not torch.is_tensor(x) or x.layout != torch.strided or x.ndim != 2 or not x.is_floating_point()
                    or x.shape[0] != len(labels[role]) or not plan["width"] <= x.shape[1] <= 512
                    or not torch.isfinite(x).all()):
                raise ValueError("Finite bounded feature rows and equal probe width required")
            rows[role] = x.detach().to(device="cpu", dtype=torch.float64).clone()
        width = rows["train"].shape[1]
        if rows["val"].shape[1] != width: raise ValueError("Layer role widths differ")
        generator = torch.Generator(device="cpu").manual_seed(plan["seed"])
        columns = torch.arange(width) if width == plan["width"] else torch.randperm(width, generator=generator)[:plan["width"]].sort().values
        train = rows["train"][:, columns]
        mean, sd = train.mean(0), train.std(0, unbiased=False)
        scale = torch.where(sd > plan["scale_floor"], sd, torch.ones_like(sd))
        layers[name] = {role: (x[:,columns]-mean)/scale for role,x in rows.items()}
        if any(not torch.isfinite(v).all() for v in layers[name].values()):
            raise ValueError("Standardized probe feature overflow")
        metadata[name] = {"original_width":width, "columns":columns.tolist(),
            "mean":mean, "scale":scale, "near_constant_columns":int((sd <= plan["scale_floor"]).sum()),
            "parameters":plan["classes"]*(plan["width"]+1)}
    return layers, labels, metadata


def gradient(x, labels, weight, bias, l2):
    """Autograd reference path updates only explicitly independent W,b."""
    w = weight.detach().clone().requires_grad_(); b = bias.detach().clone().requires_grad_()
    loss = F.cross_entropy(x.detach() @ w.T + b, labels) + .5*l2*w.square().sum()
    gw, gb = torch.autograd.grad(loss, (w,b))
    if not torch.isfinite(loss) or not torch.isfinite(gw).all() or not torch.isfinite(gb).all():
        raise ValueError("Nonfinite probe optimization")
    return float(loss), gw.detach(), gb.detach()


def metrics(x, y, weight, bias):
    logp = F.log_softmax(x @ weight.T + bias, dim=1)
    if not torch.isfinite(logp).all(): raise ValueError("Nonfinite probe prediction")
    classes = weight.shape[0]; pred = logp.argmax(1)
    cm = torch.bincount(y*classes+pred, minlength=classes**2).reshape(classes,classes)
    counts = cm.sum(1); supported = counts > 0
    recalls = [int(cm[i,i])/int(counts[i]) if counts[i] else None for i in range(classes)]
    aa = sum(v for v in recalls if v is not None)/int(supported.sum())
    return {"nll":float(-logp[torch.arange(len(y)),y].mean()), "oa":int(cm.diag().sum())/int(cm.sum()),
        "aa_supported":aa, "aa":aa if supported.all() else None, "recall":recalls,
        "confusion":cm.tolist(), "support":counts.tolist(), "missing_classes":torch.where(~supported)[0].tolist()}


def advance(pack, plan, *, state=None, stop_after=None):
    """One immutable budget; validation affects selection only, never updates."""
    before = fingerprint(pack); rng = torch.get_rng_state().clone()
    layers, labels, normalization = prepare(pack, plan)
    identity = digest({"policy":POLICY, "plan":plan, "input":before})
    if state is None:
        state = {"identity":identity, "cursor":0, "layers":{}}
        for name in layers:
            state["layers"][name] = {"weight":torch.zeros(plan["classes"],plan["width"],dtype=torch.float64),
                "bias":torch.zeros(plan["classes"],dtype=torch.float64), "history":[], "best":None}
    else:
        state = copy.deepcopy(state)
        checksum = state.pop("checksum", None)
        if (checksum != fingerprint(state) or state.get("identity") != identity or set(state) != {"identity","cursor","layers"}
                or type(state["cursor"]) is not int or not 0 <= state["cursor"] <= plan["steps"]
                or set(state["layers"]) != set(layers)):
            raise ValueError("Probe resume identity/state differs")
        for value in state["layers"].values():
            if (set(value) != {"weight","bias","history","best"} or len(value["history"]) != state["cursor"]
                    or value["weight"].shape != (plan["classes"],plan["width"]) or value["bias"].shape != (plan["classes"],)
                    or any(t.dtype != torch.float64 or t.device.type != "cpu" or not torch.isfinite(t).all()
                           for t in (value["weight"],value["bias"]))):
                raise ValueError("Probe checkpoint schema differs")
            history, best = value["history"], value["best"]
            if any(set(h) != {"step","train_objective_before_update","val_nll","gradient_norm"}
                   or h["step"] != i+1 or any(type(h[k]) not in (int,float) or not math.isfinite(h[k])
                       for k in ("train_objective_before_update","val_nll","gradient_norm"))
                   for i,h in enumerate(history)):
                raise ValueError("Probe history schema differs")
            if history:
                chosen = min(history, key=lambda h: h["val_nll"])
                if (not isinstance(best,dict) or set(best) != {"step","nll","weight","bias"}
                        or best["step"] != chosen["step"] or best["nll"] != chosen["val_nll"]
                        or best["weight"].shape != value["weight"].shape or best["bias"].shape != value["bias"].shape
                        or any(t.dtype != torch.float64 or t.device.type != "cpu" or not torch.isfinite(t).all()
                               for t in (best["weight"],best["bias"]))):
                    raise ValueError("Probe first-min selection differs")
            elif best is not None:
                raise ValueError("Empty probe cursor cannot have selected state")
    start = state["cursor"]
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= plan["steps"]):
        raise ValueError("Explicit bounded probe stop count required")
    end = min(plan["steps"], start + (stop_after if stop_after is not None else plan["steps"]))
    for step in range(start+1,end+1):
        for name, x in layers.items():
            value = state["layers"][name]
            loss, gw, gb = gradient(x["train"], labels["train"], value["weight"], value["bias"], plan["l2"])
            value["weight"] -= plan["lr"]*gw; value["bias"] -= plan["lr"]*gb
            val = metrics(x["val"], labels["val"], value["weight"], value["bias"])
            value["history"].append({"step":step,"train_objective_before_update":loss,"val_nll":val["nll"],
                                    "gradient_norm":float(torch.sqrt(gw.square().sum()+gb.square().sum()))})
            if value["best"] is None or val["nll"] < value["best"]["nll"]:
                value["best"] = {"step":step,"nll":val["nll"],"weight":value["weight"].clone(),"bias":value["bias"].clone()}
        state["cursor"] = step
    reports = {}
    for name,x in layers.items():
        value = state["layers"][name]; best = value["best"]
        if best is None: raise ValueError("Probe has no measured update")
        reports[name] = {"selected_step":best["step"], "normalization":{k:v.tolist() if torch.is_tensor(v) else v for k,v in normalization[name].items()},
            "metrics":{role:metrics(x[role],labels[role],best["weight"],best["bias"]) for role in ROLES},
            "last_gradient_norm":value["history"][-1]["gradient_norm"], "selected_at_budget_end":best["step"] == plan["steps"]}
    if fingerprint(pack) != before or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("Read-only feature/input/RNG contract violated")
    state["checksum"] = fingerprint(state)
    return state, {"policy":POLICY, "input_identity":before, "plan":copy.deepcopy(plan),
        "status":"completed" if end == plan["steps"] else "partial", "updates_this_call_per_probe":end-start,
        "probe_updates_this_call":(end-start)*len(layers), "backbone_updates":0, "query_labels_accessed":False,
        "input_unchanged":True, "layers":reports, "convergence_claim":False,
        "scope":"Equal parameter budget on fixed selected columns; descriptive linear separability only"}


def capture_pack(model, target, observation, *, labels, row_ids):
    """Caller supplies already authorized labels; no dataset/query loader exists."""
    from evaluation.frozen_representation_capture import capture
    from model.region_pixel_alignment import broadcast_regions
    if set(labels) != set(ROLES) or set(row_ids) != set(ROLES):
        raise ValueError("Only train/val capture roles allowed")
    n = observation["Q"].shape[0]
    for role in ROLES:
        ix, y = row_ids[role], labels[role]
        if (not torch.is_tensor(ix) or ix.dtype != torch.int64 or ix.ndim != 1 or len(ix) == 0
                or (ix < 0).any() or (ix >= n).any() or len(ix.unique()) != len(ix)
                or not torch.is_tensor(y) or y.dtype != torch.int64 or y.shape != ix.shape or (y < 0).any()):
            raise ValueError("Invalid selected pixel rows")
    joined = torch.cat([row_ids[role].cpu() for role in ROLES])
    if len(joined.unique()) != len(joined): raise ValueError("Train and validation rows must not overlap")
    captured, _, receipt = capture(model, target, observation)
    features = {}
    for stage, values in captured.items():
        pixel = {"cnn":values["cnn"], "graph":broadcast_regions(values["Q"], values["graph"])}
        for branch,x in pixel.items():
            features[stage+"_"+branch] = {role:x[row_ids[role].cpu()].detach().clone() for role in ROLES}
    return {"features":features,"labels":{k:v.detach().cpu().clone() for k,v in labels.items()},
        "row_ids":{k:v.detach().cpu().clone() for k,v in row_ids.items()},
        "identity":{"capture":receipt,"target":target,"query_labels_accessed":False,
                    "label_authorization":"caller_declared_train_val_not_independently_authenticated"}}
