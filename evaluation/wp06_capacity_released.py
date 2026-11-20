"""Equal-width readout diagnostics on two already authorized native endpoints."""

from collections import Counter
from io import BytesIO
from time import perf_counter

import numpy as np

from evaluation.frozen_training_stages import HEADS, VIEWS, file_pin, json_pin, pinned, require
from evaluation.retained_spectral_evidence import ReleasedBlocks, fit_support
from model.research_config import digest
from transfer.development_bank import _atomic_new

SCHEMA = "wp06_real_development_capacity_v1"
CASES = {"selected": ("xuzhou",347,9,90,45,(5,6)),
         "whu-honghu": ("whu-honghu",24,22,110,110,(15,21))}
LAYERS = ("vca","pre_pool","pooled","cnn","graph","fused")
WIDTH = 64
DRAWS = (11,23,47,89,131)


def capacity_protocol():
    return dict(width=WIDTH, layers=list(LAYERS), draws=list(DRAWS), k=[1,5],
                ridge=1., floor=1e-10, roster="approved_released_support_and_historical_devval",
                capacity_columns="fixed_evenly_spaced_integer_indices",
                solver="dual_fp64", score_decimals=12,
                parent_control="remove_same_per_class_from_other_parents")


def project(values):
    require(values.ndim==2 and values.shape[1]>=WIDTH and np.isfinite(values).all(),
            "Representation cannot supply the registered 64-column readout")
    columns = np.linspace(0, values.shape[1]-1, WIDTH, dtype=int)
    return values[:,columns],columns.tolist()


def assemble(bundle, case):
    scene,cursor,classes,nfit,nheld,focus = CASES[case]
    require((bundle["scene"],bundle["stage"],bundle["classes"],tuple(bundle["focus"]))==(scene,cursor,classes,focus),
            "An unregistered real endpoint reached the capacity diagnostic")
    arrays,view_proofs = {},{}
    parent_labels = {}
    for band in VIEWS:
        record = bundle["views"][band]
        for role,count in (("devtrain",nfit),("devval",nheld)):
            saved = record["captured"][role]
            require(len(saved.truth)==len(saved.identities)==len(saved.parents)==count
                    and len(set(saved.identities))==count and set(saved.truth)==set(range(1,classes+1)),
                    "Released role, support ranks or class roster differs")
            if band=="full":
                arrays[f"{role}_y"] = np.asarray(saved.truth,dtype=np.int64)
                positions = sorted(set(saved.parents))
                parent_labels[role] = {value:index for index,value in enumerate(positions,1)}
                arrays[f"{role}_parents"] = np.array([parent_labels[role][p] for p in saved.parents],dtype=np.int64)
                arrays[f"{role}_ids"] = np.asarray(saved.identities,dtype=np.int64)
            else:
                require(np.array_equal(arrays[f"{role}_y"],saved.truth)
                        and np.array_equal(arrays[f"{role}_ids"],saved.identities)
                        and np.array_equal(arrays[f"{role}_parents"],[parent_labels[role][p] for p in saved.parents]),
                        "Physical points, class order or parent blocks changed across views")
            for layer in LAYERS:
                values,columns = project(np.concatenate(saved.labels[layer]))
                arrays[f"{band}_{role}_{layer}"] = values
                if band=="full" and role=="devtrain":
                    arrays[f"columns_{layer}"] = np.asarray(columns,dtype=np.int64)
                require(np.array_equal(columns,arrays[f"columns_{layer}"]),
                        "Same-capacity feature selection changed across real roles or views")
        for head in HEADS:
            arrays[f"{band}_native_{head}"] = np.asarray(record["native"][head],dtype=np.int64)
        view_proofs[band] = dict(retained_ids=record["band_ids"],archived_agreement=record["archived_agreement"],
                                 capture_seconds=record["capture_seconds"])
    require(not set(arrays["devtrain_ids"]).intersection(arrays["devval_ids"])
            and len(view_proofs["central_gap"]["retained_ids"])==len(view_proofs["matched_dispersed"]["retained_ids"]),
            "Fit and development roles overlap or band counts differ")
    return arrays,view_proofs


def sample_design(labels, parent, focus):
    classes = np.unique(labels)
    index = {c: np.flatnonzero(labels==c) for c in classes}
    full = tuple(range(len(labels)))
    selections,seen,controls = {},{full:"all"},{}
    for draw in DRAWS:
        rng = np.random.default_rng([3871,draw])
        ranks = {c:rng.permutation(rows) for c,rows in index.items()}
        for k in (1,5):
            require(all(len(rows)>=k for rows in ranks.values()),"Released support is too small for the proposed thinning")
            chosen = tuple(sorted(int(i) for rows in ranks.values() for i in rows[:k]))
            label = f"k{k}-d{draw}"
            if chosen not in seen:
                seen[chosen] = label
                selections[label] = chosen
            controls[label] = seen[chosen]
    alternatives = {}
    for p in sorted(set(parent[np.isin(labels,focus)])):
        retained = np.flatnonzero(parent!=p)
        per_class = {c: int(((labels==c)&(parent==p)).sum()) for c in classes}
        lost = [int(c) for c in classes if per_class[c]==len(index[c])]
        outside = {c: index[c][parent[index[c]]!=p] for c in classes}
        insufficient = [int(c) for c in classes if len(outside[c])<per_class[c]]
        if lost or insufficient:
            alternatives[str(p)] = dict(status="not_estimable",
                missing_classes=lost,other_parent_control_impossible=insufficient,
                removed_by_class=[per_class[c] for c in classes])
            continue
        rng = np.random.default_rng([3901,int(p)])
        given = {int(i) for c,rows in outside.items() for i in rng.permutation(rows)[:per_class[c]]}
        leave,matched = f"parent-{int(p)}",f"matched-count-{int(p)}"
        selections[leave] = tuple(retained.tolist())
        selections[matched] = tuple(i for i in full if i not in given)
        alternatives[str(p)] = dict(status="paired",spatial=leave,count_control=matched,
                                   removed_by_class=[per_class[c] for c in classes])
    return {"all":full,**selections},controls,alternatives


def tally(y, pred, focus, classes):
    pred = np.asarray(pred)
    require(y.shape==pred.shape and all(1<=p<=classes for p in pred),"Foreign class prediction")
    return dict(n=len(y),correct=int((y==pred).sum()),
                focus={str(c):dict(n=int((y==c).sum()),correct=int(((y==c)&(pred==c)).sum()),
                    errors={str(k):int(v) for k,v in sorted(Counter(pred[(y==c)&(pred!=c)]).items())})
                       for c in focus},decision=digest([int(p) for p in pred]))


def dual_predictions(train, val, y):
    # Reuse the established Train-only scaling and FP64 dual ridge estimator.
    # Keep its unrounded coefficients and apply the registered tie convention.
    fit = ReleasedBlocks("devtrain",int(y.max()),layers=train)
    fit.truth = y.tolist()
    fit.identities = list(range(len(y)))
    fit.labels = {layer:[values] for layer,values in train.items()}
    _,ids,base,body,_ = fit_support(fit)
    decisions = {}
    for layer,x in val.items():
        center,scale,support,_,dual = body[layer]
        scores = ((x-center)/scale @ support.T) @ dual + base
        decisions[layer] = ids[np.round(scores,12).argmax(1)]
    return decisions


def primal_predictions(train,val,y):
    # Independent feature-space solve. Population scaling and intercept are
    # reconstructed from the sealed support rows, without the dual fitter.
    classes = np.unique(y)
    response = (y[:,None] == classes).astype(np.float64)
    intercept = response.mean(0)
    result = {}
    for name,A in train.items():
        A = np.asarray(A,dtype=np.float64)
        shift,std = A.mean(0),A.std(0)
        scale = np.where(std>1e-10,std,1.)
        X = (A-shift)/scale
        beta = np.linalg.solve(X.T@X+np.eye(WIDTH), X.T@(response-intercept))
        score = (np.asarray(val[name],dtype=np.float64)-shift)/scale@beta+intercept
        result[name] = classes[np.round(score,12).argmax(1)]
    return result


def score(arrays, case, *, independent=False):
    _,_,classes,nfit,nheld,focus = CASES[case]
    train_y,dev_y = (arrays[f"{r}_y"] for r in ("devtrain","devval"))
    require(len(train_y)==nfit and len(dev_y)==nheld
            and set(train_y)==set(dev_y)==set(range(1,classes+1))
            and not set(arrays["devtrain_ids"]).intersection(arrays["devval_ids"]),
            "Frozen features lost their actual three-head role or integer population")
    selections, draws, parents = sample_design(train_y,arrays["devtrain_parents"],focus)
    roster = {}
    for name,mask in selections.items():
        y = train_y[list(mask)]
        roster[name] = dict(total=len(y),counts=np.bincount(y,minlength=classes+1)[1:].tolist(),
                            support_sha256=digest(arrays["devtrain_ids"][list(mask)].tolist()))
    fun = primal_predictions if independent else dual_predictions
    output = {}
    for band in VIEWS:
        val = {key:arrays[f"{band}_devval_{key}"] for key in LAYERS}
        output[band] = dict(native={h:tally(dev_y,arrays[f"{band}_native_{h}"],focus,classes) for h in HEADS},
                           matched={},locked_full={})
        for name,mask in selections.items():
            idx = list(mask)
            y = train_y[idx]
            train = {key:arrays[f"{band}_devtrain_{key}"][idx] for key in LAYERS}
            result = fun(train,val,y)
            output[band]["matched"][name] = {k:tally(dev_y,p,focus,classes) for k,p in result.items()}
        train = {key:arrays[f"full_devtrain_{key}"] for key in LAYERS}
        result = fun(train,val,train_y)
        output[band]["locked_full"] = {k:tally(dev_y,p,focus,classes) for k,p in result.items()}
    return dict(views=output,selections=roster,draws=draws,parents=parents,
                capacity=dict(slopes=WIDTH*classes,intercepts=classes),
                release=dict(fit=nfit,development=nheld,fit_parents=len(set(arrays["devtrain_parents"])),
                    evaluation_parents=len(set(arrays["devval_parents"])),
                    focus_parents={role: {str(c):sorted(set(int(p) for p in arrays[f"{role}_parents"][
                        arrays[f"{role}_y"]==c])) for c in focus} for role in ("devtrain","devval")}),
                point_order=digest(arrays["devval_ids"].tolist()))


def production(path, case, bundle):
    start = perf_counter()
    arrays,proofs = assemble(bundle,case)
    scored = score(arrays,case)
    archive = path.with_suffix(".npz")
    with BytesIO() as stream:
        np.savez_compressed(stream,**arrays)
        payload = stream.getvalue()
    if not archive.exists():
        _atomic_new(archive,payload)
    else:
        require(archive.read_bytes()==payload,"Existing capacity feature pack differs")
    result = {k:v for k,v in bundle.items() if k!="views"}
    return dict(**result,feature_pack=file_pin(archive),protocol=capacity_protocol(),
                capture=proofs,results=scored,analytic_seconds=perf_counter()-start)


def review(root, pins, code):
    outcomes = {}
    for case,link in pins.items():
        source = json_pin(root,link)
        require(source["schema"]==SCHEMA and source["producer"]==code["commit"] and source["case"]==case
                and source["optimizer_updates"]==0 and source["query_accessed"] is source["raw_gt_loaded"] is False,
                "Wrong real released-point reader or frozen scope")
        result = source["result"]
        require(result["protocol"]==capacity_protocol(),"Incorrect equal-capacity protocol")
        pack = pinned(root,result["feature_pack"])
        with np.load(pack,allow_pickle=False) as arrays:
            calculated = score(arrays,case,independent=True)
        require(calculated==result["results"],"Independent primal predictions or counts differ")
        outcomes[case] = {k:v for k,v in result.items() if k!="feature_pack"}
        outcomes[case]["producer_receipt"] = link
        outcomes[case]["independent_check"] = "all_readout_point_decisions_match"
    return outcomes
