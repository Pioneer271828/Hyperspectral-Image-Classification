"""WP12 support-only diagnostics from the independently released capacity archives."""

from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation import wp06_capacity_released as original
from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require
from evaluation.wp12_block_readouts import block_metrics
from evaluation.wp12_cross_backend import FORMAL_FLOOR
from evaluation.wp12_real_readiness import transition
from inference import support_readout as readout
from model.research_config import digest
from research_runtime import write_json

ARCHIVE = "wp02-wp05-wp06-real-readiness-20261120-qa"
PRODUCER = "35287d3d248e31492ba59523d792d7c0f0670eaa"
PINS = {
    "selected": "24a43448d8878ba140cfda6397c738cd443dcf2e259aa26244e6dc5e23721e62",
    "whu-honghu": "991b85ac4f3b8116e4bfa6ced6ecec1547f8bed9384f9441b99cf078bf8f777e",
}
SCHEMA = "wp12_deployed_support_head_v1"


def frozen_case(root, case):
    receipt_pin = dict(path=str(root/ARCHIVE/"native"/f"{case}.json"),sha256=PINS[case])
    receipt = json_pin(root,receipt_pin)
    old = receipt["result"]
    scene,stage,classes,nfit,nheld,focus = original.CASES[case]
    require(receipt["schema"]==original.SCHEMA and receipt["case"]==case and receipt["producer"]==PRODUCER
            and receipt["device"]=="cpu" and receipt["optimizer_updates"]==0
            and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False
            and (old["scene"],old["stage"],old["classes"],tuple(old["focus"]))==(scene,stage,classes,focus)
            and (old["results"]["release"]["fit"],old["results"]["release"]["development"])==(nfit,nheld)
            and old["protocol"]==original.capacity_protocol(),"Expected approved frozen head-only development evidence")
    source = dict(scene=scene,checkpoint=old["checkpoint"],state=old["state"],authority=old["authority"],
                  fit_receipt=receipt_pin["sha256"],cursor=stage,class_ids=list(range(1,classes+1)),
                  bands={v:old["capture"][v]["retained_ids"] for v in readout.VIEWS})
    return old,source


def roster(arrays, old):
    y,dev_y = arrays["devtrain_y"],arrays["devval_y"]
    si,vi = arrays["devtrain_ids"],arrays["devval_ids"]
    require(digest(vi.tolist())==old["results"]["point_order"] and len(set(si))==len(si)
            and set(si).isdisjoint(vi) and y.shape==si.shape and dev_y.shape==vi.shape
            and arrays["columns_fused"].tolist()==list(range(readout.WIDTH)),
            "Support/evaluation identity or fused capacity differs")
    selections,draws,parents = original.sample_design(y,arrays["devtrain_parents"],old["focus"])
    require(draws==old["results"]["draws"] and parents==old["results"]["parents"],
            "Archived K1/K5 and parent-block design changed")
    return selections,draws,parents


def binding(arrays, mask, classes):
    rows = list(mask)
    return dict(support_sha256=digest(arrays["devtrain_ids"][rows].tolist()),
                counts=np.bincount(arrays["devtrain_y"][rows],minlength=classes+1)[1:].tolist())


def fit_case(root, case, out):
    old,source = frozen_case(root,case)
    pack = pinned(root,old["feature_pack"])
    elapsed = {}
    with np.load(pack,allow_pickle=False) as arrays:
        selections,_,_ = roster(arrays,old)
        result = {}
        for name,mask in selections.items():
            rows = list(mask)
            y,ids = arrays["devtrain_y"][rows],arrays["devtrain_ids"][rows]
            fit = binding(arrays,mask,old["classes"])
            require(fit=={k:old["results"]["selections"][name][k] for k in fit},
                    "Actual fit members differ from frozen support")
            heads,elapsed[name] = {},{}
            for view in readout.VIEWS:
                x = arrays[f"{view}_devtrain_fused"][rows]
                started = perf_counter()
                heads[view] = readout.fit(x,y,ids)
                elapsed[name][view] = perf_counter()-started
            result[name] = dict(**fit,heads=heads)
    path = out/"heads.json"
    started = perf_counter()
    write_json(path,dict(schema=readout.SCHEMA,policy=readout.POLICY,source=source,selections=result))
    export_seconds = perf_counter()-started
    artifact = file_pin(path)
    readout.load(root,artifact,source)
    write_json(out/"fit-timing.json",dict(artifact=artifact,one_off_fit_seconds=elapsed,
               export_seconds=export_seconds,scientific_optimizer_updates=0,backbone_forwards=0))
    return artifact


def comparison(y, native, candidate, parents, focus, classes):
    before,after = ({i:(int(k),int(p)) for i,(k,p) in enumerate(zip(y,pred,strict=True))}
                    for pred in (native,candidate))
    changes = {str(k):transition(before,after,k) for k in range(1,classes+1)}
    native_correct,candidate_correct = (int((y==x).sum()) for x in (native,candidate))
    gains,losses = (sum(x[key] for x in changes.values()) for key in ("gains","losses"))
    require(candidate_correct-native_correct==gains-losses,"Paired gains and losses do not reconstruct OA")
    scores = block_metrics(y.tolist(),candidate.tolist(),parents.tolist(),classes)
    native_scores = block_metrics(y.tolist(),native.tolist(),parents.tolist(),classes)
    delta_aa = scores["class_macro_aa"]-native_scores["class_macro_aa"]
    sufficient = min(cell["n"] for cell in changes.values())>=FORMAL_FLOOR
    ordinary_drops = [k for k,cell in changes.items() if int(k) not in focus
                      and (cell["after"]-cell["before"])/cell["n"] < -.02]
    eligible = sufficient and delta_aa>=-.005 and not ordinary_drops and all(
        changes[str(k)]["after"]>=changes[str(k)]["before"] for k in focus) and any(
        changes[str(k)]["after"]>changes[str(k)]["before"] for k in focus)
    if classes==22 and any(changes[str(k)]["losses"] for k in focus):
        eligible = False
    return dict(metrics=scores,delta_aa=delta_aa,delta_oa=(gains-losses)/len(y),gained_points=gains,lost_points=losses,
                classes_improved=sum(c["after"]>c["before"] for c in changes.values()),
                classes_worsened=sum(c["after"]<c["before"] for c in changes.values()),
                all_class_transitions=changes,ordinary_class_recall_drops=ordinary_drops,
                focus={str(k):changes[str(k)] for k in focus},eval_floor_met=sufficient,
                registered_limits_passed=eligible,independent_confirmation=False)


def review_case(root, case, artifact, deployment=None):
    old,source = frozen_case(root,case)
    result = readout.load(root,artifact,source)
    focus,classes = old["focus"],old["classes"]
    with np.load(pinned(root,old["feature_pack"]),allow_pickle=False) as arrays:
        selections,draws,parents = roster(arrays,old)
        require(selections.keys()==result["selections"].keys(),"Support budget roster changed")
        y,parents_val = arrays["devval_y"],arrays["devval_parents"]
        views = {}
        for view in readout.VIEWS:
            native = {h:arrays[f"{view}_native_{h}"] for h in ("fused","graph","cnn")}
            views[view] = dict(native={h:block_metrics(y.tolist(),p.tolist(),parents_val.tolist(),classes) for h,p in native.items()},
                               fixed_full={},same_view={})
            x = arrays[f"{view}_devval_fused"]
            for name,mask in selections.items():
                saved = result["selections"][name]
                fit = binding(arrays,mask,classes)
                require(all(saved[k]==v for k,v in fit.items()),"Loaded head has foreign support membership")
                rows = list(mask)
                for mode,trained in (("fixed_full","full"),("same_view",view)):
                    p = readout.decisions(readout.scores(x,saved["heads"][trained]))
                    expected = original.primal_predictions(
                        {"fused":arrays[f"{trained}_devtrain_fused"][rows]},
                        {"fused":x},arrays["devtrain_y"][rows])["fused"]
                    require(np.array_equal(p,expected),"Reloaded decisions differ from an independent primal solve")
                    if mode=="same_view" or name=="all":
                        archived = old["results"]["views"][view]
                        archived_readout = archived["matched"][name] if mode=="same_view" else archived["locked_full"]
                        require(digest(p.tolist())==archived_readout["fused"]["decision"],
                                "Existing support-fit decisions changed")
                    views[view][mode][name] = comparison(y,native["fused"],p,parents_val,focus,classes)
            if deployment is not None:
                from evaluation.wp12_support_head_cost import verify_overlap

                views[view]["deployed_point_overlap"] = verify_overlap(
                    deployment, view, native, x, arrays["devval_ids"], result["selections"]["all"]["heads"])
    return dict(source=source,artifact=artifact,views=views,draw_aliases=draws,support_parents=parents,
                release=old["results"]["release"],capacity=dict(slopes=classes*64,intercepts=classes),
                registered_candidate=False,role="historically_exposed_devval",support_counts={
                    n:x["counts"] for n,x in result["selections"].items()})
