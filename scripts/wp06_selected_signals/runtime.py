"""External orchestration bound to the unmodified Xuzhou training runtime."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import ModuleType


COMMIT = "723b1afe43f6408ece6f34d461af8ef3c0a87365"
CAMPAIGN = "2dd3eff9c32300e1df6c330daaa7c79539301a699030c37dc2a4ae6ab8cfb6a4"
DONOR_AUDIT = "1bfd78ef6da01f8fa9cb05b98dd5625c2d6e3be46af75b30a8475ebb8fc9688d"
ATTEMPT = "d2f86e2808fb436c461f703c7e1ba40d8503e3f44777984b0c02e539d931abd6"
CATALOG = "9a00d2ea572557a5749b0bc0e9f05318d43f1b7c39094cee57904db73c0dc0b4"
HEADS = ("fused","graph","cnn")
EPSILONS = (0.001,0.01)
BANDS = (0,217,435)
CAP = 28*28*436
POLICY = "wp06_xuzhou_selected_spectral_signals_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path":str(path),"sha256":sha(path)}


def checked(value):
    path = Path(value["path"])
    require(path.resolve(strict=True)==path and sha(path)==value["sha256"],f"Changed pinned file: {path}")
    return path


def read(value):
    return json.loads(checked(value).read_text())


def write(path, value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,separators=(",",":"),allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return pin(path)


def request_population(plan, roles, classes, band_ids):
    """Choose by block geometry alone. No cube, label, score or model argument."""
    require(classes==list(range(1,10)) and len(band_ids)==436 and len(set(band_ids))==436,
            "Expected the released output and band dimensions")
    require(len(roles)==len(plan["tiles"]), "Incomplete role population")
    width = plan["shape"][1]
    rows = []
    for tile,role in zip(plan["tiles"],roles,strict=True):
        require({k:v for k,v in role.items() if k!="role"}==tile,"Role geometry differs")
        if role["role"] not in ("devtrain","devval"):
            continue
        require(tile["partition"]=="train","Development role uses a protected partition")
        y0,y1,x0,x1 = tile["box"]
        h,w = y1-y0,x1-x0
        require(1<h<=28 and 1<w<=28,"Unexpected released block geometry")
        center = ((y0+y1-1)//2,(x0+x1-1)//2)
        corners = [(y,x) for y in (y0,y1-1) for x in (x0,x1-1)]
        far = min(corners,key=lambda yx:(-sum((a-b)**2 for a,b in zip(yx,center,strict=True)),yx))
        distance = math.dist(center,far)/math.hypot(h-1,w-1)
        anchors = []
        for name,(y,x),center_distance in (("center",center,0.),("far_corner",far,distance)):
            anchors.append({"name":name,"global_yx":[y,x],"id":y*width+x,"distance_to_center":center_distance,
                            "distance_to_other_target":distance,"mean_to_both_targets":distance/2})
        ids = sorted(a["id"] for a in anchors)
        sources = [[i,b] for i in ids for b in BANDS]
        rows.append({"tile_id":tile["id"],"role":role["role"],"box":tile["box"],"target_ids":ids,"anchors":anchors,
                     "sources":sources,"physical_band_ids":[band_ids[i] for i in BANDS],
                     "requests":[{"class_id":c,"epsilon":eps} for c in classes for eps in EPSILONS]})
    require([sum(r["role"]==key for r in rows) for key in ("devtrain","devval")]==[73,5],
            "Expected all 73 training and five validation blocks")
    return rows


def bootstrap(producer):
    producer = Path(producer).resolve(strict=True)
    require(Path.cwd()==producer,"Run from the frozen producer checkout")
    sys.path.insert(0,str(producer))
    import torch
    import research_runtime as rr
    from transfer.container_runtime import training_container_kind
    from transfer import matched_route_admission as admission
    torch.set_num_threads(1)
    require(training_container_kind() and __debug__ and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
            Path(rr.__file__).resolve()==producer/"research_runtime.py"
            and Path(admission.__file__).resolve()==producer/"transfer/matched_route_admission.py",
            "The original authorized CPU runtime must supply the training imports")
    code = rr.code_snapshot()
    require(code["commit"]==COMMIT,"Wrong frozen producer commit")
    rr.require_frozen_code(code)
    return rr,admission,code


def enrollment(root, rr, admission, code):
    root = Path(root).resolve(strict=True)
    required = {"campaign":(root/"campaign.json",CAMPAIGN),
                "audit":(root/"inputs/donor-audit.json",DONOR_AUDIT),
                "catalog":(root/"inputs/catalog.json",CATALOG),
                "attempt":(root/"attempts/donor-full/status.json",ATTEMPT)}
    known = {name:{"path":str(path),"sha256":h} for name,(path,h) in required.items()}
    campaign,audit,catalog,attempt = [read(known[k]) for k in ("campaign","audit","catalog","attempt")]
    require(audit["campaign"]==known["campaign"] and audit["status"]=="passed"
        and audit["result"]["attempts"]["full"]==known["attempt"] and audit["result"]["selected_step"]==246
        and attempt["registration"]==campaign["registrations"]["donor"] and attempt["code"]==code
        and catalog["policy"]==admission.CATALOG and len(catalog["candidates"])==1,
        "The accepted Xuzhou campaign, attempt or exact producer source differs")
    candidate = catalog["candidates"][0]
    summary,_,_,pins = admission.authorize(candidate,code)
    require(candidate["attempt"]==known["attempt"] and candidate["registration"]==attempt["registration"]
        and summary["selected_step"]==246 and summary["scene"]=="xuzhou" and summary["budget"]==256,
        "Expected the original accepted dynamic Xuzhou selected state")
    all_pins = {p["path"]:p for p in [*known.values(),*pins,*campaign["inputs"].values()]}
    admission.check_pins(all_pins.values())
    return known,candidate,summary,list(all_pins.values())


def verify_deployment(deployment, caller):
    value = read(deployment)
    paths = {row["name"]:checked(row["pin"]) for row in value["files"]}
    require(len(paths)==len(value["files"]) and paths[caller]==Path(sys.argv[0]).resolve()
            and paths["runtime.py"]==Path(__file__).resolve(),"Wrong driver deployment")
    return value,paths


def engine(paths):
    """Load only the two verified diagnostic files in their own package namespace."""
    private = "wp06_registered_signal_engine"
    block,selected = paths["block_input_diagnostics.py"],paths["selected_signal_diagnostics.py"]
    require(block.parent==selected.parent and private not in sys.modules,"Unexpected diagnostic engine path")
    package = ModuleType(private)
    package.__path__ = [str(block.parent)]
    sys.modules[private] = package
    loaded = importlib.import_module(private+".selected_signal_diagnostics")
    require(Path(loaded.__file__).resolve()==selected
        and Path(sys.modules[private+".block_input_diagnostics"].__file__).resolve()==block,
        "Diagnostic extension import path differs")
    return loaded


def freeze_scope(out, donor, deployment, caller, rr, admission, code):
    _,paths = verify_deployment(deployment,caller)
    known,candidate,summary,pins = enrollment(donor,rr,admission,code)
    manifest = read(summary["development"])
    rows = request_population(manifest["plan"],manifest["record"]["roles"],manifest["record"]["classes"],
                              manifest["bands"]["band_ids"])
    count = sum(len(t["requests"]) for t in rows)
    out = rr.external_directory(out)
    require(not any(out==Path(p["path"]).parent or out in Path(p["path"]).parents or
                    Path(p["path"]).parent in out.parents for p in pins),"Output overlaps a protected input")
    out.mkdir(exist_ok=False)
    return write(out/"scope.json",{"policy":POLICY,"deployment":deployment,"donor_root":str(Path(donor).resolve()),
        "known":known,"candidate":candidate,"producer":code,"selected_state_hash":summary["selected_state"],
        "input_pins":pins,"rows":rows,"requests":count,"production_forward_budget":14*count,
        "production_gradient_calls":3*count,"reference_forwards":len(rows)*(1+2*len(BANDS)*2*len(EPSILONS)),
        "restore_forward_budget_each":10,"distance":"block diagonal; center vs corner relative to center; same mean to both",
        "target":"mean probability at both geometrically registered output locations; no target label",
        "epsilon_units":"training-minmax normalized input, unclipped","spectral_cap":CAP,
        "optimizer_updates":0,"diagnostic_model_forwards_at_registration":0,"query":0})


def load_scope(scope_pin, caller, rr, admission, code):
    scope = read(scope_pin)
    _,paths = verify_deployment(scope["deployment"],caller)
    known,candidate,summary,pins = enrollment(scope["donor_root"],rr,admission,code)
    m = read(summary["development"])
    rows = request_population(m["plan"],m["record"]["roles"],m["record"]["classes"],m["bands"]["band_ids"])
    require(scope["policy"]==POLICY and scope["producer"]==code and scope["input_pins"]==pins
            and scope["known"]==known and scope["candidate"]==candidate and scope["rows"]==rows
            and scope["requests"]==1404 and scope["spectral_cap"]==CAP
            and scope["selected_state_hash"]==summary["selected_state"]
            and scope["query"]==scope["optimizer_updates"]==scope["diagnostic_model_forwards_at_registration"]==0,
            "Scope differs from its frozen selected model and full development population")
    return scope,paths


def state_snapshot(run, rr):
    import numpy as np
    import torch
    from evaluation.linear_probe import fingerprint
    from transfer.registered_matched_route import snapshot
    np_rng = np.random.get_state()
    return {"registered":snapshot(run),"gradients":fingerprint([[p.grad for p in model.parameters()]
              for model in (run.model,run.source)]),
            "training_modes":[[m.training for m in model.modules()] for model in (run.model,run.source)],
            "numpy_global":rr.tensor_hash(torch.from_numpy(np_rng[1].copy())),"numpy_flags":[np_rng[0],*np_rng[2:]],
            "python_global":random.getstate(),
            "batch_norm_modules":sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
                                     for model in (run.model,run.source) for m in model.modules())}


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--producer",required=True,type=Path)
    p.add_argument("--scope",nargs=2,metavar=("PATH","SHA256"))
    return p


def cli_pin(pair):
    path,checksum = pair
    return {"path":str(Path(path).resolve(strict=True)),"sha256":checksum}
