"""Reload a support-fitted affine head without training labels or backbone changes."""

import re
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import json_pin, require
from evaluation.retained_spectral_evidence import ReleasedBlocks, fit_support
from inference.native_deployment import Runtime
from model.research_config import digest

SCHEMA = "frozen_support_ridge_head_v1"
VIEWS = ("full", "central_gap", "matched_dispersed")
WIDTH = 64
POLICY = dict(estimator="support_only_dual_ridge_fp64", ridge=1.0, std_floor=1e-10,
              decision_decimals=12, tie="lowest_class_id", feature="native_fused", width=WIDTH)


def fit(values, labels, ids):
    require(values.ndim==2 and values.shape==(len(labels), WIDTH) and labels.ndim==ids.ndim==1
            and len(labels)==len(ids)>0 and np.isfinite(values).all()
            and labels.dtype.kind in "iu" and ids.dtype.kind in "iu",
            "Expected finite native fused support features and integer point/class identities")
    pts = ReleasedBlocks("devtrain", int(np.max(labels)), layers=("fused",))
    pts.truth = labels.tolist()
    pts.identities = ids.tolist()
    pts.labels = {"fused": [values]}
    _, classes, intercept, fields, _ = fit_support(pts)
    require(classes.tolist() == list(range(1, pts.classes+1))
            and len(set(pts.identities)) == len(labels), "Expected a full class dictionary and unique frozen support")
    mean, scale, support, _, dual = fields["fused"]
    return dict(mean=mean.tolist(),scale=scale.tolist(),weight=(support.T @ dual).tolist(),
                intercept=intercept.tolist())


def load(root, pin, source):
    item = json_pin(root,pin)
    require(item.keys() == {"schema","policy","source","selections"}
            and item["schema"] == SCHEMA and item["policy"] == POLICY
            and item["source"] == source and bool(item["selections"]),
            "Foreign support-readout schema, fit or source binding")
    require(source["class_ids"] == list(range(1,len(source["class_ids"])+1))
            and len(source["class_ids"])>=2 and set(source["bands"])==set(VIEWS)
            and all(re.fullmatch("[0-9a-f]{64}",source[key]) for key in ("checkpoint","state","authority","fit_receipt")),
            "Invalid class axis, checkpoint or release identity")
    full = source["bands"]["full"]
    require(len(full)==len(set(full)) and all(v == [i for i in full if i in set(v)] and v
            for v in source["bands"].values())
            and len(source["bands"]["central_gap"])==len(source["bands"]["matched_dispersed"])<len(full),
            "Invalid retained spectral columns")
    shapes = dict(mean=(WIDTH,),scale=(WIDTH,),weight=(WIDTH,len(source["class_ids"])),
                  intercept=(len(source["class_ids"]),))
    for selection in item["selections"].values():
        require(selection.keys() == {"support_sha256","counts","heads"}
                and re.fullmatch("[0-9a-f]{64}",selection["support_sha256"])
                and len(selection["counts"])==len(source["class_ids"])
                and all(type(n)==int and n>0 for n in selection["counts"])
                and set(selection["heads"])==set(VIEWS),"Wrong released support or training-view roster")
        for record in selection["heads"].values():
            require(record.keys()==shapes.keys(),"Wrong affine tensor roster")
            for key,shape in shapes.items():
                value = np.asarray(record[key],dtype=np.float64)
                require(value.shape==shape and np.isfinite(value).all()
                        and (key!="scale" or (value>0).all()),"Invalid affine coefficients")
    return item


def scores(values, head):
    values = np.asarray(values,dtype=np.float64)
    require(values.ndim==2 and values.shape[1]==WIDTH and np.isfinite(values).all(),
            "Expected finite native fused feature rows")
    return ((values-head["mean"])/head["scale"]) @ head["weight"] + head["intercept"]


def torch_scores(values, head):
    mean,scale,weight,intercept = head
    return ((values.to(torch.float64)-mean)/scale) @ weight + intercept


def decisions(value):
    return np.round(value, POLICY["decision_decimals"]).argmax(1)+1


class SupportRuntime:
    """Attach a readout to the existing full path for one scoped request at a time."""

    def __init__(self, runtime, artifact, selection):
        self.runtime = runtime
        self.source = artifact["source"]
        self.heads = artifact["selections"][selection]["heads"]
        scene = self.source["scene"]
        require(isinstance(runtime,Runtime) and runtime.identity.get("full")==self.source["state"]
                and runtime.full.heads[scene].in_features==WIDTH
                and runtime.full.heads[scene].out_features==len(self.source["class_ids"]),
                "Support head requires its exact frozen native backbone and class axis")
        self.coefficients = {view:tuple(torch.tensor(head[k],device=runtime.device,dtype=torch.float64)
                        for k in ("mean","scale","weight","intercept")) for view,head in self.heads.items()}

    def run(self, request, mode):
        require(mode in ("native","fixed_full","same_view"),"Explicit fitted-head mode required")
        require(request.scene==self.source["scene"] and request.x.device==self.runtime.device
                and request.meta is not None and request.band in self.source["bands"]
                and list(request.meta.band_ids)==self.source["bands"][request.band],
                "Foreign scene, device or retained columns")
        native_head = self.runtime.full.heads[request.scene]
        observed = []

        def capture(_, args):
            observed.append(torch_scores(args[0],self.coefficients[fit_view]))

        fit_view = "full" if mode=="fixed_full" else request.band
        start = perf_counter()
        if mode=="native":
            result = self.runtime.run([request],"full")
        else:
            with native_head.register_forward_pre_hook(capture):
                result = self.runtime.run([request],"full")
            require(len(observed)==1 and observed[0].shape==result["predictions"][0][0].shape,
                    "Native fused input was captured more than once or had a foreign shape")
        self.runtime.sync()
        native = result["predictions"][0]
        focal = native[0].argmax(1).cpu().numpy()+1 if mode=="native" else decisions(observed[0].cpu().numpy())
        return dict(native=native,fused_decision=focal,elapsed_seconds=perf_counter()-start,
                    graph_ms=result["graph_ms"],counts=result["counts"])
