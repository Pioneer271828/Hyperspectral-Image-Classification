"""E0 boundary and narrow-region strata for separately authorized evaluation annotations."""

import re

import numpy as np

from evaluation.frozen_training_stages import json_pin, require
from evaluation.spatial_diagnostics import spatial_support
from evaluation.spatial_metrics import boundary_bands
from model.research_config import digest

SCHEMA = "wp02_released_spatial_annotations_v1"
STRATA = ("boundary","observed_interior","narrow")


def read_annotations(root,bank,roles,link):
    result = json_pin(root,link)
    require(set(result)=={"schema","bank","roles","scope","source","tiles"} and result["schema"]==SCHEMA
            and result["bank"]==bank.pin["sha256"] and result["roles"]==digest(roles)
            and result["scope"]==("synthetic" if bank.release.profile=="synthetic" else "external_real")
            and re.fullmatch("[0-9a-f]{64}",result["source"]) and bool(result["tiles"])
            and set(result["tiles"])<={str(tile) for tile in roles["evaluation"]},
            "Spatial annotations must bind the authorized evaluation role, Bank and source")
    return result


def classify(bank, roles, annotations, role, tile, points):
    require(role=="evaluation" and tile in roles[role] and annotations["bank"]==bank.pin["sha256"]
            and annotations["roles"]==digest(roles),"Spatial strata consume only authenticated evaluation blocks")
    datum = annotations["tiles"].get(str(tile))
    if datum is None:
        return None
    require(set(datum)=={"labels","geometry","narrow"},"Explicit bounded class geometry and narrow mask required")
    row = bank.role_row("devval",tile)
    y0,y1,x0,x1 = row["box"]
    shape = (y1-y0,x1-x0)
    labels,geometry,narrow = (np.asarray(datum[k]) for k in ("labels","geometry","narrow"))
    require(labels.shape==geometry.shape==narrow.shape==shape and np.issubdtype(labels.dtype,np.integer)
            and geometry.dtype==narrow.dtype==np.bool_ and not (narrow & ~geometry).any()
            and np.all(labels[~geometry]==0) and np.all((1<=labels[geometry])&(labels[geometry]<=bank.release.classes)),
            "Spatial annotation shape, valid class geometry or narrow marking is inconsistent")
    ids = np.asarray([p[0] for p in points],dtype=np.int64)
    fullwidth = bank.release.shape[1]
    y,x = ids//fullwidth-y0,ids%fullwidth-x0
    require(np.all((0<=y)&(y<shape[0])&(0<=x)&(x<shape[1]))
            and [(int(i),int(t)) for i,t in points]==[tuple(point) for point in row["samples"]],
            "Spatial classification points differ from the authenticated evaluation order")
    truth = np.asarray([v[1] for v in points],dtype=np.int64)
    require(np.all(labels[y,x][geometry[y,x]]==truth[geometry[y,x]]),
            "Annotation class IDs differ from published evaluation labels")
    scored = np.zeros(shape,dtype=bool)
    scored[y,x] = geometry[y,x]
    labels_in_geometry = spatial_support(labels,range(1,bank.release.classes+1),geometry_mask=geometry,
                                         score_mask=scored,widths=(1,))
    boundary = boundary_bands(labels,geometry,widths=(1,))[1]
    # A point on the unobserved side of a label hole has no certified four-neighbor interior.
    interior = np.zeros(shape,dtype=bool)
    interior[1:-1,1:-1] = (geometry[1:-1,1:-1]&geometry[:-2,1:-1]&geometry[2:,1:-1]&
                             geometry[1:-1,:-2]&geometry[1:-1,2:]&~boundary[1:-1,1:-1])
    selected = dict(boundary=boundary[y,x].tolist(),observed_interior=interior[y,x].tolist(),
                    narrow=narrow[y,x].tolist())
    require(not any(a and b for a,b in zip(selected["boundary"],selected["observed_interior"],strict=True)),
            "Boundary and known interior overlap")
    return dict(strata=selected,e0_support=labels_in_geometry,n=len(points),
                sources="released_class_geometry_and_holder_narrow_mask")


def score_stratum(truth,before,after,indices,classes,scene,*,threshold,real):
    from evaluation.frozen_training_stages import HEADS
    from evaluation.wp12_paired_head_review import pair_scores
    from evaluation.wp12_release_qualification import classification_status

    if not indices:
        return dict(n=0,heads=None,reason="no_authorized_points")
    require(len(set(indices))==len(indices) and all(0<=i<len(truth) for i in indices),
            "Malformed authorized spatial point set")
    labels = [truth[i] for i in indices]
    views = {head:pair_scores(labels,[before[head][i] for i in indices],[after[head][i] for i in indices],
                              classes,scene,head_kind=head,guard_population=threshold) for head in HEADS}
    return dict(n=len(indices),heads=views,qualification={head:classification_status(metrics,accuracy_allowed=True,
                cost_allowed=False,real=real) for head,metrics in views.items()})
