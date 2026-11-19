"""Independent released-point spectral checks and simple support-only readouts."""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, require
from model.region_pixel_alignment import broadcast_regions
from transfer.native_scale_observation import forward


def column_positions(valid, name):
    locations = np.flatnonzero(np.asarray(valid,dtype=bool))
    count = len(locations)//2
    require(len(locations) >= 4, "Physical controls require four valid bands")
    if name == "full":
        return locations
    if name == "central_gap":
        start = (len(locations)-count)//2
        return np.concatenate((locations[:start],locations[start+count:]))
    if name == "matched_dispersed":
        return locations[np.linspace(0,len(locations)-1,len(locations)-count,dtype=int)]
    raise ValueError("Unknown frozen control view")


def labels_and_cells(row, shape):
    width = shape[1]
    y0,y1,x0,x1 = row["box"]
    ids = np.asarray([point[0] for point in row["samples"]],dtype=np.int64)
    y,x = np.divmod(ids,width)
    require(len(set(ids.tolist())) == len(ids) and ((y0 <= y)&(y < y1)&(x0 <= x)&(x < x1)).all(),
            "Released label coordinate is duplicate or leaves its tile")
    return (y-y0)*(x1-x0)+(x-x0), np.asarray([point[1] for point in row["samples"]],dtype=np.int64)


def normalize(raw, minimum, maximum):
    raw = np.asarray(raw,dtype=np.float32)
    low,high = (np.asarray(x,dtype=np.float32) for x in (minimum,maximum))
    scale = np.where(high > low, high-low, 1)
    return (raw-low)/scale, scale


def verify_loader(row, shape, full_raw, low, high, positions, actual, labels, targets, bands):
    independent_idx, truth = labels_and_cells(row,shape)
    given_ids = tuple(bands.band_ids[i] for i in positions)
    require(actual.view.bands.band_ids == given_ids
            and actual.receipt["source"]["positions"] == positions.tolist(),
            "Physical band order differs from independent indices")
    require(np.array_equal(targets.cells,independent_idx)
            and np.array_equal(labels,np.asarray(row["samples"],dtype=np.int64))
            and np.array_equal(actual.view.observation.global_ids.reshape(-1)[independent_idx],labels[:,0]),
            "Released supervision identities differ from the actual loader lattice")
    raw = full_raw[...,positions]
    low, high = np.asarray(low)[positions],np.asarray(high)[positions]
    expected, scale = normalize(raw,low,high)
    got = actual.view.observation.spectral
    np.testing.assert_allclose(got,expected,rtol=0,atol=2e-6,err_msg="Support normalization/column pairing differs")
    reconstructed = got*scale+np.asarray(low,dtype=np.float32)
    error = float(np.max(np.abs(reconstructed-raw)))
    require(error <= 3e-6 and np.isfinite(got).all(), "Production loader inverse band transform differs")
    return truth,independent_idx,raw,got,error


def mean_scores(truth, probabilities, classes):
    return label_scores(truth, probabilities.argmax(-1)+1, classes, probabilities=probabilities)


def label_scores(truth, predicted, classes, *, probabilities=None):
    truth,predicted = np.asarray(truth),np.asarray(predicted)
    require(truth.ndim == predicted.ndim == 1 and len(truth) == len(predicted) and len(truth)
            and min(truth.min(),predicted.min()) >= 1 and max(truth.max(),predicted.max()) <= classes,
            "Expected aligned foreground class decisions")
    matrix = np.zeros((classes,classes),dtype=np.int64)
    np.add.at(matrix,(truth-1,predicted-1),1)
    support = matrix.sum(1)
    recall = [float(matrix[i,i]/n) if n else None for i,n in enumerate(support)]
    result = dict(confusion=matrix.tolist(), support=support.tolist(), recall=recall,
                  oa=float(matrix.trace()/len(truth)), aa=float(np.mean([x for x in recall if x is not None])))
    if probabilities is not None:
        p = np.asarray(probabilities,dtype=np.float64)
        require(p.shape == (len(truth),classes) and np.isfinite(p).all()
                and (p >= 0).all() and np.allclose(p.sum(1),1,rtol=0,atol=2e-6), "Invalid class probabilities")
        result["nll"] = float(-np.log(np.maximum(p[np.arange(len(truth)),truth-1],1e-8)).mean())
    return result


def closest(x, centers):
    return ((x[:,None,:]-centers[None,:,:])**2).sum(-1).argmin(1)+1


def support_references(x, y, X, classes):
    """Train centroids and an L2-regularized linear readout on support alone."""
    require(x.ndim == X.ndim == 2 and x.shape[1] == X.shape[1] and (np.bincount(y,minlength=classes+1)[1:] > 0).all(),
            "Support reference requires every class and one aligned feature axis")
    x,X = x.astype("float64"),X.astype("float64")
    centers = np.stack([x[y == label].mean(0) for label in range(1,classes+1)])
    mean = x.mean(0)
    scale = np.maximum(x.std(0),1e-5)
    features = (x-mean)/scale
    gram = features@features.T
    targets = np.eye(classes)[y-1]
    prior = targets.mean(0)
    dual = np.linalg.solve(gram+1.0*np.eye(len(y)),targets-prior)
    linear = (((X-mean)/scale)@features.T@dual+prior).argmax(1)+1
    return {"centroid":closest(X,centers),"ridge":linear}


def native_signatures(model, actual, producer, role, *, execution=None, graph_scale=100, flat_coordinates=None):
    """Capture aligned representations at the existing native forward boundary."""
    backbone = model.backbone
    features = {}
    original = backbone.forward_stages

    def receive_vca(module, args, output):
        features["vca"] = (output[0] if isinstance(output,tuple) else output).detach()

    def capture(cnn_input, region_input, q, a, *args, **kwargs):
        pixels = cnn_input.reshape(-1,cnn_input.shape[-1])
        features["pre_pool"] = pixels.detach()
        features["pooled"] = broadcast_regions(q,region_input).detach()
        features["regions"] = int(q.shape[1])
        features["pool_variance_ratio"] = float(features["pooled"].var(0,unbiased=False).sum()/
                                                  torch.clamp(pixels.var(0,unbiased=False).sum(),min=1e-12))
        cnn,graph,extra = original(cnn_input,region_input,q,a,*args,**kwargs)
        features["cnn"], features["graph"] = cnn.detach(), graph.detach()
        return cnn,graph,extra

    with torch.inference_mode(), ExitStack() as stack:
        handle = backbone.adapter.register_forward_hook(receive_vca)
        stack.callback(handle.remove)
        stack.enter_context(patch.object(backbone,"forward_stages",new=capture))
        if flat_coordinates is not None:
            split = backbone._split_spectral_extra

            def translated(x, wavelengths):
                spectral, coordinates = split(x,wavelengths)
                moved = coordinates.clone()
                moved[...,0] -= flat_coordinates
                return spectral,moved

            stack.enter_context(patch.object(backbone,"_split_spectral_extra",new=translated))
        output, graph = forward(model,actual,producer,role,execution=execution,graph_scale=graph_scale)
    for name,value in list(features.items()):
        if torch.is_tensor(value):
            features[name] = value.reshape(-1,value.shape[-1]).cpu().numpy()
    require(set(features) == {"vca","pre_pool","pooled","regions","pool_variance_ratio","cnn","graph"},
            "Native stage capture missed an authenticated representation")
    return features, {head:value.detach().cpu().numpy() for head,value in zip(HEADS,output[:3],strict=True)}, graph
