"""Complete and sparse supervision on the native area lattice."""

from dataclasses import dataclass

import numpy as np
import torch

from model.task_losses import probability_loss_components
from transfer.development_blocks import array_hash, sample_identity
from transfer.native_scale_observation import _immutable


PURITY_NUMERATOR = 3
PURITY_DENOMINATOR = 4
MODES = ("mixture", "high_purity")


@dataclass(frozen=True)
class Targets:
    histogram: np.ndarray
    area: np.ndarray
    complete: bool

    def __post_init__(self):
        hist, area = self.histogram, self.area
        if (not isinstance(hist,np.ndarray) or not isinstance(area,np.ndarray)
                or hist.dtype!=np.int64 or area.dtype!=np.int64 or hist.ndim!=2 or not hist.size
                or area.shape!=(len(hist),) or (hist<0).any() or (area<=0).any()
                or (hist.sum(axis=1)>area).any() or (self.complete and not np.array_equal(hist.sum(axis=1),area))):
            raise ValueError("Native label counts must fit the admitted original-pixel area")
        object.__setattr__(self,"histogram",_immutable(hist))
        object.__setattr__(self,"area",_immutable(area))

    def summary(self):
        hist = self.histogram
        known = hist.sum(axis=1)
        dominant = hist.max(axis=1)
        eligible = (known==self.area) & (PURITY_DENOMINATOR*dominant >= PURITY_NUMERATOR*self.area)
        present = (hist>0).sum(axis=1)
        winners = hist.argmax(axis=1)
        return dict(known_per_class=hist.sum(axis=0).tolist(),
                    class_cells=(hist>0).sum(axis=0).tolist(),
                    known_conflicting_cells=int((present>1).sum()),
                    conflicting_points_per_class=hist[present>1].sum(axis=0).tolist(),
                    unknown=int((self.area-known).sum()),
                    known_count_histogram=np.bincount(known).tolist(),
                    purity_available=self.complete,
                    purity_lower_bound_numerator=dominant.tolist(),
                    purity_exact=None if not self.complete else (dominant/self.area).tolist(),
                    high_purity_cells_per_class=None if not self.complete else
                        np.bincount(winners[eligible],minlength=hist.shape[1]).tolist(),
                    high_purity_represented_pixels=None if not self.complete else
                        hist[eligible].sum(axis=0).tolist(),
                    valid_hard_area=None if not self.complete else int(self.area[eligible].sum()),
                    area=int(self.area.sum()), histogram=array_hash(hist),
                    trainable=self.complete)

    def weights(self, mode):
        if not self.complete or mode not in MODES:
            raise ValueError("Training requires complete known synthetic supervision and a registered mode")
        if mode=="mixture":
            p = self.histogram/self.area[:,None]
            valid = np.ones(len(self.area),dtype=bool)
        else:
            winner = self.histogram.argmax(axis=1)
            valid = PURITY_DENOMINATOR*self.histogram.max(axis=1) >= PURITY_NUMERATOR*self.area
            p = np.eye(self.histogram.shape[1],dtype=np.float32)[winner]*valid[:,None]
        if not valid.any():
            raise ValueError("No eligible native cells for supervised update")
        weights = (self.area*valid).astype(np.float32)
        return torch.tensor(p,dtype=torch.float32),torch.tensor(
            np.broadcast_to(weights[:,None],p.shape).copy(),dtype=torch.float32)


def complete_targets(actual, labels):
    """Map a complete fixture label raster. Real observations are never admitted here."""
    if actual.receipt["synthetic_profile"] is None:
        raise ValueError("Complete-label native training is synthetic-only")
    classes = actual.classes
    cell = actual.lattice.membership()
    if (not isinstance(labels,np.ndarray) or labels.shape!=cell.shape or labels.dtype!=np.int64
            or (labels<1).any() or (labels>classes).any()
            or array_hash(labels)!=actual.receipt["source"]["labels"]):
        raise ValueError("Fixture labels differ from the frozen complete raster")
    hist = np.bincount((cell*classes+labels-1).ravel(),minlength=len(actual.lattice.cells)*classes)
    return Targets(hist.reshape(-1,classes),actual.lattice.areas(),True)


def sparse_targets(actual, row):
    """Map released points only. Sparse provenance never conveys training authority."""
    source = actual.receipt["source"]
    if (actual.receipt["synthetic_profile"] is not None
            or (row["id"],row["role"],row["box"])!=(source["tile"],source["role"],source["box"])
            or sample_identity(row,(*actual.lattice.scene_shape,0),actual.classes)!=source["samples"]):
        raise ValueError("Released samples differ from the admitted native source")
    y0,_,x0,_ = actual.lattice.box
    width = actual.lattice.scene_shape[1]
    members = actual.lattice.membership()
    hist = np.zeros((len(actual.lattice.cells),actual.classes),dtype=np.int64)
    for pixel, label in row["samples"]:
        y,x = divmod(pixel,width)
        hist[members[y-y0,x-x0],label-1] += 1
    return Targets(hist,actual.lattice.areas(),False)


def loss(outputs, target, mask):
    return probability_loss_components(outputs,target.to(outputs[0]),mask.to(outputs[0]))


def metrics(output, actual, labels, targets, *, detailed):
    """Area-weighted NLL and projected original-pixel classification metrics."""
    h = targets.histogram
    count = int(h.sum())
    valid = PURITY_DENOMINATOR*h.max(axis=1) >= PURITY_NUMERATOR*targets.area
    result = {}
    for name,p in zip(("fused","graph","cnn"),output[:3],strict=True):
        probs = p.detach().numpy()
        nll = float(-(h*np.log(np.clip(probs.astype(np.float64),1e-8,1.0))).sum())
        result[name] = dict(nll_sum=nll,count=count)
        if detailed:
            classes = actual.classes
            projected = actual.lattice.project(probs).argmax(axis=1)
            confusion = np.bincount((labels.ravel()-1)*classes+projected,minlength=classes**2)
            confusion = confusion.reshape(classes,classes)
            selected = probs.argmax(axis=1)[valid]
            winner = h.argmax(axis=1)[valid]
            coarse = np.bincount(winner*classes+selected,minlength=classes**2).reshape(classes,classes)
            result[name].update(projected_confusion=confusion.tolist(), coarse_hard_confusion=coarse.tolist(),
                                coarse_hard_count=int(valid.sum()), coarse_hard_correct=int((selected==winner).sum()))
    return result
