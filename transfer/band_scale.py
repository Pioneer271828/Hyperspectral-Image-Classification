"""Common released-point band, native lattice and propagation comparison contract."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch

from model.research_config import digest
from scripts.run_frozen_band_region import check
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN, DILATIONS
from transfer.development_blocks import array_hash
from transfer.multiscale_region_graph import GRAPH_RADII, MultiRangeBackbone
from transfer.native_scale_fixture import profiles
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_view

BANDS = ("full","central_gap")
FACTORS = (1,2)
POLICIES = ("native","cnn1","graph1","both1")
HEADS = ("fused","graph","cnn")
NAME = "wp05_wp06_wp07_native_band_scale_v1"
PROFILES = {p.key:p for p in profiles() if p.key in ("a","b")}


@dataclass(frozen=True)
class SelectorCondition:
    observation: str
    variant: str
    source_key: tuple
    bands: object
    coverage: float | None
    gsd_m: float | None

    def values(self):
        return self.bands,self.coverage,self.gsd_m

    def validate(self, actual, receipt):
        if self.observation != receipt["observation"] or self != condition(actual,self.variant):
            raise ValueError("Selector metadata belongs to a different native view or profile")


def condition(actual, variant):
    record = actual.receipt
    name = record["synthetic_profile"]
    if variant not in ("matched","missing","permuted") or name not in PROFILES:
        raise ValueError("Verified synthetic selector profile required")
    donor = ({"a":"b","b":"a"}[name] if variant=="permuted" else name)
    profile = PROFILES[donor]
    source = record["source"]
    bands = actual.view.bands
    coverage = len(source["positions"])/source["master_band_count"]
    if variant=="missing":
        bands,coverage,gsd = None,None,None
    else:
        shift = profile.wavelength_shift_nm-PROFILES[name].wavelength_shift_nm
        bands = replace(bands,wavelengths=tuple(w+shift for w in bands.wavelengths),
                        fwhm=(profile.fwhm_nm,)*len(bands.band_ids),fwhm_unit="nm")
        gsd = profile.base_gsd_m*record["factor"]
    return SelectorCondition(record["observation"],variant,(record["factor"],source["band_view"],donor),
                             bands,coverage,gsd)


def released_view(bank, row, code, band, factor, *, profile=None, labels=None):
    if band not in BANDS or factor not in FACTORS or row["role"] not in ("devval","devtrain"):
        raise ValueError("Only registered band and native scale views may be materialized")
    return bank_view(bank,row,code,band,factor=factor,profile=profile,synthetic_labels=labels)


def evidence(actual, targets):
    raw = actual.receipt
    source = raw["source"]
    return dict(view=raw["observation"], bank=source["bank"], band_ids=source["band_ids"],
                parent_stats=source["statistics"]["parent_statistics_hash"], retained_stats=source["retained_statistics"],
                band_order=raw["bands"], normalized=source["normalized"], spectral=raw["spectral"],
                role=raw["role"], factor=raw["factor"], native=raw["grid_shape"], footprint=raw["footprints"],
                physical=raw["physical"], released=targets.summary())


def can_cross(model):
    return (isinstance(model,ConditionalMultiscaleVCASAHGCN)
            and isinstance(model.backbone,MultiRangeBackbone)
            and getattr(model.backbone.CNN_Branch[0],"content_selector",None) is not None
            and model.backbone.range_content is not None)


@contextmanager
def fixed_mix(model, policy):
    """Lock a coefficient at one while executing the same original candidate bank."""
    if policy not in POLICIES or (policy!="native" and (model.training or not can_cross(model))):
        raise ValueError("Frozen selector cross requires the trained shared candidate bank")
    def narrow(_module,_inputs,logits):
        output = torch.full_like(logits,float("-inf"))
        output[...,0] = 0.
        return output
    with ExitStack() as stack:
        if policy in ("cnn1","both1"):
            first = model.backbone.CNN_Branch[0]
            stack.callback(first.content_selector.register_forward_hook(narrow).remove)
        if policy in ("graph1","both1"):
            stack.callback(model.backbone.range_content.register_forward_hook(narrow).remove)
        yield


def point_groups(actual, targets, state):
    """Summarize only authenticated released labels or unlabeled graph memberships."""
    source_width = actual.lattice.scene_shape[1]
    local_width = actual.lattice.shape[1]
    labels_by_pixel = dict(targets.points)
    per_cell = targets.histogram
    members = state.Q.coalesce()
    pairs, weights = members.indices().detach().cpu().numpy(), members.values().detach().cpu().numpy()
    hard_region = np.empty(len(actual.lattice.cells),dtype=np.int64)
    # COO ordering makes the lowest region the deterministic tie break.
    for cell in range(len(hard_region)):
        mask = pairs[0]==cell
        hard_region[cell] = pairs[1,mask][weights[mask].argmax()]
    purity = np.zeros((state.Q.shape[1],actual.classes),dtype=np.int64)
    np.add.at(purity,hard_region,per_cell)
    groups = []
    for (pixel,label),cell in zip(targets.points,targets.cells,strict=True):
        yy,xx = divmod(pixel,source_width)
        ids = [(yy+a)*source_width+xx+b for a,b in ((0,-1),(0,1),(-1,0),(1,0))
               if 0<=xx+b<source_width and 0<=yy+a<actual.lattice.scene_shape[0]]
        neighbors = [labels_by_pixel[i] for i in ids if i in labels_by_pixel]
        groups.append(dict(released_boundary=("known_neighbor_disagrees" if any(v!=label for v in neighbors)
                                              else "unknown_from_sparse_labels"),
                           complete_boundary=None, thin_structure=None,
                           cell_known_classes=int((per_cell[cell]>0).sum()),
                           region_known_classes=int((purity[hard_region[cell]]>0).sum()),
                           cell_known=int(per_cell[cell].sum()), cell_area=int(targets.area[cell]),
                           edge_cell=int(cell%local_width in (0,local_width-1)
                                         or cell//local_width in (0,actual.lattice.shape[0]-1))))
    return groups


@torch.no_grad()
def measure(model, actual, targets, code, *, policy="native", selector=None, groups=False):
    if model.training:
        raise ValueError("Frozen band-scale comparisons require eval mode")
    out, collected, times = None, [], {}
    builder = model.backbone.learnable_graph.build_graph
    def capture(*args,**kwargs):
        state = builder(*args,**kwargs)
        collected.append(state)
        return state
    start = perf_counter()
    with fixed_mix(model,policy):
        out,trace = forward(model,actual,code,actual.receipt["role"],condition=selector,
                            timed=times,graph_override=capture,
                            graph_scale=(50 if actual.dataset=="paviaU" else 16 if selector is not None else 100))
    check(len(collected)==1,"One actual shared Q required")
    traces = trace["weights"]
    first = model.backbone.CNN_Branch[0]
    kernel = tuple(first.depth_conv.kernel_size)
    dilation = traces.get("cnn",{}).get("scales",[1])
    lattice_factor = actual.lattice.factor
    footprint = {str(d):[lattice_factor*(1+(k-1)*d) for k in kernel] for d in dilation}
    result = dict(graph=trace, elapsed_seconds=perf_counter()-start,
                  forward_seconds=times["forward_seconds"], build_seconds=sum(times["graph_seconds"]),
                  cnn_kernel=list(kernel), cnn_dilations=dilation, cnn_pixel_span=footprint,
                  radii=traces.get("graph",{}).get("scales",[1]),
                  sparse_nnz=trace["range_nnz"] or [trace["a0_nnz"]],
                  parameters=sum(p.numel() for p in model.parameters()),
                  scores={head:tensor[targets.cells.tolist()].cpu().tolist() for head,tensor in zip(HEADS,out[:3],strict=True)})
    if groups:
        result["groups"] = point_groups(actual,targets,collected[0])
    return result


def validate_cross(rows):
    native = rows["native"]
    for key,row in rows.items():
        for field in ("q","a0","grid","q_nnz","a0_nnz"):
            check(row["graph"][field]==native["graph"][field],"Crossed selector changed input or shared Q/A")
        check(row["parameters"]==native["parameters"] and row["cnn_dilations"]==native["cnn_dilations"]
              and row["sparse_nnz"]==native["sparse_nnz"],"Frozen scale intervention changed arithmetic roster")
        if key in ("cnn1","both1"):
            check(row["graph"]["weights"]["cnn"]["weights"]==[[1.,0.,0.]],"Wrong forced CNN scale")
        if key in ("graph1","both1"):
            check(row["graph"]["weights"]["graph"]["weights"]==[1.,0.],"Wrong forced graph range")
    check(rows["cnn1"]["scores"]["graph"]==native["scores"]["graph"] and
          rows["graph1"]["scores"]["cnn"]==native["scores"]["cnn"],"Independent branch was changed")
