"""Bounded area observations on a native grid, with an explicit original-pixel projection."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from time import perf_counter

import numpy as np

from loadData.observation_sampling import ObservationView
from loadData.spatial_protocol import PartitionObservation, forward_partition
from model.coverage_config import coverage_forward_kwargs
from model.research_config import canonical_json, digest
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.development_bank_consumers import heads
from transfer.development_block_probe import _graph_trace
from transfer.development_blocks import array_hash
from transfer.native_support import _measure


FACTORS = (1, 2, 4)
EDGE_POLICY = "anchored_partial_cells_no_padding"
COORDINATES = "original_centers_at_integers_edges_at_half_integers_yx"


def _immutable(value):
    """The backing bytes are immutable, even if callers change NumPy flags."""
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class Lattice:
    box: tuple[int, int, int, int]
    scene_shape: tuple[int, int]
    factor: int
    cells: tuple[tuple[int, int, int, int], ...]
    shape: tuple[int, int]

    @classmethod
    def make(cls, box, scene_shape, factor, edge=EDGE_POLICY):
        box, scene_shape = tuple(box), tuple(scene_shape)
        if (type(factor) is not int or factor not in FACTORS or edge != EDGE_POLICY
                or len(box) != 4 or len(scene_shape) != 2
                or any(type(value) is not int for value in (*box, *scene_shape))):
            raise ValueError("Invalid native-grid geometry or edge policy")
        y0, y1, x0, x1 = box
        height, width = scene_shape
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("Native grid must stay inside the original block")
        cells = tuple((y, min(y+factor, y1), x, min(x+factor, x1))
                      for y in range(y0, y1, factor) for x in range(x0, x1, factor))
        shape = (math.ceil((y1-y0)/factor), math.ceil((x1-x0)/factor))
        return cls(box, scene_shape, factor, cells, shape)

    def areas(self):
        return np.array([(b-a)*(d-c) for a,b,c,d in self.cells], dtype=np.int64)

    def centers(self):
        return np.array([((a+b-1)/2, (c+d-1)/2) for a,b,c,d in self.cells], dtype=np.float64)

    def coordinates(self):
        denom = [max(size-1, 1) for size in self.scene_shape]
        return (self.centers()/denom).astype(np.float32).reshape((*self.shape, 2))

    def footprints(self, base_gsd_m=None):
        scale = 1.0 if base_gsd_m is None else base_gsd_m
        return [[(edge-0.5)*scale for edge in box] for box in self.cells]

    def membership(self):
        y0,y1,x0,x1 = self.box
        y, x = np.ogrid[:y1-y0, :x1-x0]
        return (y//self.factor)*self.shape[1] + x//self.factor

    def project(self, probabilities):
        """Copy coarse probabilities to each represented original pixel for scoring."""
        if probabilities.ndim != 2 or probabilities.shape[0] != len(self.cells):
            raise ValueError("Projection requires one row per native cell")
        return probabilities[self.membership().ravel()]


def reduce_area(values, lattice):
    y0,y1,x0,x1 = lattice.box
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 3
            or values.shape[:2] != (y1-y0, x1-x0) or not values.shape[-1] or not np.isfinite(values).all()):
        raise ValueError("Expected finite FP32 values on the entire original block")
    if lattice.factor == 1:
        return values
    y = tuple(range(0,y1-y0,lattice.factor))
    x = tuple(range(0,x1-x0,lattice.factor))
    total = np.add.reduceat(np.add.reduceat(values.astype(np.float64), y, axis=0), x, axis=1)
    return (total / lattice.areas().reshape((*lattice.shape,1))).astype(np.float32)


@dataclass(frozen=True)
class NativeObservation:
    dataset: str
    classes: int
    view: ObservationView
    lattice: Lattice
    _manifest: str

    @property
    def receipt(self):
        return json.loads(self._manifest)

    def verify(self, producer, role):
        receipt = self.receipt
        obs = self.view.observation
        if (receipt["producer"] != producer or receipt["role"] != role or obs.partition != receipt["partition"]
                or self.dataset != receipt["dataset"] or self.classes != receipt["classes"]
                or list(self.lattice.box) != receipt["box"] or list(self.lattice.scene_shape) != receipt["scene_shape"]
                or self.lattice.factor != receipt["factor"] or list(self.lattice.shape) != receipt["grid_shape"]
                or digest(self.lattice.cells) != receipt["original_boxes"]
                or self.view.bands.fingerprint != receipt["bands"] or obs.plan_hash != receipt["namespace"]
                or any(array_hash(array) != receipt[name] for name,array in (
                    ("spectral",obs.spectral), ("coords",obs.coordinates), ("local_ids",obs.global_ids)))):
            raise ValueError("Native observation differs from the admitted producer, role or grid")
        return receipt


def _materialize(dataset, classes, view, lattice, source, role, producer, *, profile=None):
    if not isinstance(producer,str) or not producer:
        raise ValueError("Native producer required")
    if profile is not None and (profile.unit != "m/pixel" or not math.isfinite(profile.base_gsd_m)
                                or profile.base_gsd_m <= 0):
        raise ValueError("Synthetic source GSD must be positive meters per original pixel")
    shape = lattice.shape
    namespace = digest(dict(schema="native_cell_yx_v1", dataset=dataset, source=source,
                            box=lattice.box, scene_shape=lattice.scene_shape, factor=lattice.factor,
                            role=role, edge=EDGE_POLICY))
    spectral = _immutable(reduce_area(view.observation.spectral,lattice))
    coords = _immutable(lattice.coordinates())
    # Local negative IDs are disjoint from original global IDs. The persistent
    # identity is the pair (namespace, local ID), including at factor 1.
    ids = _immutable((-1-np.arange(math.prod(shape),dtype=np.int64)).reshape(shape))
    native = PartitionObservation(view.observation.partition,spectral,coords,ids,namespace)
    gsd = None if profile is None else profile.base_gsd_m
    receipt = dict(dataset=dataset, classes=classes, producer=producer, source=source, role=role,
        partition=native.partition, box=lattice.box, scene_shape=lattice.scene_shape, factor=lattice.factor,
        namespace=namespace, id_rule="negative_local_ordinal_in_namespace", edge=EDGE_POLICY,
        coord_rule=COORDINATES, grid_shape=shape, pixels=math.prod(shape), original_pixels=int(lattice.areas().sum()),
        bands=view.bands.fingerprint, spectral=array_hash(spectral), coords=array_hash(coords),
        local_ids=array_hash(ids), area=array_hash(lattice.areas()), footprints=digest(lattice.footprints(gsd)),
        original_boxes=digest(lattice.cells), membership=array_hash(lattice.membership()),
        physical=dict(unit=None if gsd is None else "m/pixel", original_gsd_m=gsd,
                      nominal_gsd_m=None if gsd is None else gsd*lattice.factor,
                      footprint_unit="original_pixel" if gsd is None else "m",
                      partial_cells=int((lattice.areas()<lattice.factor**2).sum()),
                      edge_dimensions=[[b-a,d-c] for a,b,c,d in lattice.cells
                                       if b-a != lattice.factor or d-c != lattice.factor]),
        synthetic_profile=None if profile is None else profile.key)
    receipt["observation"] = digest(receipt)
    return NativeObservation(dataset,classes,replace(view,observation=native),lattice,canonical_json(receipt))


def from_bank(bank, role, tile, band_spec, factor, producer, *, edge=EDGE_POLICY, engineering=False):
    """Read one authorized real block. No physical or label override is accepted."""
    if role not in ("devtrain","devval") or not isinstance(producer,str) or not producer:
        raise ValueError("Bank role and producer required")
    request = deepcopy(band_spec)
    row = bank.role_row(role,tile)
    release = bank.release
    if (release.profile == "synthetic") != engineering:
        raise ValueError("Native bank source scope differs from its registered execution")
    lattice = Lattice.make(row["box"],release.shape[:2],factor,edge)
    scene, _, view, context = bank.materialize(row,request)
    return _materialize(scene.dataset,scene.classes,view,lattice,context,role,producer), row


def forward(model, actual, producer, role, *, condition=None, timed=None, execution=None):
    receipt = actual.verify(producer, role)
    if condition is not None:
        condition.validate(actual, receipt)
    elif receipt["synthetic_profile"] is not None:
        raise ValueError("Synthetic forward requires a frozen condition entry")
    if model.backbone.adapter.cache_attention:
        raise ValueError("Native observations require uncached spectral adaptation")
    builder = model.backbone.learnable_graph
    graphs, timing = [], {} if timed is None else timed
    bands = actual.view.bands
    kwargs = coverage_forward_kwargs(model,bands)
    if bands.fwhm is not None:
        kwargs["fwhm"] = np.asarray(bands.fwhm,dtype=np.float32)
        kwargs["fwhm_unit"] = bands.fwhm_unit
    args = (model,actual.dataset,actual.view.observation,np.asarray(bands.wavelengths,dtype=np.float32),
            16 if condition is not None else (50 if actual.dataset=="paviaU" else 100))
    started = perf_counter()
    partition = forward_partition if execution is None else execution.partition
    capture = (lambda state: graphs.append(dict(**_graph_trace(state), q_shape=list(state.Q.shape)))
               if execution is None else graphs.append(state))
    measured = (_measure(builder,"build_graph",timing,"graph_seconds",capture) if execution is None
                else execution.measure_graph(builder,timing,capture))
    with measured:
        if isinstance(model, ConditionalMultiscaleVCASAHGCN):
            metadata, coverage, gsd = (bands, None, None) if condition is None else condition.values()
            with model.partition_bands(metadata), model.physical_conditions(coverage,gsd):
                out = partition(*args,wavelength_unit=bands.wavelength_unit,**kwargs)
        else:
            out = partition(*args,wavelength_unit=bands.wavelength_unit,**kwargs)
    timing["forward_seconds"] = perf_counter()-started
    if execution is not None:
        graphs = [execution.graph_trace(state) for state in graphs]
    if len(graphs) != 1 or graphs[0]["q_shape"][0] != receipt["pixels"]:
        raise ValueError("Q must be built on the actual native pixels")
    cnn, graph = out[-1].get("multiscale_cnn"), out[-1].get("multiscale_graph")
    traces = {key:dict(scales=list(value[name]),weights=value["weights"].detach().tolist())
              for key,value,name in (("cnn",cnn,"dilations"),("graph",graph,"radii"))
              if value and "weights" in value}
    return out, dict(**graphs[0],weights=traces,range_nnz=None if graph is None else list(graph["nnz"]))


def head_hashes(output, actual):
    return heads(output,(actual,None,actual.view,None))
