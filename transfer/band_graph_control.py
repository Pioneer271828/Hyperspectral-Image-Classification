"""One-view matched deletion on the existing sparse GraphState path."""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import torch

from evaluation.graph_edge_intervention import delete_support
from evaluation.matched_edge_controls import freeze, verify
from model.graph_state import GraphState
from model.research_config import digest
from research_runtime import tensor_hash

FIXED_ARMS = ("fixed_none", "important", "low_weight", "random:7719")
ARMS = ("native", *FIXED_ARMS)
BINS = [4, 12, 36]


def observation_id(actual):
    receipt = actual.receipt
    source = receipt["source"]
    return digest({name: receipt[name] for name in ("observation","spectral","coords","local_ids","namespace","role")}
                  | {name: source[name] for name in ("band_ids","positions","retained_statistics","bank")})


def row_coo(indices, weights, n):
    matrix = torch.sparse_coo_tensor(torch.as_tensor(indices,dtype=torch.long).T,
                                    torch.as_tensor(weights,dtype=torch.float32),(n,n)).coalesce()
    ix, values = matrix.indices(), matrix.values()
    mass = values.new_zeros(n).scatter_add_(0,ix[0],values)
    return torch.sparse_coo_tensor(ix,values/mass[ix[0]],matrix.shape).coalesce()


def controlled_state(actual, orientation):
    h,w = actual.lattice.shape
    symmetric = orientation == "vertical"
    if orientation not in ("vertical","horizontal") or (h,w) != ((32,12) if symmetric else (12,32)):
        raise ValueError("Unexpected guarded synthetic layout")
    yy,xx = np.mgrid[:h,:w]
    axis, span = (xx,yy) if symmetric else (yy,xx)
    groups = np.where(span < 4, axis//3, np.where(span >= 28, 4+axis//3, 8))
    byte_id = observation_id(actual)
    permutation = np.random.default_rng(int(byte_id[:12],16)).permutation(9)
    encoded = permutation[groups.ravel()]
    q = torch.sparse_coo_tensor(torch.as_tensor(np.stack((np.arange(h*w),encoded))),
                               torch.ones(h*w),(h*w,9)).coalesce()
    centers = np.stack([[yy[groups==i].mean(),xx[groups==i].mean()] for i in range(9)]).astype(np.float32)
    xy = np.empty_like(centers)
    xy[permutation] = centers
    edges, weights = [(int(i),int(i)) for i in permutation], [.6]*9
    for i in range(4):
        a,b,c=(int(permutation[j]) for j in (i,4+i,4+(i+1)%4))
        primary, secondary = ((a,b),(a,c))
        edges.extend((primary,secondary)); weights.extend((2.,.3))
        if symmetric:
            edges.extend(((b,a),(c,a))); weights.extend((2.,.3))
    a0=row_coo(edges,weights,9)
    state=GraphState(q,a0,torch.tensor(xy / np.array((h-1,w-1),dtype=np.float32)),(h,w),(1,9))
    state.validate()
    truth = dict(important=sorted(int(permutation[i])*9+int(permutation[4+i]) for i in range(4)),
                 isolated_self=int(permutation[8]), permutation=permutation.tolist())
    if symmetric:
        truth["important"] = sorted(truth["important"] + [(x%9)*9+x//9 for x in truth["important"]])
    return state, torch.tensor(xy), truth


@dataclass(frozen=True)
class ViewControl:
    identity: str
    state: GraphState
    coordinates: torch.Tensor
    plan: dict
    construction: float
    topology: dict

    @staticmethod
    def create(actual, orientation):
        started=perf_counter()
        state, coords, topology = controlled_state(actual, orientation)
        graph_time=perf_counter()-started
        plan = freeze(state.A0,coords,per_row=1,distance_bins=BINS,seeds=[7719],time_limit=3.,
                      symmetric=orientation=="vertical")
        verify(plan,state.A0,coords,expected_hash=plan["plan_hash"])
        if topology["important"] != plan["baseline"]["removed"]:
            raise ValueError("Matched baseline does not remove the registered correct remote partner")
        return ViewControl(observation_id(actual),state,coords,plan,graph_time,topology)

    @property
    def fingerprint(self):
        return digest(dict(view=self.identity,q=tensor_hash(self.state.Q),a=tensor_hash(self.state.A0),
                           plan=digest({k:v for k,v in self.plan.items() if k not in ("elapsed_seconds","plan_hash")})))

    def removal(self, arm):
        if arm == "fixed_none": return []
        if arm == "important": return self.plan["baseline"]["removed"]
        if arm not in self.plan["controls"]:
            raise ValueError("Unknown fixed graph arm")
        entry = self.plan["controls"][arm]
        if entry["status"] != "optimal":
            raise ValueError("Matched mask unavailable for this released view")
        return entry["removed"]

    def compatible(self, actual):
        if self.identity != observation_id(actual):
            raise ValueError("Frozen graph belongs to a different band, role, geometry or payload")

    def actual(self, actual, arm):
        self.compatible(actual)
        removed=self.removal(arm)
        result=replace(self.state,A0=delete_support(self.state.A0,removed))
        if result.A0._nnz() != self.state.A0._nnz()-len(removed):
            raise ValueError("Stored graph edge count differs from physical deletion")
        return result


def fixed_graph(observation, control, arm):
    """Supply the current observation's sealed GraphState to the native timer."""
    if arm == "native":
        return None
    state = control.actual(observation,arm)

    def provide(features, *args, **kwargs):
        graph = GraphState(state.Q.to(features.device), state.A0.to(features.device),
                           state.region_positions.to(features.device), state.image_shape,state.grid_shape)
        return graph.validate(features)

    return provide
