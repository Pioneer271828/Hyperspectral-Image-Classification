"""Native-grid execution with the frozen P4 arithmetic and explicit device boundaries."""

from collections import Counter
from contextlib import contextmanager
import os
import sys
from time import perf_counter

import numpy as np
import torch

from model.research_config import digest
from research_runtime import hash_file, tensor_hash
from transfer import p4_runtime_device as p4


def settings():
    return (torch.get_num_threads(), torch.get_float32_matmul_precision(),
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)


class Execution:
    def __init__(self, numeric, calls, inputs=None):
        self.numeric = numeric
        self.device = torch.device(numeric["device"])
        self.calls = calls
        self.inputs = inputs
        self.capture_values = False
        self.graphs = ()

    def sync(self):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)

    def partition(self, model, dataset, observation, wavelengths, scale, **kwargs):
        # Retain the original partition guards and prepare the exact HWC
        # model input on the host before explicitly timing its transfer.
        if (model.backbone.graph_mode!="learned" or
                (observation.partition!="train" and model.training)):
            raise ValueError("Native device execution requires an admitted learned-graph partition")
        if next(model.parameters()).device!=self.device:
            raise ValueError("Model does not reside on its registered device")
        start = perf_counter()
        host = torch.from_numpy(np.concatenate((observation.spectral,observation.coordinates),axis=-1))
        centers = torch.from_numpy(wavelengths)
        widths = kwargs.pop("fwhm",None)
        widths = None if widths is None else torch.from_numpy(widths)
        prepared = perf_counter()-start
        self.sync()
        start = perf_counter()
        value = host.to(self.device)
        centers = centers.to(self.device)
        widths = None if widths is None else widths.to(self.device)
        self.sync()
        transfer = perf_counter()-start
        start = perf_counter()
        output = model(dataset,value,wavelengths=centers,fwhm=widths,superpixel_scale=scale,
                       return_graph_info=True,**kwargs)
        self.sync()
        self.phase_times = dict(input_prepare_seconds=prepared,transfer_seconds=transfer,
                                network_with_graph_seconds=perf_counter()-start)
        return output

    @contextmanager
    def measure_graph(self, builder, timing, capture):
        if "build_graph" in vars(builder):
            raise ValueError("Native graph recorder already attached")
        original = builder.build_graph

        def build(*args,**kwargs):
            self.sync()
            start = perf_counter()
            state = original(*args,**kwargs)
            self.sync()
            timing.setdefault("graph_seconds",[]).append(perf_counter()-start)
            capture(state)
            return state

        builder.build_graph = build
        try:
            yield
        finally:
            del builder.build_graph
            timing.update(getattr(self,"phase_times",{}))

    def graph_trace(self, state):
        q,a = (graph.detach().coalesce() for graph in (state.Q,state.A0))
        if q.device!=self.device or a.device!=self.device:
            raise ValueError("Native graphs must remain resident on the registered device")
        if self.capture_values:
            self.graphs = tuple((graph.indices().cpu().numpy(),graph.values().cpu().numpy()) for graph in (q,a))
        return dict(q=tensor_hash(q),a0=tensor_hash(a),grid=list(state.grid_shape),
                    q_nnz=q._nnz(),a0_nnz=a._nnz(),q_shape=list(q.shape),
                    q_indices=tensor_hash(q.indices()),a0_indices=tensor_hash(a.indices()))

    def validate_history(self, spec, packet, cursor):
        if self.inputs is None:
            raise ValueError("Device training requires frozen input identities")
        for role, rows in (("devtrain",packet["history"]),
                           ("devval",[row for event in packet["validation"] for row in event["rows"]])):
            for row in rows:
                expected = self.inputs[input_key(spec["classes"],role,row)]
                if any(row[k]!=expected[k] for k in ("observation","labels","area")):
                    raise ValueError("Checkpoint native observation identity differs")
                if role=="devtrain" and row["histogram"]!=expected["histogram"]:
                    raise ValueError("Checkpoint supervision identity differs")

    def validate_observation(self, classes, role, event, actual, labels, targets):
        expected = self.inputs[input_key(classes,role,event)]
        if (expected["observation"]!=actual.receipt["observation"] or expected["labels"]!=labels
                or expected["histogram"]!=targets.summary()["histogram"] or expected["area"]!=int(targets.area.sum())):
            raise ValueError("Materialized native input differs from the frozen roster")


def input_key(classes,role,event):
    return f'{classes}/{role}/{event["tile"]}/{event["factor"]}/{event["band"]}/{event["profile"]}'


@contextmanager
def arithmetic(protocol, uuid=None, *, inputs=None):
    if protocol not in (p4.CPU,p4.CUDA_FP64_WEIGHT):
        raise ValueError("Native numerical execution requires a registered P4 protocol")
    before = settings()
    calls = Counter(adaptive_avg_pool2d=0,sparse_mm=0)
    try:
        with p4.arithmetic(protocol,uuid,kernel_calls=calls) as numeric:
            numeric = dict(numeric, numpy=np.__version__, python=sys.version.split()[0],
                           p4_arithmetic_sha256=hash_file(p4.__file__),
                           input_roster=None if inputs is None else digest(inputs))
            if uuid is not None and (not numeric["deterministic_algorithms"] or numeric["deterministic_warn_only"]):
                raise ValueError("CUDA execution requires strict deterministic algorithms")
            yield Execution(numeric,calls,inputs)
    finally:
        if settings()!=before:
            raise RuntimeError("P4 arithmetic did not restore the process settings")
