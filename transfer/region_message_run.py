"""Registered training and support accounting for block-local WP08 messages."""

from contextlib import contextmanager
import math
from time import perf_counter

import torch

from inference.light_cnn import validate_source
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import _forward, state_hash
from transfer.multiscale_run import _measure_graph_build
from transfer.region_message_attention import (
    ATTENTION_MODES, TRACE_FIELDS, RegionMessageVCASAHGCN, profile_region_messages,
)


GATES = dict(alignment_mode="region", interaction_mode="gated", interaction_direction="both",
             interaction_strength=.1, interaction_hidden_dim=32, interaction_chunk_pixels=16384,
             graph_update_mode="none")
ATTENTION = dict(width=32, landmark_grid=(2, 2), top_k=3, query_chunk=16)
OUTER = "backbone.interaction."
WRAPPED_OUTER = OUTER + "exchange."
PROJECTIONS = ("query", "key", "value", "output")
DIRECTIONS = ("graph_to_cnn", "cnn_to_graph")


def initialize(baseline, trial, seed):
    """Copy common weights through a complete name map before strict loading."""
    validate_source(baseline)
    if trial == "off":
        return baseline
    if trial not in ("aligned", *ATTENTION_MODES):
        raise ValueError("Unknown registered message arm")
    config = baseline.export_config()
    kwargs = {**config["backbone_kwargs"], **GATES}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        aligned = MultiDatasetVCASAHGCN(config["class_counts"], **kwargs).eval()
    shared = baseline.state_dict()
    aligned_weights = aligned.state_dict()
    new_names = set(aligned_weights) - shared.keys()
    if not new_names or any(not name.startswith(OUTER) for name in new_names):
        raise ValueError("Only original outer exchange weights may extend the baseline")
    aligned.load_state_dict({**aligned_weights, **shared}, strict=True)
    model = aligned
    if trial != "aligned":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = RegionMessageVCASAHGCN(config["class_counts"], region_attention={"mode": trial, **ATTENTION},
                                           **kwargs).eval()
        names = {name: WRAPPED_OUTER + name.removeprefix(OUTER) if name.startswith(OUTER) else name
                 for name in aligned.state_dict()}
        weights = model.state_dict()
        mapped = {names[name]: value for name, value in aligned.state_dict().items()}
        expected = {f"{OUTER}{branch}.{projection}.weight" for branch in ("to_pixels", "to_regions")
                    for projection in PROJECTIONS}
        if len(mapped) != len(names) or set(weights) - mapped.keys() != expected:
            raise ValueError("Outer gate remapping or projection inventory differs")
        model.load_state_dict({**weights, **mapped}, strict=True)
        if any(not torch.equal(model.state_dict()[key], value) for key, value in mapped.items()):
            raise ValueError("Mapped original or outer exchange tensor changed")
    if any(not torch.equal(model.state_dict()[name], value) for name, value in shared.items()):
        raise ValueError("Common backbone/head initialization changed")
    model.training_config = dict(getattr(baseline, "training_config", {}))
    model.backbone.adapter.cache_attention = False
    return model


def _group(name):
    if name.startswith((OUTER + "to_pixels.", OUTER + "to_regions.")):
        return "message_projection"
    if name.startswith(OUTER):
        return "outer_exchange"
    return "shared"


def initialization_evidence(model):
    counts = dict.fromkeys(("shared", "outer_exchange", "message_projection"), 0)
    tensors = {name: {} for name in counts}
    for name, parameter in model.named_parameters():
        group = _group(name)
        counts[group] += parameter.numel()
        canonical = OUTER + name.removeprefix(WRAPPED_OUTER) if name.startswith(WRAPPED_OUTER) else name
        tensors[group][canonical] = dict(shape=list(parameter.shape), dtype=str(parameter.dtype),
                                         sha256=tensor_hash(parameter))
    return dict(counts=counts, fingerprints={name: digest(values) for name, values in tensors.items()},
                gate_map="backbone.interaction.* -> backbone.interaction.exchange.*"
                         if type(model) is RegionMessageVCASAHGCN else "identity",
                exchange_and_message={k: v for k, v in tensors.items() if k != "shared"})


def _require_graph(source):
    backbone = source.backbone
    if (backbone.graph_mode != "learned" or backbone.graph_assignment_mode != "learned"
            or backbone.graph_adjacency_mode != "learned" or backbone.learnable_graph.graph_offsets.shape[0] != 9
            or any(getattr(backbone, key) is not None for key in ("graph_router", "edge_updater",
                                                               "coverage_modulation", "region_residual"))):
        raise ValueError("Region messages require the original learned graph and separate controls")


@contextmanager
def _profile_forward(model, timings, phase, verified):
    """Observe the graph delivered to both original stages on this private model."""
    backbone = model.backbone
    patches = ("graph_stage1", "graph_stage2")
    if any(name in vars(backbone) for name in patches):
        raise ValueError("Graph stage observer is already bound")
    originals = {name: getattr(backbone, name) for name in patches}
    graph_times = timings.setdefault(phase + "_graph_build", [])
    message_times = {}
    current = []

    def capture(state):
        if current:
            raise ValueError("Prior graph was not consumed by both original stages")
        A = state.A0
        current.extend((A, A._version, A._nnz(), 0))

    def observe(number, apply):
        def checked(values, adjacency, dataset_name=None):
            if (len(current) != 4 or adjacency is not current[0] or adjacency._version != current[1]
                    or adjacency._nnz() != current[2] or current[3] != number - 1):
                raise ValueError("Message exchange changed same-forward graph support")
            output = apply(values, adjacency, dataset_name)
            if number == 1:
                current[3] = 1
            else:
                verified.append(current[2])
                current.clear()
            return output
        return checked

    exchange = backbone.interaction
    # The aligned control calls the frozen exchange directly.
    aligned = exchange is not None and type(model) is MultiDatasetVCASAHGCN
    if aligned:
        if "forward_aligned" in vars(exchange):
            raise ValueError("Aligned exchange observer is already bound")
        original_exchange = exchange.forward_aligned

        def timed_exchange(*args, **kwargs):
            started = perf_counter()
            output = original_exchange(*args, **kwargs)
            message_times.setdefault("aligned_exchange", []).append(perf_counter() - started)
            return output

        exchange.forward_aligned = timed_exchange
    for index, name in enumerate(patches, 1):
        setattr(backbone, name, observe(index, originals[name]))
    try:
        with _measure_graph_build(model, graph_times, capture), profile_region_messages(message_times):
            yield
        if current:
            raise ValueError("Incomplete same-forward graph support observation")
    finally:
        for name in patches:
            delattr(backbone, name)
        if aligned:
            del exchange.forward_aligned
        for name, values in message_times.items():
            timings.setdefault(phase + "_message_" + name, []).extend(values)


class RegionMessageRun(CurriculumRun):
    FORMAT = "vca-region-message-run-v1"
    FAILURE_FORMAT = "vca-region-message-failure-v1"

    def __init__(self, baseline, scene, record, *, trial, selector_seed, **kwargs):
        if type(selector_seed) is not int or not 0 <= selector_seed < 2**32:
            raise ValueError("Explicit initialization seed required")
        if any(kwargs.get(key) is not None for key in ("policy", "expected_policy_hash", "observation_policy",
                                                       "expected_observation_hash")):
            raise ValueError("Message trial excludes sample and observation curricula")
        kwargs.setdefault("policy", None)
        kwargs.setdefault("expected_policy_hash", None)
        self.trial = trial
        self.seed = selector_seed
        self.baseline = baseline
        self.baseline_hash = state_hash(baseline)
        source = initialize(baseline, trial, selector_seed)
        self.initialization = initialization_evidence(source)
        self.tracked = {name: p.numel() for name, p in source.named_parameters() if _group(name) != "shared"}
        self.timings = {"train_forward": [], "train_backward": [], "validation": []}
        kwargs.setdefault("expected_source_state", state_hash(source))
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        _require_graph(model)
        if self.trial == "off":
            return validate_source(model)
        backbone = model.backbone
        expected_type = MultiDatasetVCASAHGCN if self.trial == "aligned" else RegionMessageVCASAHGCN
        if (type(model) is not expected_type or any(m.training for m in model.modules())
                or backbone.adapter.cache_attention
                or any(t.device.type != "cpu" or t.dtype != torch.float32 or not torch.isfinite(t).all()
                       for t in model.state_dict().values())
                or any(getattr(backbone.interaction_config, key) != expected for key, expected in GATES.items())):
            raise ValueError("Expected sealed FP32 region exchange and fixed outer configuration")
        exchange = backbone.interaction if self.trial == "aligned" else backbone.interaction.exchange
        for direction in DIRECTIONS:
            outer = getattr(exchange, direction)
            if outer.strength != GATES["interaction_strength"] or outer.chunk_size != GATES["interaction_chunk_pixels"]:
                raise ValueError("Outer exchange runtime controls changed")
        if self.trial != "aligned":
            for branch in ("to_pixels", "to_regions"):
                module = getattr(backbone.interaction, branch)
                if (module.mode != self.trial or module.query.out_features != ATTENTION["width"]
                        or any(getattr(module, key) != ATTENTION[key] for key in ("landmark_grid", "top_k", "query_chunk"))):
                    raise ValueError("Message attention runtime controls changed")

    def _contract_extensions(self):
        return {"region_message": dict(arm=self.trial, seed=self.seed, initialization=self.initialization,
                                       source_baseline=self.baseline_hash, gsd=None, coverage=None,
                                       band_sha256=self.scene.bands.fingerprint, graph_support="original_A0")}

    def _check_extensions(self):
        if (state_hash(self.baseline) != self.baseline_hash
                or self.spec["region_message"] != self._contract_extensions()["region_message"]):
            raise ValueError("Registered message inputs or initialization changed")

    def _training_forward(self, tile, step, extra):
        started = perf_counter()
        verified = []
        with _profile_forward(self.model, self.timings, "train", verified):
            outputs = _forward(self.model, self.scene, tile, **extra)
        if verified != [outputs[3]["graph_nnz"]]:
            raise ValueError("Training forward omitted graph support proof")
        outputs[3]["same_support_nnz"] = verified[0]
        self.timings["train_forward"].append(perf_counter() - started)
        return outputs

    def _backward(self, loss):
        started = perf_counter()
        super()._backward(loss)
        self.timings["train_backward"].append(perf_counter() - started)

    def _evaluate(self, model):
        started = perf_counter()
        verified = []
        with _profile_forward(model, self.timings, "validation", verified):
            outputs = super()._evaluate(model)
        expected = sum(r["role"] == "devval" for r in self.record["roles"])
        if len(verified) != expected:
            raise ValueError("Validation omitted graph support proof")
        self.timings["validation"].append(perf_counter() - started)
        return outputs

    def _history_extensions(self, outputs):
        info = outputs[3]
        exchange = info.get("interaction", {})
        return {"message": dict(graph_nnz=info["same_support_nnz"], regions=info["token_count"],
                               assignment_nnz=info["assignment_nnz"], branches=exchange.get("region_message", {}),
                               gates={k: float(v) for k, v in exchange.items() if k.endswith(("_mean", "_std"))})}

    def _parameter_history(self, before):
        return {"message_parameters": {
            name: dict(gradient_l1=None if p.grad is None else float(p.grad.abs().sum()),
                       changed=int((before[name] != p.detach()).sum()))
            for name, p in self.model.named_parameters() if name in self.tracked}}

    def _history_extension_keys(self):
        return {"message", "message_parameters"}

    def _validate_history_extensions(self, row):
        message = row["message"]
        n = message["regions"]
        if (set(message) != {"graph_nnz", "regions", "assignment_nnz", "branches", "gates"}
                or any(type(message[k]) is not int for k in ("regions", "assignment_nnz", "graph_nnz"))
                or not 1 <= n <= message["assignment_nnz"] or not n <= message["graph_nnz"] <= n*n):
            raise ValueError("Invalid original graph trace")
        branches = message["branches"]
        gates = message["gates"]
        if (set(branches) != (set(DIRECTIONS) if self.trial in ATTENTION_MODES else set())
                or set(gates) != ({f"{d}_gate_{m}" for d in DIRECTIONS for m in ("mean", "std")}
                                  if self.trial != "off" else set())
                or any(type(v) is not float or not 0 <= v <= 1 for v in gates.values())):
            raise ValueError("Invalid message direction or gate trace")
        for trace in branches.values():
            count = trace["valid_regions"]
            landmarks = trace["landmarks"]
            k = min(ATTENTION["top_k"], count) if self.trial in ("route", "combined") else 0
            pairs = landmarks * count
            compressed = self.trial in ("compress", "combined")
            raw = {"local": count, "dense": count*count}.get(self.trial, k*count)
            if (set(trace) != set(TRACE_FIELDS) or any(type(v) is not int for v in trace.values())
                    or not 0 <= count <= n or not 0 <= landmarks <= min(count, 4)
                    or bool(landmarks) != bool(count and self.trial in ("compress", "route", "combined"))
                    or trace != dict(regions=n, valid_regions=count, landmarks=landmarks, k=k, raw_query_kv_pairs=raw,
                                     compressed_query_kv_pairs=pairs if compressed else 0, probe_pairs=pairs,
                                     compression_raw_value_pairs=pairs if compressed else 0, route_score_pairs=pairs if k else 0)):
                raise ValueError("Invalid registered message access counts")
        parameters = row["message_parameters"]
        if set(parameters) != set(self.tracked):
            raise ValueError("Invalid exchange parameter inventory")
        for name, evidence in parameters.items():
            g, changed = evidence["gradient_l1"], evidence["changed"]
            if (set(evidence) != {"gradient_l1", "changed"} or type(changed) is not int
                    or not 0 <= changed <= self.tracked[name] or (g is None and changed)
                    or (g is not None and (type(g) is not float or not math.isfinite(g) or g < 0))
                    or ((g is not None) != (name in row["gradient_parameters"]))):
                raise ValueError("Invalid exchange gradient or update evidence")

    @classmethod
    def load(cls, baseline, scene, record, path, *, plan, trial, selector_seed,
             expected_sha256, expected_contract_hash):
        return super().load(baseline, scene, record, path, plan=plan, policy=None, expected_sha256=expected_sha256,
                            expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(trial=trial, selector_seed=selector_seed))
