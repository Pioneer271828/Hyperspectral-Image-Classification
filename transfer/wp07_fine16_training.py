"""Private released-point trainer using the existing native update/recovery methods."""
from copy import deepcopy
from contextlib import contextmanager
from io import BytesIO
from itertools import product
import math
from time import perf_counter

import numpy as np
import torch

from model.research_config import digest
from research_runtime import tensor_hash
from transfer import native_released_training as synthetic
from transfer import wp07_graph_range as graph
from transfer.development_bank_training import _initial
from transfer.development_bank import _atomic_new
from transfer.g3_training import state_hash
from transfer.native_released_points import HEADS, LOSS, MODE, scores
from transfer.native_released_real import ReleasedBankValidation, RealReleasedRun, registered_case
from transfer.native_scale_fixture import Fixture, condition_for
from transfer.native_scale_observation import head_hashes
from transfer.native_scale_training import Run as NativeRun, baseline
from transfer.native_scale_training import evaluate_validation
from transfer import native_released_campaign as campaign
from transfer.xuzhou_multiscale_audit import _exact


FORMAT = "wp07_b_frozen_q_range_v1"


@contextmanager
def response_observer(run):
    """Observe actual nonlinear branch states; never change inputs or parameters."""
    graph.require(not torch.is_grad_enabled() and not run.model.training,
                  "Response diagnostics require eval and no_grad")
    rows, captured = [], {}
    backbone = run.model.backbone

    def graph_input(module, args):
        captured['graph_input'] = tensor_hash(args[0])

    def cnn_input(module, args):
        captured['cnn_input'] = tensor_hash(args[0])
        rows[-1]['cnn_input'] = captured['cnn_input']

    def states(module, args, output):
        entry = run._current_entry
        a, b = output.detach().split(output.shape[0]//2)
        q = entry['q'].to(device=a.device, dtype=a.dtype)
        a, b = torch.sparse.mm(q, a).double(), torch.sparse.mm(q, b).double()
        delta = a-b
        rows.append({'graph_input': captured['graph_input'],
            'near_state': tensor_hash(a), 'wide_state': tensor_hash(b),
            'cells': a.shape[0], 'channels': a.shape[1],
            'strict_nonzero_delta_cells': int((delta != 0).any(1).sum()),
            'rms_delta': float(delta.square().mean().sqrt()),
            'absmax_delta': float(delta.abs().max()),
            'relative_l2_delta': float(delta.norm()/a.norm().clamp_min(1e-12)),
            'cosine': float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp_min(1e-12)),
            'scope': 'actual learned two-layer branch states broadcast with frozen Q; fixed same-weight forward; excludes upstream dependency claims'})

    handles = [backbone.graph_stage1.register_forward_pre_hook(graph_input),
               backbone.graph_stage2.register_forward_hook(states),
               backbone.cnn_stage1.register_forward_pre_hook(cnn_input)]
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def key(role, event):
    return f'{role}/{event["tile"]}/{event["band"]}'


class BankProvider(ReleasedBankValidation):
    validate_history = RealReleasedRun.validate_history

    def __init__(self, root, receipt, seed):
        case = registered_case(receipt, f'{receipt["scene"]}/{seed}/1/off', engineering=False)
        super().__init__(root, receipt, case)
        self.synthetic = False
        self.graph_scale = 16

    def events(self):
        train = [dict(**r, step=i, factor=1, profile=None, condition="missing")
                 for i, r in enumerate(self.case["schedule"], 1)]
        return train, self.validation_events(1), self.case["events"]


class FixtureProvider:
    synthetic = True
    classes = 9
    graph_scale = 16
    scores = staticmethod(scores)

    def __init__(self):
        self.fixtures = {role: Fixture(9, role) for role in ("devtrain", "devval")}

    def baseline(self, seed):
        return baseline(9, seed)

    def events(self):
        train = [dict(step=i, tile=tile, factor=1, band="full", profile="a", condition="matched")
                 for i, tile in enumerate(self.fixtures["devtrain"].tiles, 1)]
        val = [dict(tile=tile, factor=1, band=band, profile="b", condition="matched")
               for tile, band in product(self.fixtures["devval"].tiles, ("full", "even_retained_positions"))]
        return train, val, [0, 3, 6]

    def materialize(self, role, event, producer):
        return synthetic.materialize(self.fixtures[role], event, producer)

    def selection_condition(self, actual, spec):
        return condition_for(actual, "matched")

    def validate_observation(self, spec, actual, labels, targets, event):
        graph.require(actual.receipt["role"] in self.fixtures and actual.receipt["synthetic_profile"] is not None,
                      "Fixture scope differs")

    def validate_history(self, spec, value):
        graph.require(all(r["released"]["points"] > 0 for r in value["history"]), "Fixture points missing")


def fixture_cache(provider, producer):
    model = graph.initialize(provider.baseline(64), 7385)
    entries = {}
    train, val, _ = provider.events()
    for role, events in (("devtrain", train), ("devval", val)):
        for event in events:
            actual, _, _ = provider.materialize(role, event, producer)
            condition = provider.selection_condition(actual, None)
            entry = graph.capture_geometry(model, actual, producer, role, provider.graph_scale, condition=condition)
            graph.compare_supports(entry["near"], entry["wide"], require_distinct=True)
            entries[key(role, event)] = entry
    return entries


def contract(provider, seed, arm, producer, numeric, cache_pin, authority):
    graph.require_checks()
    graph.require(seed in (64, 512, 2048) and arm in graph.ARMS, "Unknown range case")
    source = provider.baseline(seed)
    initial = graph.initialize(source, seed+7321)
    train, val, events = provider.events()
    return {"format": FORMAT, "producer": producer, "classes": provider.classes, "seed": seed,
            "arm": arm, "supervision": MODE, "objective": LOSS, "training": train, "validation": val,
            "steps": len(train), "prefix": events[1], "events": events, "numeric": deepcopy(numeric),
            "module_seed": seed+7321, "rng_seed": seed+15000, "learning_rate": 0.0001,
            "initialization": _initial(initial, source), "source_state": state_hash(source),
            "parameters": sum(p.numel() for p in initial.parameters()), "registered_graph_scale": provider.graph_scale,
            "cache": cache_pin, "authority": authority, "synthetic": provider.synthetic,
            "selection": "first_min_released_point_fused_nll", "cnn_dilation": 1,
            "graph_regularizers": "frozen_zero_with_original_zero_loss_weights"}


class RangeRun(NativeRun):
    """Reuse native start/update/evaluate/pack/restore without widening its public provider interface."""
    def __init__(self, spec, provider, entries, producer, execution):
        graph.require_checks()
        expected = contract(provider, spec["seed"], spec["arm"], producer, execution.numeric,
                            spec["cache"], spec["authority"])
        graph.require(spec == expected, "Frozen range registration, source, numerical protocol or budget differs")
        self._contract, self.producer, self.execution = deepcopy(spec), producer, execution
        self._released, self._entries = provider, entries
        self.device = execution.device
        self.model = graph.initialize(provider.baseline(spec["seed"]), spec["module_seed"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=spec["learning_rate"], foreach=False, fused=False)
        self._train, self._val = "devtrain", "devval"
        self._history, self._validation, self._selected = [], [], None
        self._fresh = tuple(spec["initialization"]["fresh"])
        self._capacity, self._region_action = None, None
        self._selection_override = None
        self.timing = dict(train_update_seconds=0., validation_seconds=0., transform_seconds=0.,
                           forward_seconds=0., graph_seconds=0.)

    def _forward(self, actual, role, intervention):
        entry = self._current_entry
        graph.require(graph.identity(actual) == entry["identity"], "Frozen graph is bound to a different input")
        if self._selection_override is not None:
            graph.require(not self.model.training and not torch.is_grad_enabled(),
                          "Selector interventions are read-only inference, never training")
        arm = self._selection_override or self._contract["arm"]
        with graph.frozen_support(self.model, entry, arm):
            output, trace, condition = super()._forward(actual, role, intervention)
        graph.require(trace["q"] == tensor_hash(entry["q"]) and trace["a0"] == tensor_hash(entry["near"]),
                      "Frozen Q or near weights changed")
        graph.require(trace["range_nnz"] == [entry["near"]._nnz(), entry["wide"]._nnz()], "Frozen ranges changed")
        trace["range_identity"] = {k: graph.sparse_digest(entry[k]) for k in ("q", "near", "wide")}
        trace["selection_override"] = self._selection_override
        return output, trace, condition

    @contextmanager
    def selector_intervention(self, arm):
        graph.require(arm in graph.ARMS and self._selection_override is None, "Invalid or nested selector intervention")
        self._selection_override = arm
        try:
            yield
        finally:
            self._selection_override = None

    def _materialize(self, role, event):
        result = super()._materialize(role, event)
        entry = self._entries[key(role, event)]
        graph.require(graph.identity(result[0]) == entry["identity"], "Input differs from geometry enumeration")
        self._current_entry = entry
        return result

    def restore(self, value, cursor):
        super().restore(value, cursor)
        for role, rows in (("devtrain", value["history"]),
                           ("devval", [r for v in value["validation"] for r in v["rows"]])):
            for row in rows:
                entry = self._entries[key(role, row)]
                graph.require(row["graph"]["range_identity"] == {k: graph.sparse_digest(entry[k]) for k in ("q", "near", "wide")},
                              "Checkpoint history changed its frozen graph")


def execute(spec, phase, path, provider, entries, producer, execution, prior=None):
    run = RangeRun(spec, provider, entries, producer, execution)
    start, end = {"full": (0, spec["steps"]), "prefix": (0, spec["prefix"]),
                  "resume": (spec["prefix"], spec["steps"]), "completed": (spec["steps"], spec["steps"])}[phase]
    graph.require(not path.exists() and (prior is not None) == (start > 0), "Phase predecessor or destination differs")
    if start:
        run.restore(prior, start)
    else:
        run.start()
    for event in spec["training"][start:end]:
        run.update(event)
    value = run.pack()
    if phase == "completed":
        graph.require(_exact(prior, value), "Completed range re-entry changed state")
    with BytesIO() as buffer:
        torch.save(value, buffer)
        _atomic_new(path, buffer.getvalue())
    return {"phase": phase, "updates": end-start, "state": state_hash(run.model),
            "selected": value["selected"]["step"], "timing": run.timing}


@torch.no_grad()
def audit(spec, state, provider, entries, producer, execution, *, cost=False):
    """Independently map released original pixels and compare every saved head."""
    run = RangeRun(spec, provider, entries, producer, execution)
    run.restore(state, spec["steps"])
    run.optimizer = None
    endpoints = {}
    for name, weights, historical in (("selected", state["selected"]["state"],
                                      min(state["validation"], key=lambda r: r["fused_nll"])),
                                     ("last", state["model"], state["validation"][-1])):
        run.model.load_state_dict(weights, strict=True)
        before = state_hash(run.model)
        scalars, points = {}, []

        def observe(event, actual, output, targets):
            y0, _, x0, _ = actual.lattice.box
            width = actual.lattice.scene_shape[1]
            factor, grid_width = actual.lattice.factor, actual.lattice.shape[1]
            for head, tensor in zip(HEADS, output[:3], strict=True):
                probabilities = tensor.detach().cpu().numpy()
                bucket = scalars.setdefault((head, event["band"]), {"cm": np.zeros((provider.classes, provider.classes), dtype=np.int64), "nll": []})
                for ordinal, (pixel, label) in enumerate(targets.points):
                    y, x = divmod(pixel, width)
                    cell = ((y-y0)//factor)*grid_width+(x-x0)//factor
                    probability = probabilities[cell].astype(float)
                    bucket["cm"][label-1, int(probability.argmax())] += 1
                    bucket["nll"].append(-math.log(float(np.clip(probability[label-1], 1e-8, 1))))
                    points.append({"tile": event["tile"], "released_ordinal": ordinal, "head": head,
                                   "band": event["band"], "class_id": label, "probabilities": probability.tolist()})

        value = evaluate_validation(run.model, spec["validation"], historical["step"],
            lambda event: run._materialize("devval", event), run._forward, released=provider,
            detailed=True, observe=observe)
        metrics = campaign.scoring(value, provider.classes, historical=historical)
        for (head, band), data in scalars.items():
            score = metrics[head][band]
            graph.require(data["cm"].tolist() == score["confusion_matrix"]
                          and abs(math.fsum(data["nll"])/len(data["nll"])-score["nll"]) <= 1e-12,
                          "Independent original-pixel counts or NLL differ")
        graph.require(state_hash(run.model) == before, "Inference changed the checkpoint")
        endpoints[name] = {"metrics": metrics, "validation": value, "original_point_predictions": points}
    costs = []
    if cost:
        run.model.load_state_dict(state["selected"]["state"], strict=True)
        # A deterministic unlabelled tile identity, independent of scores and arm.
        tile = min(e["tile"] for e in spec["validation"])
        for event in (e for e in spec["validation"] if e["tile"] == tile):
            actual, _, _ = run._materialize("devval", event)
            for _ in range(2):
                run._forward(actual, "devval", event["condition"])
            execution.sync()
            if execution.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(5):
                execution.sync()
                started = perf_counter()
                output, trace, _ = run._forward(actual, "devval", event["condition"])
                execution.sync()
                samples.append((perf_counter()-started)*1000)
            reference = next(r for r in endpoints["selected"]["validation"]["rows"]
                             if r["tile"] == tile and r["band"] == event["band"])
            graph.require(head_hashes(output, actual) == reference["heads"], "Timed inference changed prediction")
            costs.append({"tile": tile, "band": event["band"], "samples_ms": samples,
                          "warmups": 2, "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated() if execution.device.type == "cuda" else None,
                          "scope": "resident_model_fixed_host_view_native_prepare_transfer_frozen_QA_forward_and_diagnostics"})
    exposed = np.asarray([r["released"]["per_class"] for r in state["history"]]).sum(0).tolist()
    return {"selected_step": state["selected"]["step"], "endpoints": endpoints, "cost": costs,
            "training_exposures_per_class": exposed, "audit_optimizer_updates": 0,
            "fresh_updates": {name: sum(bool(event["fresh"][name]["changed"]) for event in state["history"])
                              for name in spec["initialization"]["fresh"]},
            "state_unchanged": True, "query_accessed": False, "raw_gt_loaded": False}
