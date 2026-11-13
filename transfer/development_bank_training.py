"""A fixed-budget, synthetic-only continuation exercise over admitted bank views."""

from copy import deepcopy
import io
import random

import numpy as np
import torch

from joint_training import _restore_rng, _rng_state
from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from model.task_losses import detached_loss_values, probability_loss_components
from research_runtime import tensor_hash
from transfer.development_bank import BankRelease, _atomic_new, _row_key
from transfer.development_bank_consumers import ARMS, forward, heads, initialize, native_selector
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact


STEPS = 24
PREFIX = 7
EVENTS = (0, 4, 8, 12, 16, 20, 24)
SEEDS = (64, 512, 2048)
CPU = torch.device("cpu")
FORMAT = "wp02_bank_synthetic_continuation_v1"


def fixture(classes):
    scene = {9: "xuzhou", 22: "whu-honghu"}[classes]
    bands = 8 if classes == 9 else 11
    shape = (24, 66, bands)
    metadata = BandMetadata(f"bank-fixture:{classes}", tuple(range(bands)), tuple(np.linspace(450, 860, bands)),
                            "nm", "approximate", (True,) * bands)
    plan = digest(dict(kind="synthetic_bank_geometry", scene=scene, shape=shape))
    stats = dict(schema_version=1, fit_partition="train", plan_hash=plan, minimum=[-3.] * bands, maximum=[3.] * bands)
    stats["statistics_hash"] = digest(stats)
    rows, arrays = [], {}
    rng = np.random.default_rng(12973 + classes)
    for k, key in enumerate((3, 10, 20, 36, 48, 57, 81, 99)):
        role = "devtrain" if k < 6 else "devval"
        y, x = (2, 2 + k * 10) if k < 6 else (14, 2 + (k - 6) * 18)
        h, w = 8 - (k % 2), 8 - (k % 3)
        box = [y, y + h, x, x + w]
        points = [(int(y + n // w) * shape[1] + x + n % w, label)
                  for n, label in enumerate(c for c in range(1, classes + 1)
                   if (c - 1) % (6 if role == "devtrain" else 2) == (k if role == "devtrain" else k - 6))]
        arrays[key] = rng.normal(size=(h, w, bands)).astype(np.float32)
        rows.append(dict(id=key, role=role, partition="train" if k < 6 else "val", box=box,
            parent_tile=key, parent_box=box, samples=[list(p) for p in points], origin={
                "synthetic_seed": 12973 + classes, "array_sha256": tensor_hash(arrays[key])},
            representation="raw_fp32", blocked=None))
    authority = dict(minimum_buffer=2, construction="independent_9_or_22_class_engineering_fixture",
                     source_has_fresh_head=True)
    release = BankRelease(scene, "synthetic", shape, classes, plan, metadata, stats,
                          {"kind": "synthetic_per_case_initialization", "shared_state": digest(authority)},
                          authority, tuple(sorted(rows, key=_row_key)))
    return release, arrays


def _baseline(release, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MultiDatasetVCASAHGCN({release.scene: release.classes}, graph_mode="learned",
            norm_type="group_norm", latent_channels=128, adapter_embed_dim=64).eval()
    model.backbone.adapter.cache_attention = False
    return model


def _events(bank, seed):
    choices = [row["id"] for row in bank.role_rows("devtrain") if row["samples"]]
    rng = np.random.default_rng(seed)
    sequence = [int(i) for _ in range(STEPS // len(choices)) for i in rng.permutation(choices)]
    if len(sequence) != STEPS or len(choices) != 6:
        raise ValueError("Expected six synthetic supervised blocks at the fixed budget")
    return [dict(step=i, tile=key, view="full" if i % 2 else "even_retained_positions")
            for i, key in enumerate(sequence, 1)]


def _condition(name):
    if name == "full":
        return {"mode": "none"}
    if name == "even_retained_positions":
        return {"mode": "uniform_stride", "stride": 2, "offset": 0}
    raise ValueError("Unknown registered training view")


def _targets(row, materialized):
    _, _, observed, _ = materialized
    lookup = dict(row["samples"])
    return torch.tensor([lookup.get(int(pixel), 0) for pixel in observed.observation.global_ids.ravel()])


def _state(model):
    return {name: item.detach().cpu().clone() for name, item in model.state_dict().items()}


def _initial(model, baseline):
    original = baseline.state_dict()
    current = model.state_dict()
    if any(name not in current or not torch.equal(value, current[name]) for name, value in original.items()):
        raise ValueError("Synthetic variant changed shared initialization")
    return dict(source=state_hash(baseline), state=state_hash(model),
                fresh={name: tensor_hash(item) for name, item in current.items() if name not in original})


def contract(bank, seed, arm):
    release = bank.release
    if release.profile != "synthetic" or seed not in SEEDS or arm not in ARMS:
        raise ValueError("Only the preregistered synthetic bank exercise can update")
    coverage = release.coverage()
    if any(v["authorized_blocks"] != v["publishable_blocks"] or min(v["publishable_per_class"]) == 0
           for v in coverage.values()):
        raise ValueError("A training contract requires a complete labeled engineering fixture")
    baseline = _baseline(release, seed)
    model = initialize(baseline, arm, seed + 7321)
    return dict(kind=FORMAT, bank=bank.pin, scene=release.scene, seed=seed, arm=arm,
                module_seed=seed + 7321, support_seed=seed + 19000, rng_seed=seed + 15000,
                budget=STEPS, prefix=PREFIX, validation_steps=list(EVENTS), exposure=_events(bank, seed),
                validation=[row["id"] for row in bank.role_rows("devval")], learning_rate=0.0001,
                initialization=_initial(model, baseline), real_updates=0)


class Run:
    def __init__(self, bank, registered):
        if contract(bank, registered["seed"], registered["arm"]) != registered:
            raise ValueError("Synthetic source, bank, seed, role, or exposure registration differs")
        self._bank, self._contract = bank, deepcopy(registered)
        self.baseline = _baseline(bank.release, registered["seed"])
        self.model = initialize(self.baseline, registered["arm"], registered["module_seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=registered["learning_rate"])
        self.selectors = {}
        self._history, self._validation = [], []
        self._selected = None
        self.captures = 0

    @property
    def bank(self):
        return self._bank

    @property
    def contract(self):
        return deepcopy(self._contract)

    @property
    def validation(self):
        return deepcopy(self._validation)

    def _verify_input(self):
        self._bank.verify_authority(self._contract["bank"])

    def _forward(self, row, name, step):
        materialized = self._bank.materialize(row, _condition(name))
        arm = self._contract["arm"]
        selector = None
        if arm == "native":
            key = row["id"], name
            if key not in self.selectors:
                source = initialize(self.baseline, "native", self._contract["module_seed"])
                self.selectors[key] = native_selector(source, materialized, row, state_hash(self.baseline), STEPS,
                                                      self._contract["support_seed"])
                self.captures += 1
            selector = self.selectors[key]
            if row["role"] == "devtrain":
                selector.advance(step)
            elif selector.step != 0:
                raise ValueError("Native validation selector advanced")
        output, trace = forward(self.model, arm, materialized, selector)
        return materialized, output, trace

    @torch.no_grad()
    def _evaluate(self, step):
        self.model.eval()
        rows = []
        total, count = 0., 0
        for row in self._bank.role_rows("devval"):
            view, output, trace = self._forward(row, "full", 0)
            labels = _targets(row, view)
            indices = torch.where(labels > 0)[0]
            total -= float(output[0][indices, labels[indices] - 1].clamp_min(1e-8).log().sum())
            count += len(indices)
            rows.append(dict(tile=row["id"], context=digest(view[-1]), heads=heads(output, view),
                             support=trace, samples=len(indices)))
        score = total / count
        self._validation.append(dict(step=step, fused_nll=score, rows=rows))
        if self._selected is None or score < self._selected["fused_nll"]:
            self._selected = dict(step=step, fused_nll=score, state=_state(self.model))

    def update(self, event):
        self._verify_input()
        event = deepcopy(event)
        cursor = len(self._history)
        if cursor >= STEPS or event != self._contract["exposure"][cursor]:
            raise ValueError("Training cursor must select the next registered train event")
        row = self._bank.role_row("devtrain", event["tile"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        view, output, trace = self._forward(row, event["view"], event["step"])
        labels = _targets(row, view)
        present = labels > 0
        if not present.any():
            raise ValueError("Training needs a released sample in the materialized tile")
        mask = present[:, None].expand(-1, self._bank.release.classes).float()
        onehot = torch.nn.functional.one_hot((labels - 1).clamp_min(0), self._bank.release.classes).float() * mask
        losses = probability_loss_components(output, onehot, mask)
        if not torch.isfinite(losses["total"]):
            raise ValueError("Synthetic three-head loss became nonfinite")
        losses["total"].backward()
        self.optimizer.step()
        self._history.append(dict(**event, context=digest(view[-1]), support=trace, heads=heads(output, view),
            released_ids=[pixel for pixel, _ in row["samples"]], losses=detached_loss_values(losses)))
        if event["step"] in EVENTS:
            self._evaluate(event["step"])

    def _clock(self):
        return {f"{key}:{view}": item.snapshot() for (key, view), item in self.selectors.items()}

    def start(self):
        self._verify_input()
        if self._validation:
            raise ValueError("Training run has already started")
        seed = self._contract["rng_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._evaluate(0)

    def pack(self):
        return dict(format=FORMAT, contract=digest(self._contract), cursor=len(self._history),
            model=_state(self.model), optimizer=deepcopy(self.optimizer.state_dict()), rng=_rng_state(CPU),
            history=deepcopy(self._history), validation=deepcopy(self._validation), selected=deepcopy(self._selected), native_clocks=self._clock())

    def restore(self, packet, cursor):
        self._verify_input()
        packet = deepcopy(packet)
        if (packet["format"] != FORMAT or packet["contract"] != digest(self._contract)
                or packet["cursor"] != cursor or len(packet["history"]) != cursor
                or [row["step"] for row in packet["validation"]] != [i for i in EVENTS if i <= cursor]
                or any({key: row[key] for key in event} != event for row, event in
                       zip(packet["history"], self._contract["exposure"][:cursor], strict=True))):
            raise ValueError("Checkpoint cursor, contract or exposure differs")
        self.model.load_state_dict(packet["model"], strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        if not all(int(state["step"]) == cursor for state in self.optimizer.state.values()):
            raise ValueError("Optimizer cursor differs")
        self._history, self._validation, self._selected = packet["history"], packet["validation"], packet["selected"]
        selected = min(self._validation, key=lambda row: row["fused_nll"])
        if (self._selected["step"], self._selected["fused_nll"]) != (selected["step"], selected["fused_nll"]):
            raise ValueError("First minimum selection differs")
        expected = {}
        if self._contract["arm"] == "native":
            for key in self._contract["validation"]:
                expected[(key, "full")] = 0
            for row in self._history:
                expected[(row["tile"], row["view"])] = row["step"]
        if set(packet["native_clocks"]) != {f"{key}:{view}" for key, view in expected}:
            raise ValueError("Native clock roster differs")
        for (key, name), step in expected.items():
            row = self._bank.role_row("devval" if step == 0 else "devtrain", key)
            view = self._bank.materialize(row, _condition(name))
            source = initialize(self.baseline, "native", self._contract["module_seed"])
            clock = native_selector(source, view, row, state_hash(self.baseline), STEPS, self._contract["support_seed"])
            snapshot = packet["native_clocks"][f"{key}:{name}"]
            if snapshot["step"] != step:
                raise ValueError("Native clock advanced outside its registered visit")
            clock.restore(snapshot)
            self.selectors[key, name] = clock
        _restore_rng(packet["rng"], CPU)


def execute(bank, registered, stage, output, prior=None):
    run = Run(bank, registered)
    intervals = {"full": (0, STEPS), "prefix": (0, PREFIX), "resume": (PREFIX, STEPS), "completed": (STEPS, STEPS)}
    start, end = intervals[stage]
    if (start > 0) != (prior is not None) or output.exists():
        raise ValueError("Stage needs its prior cursor and unused output")
    if prior is None:
        run.start()
    else:
        run.restore(prior, start)
    for event in registered["exposure"][start:end]:
        run.update(event)
    packet = run.pack()
    if stage == "completed" and not _exact(packet, prior):
        raise ValueError("Completed synthetic consumer state changed")
    with io.BytesIO() as stream:
        torch.save(packet, stream)
        _atomic_new(output, stream.getvalue())
    return dict(stage=stage, updates=end - start, checkpoint=str(output), bank_reads=bank.reads,
                selected=packet["selected"]["step"], state=state_hash(run.model))
