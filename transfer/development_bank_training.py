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
from transfer.development_block_probe import VIEWS
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
    if bank.release.profile != "synthetic" or seed not in SEEDS or arm not in ARMS:
        raise ValueError("Only the preregistered synthetic bank exercise can update")
    coverage = bank.release.coverage()
    if any(v["authorized_blocks"] != v["publishable_blocks"] or min(v["publishable_per_class"]) == 0
           for v in coverage.values()):
        raise ValueError("A training contract requires a complete labeled engineering fixture")
    baseline = _baseline(bank.release, seed)
    model = initialize(baseline, arm, seed + 7321)
    return dict(kind=FORMAT, bank=bank.pin, scene=bank.release.scene, seed=seed, arm=arm,
                module_seed=seed + 7321, support_seed=seed + 19000, rng_seed=seed + 15000,
                budget=STEPS, prefix=PREFIX, validation_steps=list(EVENTS), exposure=_events(bank, seed),
                validation=[row["id"] for row in bank.role_rows("devval")], learning_rate=0.0001,
                initialization=_initial(model, baseline), real_updates=0)


class Run:
    def __init__(self, bank, registered):
        if contract(bank, registered["seed"], registered["arm"]) != registered:
            raise ValueError("Synthetic source, bank, seed, role, or exposure registration differs")
        self.bank, self.contract = bank, deepcopy(registered)
        self.baseline = _baseline(bank.release, registered["seed"])
        self.model = initialize(self.baseline, registered["arm"], registered["module_seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=registered["learning_rate"])
        self.selectors = {}
        self.history, self.validation = [], []
        self.selected = None
        self.captures = 0

    def _forward(self, row, name, step):
        materialized = self.bank.materialize(row, VIEWS[name])
        arm = self.contract["arm"]
        selector = None
        if arm == "native":
            key = row["id"], name
            if key not in self.selectors:
                source = initialize(self.baseline, "native", self.contract["module_seed"])
                self.selectors[key] = native_selector(source, materialized, row, state_hash(self.baseline), STEPS,
                                                      self.contract["support_seed"])
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
        for row in self.bank.role_rows("devval"):
            view, output, trace = self._forward(row, "full", 0)
            labels = _targets(row, view)
            indices = torch.where(labels > 0)[0]
            total -= float(output[0][indices, labels[indices] - 1].clamp_min(1e-8).log().sum())
            count += len(indices)
            rows.append(dict(tile=row["id"], context=digest(view[-1]), heads=heads(output, view),
                             support=trace, samples=len(indices)))
        score = total / count
        self.validation.append(dict(step=step, fused_nll=score, rows=rows))
        if self.selected is None or score < self.selected["fused_nll"]:
            self.selected = dict(step=step, fused_nll=score, state=_state(self.model))

    def update(self, event):
        row = self.bank.rows[event["tile"]]
        cursor = len(self.history)
        if row["role"] != "devtrain" or cursor >= STEPS or event != self.contract["exposure"][cursor]:
            raise ValueError("Training cursor must select the next registered train event")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        view, output, trace = self._forward(row, event["view"], event["step"])
        labels = _targets(row, view)
        present = labels > 0
        if not present.any():
            raise ValueError("Training needs a released sample in the materialized tile")
        mask = present[:, None].expand(-1, self.bank.release.classes).float()
        onehot = torch.nn.functional.one_hot((labels - 1).clamp_min(0), self.bank.release.classes).float() * mask
        losses = probability_loss_components(output, onehot, mask)
        if not torch.isfinite(losses["total"]):
            raise ValueError("Synthetic three-head loss became nonfinite")
        losses["total"].backward()
        self.optimizer.step()
        self.history.append(dict(**event, context=digest(view[-1]), support=trace, heads=heads(output, view),
            released_ids=[pixel for pixel, _ in row["samples"]], losses=detached_loss_values(losses)))
        if event["step"] in EVENTS:
            self._evaluate(event["step"])

    def _clock(self):
        return {f"{key}:{view}": item.snapshot() for (key, view), item in self.selectors.items()}

    def start(self):
        seed = self.contract["rng_seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self._evaluate(0)

    def pack(self):
        return dict(format=FORMAT, contract=digest(self.contract), cursor=len(self.history),
            model=_state(self.model), optimizer=self.optimizer.state_dict(), rng=_rng_state(CPU),
            history=self.history, validation=self.validation, selected=self.selected, native_clocks=self._clock())

    def restore(self, packet, cursor):
        if (packet["format"] != FORMAT or packet["contract"] != digest(self.contract)
                or packet["cursor"] != cursor or len(packet["history"]) != cursor
                or [row["step"] for row in packet["validation"]] != [i for i in EVENTS if i <= cursor]
                or any({key: row[key] for key in event} != event for row, event in
                       zip(packet["history"], self.contract["exposure"][:cursor], strict=True))):
            raise ValueError("Checkpoint cursor, contract or exposure differs")
        self.model.load_state_dict(packet["model"], strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        if not all(int(state["step"]) == cursor for state in self.optimizer.state.values()):
            raise ValueError("Optimizer cursor differs")
        self.history, self.validation, self.selected = packet["history"], packet["validation"], packet["selected"]
        selected = min(self.validation, key=lambda row: row["fused_nll"])
        if (self.selected["step"], self.selected["fused_nll"]) != (selected["step"], selected["fused_nll"]):
            raise ValueError("First minimum selection differs")
        expected = {}
        if self.contract["arm"] == "native":
            for key in self.contract["validation"]:
                expected[(key, "full")] = 0
            for row in self.history:
                expected[(row["tile"], row["view"])] = row["step"]
        if set(packet["native_clocks"]) != {f"{key}:{view}" for key, view in expected}:
            raise ValueError("Native clock roster differs")
        for (key, name), step in expected.items():
            row = self.bank.rows[key]
            view = self.bank.materialize(row, VIEWS[name])
            source = initialize(self.baseline, "native", self.contract["module_seed"])
            clock = native_selector(source, view, row, state_hash(self.baseline), STEPS, self.contract["support_seed"])
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
