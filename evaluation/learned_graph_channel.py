"""Paired binary recovery over independently randomized two-message channels."""
from copy import deepcopy
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evaluation.controlled_signal_learning import adjacency
from model.VCASAHGCN import SharedHGCN
from model.research_config import digest
from research_runtime import tensor_hash


SCHEMA = "wp06_synthetic_learned_channel_v1"
SEEDS = (64, 512, 2048)
ARMS = ("parallel", "bottleneck", "local", "shuffle")
ROLE_SEED = {"train": 19661, "devval": 44027, "confirmation": 90527}
CLASSES = (0, 1)
STRATA = ("reachable", "capacity", "beyond", "disconnected")
STEPS, CUT, BATCH, WIDTH, NODES = 96, 17, 12, 12, 9
EVENTS = (0, 16, 32, 48, 64, 80, 96)
EVAL_SIZE = 48
LR = .0125


def contract():
    return {"schema": SCHEMA, "scope": "synthetic_learning", "classes": CLASSES,
            "seeds": SEEDS, "arms": ARMS, "role_seeds": ROLE_SEED, "strata": STRATA,
            "steps_per_route": STEPS, "split_at": CUT, "batch_per_stratum": BATCH,
            "validation_instances_per_stratum": EVAL_SIZE, "confirmation_instances_per_stratum": EVAL_SIZE,
            "events": EVENTS, "learning_rate": LR, "hidden_width": WIDTH, "layers": 2,
            "selection": "first_min_devval_equal_stratum_binary_nll", "budget": len(SEEDS)*len(ARMS)*STEPS*2,
            "source_label": "queried one of two independent balanced sign signals",
            "pairing": "same source draws and frozen initial weights; one tie-resolved order per role",
            "torch_rng_seed": 77771,
            "source_and_context": "cpu_fp32_single_thread_original_shared_hgcn_frozen_bn_running_buffers",
            "negative_control": "fixed label-blind message derangement across examples after stage1",
            "real_updates": 0, "query_accessed": False, "raw_gt_loaded": False,
            "estimated_campaign_seconds": [20, 120]}


def edges(stratum, arm):
    common = {(n, n) for n in range(NODES)}
    if arm == "local":
        return sorted(common)
    if stratum == "beyond":
        return sorted(common | {(2, 0), (3, 2), (4, 3), (5, 4), (2, 1)})
    if stratum == "disconnected":
        return sorted(common | {(2, 0), (3, 1), (4, 3)})
    if stratum == "reachable":
        return sorted(common | {(2, 0), (5, 2)})
    # Two input signals, two hops and four edges in both arms. A single relay
    # averages both indistinguishable sources, erasing which sent each value.
    if arm == "bottleneck":
        return sorted(common | {(2, 0), (2, 1), (5, 2), (5, 3)})
    return sorted(common | {(2, 0), (3, 1), (5, 2), (5, 3)})


def instances(seed, role, stratum, index, n):
    if seed not in SEEDS or role not in ROLE_SEED or stratum not in STRATA:
        raise ValueError("Instance identity is outside the registered task")
    rng = np.random.default_rng(np.random.SeedSequence([ROLE_SEED[role], seed, STRATA.index(stratum), index]))
    # Exactly balanced joint sign combinations prevent finite-sample class imbalance.
    if n % 4:
        raise ValueError("Paired sign design requires a multiple of four")
    signs = np.tile(np.asarray(list(product((-1., 1.), repeat=2)), dtype=np.float32), (n//4, 1))
    signs = signs[rng.permutation(n)]
    ask = (np.zeros(n, dtype=int) if stratum == "reachable"
           else np.tile([0, 1], n//2)[rng.permutation(n)])
    y = (signs[np.arange(n), ask] > 0).astype(np.int64)
    context = rng.standard_normal((n, NODES, 2), dtype=np.float32)*.12
    # Sender contexts are identical, so the shared relay only observes a sum.
    context[:, :2] = 0
    x = np.zeros((n, NODES, 6), np.float32)
    x[:, :, 4:] = context
    x[:, 0, 0] = signs[:, 0]
    x[:, 1, 1] = signs[:, 1]
    x[np.arange(n), 5, 2+ask] = 1
    orders = np.stack([rng.permutation(NODES) for _ in range(n)])
    message_permutation = rng.permutation(n)
    # Rotate the randomly ordered batch, then map each original index to its donor.
    # No example donates to itself.
    donors = np.empty(n, dtype=np.int64)
    donors[np.roll(message_permutation, 1)] = message_permutation
    key = {"seed": seed, "role": role, "stratum": stratum, "index": index,
           "x": tensor_hash(x), "signs": tensor_hash(signs), "ask": tensor_hash(ask),
           "target": tensor_hash(y), "order": tensor_hash(orders), "donor": tensor_hash(donors)}
    return {"x": x, "signs": signs, "ask": ask, "y": y, "order": orders, "donors": donors, "fingerprint": digest(key)}


class ChannelModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.graph = SharedHGCN(6, WIDTH, WIDTH, dropout=0.).eval()
            self.readout = nn.Linear(WIDTH, 4)

    def forward(self, features, matrix, batch, stratum, arm, intervention="baseline", *, trace=None):
        n = len(features)
        ids = torch.as_tensor(batch["order"])
        inverse = ids.argsort(dim=1)
        batch_rows = torch.arange(n)[:, None]
        x = features[batch_rows, ids]
        adjacency_rows = matrix.indices()
        offsets = torch.arange(n)[:, None, None]*NODES
        ij = (inverse[:, adjacency_rows]+offsets).permute(1, 0, 2).reshape(2, -1)
        support = torch.sparse_coo_tensor(ij, matrix.values().repeat(n), (n*NODES, n*NODES)).coalesce()
        first = self.graph.forward_stage1(x.reshape(-1, 6), support).reshape(n, NODES, WIDTH)
        if intervention in ("relay_zero", "relay_replace") or arm == "shuffle":
            first = first.clone()
            indices = torch.arange(n)
            replace = torch.as_tensor(batch["donors"])
            # All disconnected/noise-only rows and the target's own self-message
            # remain unchanged. The registered shuffle changes only the relay.
            if arm == "shuffle":
                for original in ((2, 3) if stratum == "capacity" else (2,)):
                    relay = inverse[:, original]
                    first[indices, relay] = first[replace, relay[replace]]
            if intervention in ("relay_zero", "relay_replace"):
                original = np.where((stratum == "capacity") & (arm != "bottleneck"), batch["ask"]+2, 2)
                relay = inverse[indices, torch.as_tensor(original)]
                first[indices, relay] = (torch.zeros_like(first[indices, relay]) if intervention == "relay_zero"
                                         else first[indices, inverse[:, 6]])
        second = self.graph.forward_stage2(first.reshape(-1, WIDTH), support).reshape(n, NODES, WIDTH)
        if trace is not None:
            trace.extend((first.detach(), second.detach()))
        t = inverse[:, 5]
        scores = self.readout(second[torch.arange(n), t]).reshape(n, 2, 2)
        return scores[torch.arange(n), torch.as_tensor(batch["ask"])]


def arm_matrix(arm, stratum):
    pairs = edges(stratum, arm)
    # Same node features, layer widths and parameter counts for all supports.
    return adjacency(pairs, NODES)


def model_features(batch):
    x = torch.from_numpy(batch["x"].copy())
    # Two source values have the same feature code in every arm. Independent
    # relay type tags distinguish the two messages in the parallel graph.
    x[:, 1, 0] = x[:, 1, 1]
    x[:, 1, 1] = 0
    x[:, 2, 1] = 1
    x[:, 3, 1] = -1
    return x


def metrics(logits, labels):
    probabilities = logits.softmax(dim=1)
    chosen = logits.argmax(dim=1)
    labels = torch.as_tensor(labels)
    return {"count": len(labels), "correct": int((chosen == labels).sum()),
            "nll": float(F.cross_entropy(logits, labels)), "accuracy": float((chosen == labels).double().mean()),
            "probabilities": probabilities.detach().tolist(), "labels": labels.tolist()}


def scientific_hash(model):
    return digest({k: tensor_hash(v) for k, v in model.state_dict().items()})


class Run:
    def __init__(self, seed, arm):
        if seed not in SEEDS or arm not in ARMS:
            raise ValueError("Unregistered model seed or arm")
        self.seed, self.arm = seed, arm
        self.model = ChannelModel(seed).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR, foreach=False, fused=False)
        self.matrices = {key: arm_matrix(arm, key) for key in STRATA}
        self.initial = scientific_hash(self.model)
        self.history = []
        self.selected = None

    def batch(self, role, stratum, index=0):
        return instances(self.seed, role, stratum, index, BATCH if role == "train" else EVAL_SIZE)

    @torch.no_grad()
    def evaluate(self, role):
        if role not in ("devval", "confirmation"):
            raise ValueError("Unexpected evaluation stream")
        result = {}
        for key in STRATA:
            rows = self.batch(role, key)
            out = self.model(model_features(rows), self.matrices[key], rows, key, self.arm)
            result[key] = {**metrics(out, rows["y"]), "instances": rows["fingerprint"]}
        return result

    def event(self, step):
        validate = self.evaluate("devval")
        score = sum(v["nll"] for v in validate.values())/len(validate)
        if self.selected is None or score < self.selected["nll"]:
            self.selected = {"step": step, "nll": score, "weights": deepcopy(self.model.state_dict())}
        return {"step": step, "nll": score, "strata": validate}

    def advance(self, stop):
        if not len(self.history)-1 <= stop <= STEPS:
            raise ValueError("Update would cross the registered budget")
        if self.selected is None:
            self.history.append({"step": 0, "validation": self.event(0)})
        for step in range(len(self.history), stop+1):
            self.optimizer.zero_grad(set_to_none=True)
            receipts, losses = {}, []
            for key in STRATA:
                batch = self.batch("train", key, step)
                pred = self.model(model_features(batch), self.matrices[key], batch, key, self.arm)
                losses.append(F.cross_entropy(pred, torch.from_numpy(batch["y"])))
                receipts[key] = batch["fingerprint"]
            loss = torch.stack(losses).mean()
            loss.backward()
            grads = {name: float(value.grad.abs().sum()) for name, value in self.model.named_parameters()}
            if not torch.isfinite(loss) or not all(np.isfinite(val) for val in grads.values()):
                raise ValueError("Nonfinite paired learning")
            self.optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "instances": receipts, "gradients": grads}
            if step in EVENTS:
                row["validation"] = self.event(step)
            self.history.append(row)

    def pack(self, registration):
        return {"registration": registration, "seed": self.seed, "arm": self.arm, "initial": self.initial,
                "history": self.history, "cursor": len(self.history)-1, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "selected": self.selected, "rng": torch.get_rng_state()}

    def restore(self, packet, registration, cursor):
        history = packet["history"]
        events = [r["validation"] for r in history if "validation" in r]
        if ((packet["registration"], packet["seed"], packet["arm"], packet["initial"], packet["cursor"]) !=
                (registration, self.seed, self.arm, self.initial, cursor) or len(history) != cursor+1
                or [row["step"] for row in history] != list(range(cursor+1))
                or [row["step"] for row in events] != [s for s in EVENTS if s <= cursor]
                or any(record["instances"] != {k: self.batch("train", k, record["step"])["fingerprint"] for k in STRATA}
                       for record in history[1:])):
            raise ValueError("Checkpoint registration, event or sample identity differs")
        chosen = min(events, key=lambda row: row["nll"])
        if any(chosen[k] != packet["selected"][k] for k in ("step", "nll")):
            raise ValueError("Checkpoint selection differs from first minimum")
        if any(int(g["step"]) != cursor for g in packet["optimizer"]["state"].values()):
            raise ValueError("Adam cursor differs from the frozen schedule")
        self.model.load_state_dict(packet["model"], strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        self.history, self.selected = deepcopy(history), deepcopy(packet["selected"])
        torch.set_rng_state(packet["rng"])
