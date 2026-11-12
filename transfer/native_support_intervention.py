"""Separate authenticated interventions for native second-stage sparse propagation."""

from copy import deepcopy
from time import perf_counter

import torch

from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.controlled_regional_support import _domain, geometry, invariants, match_support
from transfer.native_support import make_selector, native_forward


def _selector(entry, trial):
    return make_selector(entry, trial, trial["support_seed"])


def _origin(entry, trial, step):
    selector = _selector(entry, trial)
    selector.advance(step)
    return selector


def prepare_block(entry, box, trial, random_trial, step, seed, owner):
    started = perf_counter()
    original = _origin(entry, trial, step)
    snapshot = original.snapshot()
    allowed = [list(edge) for edge in sorted(original.local | set(map(tuple, snapshot["pool"])))]
    points, denominator = geometry(entry, box)
    context = entry["context"]
    match = match_support(snapshot["selected"], allowed, entry["mandatory"], points,
                          context["partitions"], context["node_ids"], seed=seed)
    measured = match.pop("seconds")
    random = _origin(entry, random_trial, step)
    body = dict(version=1, owner=owner, role=context["role"], block=context["block"], box=box, points=points,
                coordinate_denominator=denominator, reference=snapshot, old_random=random.snapshot(),
                allowed=allowed, match=match)
    return dict(**body, sha256=digest(body)), dict(search_seconds=measured, total_seconds=perf_counter() - started)


class BoundIntervention:
    """Supply the original value rule while retaining separate support provenance."""

    def __init__(self, entry, trial, random_trial, receipt, arm, owner, *, expected_sha256):
        if (arm not in ("reference", "matched", "old_random") or receipt["owner"] != owner
                or receipt["version"] != 1 or receipt["sha256"] != expected_sha256
                or digest({key: value for key, value in receipt.items() if key != "sha256"}) != expected_sha256):
            raise ValueError("Intervention receipt or registered owner differs")
        reference = _selector(entry, trial).authenticate(receipt["reference"])
        old = _selector(entry, random_trial).authenticate(receipt["old_random"])
        if old.step != reference.step or old.schedule.budget(old.step) != reference.schedule.budget(reference.step):
            raise ValueError("Reference and old random schedules differ")
        points, denominator = geometry(entry, receipt["box"])
        context = entry["context"]
        if (receipt["role"] != context["role"] or receipt["block"] != context["block"]
                or receipt["points"] != points or receipt["coordinate_denominator"] != denominator):
            raise ValueError("Intervention geometry or role differs")
        allowed = sorted(reference.local | set(map(tuple, reference.pool)))
        if receipt["allowed"] != [list(edge) for edge in allowed]:
            raise ValueError("Frozen candidate pool differs")
        match = receipt["match"]
        original, pool, required = _domain(reference.selected, receipt["allowed"], entry["mandatory"], points,
                                            context["partitions"], context["node_ids"])
        alternate = set(map(tuple, match["pairs"]))
        if (not required <= alternate <= pool or len(alternate) != len(match["pairs"])
                or invariants(alternate, points) != invariants(original, points)
                or digest(sorted(alternate)) != match["support_sha256"]
                or digest(sorted(original)) != match["reference_sha256"]
                or digest(allowed) != match["candidate_sha256"]
                or match["changed_edges"] != len(original - alternate)
                or (match["status"] == "MATCHED") != (alternate != original)):
            raise ValueError("Invalid constrained support")
        if arm == "matched" and match["status"] != "MATCHED":
            raise ValueError("Nontrivial strict matched support unavailable")
        self._origin = reference
        self._scorer = old if arm == "old_random" else reference
        self._pairs = sorted(alternate) if arm == "matched" else self._scorer.selected.copy()
        self.receipt, self.arm = deepcopy(receipt), arm
        self.identity, self.context, self.n, self.cost = reference.identity, reference.context, reference.n, self._scorer.cost

    def render_live(self, adjacency, features):
        # _render is the unchanged native value mixture/normalization. The
        # distinct bound receipt above authenticates all replacement indices.
        return self._scorer._render(adjacency, features, self._pairs, self._scorer.schedule.budget(self._scorer.step))

    def snapshot(self):
        # The underlying native trace sees the authenticated controller origin.
        return self._origin.snapshot()

    def trace(self):
        return dict(kind="fixed_regional_intervention_v1", arm=self.arm, receipt=self.receipt["sha256"],
                    origin_state=self.receipt["reference"]["state_sha256"], pairs_sha256=digest(self._pairs),
                    nnz=len(self._pairs))


def forward_intervention(model, scene, record, tile, bound, *, detail=False, timing=None):
    from transfer.g3_training import state_hash
    owner_state = bound.receipt["owner"].get("state")
    if owner_state is not None and state_hash(model) != owner_state:
        raise ValueError("Intervention endpoint weights differ from the frozen model")
    key = tile["id"]
    roles = record["roles"]
    if (type(key) is not int or not 0 <= key < len(roles) or scene.plan["tiles"][key] != tile
            or roles[key]["role"] != bound.context["role"] or str(key) != bound.context["block"]
            or roles[key]["role"] not in ("devtrain", "devval") or roles[key]["box"] != bound.receipt["box"]):
        raise ValueError("Intervention cannot access this development block or role")
    if tensor_hash(torch.from_numpy(scene.cube[tile_slice(tile)].copy())) != bound.identity["block_input"]:
        raise ValueError("Intervention block bytes differ from the frozen context")
    output, frame = native_forward(model, scene, tile, selector=bound, detail=detail, timing=timing)
    raw = frame["trace"]
    raw["origin_snapshot"] = raw.pop("snapshot")
    raw["intervention"] = bound.trace()
    if raw["stage2_nnz"] != raw["intervention"]["nnz"]:
        raise RuntimeError("Native sparse propagation changed support size")
    return output, frame
