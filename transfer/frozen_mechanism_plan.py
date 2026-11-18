"""Pre-score evaluation windows and paired population identities."""

from collections import defaultdict
import hashlib
import hmac

import numpy as np

from model.research_config import digest


POLICY = "wp05_wp06_frozen_mechanism_v1"
FOCUS = {"xuzhou": (5, 6, 8, 3), "paviaU": (2, 6), "whu-honghu": (15, 21)}
EXPECTED_CONFUSIONS = {
    "xuzhou": {5: (8,), 6: (3,), 8: (5,), 3: (6,)},
    "paviaU": {2: (6,), 6: (2,)},
    "whu-honghu": {15: (18, 10, 12, 1), 21: (18, 10, 16, 12)},
}
RANDOM_SEEDS = (7719, 7720)
REGION_SEED = 8137
ARCHITECTURES = {"xuzhou": ("off", "cnn_content", "graph_average"),
                 "paviaU": ("64", "512", "2048")}
NATIVE_CELLS = ("clean", "target", "matched", "random")
FACTORIAL_CELLS = ("clean_target_graph", "target_clean_graph",
                   "clean_full_graph", "target_full_graph")
ALL_CELLS = (*NATIVE_CELLS, *FACTORIAL_CELLS)
FULL_INFORMATION = ("clean_full_graph", "target_full_graph")


def windows(count):
    """Return every declared position, including mirror and grid controls."""
    if type(count) is not int or count < 16:
        raise ValueError("A complete spectrum of at least 16 bands is required")
    out = []
    for width in (count // 4, count // 2):
        edge = count - width
        starts = defaultdict(set)
        for start in range(0, edge + 1, width // 2):
            starts[start].add("sliding")
            starts[edge - start].add("mirror")
        for start in range(0, edge + 1, width):
            starts[start].add("disjoint")
        for start in (0, edge):
            starts[start].add("end")
        prior_center = (count - width) // 2 if width == count // 2 else None
        for start, families in sorted(starts.items()):
            if start == prior_center:
                continue
            out.append(dict(id=f"w{width:03d}-at{start:03d}", width=width,
                            families=sorted(families), seed=None, first=start,
                            deleted=list(range(start, start + width))))
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            deleted = np.sort(rng.choice(count, size=width, replace=False)).tolist()
            out.append(dict(id=f"w{width:03d}-r{seed}", width=width, families=["fixed_random"],
                            seed=seed, first=None, deleted=deleted))
    for row in out:
        deleted = set(row["deleted"])
        row["retained"] = [point for point in range(count) if point not in deleted]
        row["retained_hash"] = digest(row["retained"])
    if len({x["id"] for x in out}) != len(out) or len({x["retained_hash"] for x in out}) != len(out):
        raise ValueError("Window or random control duplicated a retained spectrum")
    return out


def random_control(shape, boxes, tile):
    height, width = shape
    a, b, c, d = boxes["target"]
    side_y, side_x = b - a, d - c
    choices = [(y, y + side_y, x, x + side_x)
               for y in range(height - side_y + 1) for x in range(width - side_x + 1)
               if (y + side_y <= a or y >= b or x + side_x <= c or x >= d)
               and (y, y + side_y, x, x + side_x) != boxes["matched"]]
    if not choices:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([REGION_SEED, tile]))
    return choices[int(rng.integers(len(choices)))]


def point_alias(salt, scene, role, coordinate):
    message = f"{POLICY}/{scene}/{role}/{int(coordinate)}".encode()
    return hmac.new(bytes.fromhex(salt), message, hashlib.sha256).hexdigest()[:32]


def population_rows(bank, role):
    if role not in ("devtrain", "devval"):
        raise ValueError("Only released development populations are permitted")
    return [row for row in bank.role_rows(role) if row["samples"]]
