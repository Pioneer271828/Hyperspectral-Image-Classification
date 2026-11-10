"""Prepare label-blind roles and package one explicitly authorized sparse release."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_confirmation import unused_tiles
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.registered_development import _read


ROLES = ("calibration", "confirmation")
PLAN = "dy-held-aside-label-roles-v1"
AUTHORITY = "dy-held-aside-sparse-label-authorization-v1"
RELEASE = "dy-held-aside-sparse-curator-release-v1"
COMMITMENT = "dy-held-aside-sparse-commitment-v1"


def read(pin):
    return json.loads(_read(pin))


def verify_inputs(pins, code, out):
    require_frozen_code(code)
    if out.exists() or any(out == Path(p["path"]).resolve() or out in Path(p["path"]).resolve().parents for p in pins):
        raise ValueError("New external output outside the inputs required")
    if any(hash_file(p["path"]) != p["sha256"] for p in pins):
        raise ValueError("Pinned inputs changed")


def plan_roles(specification, out, code):
    spec = read(specification)
    if set(spec) != {"geometry", "development", "observations"}:
        raise ValueError("Pinned geometry, development record and observation manifest required")
    geometry, development, obs = (read(spec[k]) for k in ("geometry", "development", "observations"))
    expected = set(unused_tiles(geometry, development))
    if (obs["status"] != "candidates_prepared" or obs["raw_labels_read"] is not False
            or obs["plan_sha256"] != geometry["plan_hash"]
            or obs["development_sha256"] != development["validation_hash"]
            or geometry["gap"] < 1 or len(expected) < 2):
        raise ValueError("Verified held-aside geometry with two separate roles required")
    tiles = {t["tile_id"]: t for t in obs["tiles"]}
    if len(tiles) != len(obs["tiles"]) or tiles.keys() != expected:
        raise ValueError("Complete untouched candidate tiles required")
    ordered = sorted(tiles, key=lambda i: (geometry["tiles"][i]["box"][0], geometry["tiles"][i]["box"][2], i))
    middle = len(ordered) // 2
    blocks = []
    for role, ids in zip(ROLES, (ordered[:middle], ordered[middle:])):
        for i in ids:
            box = geometry["tiles"][i]["box"]
            if box != tiles[i]["box"]:
                raise ValueError("Candidate observation geometry changed")
            blocks.append(dict(id=f"tile-{i}", tile_id=i, role=role, box=box))
    verify_inputs([specification, *spec.values(), *(t["file"] for t in tiles.values())], code, out)
    result = dict(policy=PLAN, scene=obs["scene"], classes=development["classes"], blocks=blocks,
                  geometry_sha256=geometry["plan_hash"], development_sha256=development["validation_hash"],
                  observations=spec["observations"], inputs=spec,
                  allocation="row_then_column_first_half_calibration_remainder_confirmation",
                  status="roles_prepared_authorization_pending", raw_labels_read=False)
    out.mkdir()
    write_json(out / "role-plan.json", result)
    return result


def package(specification, out, code):
    spec = read(specification)
    if set(spec) != {"plan", "role", "authority", "source"} or spec["role"] not in ROLES:
        raise ValueError("One declared role and its pinned authority and source required")
    pins = [specification, spec["plan"], spec["authority"], spec["source"]]
    if len({Path(p["path"]).resolve() for p in pins}) != len(pins):
        raise ValueError("Curator data, authorization and plan must be separate artifacts")
    plan = read(spec["plan"])
    role = spec["role"]
    expected_authority = dict(policy=AUTHORITY, status="authorized", role=role,
                              plan_sha256=spec["plan"]["sha256"], source_sha256=spec["source"]["sha256"],
                              scope="held_aside_sparse_development_only", query_access=False)
    if plan["policy"] != PLAN or read(spec["authority"]) != expected_authority:
        raise ValueError("Matching role-scoped curator authorization required before reading labels")
    registered = {b["tile_id"]: b for b in plan["blocks"] if b["role"] == role}
    source = read(spec["source"])
    if (set(source) != {"policy", "role", "plan_sha256", "blocks"} or source["policy"] != RELEASE
            or source["role"] != role or source["plan_sha256"] != spec["plan"]["sha256"]
            or not isinstance(source["blocks"], list) or len(source["blocks"]) != len(registered)):
        raise ValueError("Complete single-role sparse curator release required")
    seen, blocks, labels = set(), [], {}
    counts = {c: 0 for c in plan["classes"]}
    for row in source["blocks"]:
        if set(row) != {"tile_id", "indices", "labels"} or type(row["tile_id"]) is not int:
            raise ValueError("Only local sparse label tuples are supported")
        i = row["tile_id"]
        if i not in registered or i in seen:
            raise ValueError("Released block is repeated or outside the authorized role")
        seen.add(i)
        block = registered[i]
        y0, y1, x0, x1 = block["box"]
        indices, values = row["indices"], row["labels"]
        if (not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values)
                or any(type(n) is not int or not 0 <= n < (y1-y0)*(x1-x0) for n in indices)
                or indices != sorted(set(indices)) or any(type(c) is not int or c not in counts for c in values)):
            raise ValueError("Invalid local label indices or foreground classes")
        if values:
            blocks.append(dict(id=block["id"], tile_id=i, role=role, label_indices=indices))
            labels[block["id"]] = values
            for c in values:
                counts[c] += 1
    verify_inputs(pins, code, out)
    missing = [c for c, count in counts.items() if not count]
    out.mkdir()
    if not missing:
        write_json(out / "blocks.json", sorted(blocks, key=lambda b: b["tile_id"]))
        write_json(out / "labels.json", labels)
    result = dict(status="insufficient_support" if missing else "packaged", role=role,
                  plan_sha256=spec["plan"]["sha256"], authority_sha256=spec["authority"]["sha256"],
                  source_sha256=spec["source"]["sha256"], support=counts, missing_classes=missing,
                  excluded_empty_tiles=sorted(registered.keys() - {b["tile_id"] for b in blocks}),
                  labels_sha256=None if missing else hash_file(out / "labels.json"),
                  labels_content_sha256=None if missing else digest(labels),
                  blocks_sha256=None if missing else hash_file(out / "blocks.json"),
                  raw_labels_read=False, query=0)
    write_json(out / "receipt.json", result)
    if not missing:
        write_json(out / "commitment.json", dict(policy=COMMITMENT, **{key: result[key] for key in (
            "role", "plan_sha256", "authority_sha256", "labels_sha256", "labels_content_sha256", "blocks_sha256")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "package"))
    for field in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{field}", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux label-custody checkout required")
    action = plan_roles if args.operation == "plan" else package
    action(dict(path=args.registration, sha256=args.registration_sha256), external_directory(args.out), code)


if __name__ == "__main__":
    main()
