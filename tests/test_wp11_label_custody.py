"""The label custody boundary opens only the authorized sparse role."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.wp11_confirmation import unused_tiles
from loadData.spatial_development import make_development
from loadData.spatial_tiles import make_tile_plan
from research_runtime import code_snapshot, hash_file, write_json
from scripts.prepare_wp11_label_custody import AUTHORITY, RELEASE, package, plan_roles, read


def save(root, name, value):
    path = root / f"{name}.json"
    write_json(path, value)
    return {"path": str(path), "sha256": hash_file(path)}


def fixture(root):
    geometry = make_tile_plan((80, 80), 20, 2)
    development = make_development(np.resize(np.arange(1, 10), (80, 80)), geometry, list(range(1, 10)))
    tiles = [dict(tile_id=i, box=geometry["tiles"][i]["box"], file=save(root, f"tile-{i}", {"fake": i}))
             for i in unused_tiles(geometry, development)]
    obs = dict(status="candidates_prepared", scene="xuzhou", raw_labels_read=False,
               plan_sha256=geometry["plan_hash"], development_sha256=development["validation_hash"], tiles=tiles)
    return save(root, "plan-spec", dict(geometry=save(root, "geometry", geometry),
                development=save(root, "development", development), observations=save(root, "obs", obs)))


class LabelCustodyTests(unittest.TestCase):
    def test_scoped_packages_and_insufficient_support(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            code = code_snapshot()
            out = root / "plan"
            plan = plan_roles(fixture(root), out, code)
            self.assertFalse(plan["raw_labels_read"])
            self.assertEqual(len({b["tile_id"] for b in plan["blocks"]}), len(plan["blocks"]))
            pin = dict(path=str(out / "role-plan.json"), sha256=hash_file(out / "role-plan.json"))
            role = "confirmation"
            blocks = [b for b in plan["blocks"] if b["role"] == role]
            rows = [dict(tile_id=b["tile_id"], indices=list(range(9)), labels=list(range(1, 10))) for b in blocks]
            for name, data in (("full", rows), ("missing", [dict(r, labels=[1]*9) for r in rows])):
                source = save(root, name, dict(policy=RELEASE, role=role, plan_sha256=pin["sha256"], blocks=data))
                authority = save(root, name+"-authorization", dict(policy=AUTHORITY, status="authorized", role=role,
                                 plan_sha256=pin["sha256"], source_sha256=source["sha256"],
                                 scope="held_aside_sparse_development_only", query_access=False))
                spec = dict(plan=pin, role=role, authority=authority, source=source)
                case = root / ("result-"+name)
                result = package(save(root, name+"-spec", spec), case, code)
                self.assertEqual((case / "labels.json").exists(), name == "full")
                self.assertEqual((case / "commitment.json").exists(), name == "full")
                self.assertEqual(result["status"], "packaged" if name == "full" else "insufficient_support")
                if name == "full":
                    bad = save(root, "wrong-role", dict(spec, role="calibration"))
                    with patch("scripts.prepare_wp11_label_custody.read", wraps=read) as reader:
                        with self.assertRaisesRegex(ValueError, "authorization"):
                            package(bad, root / "rejected", code)
                        self.assertNotIn(source, [call.args[0] for call in reader.call_args_list])
                    self.assertFalse((root / "rejected").exists())
