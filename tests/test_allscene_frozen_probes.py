"""Real-network synthetic schemas, mixed historical reuse, and closed gates."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from evaluation import allscene_probe_campaign as campaign, allscene_probe_registry as registry
from evaluation import authenticated_probe_pack as packs, frozen_probe_audit as audit
from evaluation import frozen_probe_campaign as pavia, frozen_probe_history as history, frozen_probe_registry as reg
from evaluation.frozen_representation_capture import capture
from g4_matched_target_study import METHODS, initialize
from g4_support_context import read
from joint_baseline import _item_fingerprint
from loadData.target_label_protocol import make_target_family
from model.research_config import digest
from research_runtime import code_snapshot, hash_file, write_json
from target_bn_calibration import GRAPH_POLICY, OBS_KEYS
from tests.test_authenticated_probe_pack import fixture


def scene_inputs(root, scene, count, template):
    side = 24
    pixels = side*side
    regions = template["Q"].shape[1]
    rng = torch.Generator().manual_seed(count)
    item = {**template, "name": scene, "class_num": count, "height": side, "width": side,
            "Q": torch.nn.functional.one_hot(torch.arange(pixels)%regions, regions).float().to_sparse().coalesce(),
            "net_input": torch.rand(side, side, template["net_input"].shape[-1], generator=rng)}
    directory = root/scene
    directory.mkdir()
    obs, record, family = (directory/n for n in ("observation.pt", "record.json", "family.json"))
    torch.save(item, obs)
    write_json(record, {"policy": GRAPH_POLICY, "graph_fit_labels": 0, "target": scene,
                        "prepared_fields": _item_fingerprint(item), "observation_sha256": hash_file(obs)})
    labels = (torch.arange(pixels)%count+1).reshape(side, side).numpy()
    write_json(family, make_target_family(labels, scene, 64))
    return item, (obs, record, family)


def make_grid(root):
    source, freeze, analysis, _, initial = fixture(root)
    template = torch.load(initial[0], map_location="cpu", weights_only=True)
    accepted = read(analysis)
    frozen = read(freeze)
    launch_path = Path(frozen["training_root"])/"launch.json"
    launch = read(launch_path)
    records, model_stages = {}, {}
    for scene, classes in registry.CLASSES.items():
        item, files = (template, initial) if scene == "xuzhou" else scene_inputs(root, scene, classes, template)
        model, _ = initialize(scene, 64, METHODS[0], "none")
        stages, _, evidence = capture(model.eval(), scene, {k:item[k] for k in OBS_KEYS})
        records[scene] = files
        model_stages[scene] = stages, evidence
        frozen["protected_files"].update({str(path): hash_file(path) for path in files})
    models = {row["name"]: row for row in frozen["models"]}
    for row in launch["cases"]:
        obs, record, family = records[row["target"]]
        row["inputs"] = {"observation": str(obs), "record": str(record), "family": str(family),
                         "record_sha256": hash_file(record), "family_sha256": hash_file(family)}
        models[row["name"]].update(case_sha256=digest(row), family_hash=read(family)["family_hash"])
    write_json(launch_path, launch)
    frozen["launch_sha256"] = frozen["protected_files"][str(launch_path)] = hash_file(launch_path)
    write_json(freeze, frozen)
    write_json(analysis, {**accepted, "freeze_sha256": hash_file(freeze), "protected_files": frozen["protected_files"]})
    pins = {str(freeze): hash_file(freeze), str(analysis): hash_file(analysis), **frozen["protected_files"],
            **{str(analysis.parent/name): sha for name, sha in accepted["outputs"].items()}}
    receipt = read(source)
    rows = {"old": [], "new": []}
    rng = torch.Generator().manual_seed(1237)
    for model in frozen["models"]:
        name, scene = model["name"], model["target"]
        stages, evidence = model_stages[scene]
        folder = root/("capture-"+name)
        folder.mkdir()
        for stage, values in stages.items():
            # Constant graph logits are deterministic even when the fixture Q
            # repeats the same membership across several classes.
            synthetic = {**values, "graph": torch.zeros_like(values["graph"]),
                         "cnn": torch.randn(values["cnn"].shape, dtype=values["cnn"].dtype, generator=rng)}
            torch.save(synthetic, folder/(stage+"-features.pt"))
            write_json(folder/(stage+"-summary.json"), {"synthetic": True})
        path = folder/"receipt.json"
        write_json(path, {**receipt, "case": name, "evidence": evidence, "protected_files": pins,
            "freeze_sha256": hash_file(freeze), "analysis_sha256": hash_file(analysis),
            "outputs": {file.name: hash_file(file) for file in folder.iterdir()}})
        rows["old" if scene == "xuzhou" else "new"].append(
            {"case": name, "receipt": str(path), "sha256": hash_file(path), "origin": "captured"})
    indices = {}
    for group, entries in rows.items():
        folder = root/(group+"-index")
        folder.mkdir()
        indices[group] = folder/"capture-index.json"
        write_json(indices[group], {"captures": entries})
    return freeze, analysis, indices, model_stages


class AllSceneProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.code = code_snapshot()
        cls.freeze, cls.analysis, indices, cls.model_stages = make_grid(cls.root)
        cls.index = indices["new"]
        previous_scope = reg.prepare(cls.freeze, hash_file(cls.freeze), cls.analysis, hash_file(cls.analysis),
            indices["old"], hash_file(indices["old"]), cls.root/"old-scope", cls.code, synthetic=True,
            plan={**reg.PLAN, "steps": 6}, paths={"full": [6], "split": [2, 4, 6], "completed": [6]})
        previous_execution = pavia.run(previous_scope, hash_file(previous_scope), cls.root/"old-run", cls.code)
        previous_review = pavia.review(previous_scope, hash_file(previous_scope), previous_execution,
                                       hash_file(previous_execution), cls.root/"old-review", cls.code)
        cls.history_paths = previous_scope, previous_execution, previous_review
        historical = {"scope": previous_scope, "execution": previous_execution, "review": previous_review,
                      "acceptance": previous_review.with_name("acceptance.json")}
        for name, value in (("COMMIT", cls.code["commit"]), ("SOURCE", cls.code["source_hash"]),
                            ("SHA256", {k: hash_file(p) for k, p in historical.items()})):
            guard = patch.object(history, name, value)
            guard.start()
            cls.addClassCleanup(guard.stop)
        cls.scope = cls.register(cls.index, cls.root/"registration")
        cls.scope_sha = hash_file(cls.scope)
        cls.execution = campaign.run(cls.scope, cls.scope_sha, cls.root/"execution", cls.code)
        cls.execution_sha = hash_file(cls.execution)
        cls.review = campaign.review(cls.scope, cls.scope_sha, cls.execution, cls.execution_sha, cls.root/"review", cls.code)

    @classmethod
    def register(cls, index, out):
        return registry.prepare(cls.freeze, hash_file(cls.freeze), cls.analysis, hash_file(cls.analysis),
            index, hash_file(index), cls.history_paths, out, cls.code, synthetic=True, steps=6)

    def test_full_mixed_acceptance_class_metrics_and_exact_budget(self):
        import csv

        self.assertEqual(read(self.history_paths[2])["status"], "passed")
        receipt = read(self.review)
        self.assertEqual((receipt["status"], receipt["accepted_cases"], receipt["accepted_probes"]), ("passed", 54, 216))
        self.assertEqual((receipt["historical_probe_updates"], receipt["new_probe_updates"],
                          receipt["total_probe_updates"], receipt["audit_probe_updates"]), (864, 1728, 2592, 0))
        self.assertEqual(read(self.review.with_name("acceptance.json"))["probes"], 216)
        report = read(self.review.with_name("summary.json"))
        self.assertEqual((report["metric_rows"], report["class_metric_rows"], report["paired_rows"],
                          report["paired_class_rows"]), (864, 11520, 720, 9600))
        self.assertEqual(set(report["scenes"]), set(registry.CLASSES))
        self.assertTrue(all(row["seed_count"] == 3 for row in report["three_seed_class_paired"]))
        with self.review.with_name("class-metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for scene, count in registry.CLASSES.items():
            scene_rows = [row for row in rows if row["scene"] == scene]
            self.assertEqual({int(r["class_id"]) for r in scene_rows}, set(range(1, count+1)))
            self.assertTrue(all(r["support"] == ("10" if r["role"] == "train" else "5") for r in scene_rows))
        with self.assertRaises(FileExistsError):
            campaign.run(self.scope, self.scope_sha, self.root/"repeat", self.code)
        with self.assertRaises(FileExistsError):
            campaign.review(self.scope, self.scope_sha, self.execution, self.execution_sha, self.review.parent, self.code)

    def test_scope_seed_plan_layers_and_historical_updates_cannot_be_recounted(self):
        original = read(self.scope)
        cases = original["cases"]
        for field, value in (("cases", [{**cases[0], "seed": cases[1]["seed"]+777}, *cases[1:]]),
                             ("layers", original["layers"][:-1]), ("planned_new_probe_updates", 2592),
                             ("probe_plans", {**original["probe_plans"],
                                 "whu-honghu": {**original["probe_plans"]["whu-honghu"], "lr": .2}})):
            path = self.root/f"bad-{field}.json"
            write_json(path, {**original, field: value})
            with self.assertRaisesRegex(ValueError, "Registered all-scene roster"):
                registry.load(path, hash_file(path), self.code)
        run = read(self.execution)
        prior = next(row for row in run["cases"] if row["disposition"] == "historical")
        prior["new_probe_updates"] = 48
        scope, _, inputs, _, reused, _ = registry.load(self.scope, self.scope_sha, self.code)
        contents = self.execution.read_bytes()
        try:
            write_json(self.execution, run)
            with self.assertRaisesRegex(ValueError, "Historical or missing-capture role"):
                campaign.validate_run(self.execution, hash_file(self.execution), self.scope, self.scope_sha,
                                      scope, inputs, reused, self.code)
        finally:
            self.execution.write_bytes(contents)

    def test_missing_or_duplicate_capture_index_and_same_class_wrong_scene(self):
        rows = read(self.index)["captures"]
        nine_class = next(row for row in rows if row["case"].startswith("paviaU"))
        wrong = read(Path(read(self.history_paths[0])["cases"][0]["capture_receipt"]))
        source = self.root/"wrong-scene-source"
        source.mkdir()
        path = source/"receipt.json"
        write_json(path, {**wrong, "case": nine_class["case"]})
        mixed = [{**r, "receipt": str(path), "sha256": hash_file(path)} if r == nine_class else r for r in rows]
        for name, modified in (("missing", rows[:18]), ("repeat", [*rows[:-1], rows[0]]), ("wrong-scene", mixed)):
            index = self.index.parent/(name+".json")
            write_json(index, {"captures": modified})
            with self.assertRaises(ValueError):
                self.register(index, self.root/("reject-"+name))

    def test_passed_history_missing_original_file_and_new_producer_tamper(self):
        old = read(self.history_paths[1])["cases"][0]["case"]
        state = self.history_paths[1].parent/old/"full-0006"/"state.pt"
        moved = state.with_suffix(".saved")
        state.rename(moved)
        try:
            with self.assertRaises((FileNotFoundError, ValueError)):
                registry.load(self.scope, self.scope_sha, self.code)
        finally:
            moved.rename(state)
        rows = read(self.index)["captures"]
        path = Path(rows[0]["receipt"])
        original = path.read_bytes()
        try:
            receipt = read(path)
            receipt["code"]["commit"] = "not-the-capture-producer"
            write_json(path, receipt)
            with self.assertRaisesRegex(ValueError, "historical capture"):
                packs.read_capture_receipt(path, hash_file(path))
            with self.assertRaisesRegex(ValueError, "Capture receipt changed"):
                registry.load(self.scope, self.scope_sha, self.code)
        finally:
            path.write_bytes(original)

    def test_actual_honghu_stage_schema_and_strict_label_guards(self):
        scope, _, cases, _, _, _ = registry.load(self.scope, self.scope_sha, self.code)
        case, bound, local = next(row for row in cases if row[0]["target"] == "whu-honghu")
        folder = self.execution.parent/case["name"]
        plan = registry.case_scope(scope, case)
        pack, state, report, probabilities, _, _ = pavia.checked_execution(folder, hash_file(folder/"execution.json"),
            plan, self.scope_sha, case, local)
        stages, evidence = self.model_stages["whu-honghu"]
        self.assertEqual(set(stages), {"stage1", "stage2"})
        self.assertEqual(evidence["stage_calls"], {"stage1": 1, "stage2": 1})
        self.assertEqual(stages["stage1"]["cnn"].shape, (576, 128))
        self.assertEqual(stages["stage2"]["graph"].shape, (6, 64))
        self.assertEqual((len(pack["labels"]["train"]), len(pack["labels"]["val"])), (220, 110))
        proof = audit.audit_pack(pack, bound, plan, case, local)
        self.assertEqual(proof["authorized_val_support"], [5]*22)
        for kind in ("missing-class", "missing-layer", "reordered", "swapped", "query"):
            bad = deepcopy(pack)
            if kind == "missing-class":
                bad["labels"]["val"][bad["labels"]["val"] == 21] = 0
            elif kind == "missing-layer":
                bad["features"].pop("stage2_graph")
            elif kind == "reordered":
                bad["row_ids"]["train"] = bad["row_ids"]["train"].roll(1)
            elif kind == "swapped":
                bad["labels"]["train"], bad["labels"]["val"] = bad["labels"]["val"], bad["labels"]["train"]
            else:
                bad["identity"]["query_labels_accessed"] = True
            with self.assertRaises(ValueError):
                audit.audit_pack(bad, bound, plan, case, local)
        self.assertEqual(audit.audit_probes(pack, plan, state, report, probabilities)["history_selection_roles"], 4)

        original = self.root/"actual-honghu"
        original.mkdir()
        path, receipt, record, family, phases, overlay, selected, pins = bound
        for stage, values in stages.items():
            torch.save(values, original/(stage+"-features.pt"))
            write_json(original/(stage+"-summary.json"), {"synthetic_initialized_network": True})
        current = {**receipt, "outputs": {p.name: hash_file(p) for p in original.iterdir()}}
        current_path = original/"receipt.json"
        write_json(current_path, current)
        case = {**case, "capture_receipt": str(current_path), "capture_receipt_sha256": hash_file(current_path)}
        stage_pins = {str(original/name): sha for name, sha in current["outputs"].items()}
        stage_pins[str(current_path)] = case["capture_receipt_sha256"]
        fresh = (current_path, current, record, family, phases, overlay, selected, {**pins, **stage_pins})
        local = {**{name: sha for name, sha in local.items() if Path(name).parent != path.parent}, **stage_pins}
        out = self.root/"actual-honghu-probe"
        sha = pavia.run_case(out, fresh, case, local, plan, self.scope_sha, lambda receipt: None)
        actual, state, report, probabilities, counts, _ = pavia.checked_execution(out, sha, plan, self.scope_sha, case, local)
        self.assertEqual(counts, {"full": 24, "split": 24, "completed": 0})
        self.assertEqual(audit.audit_pack(actual, fresh, plan, case, local)["authorized_train_support"], [10]*22)
        self.assertEqual(audit.audit_probes(actual, plan, state, report, probabilities)["history_selection_roles"], 4)

    def test_numerical_failure_is_retained_without_accepting_successful_subset(self):
        oracle = audit.numpy_logp
        seen = []

        def alter(*args):
            result = oracle(*args)
            if not seen:
                result[0, result[0].argmax()] = -1e6
            seen.append(True)
            return result

        with patch.object(audit, "numpy_logp", side_effect=alter):
            path = campaign.review(self.scope, self.scope_sha, self.execution, self.execution_sha,
                                   self.root/"negative-review", self.code)
        receipt = read(path)
        self.assertEqual((receipt["status"], receipt["accepted_cases"], receipt["reviewed_probes"], receipt["failed_probes"]),
                         ("incomplete_or_failed", 53, 216, 1))
        self.assertFalse(path.with_name("acceptance.json").exists())
