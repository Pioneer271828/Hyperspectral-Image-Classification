"""Released two-scene campaign on the existing native optimizer."""

from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch

from evaluation.frozen_training_stages import VIEWS, file_pin, json_pin, pinned, require
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from transfer import development_bank as custody, native_coverage_queue as coverage
from transfer.development_bank_training import _initial
from transfer.native_released_points import LOSS, MODE
from transfer.spatial_coverage_protocol import bank_view, counts, release_statistics
from transfer.spatial_coverage_training import SpatialReleasedFixture, shared_source

SCHEMA = "wp05_native_matched_1500_v1"
SCENES = {"xuzhou": 9, "whu-honghu": 22}
SEEDS = (64, 512, 2048)
ARMS = ("none", "continuous")
STEPS = 1500
CLOCK = tuple(range(0, STEPS+1, 150))
CHECKPOINT_INTERVAL = 150
TRAIN_VIEWS = ("full", "central_gap", "matched_dispersed")
SELECTION = "first_min_full_band_fused_development_nll"
DESIGN = dict(steps=STEPS, selection_steps=list(CLOCK), endpoint_steps=[0, 750, 1500],
              selection=SELECTION, support_seed=2048, k=5, train_views=list(TRAIN_VIEWS),
              deleted_only=[1, 2], redundant=[3, 4], rare_recall_max_loss=0.02,
              synthetic_redundant_min_aa=0.65, synthetic_deleted_max_aa=0.65)

def design():
    return {f"{scene}-m{seed}-{arm}": dict(scene=scene, model_seed=seed, spectral_arm=arm)
            for scene, seed, arm in product(SCENES, SEEDS, ARMS)}


def synthetic_definition(scene):
    """Preserve native block geometry and deliberately separate two signal regimes."""
    base, _ = coverage.generated(scene, DESIGN["support_seed"], 5, "a")
    shape = (47, base.shape[1], base.shape[2])
    rows = [deepcopy(row) for row in base.rows]
    for i, row in enumerate(rows[6:], 1):
        clone = deepcopy(row)
        clone["id"] += 100
        clone["parent_tile"] = clone["id"]
        clone["box"] = [34, 40, *clone["box"][2:]]
        clone["parent_box"] = clone["box"]
        rows.append(clone)
    rows.sort(key=custody._row_key)
    roles = dict(support=sorted(row["id"] for row in rows if row["role"] == "devtrain"),
                 calibration=sorted(row["id"] for row in rows if 60 < row["id"] < 110),
                 evaluation=sorted(row["id"] for row in rows if row["id"] > 110))
    seed = coverage.source_seed(scene)
    classes = base.classes
    valid = list(np.flatnonzero(base.bands.valid))
    cut = (len(valid)-len(valid)//2)//2
    central = valid[cut:cut+len(valid)//2]
    outside = [index for index in valid if index not in central]
    rng = np.random.default_rng([seed, 7919])
    # Give the removed-only pair identical outside-band signatures. Give the
    # redundant pair distinct signatures on both sides of the missing window.
    book = rng.normal(0, 1.9, size=(classes, shape[-1])).astype(np.float32)
    book[1, outside] = book[0, outside]
    book[0, central], book[1, central] = -2.5, 2.5
    for index in (2, 3):
        book[index, outside] = (-1 if index == 2 else 1) * np.linspace(.8, 2.6, len(outside))
        book[index, central] = (-1 if index == 2 else 1) * 2
    arrays = {}
    for ids, role in ((roles["support"], "support"), (roles["calibration"], "calibration"),
                      (roles["evaluation"], "evaluation")):
        block = [row for row in rows if row["id"] in ids]
        for row in block:
            row["samples"] = []
        rng = np.random.default_rng([seed, DESIGN["support_seed"] if role == "support" else 773, ids[0]])
        cells = {row["id"]: rng.permutation(42).tolist() for row in block}
        assignments = [(label, i) for label in range(1, classes+1)
                       for i in range(2 if role == "evaluation" and label > classes-2 else 5)]
        rng.shuffle(assignments)
        for at, (label, _) in enumerate(assignments):
            row = block[at % len(block)]
            dy, dx = divmod(cells[row["id"]].pop(), 7)
            y, _, x, _ = row["box"]
            row["samples"].append([(y+dy)*shape[1]+x+dx, label])
        for row in block:
            row["samples"].sort()
            y, _, x, _ = row["box"]
            bg = rng.integers(0, classes, size=(6, 7))
            for pixel, label in row["samples"]:
                py, px = divmod(pixel, shape[1])
                bg[py-y,px-x] = label-1
            values = (book[bg] + rng.normal(0, 0.045, size=(6,7,shape[-1]))).astype(np.float32)
            arrays[row["id"]] = values
            row["origin"] = dict(synthetic_seed=seed, array_sha256=custody.array_hash(values))
    plan = digest([SCHEMA, scene, shape, roles, [row["box"] for row in rows]])
    stats = release_statistics(arrays, rows, plan)
    report = dict(minimum_buffer=3, construction=SCHEMA, support_seed=2048, k=5,
                  roles_sha256=digest(roles), planted_null_spectral_signal=[1, 2],
                  redundant_signal=[3, 4], confirmation_rare=[classes-1, classes])
    release = custody.BankRelease(scene, "synthetic", shape, classes, plan, base.bands, stats,
                                  base.source, report, tuple(rows))
    return release, arrays, roles


def register_synthetic(out, code):
    out.mkdir(exist_ok=True)
    folder = out/"inputs"
    folder.mkdir(exist_ok=True)
    inputs = {}
    for scene in SCENES:
        release, arrays, roles = synthetic_definition(scene)
        path = folder/scene
        if not (path/"bank.json").exists():
            custody.publish(path, folder, release, lambda row: arrays[row["id"]], resume=path.exists())
        pin = dict(root=str(path), sha256=file_pin(path/"bank.json")["sha256"],
                   registration=file_pin(path/"registration.json")["sha256"])
        custody.Bank(pin, folder, release).verify_authority(pin)
        weights = folder/f"{scene}-source.pt"
        if not weights.exists():
            save_model_checkpoint(shared_source(scene, coverage.source_seed(scene)), weights)
        inputs[scene] = dict(bank=pin, roles=roles, source=dict(weights=file_pin(weights),
                             sidecar=file_pin(weights.with_suffix(".pt.json"))))
    return dict(schema=SCHEMA, producer=code["commit"], source_hash=code["source_hash"], scope="synthetic_only",
                design=DESIGN, inputs=inputs, cases=design(), optimization_updates=12*STEPS,
                estimate=dict(per_case_wall_seconds=[60,220], two_workers_wall_seconds=[400,1550],
                              separate_continuous_reference_updates=STEPS,
                              resource_threshold=dict(free_mib=12288,max_utilization_percent=60)),
                query_accessed=False, raw_gt_loaded=False)


def real_inputs(root, out, code, spec_pin, approval_pin):
    """Intake provided approvals only. No filesystem output is created here."""
    require(spec_pin is not None and approval_pin is not None,
            "Real coverage requires both the campaign request and a separate execution approval")
    approval = json_pin(root, approval_pin)
    request = json_pin(root, spec_pin)
    require(set(request) == {"schema", "inputs", "physical_devices"}
            and request["schema"] == SCHEMA and set(request["inputs"]) == set(SCENES)
            and set(request["physical_devices"]) == set(design()), "Real campaign roster differs")
    from transfer.p4_runtime_device import UUID

    require(all(isinstance(v, str) and UUID.fullmatch(v) for v in request["physical_devices"].values()),
            "Each real case requires an approved physical UUID")
    expected = dict(schema=SCHEMA, decision="approved", scope="external_real_execution", request=spec_pin,
                    source_commit=code["commit"], source_hash=code["source_hash"], destination=str(out),
                    design_sha256=digest(DESIGN), optimizer_updates=12*STEPS,
                    raw_gt_authorized=False, query_authorized=False)
    require(approval == expected, "Separate execution approval does not bind this destination/budget/source")
    from transfer.native_independent_release import open_independent

    inputs = {}
    for scene, entry in request["inputs"].items():
        require(set(entry) == {"manifest", "authority", "source"}
                and set(entry["source"]) == {"weights", "sidecar"}, "Incomplete independent release or source")
        bank, report, roles = open_independent(root, entry["manifest"], entry["authority"])
        require(report["source_mode"] == "external_real" and bank.release.scene == scene
                and report["uses"]["independent_accuracy"]["eligible"]
                and report["physical_wavelengths_verified"]
                and bank.release.custody.get("support_seed") == DESIGN["support_seed"]
                and report["role_reports"]["support"]["per_class"] == [5]*bank.release.classes,
                "Real K5 support, disjoint confirmation and verified physical band release required")
        weights, sidecar = (pinned(root, entry["source"][key]) for key in ("weights", "sidecar"))
        require(sidecar == weights.with_suffix(".pt.json")
                and bank.release.source.get("shared_state"), "Released source identity missing")
        inputs[scene] = dict(bank=bank.pin, roles=roles, source=entry["source"],
                             manifest=entry["manifest"], authority=entry["authority"])
    return dict(schema=SCHEMA, producer=code["commit"], source_hash=code["source_hash"], scope="external_real",
                design=DESIGN, inputs=inputs, cases=design(), physical_devices=request["physical_devices"],
                approval=approval_pin, request=spec_pin, optimization_updates=12*STEPS,
                query_accessed=False, raw_gt_loaded=False)


class MatchedProvider(SpatialReleasedFixture):
    """Adapt only released observations. The model, loss and updates stay native."""

    def __init__(self, root, code, queue, name):
        spec = queue["cases"][name]
        scene, seed, arm = spec["scene"], spec["model_seed"], spec["spectral_arm"]
        entry = queue["inputs"][scene]
        if queue["scope"] == "synthetic_only":
            release, _, roles = synthetic_definition(scene)
            bank_path = Path(entry["bank"]["root"])
            require(root in bank_path.parents and bank_path.name == scene, "Synthetic Bank escaped the registered root")
            bank = custody.Bank(entry["bank"], bank_path.parent, release)
        else:
            from transfer.native_independent_release import open_independent

            bank, result, roles = open_independent(root, entry["manifest"], entry["authority"])
            require(result["uses"]["independent_accuracy"]["eligible"], "Independent real release expired")
        require(bank.pin == entry["bank"] and roles == entry["roles"], "Bank or roles differ from long registration")
        for pin in entry["source"].values():
            pinned(root, pin)
        require(Path(entry["source"]["sidecar"]["path"]) == Path(entry["source"]["weights"]["path"]).with_suffix(".pt.json"),
                "Source sidecar is foreign to the published source")
        self.bank, self.roles, self.source_files = bank, roles, entry["source"]
        self.scene, self.classes = scene, bank.release.classes
        self.arm, self.seed, self.producer = arm, seed, code["commit"]
        self.roster, self.cache = {}, {}
        self.training_scope = queue["scope"]

    def initialization(self, seed):
        target, witness = super().initialization(seed)
        require(witness["source"] == self.bank.release.source["shared_state"], "Released source state differs")
        return target, witness

    def materialize(self, role, event, producer):
        expected = {"devtrain": "support", "devval": "calibration"}[role]
        require(producer == self.producer and event["tile"] in self.roles[expected]
                and event["factor"] == 1 and (event["band"] in TRAIN_VIEWS if role == "devtrain" else event["band"] == "full"),
                "Optimizer/selector attempted to consume a different role or view")
        key = self.key(role, event)
        if key not in self.cache:
            self.cache[key] = bank_view(self.bank, self.bank.role_row(role, event["tile"]), producer, event["band"])
        return self.cache[key]

    def contract(self, classes, seed, arm, factor, producer, *, numeric=None):
        require((classes, seed, arm, factor, producer) == (self.classes, self.seed, "off", 1, self.producer),
                "Long case model seed or native configuration differs")
        self.bank.verify_authority(self.bank.pin)
        tiles = self.roles["support"]
        require(tiles and all(self.bank.role_row("devtrain", tile)["samples"] for tile in tiles),
                "Long exposure needs a labeled roster")
        rng = np.random.default_rng(seed)
        train = []
        clock = 0
        while len(train) < STEPS:
            order = rng.permutation(tiles).tolist()
            for tile in order:
                band = TRAIN_VIEWS[(clock + tiles.index(tile)) % len(TRAIN_VIEWS)]
                train.append(dict(step=len(train)+1, tile=tile, band=band, factor=1, profile=None, condition="missing"))
            clock += 1
        train = train[:STEPS]
        val = [dict(tile=tile, band="full", factor=1, profile=None, condition="missing")
               for tile in self.roles["calibration"]]
        if not self.roster:
            for role, events in (("devtrain", train), ("devval", val)):
                for event in events:
                    key = self.key(role, event)
                    if key not in self.roster:
                        actual, _, targets = self.materialize(role, event, producer)
                        self.roster[key] = self.evidence(actual, targets)
        source, witness = self.initialization(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+7321)
            model = self.model_factory(source, arm, seed+7321)
        baseline = coverage.CoverageFixture._unconditioned(source, seed+7321)
        return dict(format=SCHEMA, producer=producer, classes=classes, seed=seed, arm=arm, factor=1,
                    supervision=MODE, objective=LOSS, training=train, validation=val, inputs=deepcopy(self.roster),
                    steps=STEPS, prefix=750, events=list(CLOCK), learning_rate=.0001, module_seed=seed+7321,
                    rng_seed=seed+15000, registered_graph_scale=100, initialization=_initial(model, baseline),
                    parameters=sum(p.numel() for p in model.parameters()), bank=self.bank.pin, source=witness,
                    selection=SELECTION, numeric=numeric, frozen_stage_protocol=SCHEMA,
                    raw_gt_loaded=False, query_accessed=False, real_updates=STEPS if not self.is_synthetic else 0)

    def verify_confirmation(self):
        fit = [self.bank.role_row("devtrain", i) for i in self.roles["support"]]
        select, confirm = ([self.bank.role_row("devval", i) for i in self.roles[role]]
                           for role in ("calibration", "evaluation"))
        require(all(custody._gap(a["parent_box"], b["parent_box"]) >= self.bank.release.custody["minimum_buffer"]
                    and a["parent_tile"] != b["parent_tile"] for a in fit+select for b in confirm)
                and min(counts(confirm, self.classes)) > 0, "Independent confirmation geometry/points differ")
        return confirm
