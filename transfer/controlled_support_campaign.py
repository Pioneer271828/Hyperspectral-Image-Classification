"""Register and recover only complete, predeclared fixed-support triplets."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

from research_runtime import external_directory
from transfer import native_support_campaign as native, registered_development as dev, xuzhou_multiscale as stages
from transfer.controlled_regional_support import ARMS
from transfer.controlled_support_run import ControlledSupportRun
from transfer.native_support import make_selector
from transfer.native_support_audit import load_frozen
from transfer.native_support_intervention import prepare_block


OLD = {
    "registration": "d012597b407f59bdb83ead39ba58317b7d9a8f495700ea3880b673ac0759fa70",
    "freeze": "faedf4ac600f976c07629f0f0bf604d2d936db719f5e9b253e6061b88c2794a1",
    "audit": "0dde73772cf6eaf07291573feba12a00f18cf0f50a5dac12967366de9bc740a0",
    "real": "a138590a2ac788c0fdf9c0719259f8f06e432d2d58789c2001cdc988cff53467",
}
PRODUCER = "ac5d5663b1ce992183ea36ce7cb570ea3a7d742c"
MAXIMUM_UPDATES = 2592
BUDGET = 48


def read(pin):
    return json.loads(dev._read(pin))


def originals(pins):
    if set(pins) != set(OLD) or any(pins[key]["sha256"] != value for key, value in OLD.items()):
        raise ValueError("Reviewed Issue447 evidence required")
    producer = native.pinned_producer(pins["registration"], PRODUCER)
    _, old, frozen = load_frozen(pins["registration"], pins["freeze"], producer)
    audit, real = read(pins["audit"]), read(pins["real"])
    if (audit["registration"] != pins["registration"] or audit["freeze"] != pins["freeze"]
            or real["query"] or real["updates"] or audit["audit_updates"] or len(audit["results"]) != 75):
        raise ValueError("Reviewed endpoint or custody chain differs")
    return old, frozen, audit, real


def trial(case):
    return native._trial("joint", case["support_seed"])


def random_trial(case):
    return native._trial("random", case["support_seed"])


def intervention_seed(case, tile):
    return 41000 + case["support_seed"] * 257 + tile


def _factory(case, data, package, arm):
    baseline, scene, record, plan, bank = data
    intervention = dict(owner=package["owner"], blocks=package["blocks"])
    settings = dict(selector_seed=case["scorer_seed"], bank=bank, random_trial=random_trial(case),
                    intervention=intervention, arm=arm)
    active_trial = trial(case)

    def create(*_):
        return ControlledSupportRun(baseline, scene, record, plan=plan, trial=active_trial, **settings,
                                    lr=1e-4, torch_seed=case["torch_seed"], expected_plan_hash=plan.plan_hash)

    def load(source, current_scene, release, path, *, selector_seed, trial, **kwargs):
        return ControlledSupportRun.load(source, current_scene, release, path, trial=trial, **settings, **kwargs)

    return create, SimpleNamespace(load=load)


def register(root, pins, code):
    old, frozen, _, _ = originals(pins)
    root = external_directory(root)
    root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for key, case in old["cases"].items():
        if case["work_package"] != "WP08":
            continue
        data = native._case_data(case)
        _, scene, record, plan, bank = data
        package = dict(owner=dict(case=key, purpose="fixed_recovery", bank=case["bank"], prior=pins["registration"],
                                 donor=frozen["rows"][f"{key}/fixed"]["endpoints"]["last"]),
                       blocks={}, search_cost={})
        joint = trial(case)
        for block, entry in bank.items():
            if entry["context"]["role"] != "devtrain":
                continue
            tile = int(block)
            receipt, cost = prepare_block(entry, record["roles"][tile]["box"], joint, random_trial(case),
                                          BUDGET, intervention_seed(case, tile), package["owner"])
            donor = native._load_pt(package["owner"]["donor"])
            if donor["donor"]["snapshots"][block] != receipt["reference"]:
                raise ValueError("Fixed reference differs from reviewed final donor")
            package["blocks"][block], package["search_cost"][block] = receipt, cost
        if len(package["blocks"]) != 2 or plan.updates != BUDGET:
            raise ValueError("Two registered training blocks and the 48-step exposure plan required")
        package["eligible"] = all(block["match"]["status"] == "MATCHED" for block in package["blocks"].values())
        package["status"] = "ELIGIBLE" if package["eligible"] else "INFEASIBLE_FOR_REGISTERED_TRIPLET"
        package["contracts"] = {arm: _factory(case, data, package, arm)[0]().contract_hash
                                for arm in ARMS} if package["eligible"] else {}
        cases[key] = package
    if len(cases) != 9:
        raise ValueError("Original WP08 three-by-three design required")
    count = sum(case["eligible"] for case in cases.values()) * len(ARMS) * BUDGET * 2
    return dev._json_new(root / "registration.json", dict(issue=449, version=1, scope="synthetic",
        code={key: code[key] for key in ("commit", "branch", "source_hash")}, original=pins, cases=cases,
        maximum_updates=MAXIMUM_UPDATES, registered_updates=count, extra_engineering_updates=0,
        validation="original_step_zero_identical_in_all_arms", query=0, real_updates=0))


def load(pin, code):
    reg = read(pin)
    if (reg["code"] != {key: code[key] for key in ("commit", "branch", "source_hash")} or reg["scope"] != "synthetic"
            or reg["maximum_updates"] != MAXIMUM_UPDATES or reg["real_updates"] or reg["query"]):
        raise ValueError("Controlled registration or execution source differs")
    old, frozen, audit, real = originals(reg["original"])
    expected = sum(package["eligible"] for package in reg["cases"].values()) * len(ARMS) * BUDGET * 2
    if expected != reg["registered_updates"]:
        raise ValueError("Controlled update envelope changed")
    return Path(pin["path"]).parent, reg, old, frozen, audit, real


@contextmanager
def _queue(root):
    with (root / "queue.lock").open("ab") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def stage(pin, code, key, arm, phase, prior):
    root, reg, old, _, _, _ = load(pin, code)
    package = reg["cases"][key]
    if not package["eligible"] or arm not in ARMS:
        raise ValueError("Unregistered or incomplete controlled triplet")
    case = old["cases"][key]
    data = native._case_data(case)
    baseline, scene, record, plan, _ = data
    create, loader = _factory(case, data, package, arm)
    with _queue(root):
        return stages.execute_stage(root / "matrix" / key, pin, arm, phase, prior,
            value=dict(budget=BUDGET, contracts=package["contracts"]), baseline=baseline, scene=scene,
            record=record, plan=plan, arms=dict.fromkeys(ARMS, trial(case)), stages=native._stages(BUDGET),
            run_factory=create, run_type=loader)
