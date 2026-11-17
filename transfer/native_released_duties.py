"""Prepare the independent eight-arm and same-capacity bank design."""
from pathlib import Path

from transfer import native_released_real as real, native_released_preflight as preflight
from transfer import registered_development as dev
from transfer.native_scale_campaign import read
from transfer.native_scale_training import SEEDS
from transfer.spatial_multiscale import initialize


def expand(row):
    original = {(r["model_seed"], r["factor"]): r for r in row["draft"]["cases"]}
    cases = [{**original[seed, factor], "arm": arm, "id": f'{row["scene"]}/{seed}/{factor}/{arm}'}
             for seed in SEEDS for factor, arm in real.DUTY_COMPARISONS]
    return {**row, "design": real.DUTY, "draft": {**row["draft"], "cases": cases,
            "planned_updates": sum(c["planned_updates"] for c in cases)}}


def prepare(root, manifest, out):
    from transfer.native_released_campaign import REAL_PRE_FLIGHT_SHA

    if manifest["sha256"] != REAL_PRE_FLIGHT_SHA:
        raise ValueError("Expected the frozen two-bank released-point preflight")
    source = read(manifest, root)
    out.mkdir()
    pins = {}
    inventory = {}
    for scene, item in source["scenes"].items():
        row = read(item, root)
        bank, prior, intake = preflight.open_bank(root, scene)
        if bank.pin != row["bank"] or prior != row["prior_production"] or bank.reads:
            raise ValueError("Expanded duties must preserve original admitted bank")
        expanded = expand(row)
        expanded["origin"] = item
        pins[scene] = dev._json_new(out / f"{scene}.json", expanded)
        arms = dict.fromkeys(arm for _, arm in real.DUTY_COMPARISONS)
        for arm in arms:
            candidate = initialize(intake.baseline, "joint" if arm == "constant_joint" else arm, 7397)
            arms[arm] = sum(p.numel() for p in candidate.parameters())
        inventory[scene] = {"parameters": arms, "draws": len(expanded["draft"]["cases"]),
                            "case_steps": expanded["draft"]["cases"][0]["planned_updates"],
                            "train_support_per_class": [5]*bank.release.classes,
                            "point_exposures_per_case": [10]*bank.release.classes}
    return dev._json_new(out / "preflight.json", {**source, "scenes": pins, "origin": manifest,
                                               "design": real.DUTY, "inventory": inventory,
                                               "paviaU": "DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED"})


def validate_expansion(root, candidate):
    from transfer.native_released_campaign import REAL_PRE_FLIGHT_SHA

    origin = read(candidate["origin"], root)
    if candidate["origin"]["sha256"] != REAL_PRE_FLIGHT_SHA or set(origin["scenes"]) != set(candidate["scenes"]):
        raise ValueError("Expanded duties do not bind the original preflight")
    for name, pin in candidate["scenes"].items():
        current = read(pin, root)
        item = origin["scenes"][name]
        if current != {**expand(read(item, root)), "origin": item}:
            raise ValueError("Expanded duties changed original observations or budgets")
