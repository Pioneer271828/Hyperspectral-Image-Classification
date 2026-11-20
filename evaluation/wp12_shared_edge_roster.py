"""Identify complete frozen development cohorts before reading mechanism scores."""

from evaluation import frozen_training_stages as frozen
from evaluation import wp12_mechanism_real as real
from model.research_config import digest
from scripts.run_wp12_retained_evidence import RELEASE, pin
from transfer.development_training import validate
from transfer.native_released_preflight import open_bank
from transfer.spatial_coverage_protocol import retained_positions


def per_class(points, classes):
    labels = [label for _, label in points]
    return [labels.count(label) for label in range(1, classes + 1)]


def xuzhou_population(root):
    plan = frozen.stage_plan(root, pin(root, RELEASE))
    manifest = frozen.json_pin(root, plan["development"])
    record, geometry = manifest["record"], manifest["plan"]
    validate(record, geometry)
    release = frozen.json_pin(root, plan["release"])
    config = frozen.json_pin(root, release["full_metadata"])["training_config"]
    endpoint = plan["stages"]["selected"]
    scene, classes = manifest["dataset"], plan["identities"]["classes"]
    width = geometry["shape"][1]
    support, evaluation = record["train_support"], record["validation_support"]
    train = list(zip(support["indices"], support["labels"], strict=True))
    val = list(zip(evaluation["indices"], evaluation["labels"], strict=True))
    rows = []
    for tile in record["roles"]:
        if tile["role"] != "devval":
            continue
        y0, y1, x0, x1 = tile["box"]
        points = [(idx, label) for idx, label in val
                  if y0 <= idx // width < y1 and x0 <= idx % width < x1]
        rows.append(dict(id=tile["id"], parent_tile=tile["id"], samples=points))
    info = dict(support_seed=support["seed"], validation_seed=evaluation["seed"],
                support_sha256=plan["identities"]["support"], validation_sha256=plan["identities"]["validation"],
                space_sha256=plan["identities"]["space"], statistics_sha256=plan["statistics"],
                support_per_class=per_class(train, classes), evaluation_per_class=per_class(val, classes),
                support_evaluation_overlap=0, support_evaluation_parent_overlap=0,
                model_seed=config.get("model_seed"), original_sampler_seed=config.get("sampler_seed"),
                space_seed=geometry["seed"])
    identity = dict(scene=scene, classes=classes, step=endpoint["step"], checkpoint=endpoint["state"]["sha256"],
                    authority=plan["development"]["sha256"], state=release["full_state"], release=info)
    return identity, rows, plan["bands"]


def honghu_population(root):
    case = "whu-honghu"
    archive = frozen.json_pin(root, pin(root, (f"{real.BAND}/{case}.json", real.BAND_PINS[case])))
    bank, _, _ = open_bank(root, case)
    release = bank.release
    frozen.require(archive["case"] == archive["scene"] == case == release.scene
                   and archive["preflight_population"] == bank.pin["sha256"]
                   and archive["role"] == "devval" and archive["real_updates"] == 0
                   and archive["raw_gt_loaded"] is archive["query_accessed"] is False,
                   "The released HongHu endpoint or development bank differs")
    train = [row for row in bank.role_rows("devtrain") if row["samples"]]
    val = [row for row in bank.role_rows("devval") if row["samples"]]
    support = [point for row in train for point in row["samples"]]
    evaluation = [point for row in val for point in row["samples"]]
    info = dict(support_seed=release.custody["release_seed"], validation_seed=release.custody["validation_seed"],
                support_sha256=digest(support), validation_sha256=digest(evaluation),
                space_sha256=release.plan_hash, statistics_sha256=release.statistics["statistics_hash"],
                support_per_class=per_class(support, release.classes),
                evaluation_per_class=per_class(evaluation, release.classes),
                support_evaluation_overlap=len({i for i, _ in support} & {i for i, _ in evaluation}),
                support_evaluation_parent_overlap=len({r["parent_tile"] for r in train}
                                                      & {r["parent_tile"] for r in val}),
                model_seed=archive["model_seed"])
    identity = dict(scene=case, classes=release.classes, step=archive["selected_step"],
                    checkpoint=archive["checkpoint"], authority=bank.pin["sha256"], state=archive["model"], release=info)
    bands = {band: [release.bands.band_ids[i] for i in retained_positions(release.bands, band)]
             for band in frozen.VIEWS}
    return identity, val, bands


def focal_roster(rows, focus):
    """Get the first released block for each focus class within the full ordered population."""
    groups, truth, point_ids = [], [], []
    registered, ordinal = set(), 0
    for row in rows:
        if not row["samples"]:
            continue
        ordinal += 1
        ids, labels = map(list, zip(*row["samples"], strict=True))
        first = sorted((set(labels) & set(focus)) - registered)
        if first:
            registered.update(first)
            groups.append(dict(ordinal=ordinal, tile=row["id"], parent=row["parent_tile"], offset=len(truth),
                               points_sha256=digest(ids), truth=labels, first_focus=first))
        truth.extend(labels)
        point_ids.extend(ids)
    frozen.require(registered == set(focus) and len(point_ids) == len(set(point_ids)),
                   "Released points repeat or a focal class has no authorized development block")
    return groups, truth


def authority(root, case):
    if case == "selected":
        identity, rows, columns = xuzhou_population(root)
    else:
        frozen.require(case == "whu-honghu", "Unknown frozen focal release")
        identity, rows, columns = honghu_population(root)
    scene = identity["scene"]
    focus = list(real.FOCUS[scene])
    identity.update(case=case, focus=focus, rivals={str(k): v for k, v in real.RIVALS[scene].items()})
    groups, truth = focal_roster(rows, focus)
    archive = real.locked_history(root, case, identity["checkpoint"])
    old = archive["result"]
    frozen.require(all(old[key] == identity[key] for key in ("case", "scene", "classes", "authority", "state"))
                   and old["stage"] == identity["step"]
                   and [truth.count(c) for c in range(1, identity["classes"] + 1)]
                       == identity["release"]["evaluation_per_class"]
                   and all(old["views"][band]["truth"] == truth and old["views"][band]["band_ids"] == columns[band]
                           for band in frozen.VIEWS),
                   "Archived observations, endpoint, or released denominators differ")
    return identity, groups, old["views"]
