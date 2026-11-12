"""Adapters for the accepted observation, role, release, and source receipts."""

from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
import torch

from loadData.data_read import read_band_metadata
from model.research_config import canonical_json, digest
from research_runtime import hash_file, tensor_hash
from train_spatial import SpatialConfig
from transfer import p4_guarded, p4_spatial, registered_development as transport, wp02_route_development as wp02
from transfer.development_blocks import Release, SCENES, inside, read_pin
from transfer.development_mat import open_guarded
from transfer.g3_training import state_hash
from transfer.native_support_real import WP07_REGISTRATION, WP08_SOURCE_CAMPAIGN, CURATOR_RECEIPT
from transfer.xuzhou_multiscale import baseline_from_config


HISTORICAL = {
    "metadata": ("g4-observation-provenance-issue194-audit01/receipt.json",
                 "a311a4a08f3a6bfa3d639d4cde65444bc6f9deff2bd86727b884d526efaebdff"),
    "xuzhou": ("wp08-native-control01/real-zero-update-reviewed01.json",
               "a138590a2ac788c0fdf9c0719259f8f06e432d2d58789c2001cdc988cff53467"),
    "pavia": ("p4-pavia-spatial-real01/registration.json",
              "fc61d64f204a78595856cd84394809911cce89eb83e19a097ecb3fb1980d4ad2"),
    "pavia_acceptance": ("p4-pavia-spatial-real01/acceptance.json",
                         "413b23ab9350201bbd2cf10de675ff355a06a2f7d2207774ec487da709d6509c"),
    "pavia_staging": ("p4-g2-pavia-observation-staging01/receipt.json",
                      "4b319a8d8430f050e73ca26b84f616eb97de24c1396fb0e3415ada4ea420bd45"),
    "pavia_online": ("e4b-online-tta-pavia01/receipt.json",
                     "2435a74dc65e27b496db58b998498fb84cb91792facccdef7313ef445c6b410a"),
    "honghu": ("wp05-honghu-guarded-short01/registration.json",
               "67fc5cf386960172d8d7c4b925efdcb12cf79b81a6004616ac2faa049429ec13"),
    "honghu_split": ("wp05-honghu-guarded-short01/registration-split.json",
                     "b339a1c471061a08085c92c5bbf494165994bb0eae1fe0ce6a4a9d9073b3bc"),
    "honghu_acceptance": ("wp05-honghu-guarded-independent02/receipt.json",
                          "80063151738651736512b47a66ef907f2bb22fb80376fea1023ea5c81505aa72"),
    "wp12": ("wp12-controlled-support-reviewed01/controlled-supplement.json",
             "b813272abff44c3dd72ef5eb580c1cd6c6777725e8a1395b2295d4aeb33cf9cf"),
}


def historical(root, key):
    name, sha256 = HISTORICAL[key]

    return {"path": str(root / name), "sha256": sha256}


def _json(pin, root):
    return json.loads(read_pin(pin, root))


def _pin(registration, path):
    return {"path": path, "sha256": registration["pins"][path]}


def _protect(root, files):
    for pin in files:
        if hash_file(inside(pin["path"], root)) != pin["sha256"]:
            raise ValueError(f"Historical input bytes changed: {pin['path']}")


def _band_metadata(root, scene):
    pin = historical(root, "metadata")
    value = _json(pin, root)
    if value["status"] != "passed" or not value["full_target_scene_transductive"] or value["spatial_isolation"]:
        raise ValueError("Historical metadata has an unexpected provenance scope")
    row = wp02._only(value["rows"], lambda row: row["target"] == scene)
    shape, _ = SCENES[scene]
    bands = read_band_metadata(scene, shape[-1])
    if (row["shape"] != list(shape) or row["wavelength_quality"] != bands.wavelength_quality
            or row["wavelength_sha256"] != tensor_hash(torch.tensor(bands.wavelengths, dtype=torch.float32))
            or row["fwhm_present"] or row["original_sensor_band_mapping_authenticated"]
            or row["per_band_centers_authenticated"] or row["srf_authenticated"]):
        raise ValueError("Delivered bands differ from audited approximate metadata")
    physical = {name: row[name] for name in ("band_index_scope", "original_sensor_band_mapping_authenticated",
                "per_band_centers_authenticated", "fwhm_present", "srf_authenticated", "wavelength_sha256")}
    physical.update(gsd_status="unverified_not_used", radiometry="historical_preprocessing_only",
                    observation_scope="metadata_only_from_historical_transductive_receipt",
                    metadata_receipt=pin)
    return bands, physical


def _rows(tiles, labels, shape, origin, parents=None):
    labels_by_tile = {tile["id"]: local for role in ("train", "val") for tile, local in labels[role]}
    rows = []
    for role, tile in zip(("devtrain", "devval"), tiles, strict=True):
        y0, y1, x0, _ = tile["box"]
        yy, xx = np.nonzero(labels_by_tile[tile["id"]])
        local = labels_by_tile[tile["id"]]
        points = [[int((y0 + y) * shape[1] + x0 + x), int(local[y, x])] for y, x in zip(yy, xx, strict=True)]
        rows.append(dict(tile, role=role, parent_tile=tile["id"] if parents is None else parents[tile["id"]],
                         origin=origin(tile), samples=points))
    return tuple(rows)


@dataclass
class Admitted:
    release: Release
    baseline: torch.nn.Module
    reader: object
    protected: tuple
    regression: dict | None = None


def xuzhou(root):
    accepted = historical(root, "xuzhou")
    value = _json(accepted, root)
    prior_pin, campaign_pin, curator_pin = value["provenance"]
    if tuple(p["sha256"] for p in value["provenance"]) != (
            WP07_REGISTRATION, WP08_SOURCE_CAMPAIGN, CURATOR_RECEIPT):
        raise ValueError("Historical Xuzhou custody differs")
    prior, campaign, curator = [_json(pin, root) for pin in value["provenance"]]
    if (prior["issue"] != 411 or campaign["issue"] != 407 or curator["status"] != "completed"
            or any(value[key] for key in ("updates", "raw_gt", "query", "labels_added"))):
        raise ValueError("Historical Xuzhou acceptance differs")
    old = value["scenes"]["xuzhou"]
    development = campaign["development"]
    package = _json(development, root)
    if (old["released_role_manifest"] != development or package["dataset"] != "xuzhou"
            or old["source"] != prior["source"] or old["sidecar"] != prior["sidecar"]):
        raise ValueError("Historical Xuzhou source/observation chain differs")
    plan, record = package["plan"], package["record"]
    for pin in record["wp02"].values():
        inside(pin["path"], root.parent if pin == record["wp02"]["image"] else root)
    transport._released_record(record)
    original_stats = wp02.verify_record(record, plan, "xuzhou")
    shape, classes = SCENES["xuzhou"]
    bands, physical = _band_metadata(root, "xuzhou")
    stats = package["statistics"]
    if (canonical_json(package["bands"]) != canonical_json(bands.__dict__)
            or any(stats[key] != np.asarray(original_stats[key], dtype=np.float32).tolist()
                   for key in ("minimum", "maximum"))):
        raise ValueError("Xuzhou observation bands or training extrema differ")
    roles = [next(row for row in record["roles"] if row["role"] == role) for role in ("devtrain", "devval")]
    members = {row["id"]: row for row in package["blocks"]}
    rows = []
    for role, pin in zip(roles, old["role_arrays"], strict=True):
        entry = members[role["id"]]
        expected = {"path": str(Path(development["path"]).parent / f"tile-{role['id']}.npy"),
                    "sha256": entry["sha256"]}
        if pin != expected or entry["role"] != role["role"] or entry["name"] != Path(pin["path"]).name:
            raise ValueError("Xuzhou block differs from prior admitted NPY")
        key = "train_support" if role["role"] == "devtrain" else "validation_support"
        y0, y1, x0, x1 = role["box"]
        points = [[i, y] for i, y in zip(record[key]["indices"], record[key]["labels"], strict=True)
                  if y0 <= i // shape[1] < y1 and x0 <= i % shape[1] < x1]
        rows.append(dict(role, parent_tile=role["id"], origin=pin, samples=points))
    model_files = (prior["source"], prior["sidecar"])
    for pin in model_files:
        inside(pin["path"], root)
    baseline = transport._load_source(*model_files)
    if state_hash(baseline) != prior["source_state"] or prior["source_state"] != old["source_state"]:
        raise ValueError("Xuzhou source model differs")
    source = dict(weight=model_files[0], sidecar=model_files[1], state=old["source_state"],
                  initialization="exact_issue411_initial", shared_state=old["source_state"])
    custody = dict(acceptance=accepted, development=development, label_release=record["wp02"]["release"],
                   prior_registration=prior_pin, prior_route=campaign_pin, curator=curator_pin,
                   minimum_buffer=2 * plan["gap"], block_selection="original_issue447_pair")
    release = Release("xuzhou", "nested_development", shape, classes, plan["plan_hash"], bands, stats,
                      tuple(rows), custody, physical, source)
    return Admitted(release, baseline, lambda row: np.load(io.BytesIO(read_pin(row["origin"], root)),
                    allow_pickle=False), (accepted, *value["provenance"], development, *model_files,
                    *old["role_arrays"], *[p for k, p in record["wp02"].items() if k != "image"],
                    physical["metadata_receipt"]), old)


def _spatial_source(root, registration, scene, *, state_required):
    evidence = registration["source"]
    weight, sidecar, result = [_pin(registration, name) for name in
                               (evidence["weight"], evidence["weight"] + ".json", evidence["result"])]
    for pin in (weight, sidecar, result):
        inside(pin["path"], root)
    old = transport._load_source(weight, sidecar)
    record = _json(result, root)
    others = set(SCENES) - {scene}
    if (set(old.heads) != others or set(record["per_dataset_updates"]) != others
            or old.backbone.graph_mode != "static" or old.backbone.norm_type != "domain_bn"
            or old.backbone.region_residual_mode != "none" or record["status"] != "completed"):
        raise ValueError("Historical shared source is outside the accepted spatial protocol")
    authority = dict(weight=weight, sidecar=sidecar, result=result, state=state_hash(old))
    if not state_required:
        return authority, (weight, sidecar, result), None
    config = registration["config"]
    target = baseline_from_config(config, scene=scene, classes=SCENES[scene][1])
    # P4's established transfer supplies exactly the non-domain, non-graph shared tensors.
    shared_model, shared_info = p4_spatial.initialize(SpatialConfig(config), weight["path"], result["path"])
    keys = shared_info["copied_keys"]
    original = old.backbone.state_dict()
    shared = shared_model.backbone.state_dict()
    target_state = target.backbone.state_dict()
    expected = {name for name in target_state if name.startswith(p4_spatial.SHARED) and ".norms." not in name
                and name in original}
    if set(keys) != expected or any(not torch.equal(shared[name], original[name]) for name in keys):
        raise ValueError("Shared P4 tensor inventory differs from the common zero-update backbone")
    target.backbone.load_state_dict({**target_state, **{name: shared[name] for name in keys}}, strict=True)
    if any(not torch.equal(target.backbone.state_dict()[name], original[name]) for name in keys):
        raise ValueError("GroupNorm zero-update bridge changed shared source parameters")
    fresh = {f"backbone.{name}": tensor_hash(value) for name, value in target.backbone.state_dict().items()
             if name not in keys}
    fresh.update({name: tensor_hash(value) for name, value in target.state_dict().items()
                  if not name.startswith("backbone.")})
    authority.update(initialization="p4_exact_shared_into_wp07_group_norm", baseline_state=state_hash(target),
                     shared_keys=keys, shared_state=digest({n: tensor_hash(original[n]) for n in keys}),
                     fresh=fresh, target_seed=64)
    return authority, (weight, sidecar, result), target


def pavia_status(root):
    registration_pin, accepted_pin = [historical(root, key) for key in ("pavia", "pavia_acceptance")]
    registration, accepted = [_json(pin, root) for pin in (registration_pin, accepted_pin)]
    bands, physical = _band_metadata(root, "paviaU")
    if (accepted["status"] != "passed" or accepted["commit"] != registration["commit"]
            or accepted["raw_gt_loaded"] or accepted["query_accessed"] or accepted["exact_complete_state_pairs"] != 8):
        raise ValueError("PaviaU fixed spatial acceptance differs")
    case = wp02._only(registration["cases"], lambda row: (row["support_seed"], row["k"], row["method"]) ==
                     (512, 5, "shared_transfer"))
    plan_pin, stats_pin, labels_pin = [_pin(registration, path) for path in
                                       (registration["plan"], registration["statistics"], case["release"])]
    plan, stats, labels = [_json(pin, root) for pin in (plan_pin, stats_pin, labels_pin)]
    p4_spatial.load_release(labels, plan, 512, 5)
    source, files, _ = _spatial_source(root, registration, "paviaU", state_required=False)
    online = historical(root, "pavia_online")
    prior = _json(online, root)
    if (prior["scene"] != "paviaU" or prior["training_partition_tile_ids"] != [0, 1]
            or any(plan["tiles"][i]["partition"] != "train" for i in prior["training_partition_tile_ids"])
            or prior["query_accessed"] or prior["raw_labels_accessed"]):
        raise ValueError("PaviaU training-only packaged observations differ")
    staging = historical(root, "pavia_staging")
    staged = _json(staging, root)
    if (staged["scene"] != "paviaU" or staged["archives_persisted"]
            or staged["raw_labels_opened"] or staged["query_opened"]):
        raise ValueError("PaviaU staging scope differs")
    image = _pin(registration, registration["image"])
    image_path = inside(image["path"], root.parent / "datasets")
    with image_path.open("rb") as stream:
        header = stream.read(136)
    if header[124:136] != bytes.fromhex("0001494d0f000000fd1b1302"):
        raise ValueError("PaviaU original image format identity differs")
    _protect(root, (registration_pin, accepted_pin, plan_pin, stats_pin, labels_pin, online, staging, *files,
                    physical["metadata_receipt"]))
    return dict(status="BLOCKED", reason_code="DEVVAL_BOUNDED_ARRAY_NOT_PUBLISHED",
                missing="A hash-bound PaviaU fixed-role devval NPY or a seekable source for that released rectangle",
                profile="fixed_spatial", original_image=image, original_format="compressed_mat_v5",
                training_only_observation_receipt=online, training_only_tiles=[0, 1],
                nonpersistent_staging=staging, role_plan=plan_pin,
                label_release=labels_pin, training_statistics=stats_pin, statistics_hash=stats["statistics_hash"],
                source=source, physical=physical, bands=len(bands.band_ids), actual_forwards=0)


def honghu(root):
    registration_pin, split_pin, accepted_pin = [historical(root, key) for key in (
        "honghu", "honghu_split", "honghu_acceptance")]
    registration, split, accepted = [_json(pin, root) for pin in (registration_pin, split_pin, accepted_pin)]
    image = _pin(split, split["image"])
    original_image = _pin(registration, registration["image"])
    inside(original_image["path"], root.parent / "datasets")
    if (accepted["status"] != "passed" or accepted["tested_commit"] != registration["commit"]
            or accepted["raw_gt_loaded"] or accepted["query_accessed"] or accepted["case_pairs"] != 8
            or accepted["protected_files"].get(image["path"]) != image["sha256"]
            or accepted["protected_files"].get(split_pin["path"]) != split_pin["sha256"]
            or any(split[key] != registration[key] for key in ("plan", "statistics", "development", "source", "guarded", "cases"))):
        raise ValueError("HongHu guarded bounded observation is outside the accepted counterfactual")
    plan_pin, stats_pin, roles_pin = [_pin(registration, registration[key]) for key in (
        "plan", "statistics", "development")]
    evidence = [_pin(registration, path) for path in registration["guarded"].values()]
    _protect(root, evidence)
    plan, stats, geometry = [_json(pin, root) for pin in (plan_pin, stats_pin, roles_pin)]
    guarded_plan, allowed = p4_guarded.bind_inputs(registration, plan, geometry)
    case = wp02._only(registration["cases"], lambda row: (row["support_seed"], row["k"], row["method"]) ==
                     (512, 5, "shared_transfer"))
    labels_pin = _pin(registration, case["release"])
    labels = p4_spatial.load_release(_json(labels_pin, root), plan, 512, 5, geometry, guarded_tiles=allowed)
    bands, physical = _band_metadata(root, "whu-honghu")
    shape, classes = SCENES["whu-honghu"]
    if stats["development_roles_hash"] != guarded_plan["plan_hash"] or stats["fit_role"] != "devtrain":
        raise ValueError("HongHu training statistics escape the guarded protocol")
    special = {row["role"]: row for row in geometry["roles"] if "parent_tile" in row}
    tiles = [guarded_plan["tiles"][special[role]["id"]] for role in ("devtrain", "devval")]
    parents = {row["id"]: row["parent_tile"] for row in special.values()}
    rows = _rows(tiles, labels, shape, lambda _: image, parents)
    reader = open_guarded(image, root, "WHU_Hi_HongHu", shape, rows)
    source, files, baseline = _spatial_source(root, registration, "whu-honghu", state_required=True)
    custody = dict(acceptance=accepted_pin, registration=registration_pin, guarded_split=split_pin,
                   original_image=original_image, geometry=roles_pin, role_plan=plan_pin,
                   label_release=labels_pin, training_statistics=stats_pin, guarded_evidence=evidence,
                   minimum_buffer=plan["gap"], block_selection="both_preexisting_guarded_siblings")
    release = Release("whu-honghu", "guarded_development", shape, classes, guarded_plan["plan_hash"],
                      bands, stats, rows, custody, physical, source)
    return Admitted(release, baseline, reader, (image, registration_pin, split_pin, accepted_pin, plan_pin,
                    stats_pin, roles_pin, labels_pin, *evidence, *files, physical["metadata_receipt"]))
