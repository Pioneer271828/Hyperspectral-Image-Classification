"""Load published native, WP07, or original CPU endpoints by explicit pins."""

from copy import deepcopy

import torch

from loadData.spatial_tiles import tile_slice
from research_runtime import tensor_hash

from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.g3_training import state_hash
from transfer.native_support import NativeSupportModel, bank_digest, make_selector, native_forward


def load_endpoint(pin, registration, case, data, *, device="cpu"):
    if device != "cpu":
        raise ValueError("This experiment endpoint is validated for CPU FP32 only")
    from transfer.native_support_campaign import _load_pt
    value = _load_pt(pin)
    if (value["version"] != 2 or value["registration"] != registration
            or value["development"] != case["development"]):
        raise ValueError("Endpoint registration or development identity differs")
    baseline, scene, _, _, bank = data
    config = value["config"]
    if config["format_version"] == 2:
        expected = {"version": 2, "scorer_seed": case["scorer_seed"], "device": "cpu_fp32"}
        if config["regional_support"] != expected or bank_digest(bank) != value["bank"]:
            raise ValueError("Native endpoint scorer or bank identity differs")
        model = NativeSupportModel(config["class_counts"], scorer_seed=case["scorer_seed"], **config["backbone_kwargs"])
    elif value["work_package"] == "WP07":
        with torch.random.fork_rng(devices=[]):
            model = ConditionalMultiscaleVCASAHGCN(config["class_counts"], **config["backbone_kwargs"])
    else:
        model = deepcopy(baseline)
    if value["endpoint"] not in ("selected", "last") or not 1 <= value["step"] <= case["budget"]:
        raise ValueError("Invalid published endpoint identity")
    model.load_state_dict(value["state"], strict=True)
    model.training_config = deepcopy(value["training_config"])
    model.backbone.adapter.cache_attention = False
    if state_hash(model) != value["state_hash"] or model.export_config() != config:
        raise ValueError("Endpoint tensors or architecture differ")
    return model.eval(), value


def predict_registered_tile(model, value, scene, record, tile_id, bank, *, detail=False, timing=None):
    if type(tile_id) is not int or not 0 <= tile_id < len(record["roles"]) or record["roles"][tile_id]["role"] != "devval":
        raise ValueError("Published endpoint accepts registered development validation blocks only")
    tile = scene.plan["tiles"][tile_id]
    from transfer.g3_training import _forward
    if isinstance(model, NativeSupportModel):
        settings = value["trial"]
        entry = bank[str(tile_id)]
        block_hash = tensor_hash(torch.from_numpy(scene.cube[tile_slice(tile)].copy()))
        if (entry["identity"]["block_input"] != block_hash or entry["context"]["role"] != "devval"
                or entry["identity"]["admission"] != value["development"]["sha256"]):
            raise ValueError("Published endpoint block array, role or custody changed")
        selector = make_selector(entry, settings, settings["support_seed"])
        return native_forward(model, scene, tile, selector=selector, detail=detail, timing=timing)
    if isinstance(model, ConditionalMultiscaleVCASAHGCN):
        with model.physical_conditions(**(value["conditions"] or {"coverage_fraction": None, "gsd_m": None})):
            return _forward(model, scene, tile, multiscale_condition_only=True), {}
    return _forward(model, scene, tile), {}
