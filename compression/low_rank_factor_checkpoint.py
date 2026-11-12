"""Authenticate the exact factor calibration or integer deployment before reload."""

from copy import deepcopy
import hashlib
import io
from pathlib import Path

import torch

from compression.int8_checkpoint import _check_state
from compression.low_rank_factor_precision import template, validate
from compression.precision_cost import _state_identity
from transfer.curriculum_training import CurriculumRun


FORMAT = "vca-low-rank-factor-int8-v1"


def save(model, path):
    plan = validate(model)
    state = model.state_dict()
    _check_state(state)
    packet = dict(format=FORMAT, plan=plan, torch_version=str(torch.__version__),
                  trainable={k: p.requires_grad for k, p in model.named_parameters()},
                  training_config=deepcopy(model.training_config), state=state, state_identity=_state_identity(model))
    return CurriculumRun._write(packet, path)


def load(source, path, expected_sha256, *, expected_contract_hash):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Factor precision checkpoint hash mismatch")
    packet = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (set(packet) != {"format", "plan", "torch_version", "trainable", "training_config", "state", "state_identity"}
            or packet["format"] != FORMAT or packet["torch_version"] != str(torch.__version__)
            or packet["plan"]["contract_hash"] != expected_contract_hash):
        raise ValueError("Factor precision checkpoint format, contract or Torch version differs")
    plan = packet["plan"]
    _check_state(packet["state"])
    model = template(source, plan["contract"], plan["phase"])
    model.load_state_dict(packet["state"], strict=True)
    model._factor_precision = deepcopy(plan)
    params = dict(model.named_parameters())
    if (params.keys() != packet["trainable"].keys()
            or any(type(v) is not bool for v in packet["trainable"].values())):
        raise ValueError("Factor precision trainability differs")
    for name, parameter in params.items():
        parameter.requires_grad_(packet["trainable"][name])
    model.training_config = deepcopy(packet["training_config"])
    model.eval()
    validate(model)
    if _state_identity(model) != packet["state_identity"]:
        raise ValueError("Restored factor precision state differs")
    return model
