"""Explicit M4 source v2 validation policy with fixed-order FP32 row reductions."""
import copy

import torch
from torch.overrides import TorchFunctionMode

import m4_source_study as legacy
from joint_training import validation_snapshot
from m4_transfer import source_options

POLICY = {"name": "fp32_sorted_row_segment_sum_v1", "scope": "validation_only",
          "rtol": 1e-5, "atol": 1e-6, "classification_changes": 0}
EXPECTED = {**legacy.EXPECTED, "protocol": "g4_m4_source_pretrain_full_v2",
            "validation_arithmetic": POLICY}
MODES = legacy.MODES


def ordered_mm(matrix, features):
    if (torch.is_grad_enabled() or matrix.layout != torch.sparse_coo
            or matrix.dtype != torch.float32 or features.dtype != matrix.dtype):
        raise ValueError("Ordered validation requires no-grad FP32 COO operands")
    csr = matrix.coalesce().to_sparse_csr()
    lengths = csr.crow_indices().diff()
    weights = csr.values()[:, None]
    blocks = []
    for channels in features.split(16, dim=1):
        messages = weights * channels.index_select(0, csr.col_indices())
        blocks.append(torch.segment_reduce(messages, "sum", lengths=lengths))
    return torch.cat(blocks, dim=1)


class OrderedValidation(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.sparse.mm:
            if kwargs:
                raise ValueError("Unexpected sparse multiplication arguments")
            return ordered_mm(*args)
        return func(*args, **(kwargs or {}))


def full_settings(held_out, seed, mode):
    settings = legacy.full_settings(held_out, seed, mode)
    settings.update(protocol=EXPECTED["protocol"], m4_source_study=copy.deepcopy(EXPECTED))
    return settings


def load_study():
    legacy.load_study()
    return copy.deepcopy(EXPECTED)


def short_settings(held_out, seed, mode):
    if mode not in MODES:
        raise ValueError("Registered active M4 source modes required")
    settings = source_options(held_out, mode, seed=seed)
    settings.update(protocol="g4_m4_source_ordered_engineering_v2",
                    source_validation_arithmetic=copy.deepcopy(POLICY))
    return settings


def select_case(launch, source, held_out, seed, mode):
    if launch["study"] != EXPECTED:
        raise ValueError("Ordered v2 study differs")
    translated = copy.deepcopy(launch)
    translated["study"] = copy.deepcopy(legacy.EXPECTED)
    for case in translated["cases"]:
        if case["settings"] != full_settings(case["held_out"], case["seed"], case["mode"]):
            raise ValueError("Ordered v2 case differs")
        case["settings"] = legacy.full_settings(case["held_out"], case["seed"], case["mode"])
    selected = legacy.select_case(translated, source, held_out, seed, mode)
    return next(case for case in launch["cases"] if case["name"] == selected["name"])


class M4SourceOrderedRuntime(legacy.M4SourceFullRuntime):
    def validate(self, settings):
        expected = full_settings(settings.get("held_out"), settings.get("model_seed"), settings.get("m4_mode"))
        actual = {k: v for k, v in settings.items() if k not in ("executed_optimizer_steps", "selected_step")}
        if actual != expected:
            raise ValueError("Ordered source v2 settings differ")
        return True

    @staticmethod
    def validation_snapshot(model, items):
        with OrderedValidation():
            return validation_snapshot(model, items)

    def checkpoint_metadata(self, model):
        return {"source_validation_arithmetic": copy.deepcopy(POLICY)}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        model = super().load_selected(path, items, raw_source_path, device=device)
        if model.training_config.get("source_validation_arithmetic") != POLICY:
            raise ValueError("Selected source validation arithmetic differs")
        return model


class M4SourceOrderedShortRuntime(M4SourceOrderedRuntime):
    def validate(self, settings):
        expected = short_settings(settings.get("held_out"), settings.get("model_seed"), settings.get("m4_mode"))
        actual = {k: v for k, v in settings.items() if k not in ("executed_optimizer_steps", "selected_step")}
        if actual != expected:
            raise ValueError("Ordered short engineering settings differ")
        return True

