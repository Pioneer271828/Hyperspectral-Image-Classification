"""Independent verification inputs and opt-in source-bound audit backends."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.audit_oss_b_metrics import oracle as torchmetrics_oracle
from transfer.oss_a_dependency import compare_stage1


DEPGRAPH = "e80127d7c6935c319a7dd6719b8a72e3c1fcbff9"
HEADS = ("fused", "graph", "cnn")


def confusion(labels, heads, classes, *, role, active_heads, crosscheck=None):
    if role not in ("train", "val"):
        raise ValueError("Only authorized train and validation roles may be scored")
    if active_heads not in (HEADS, HEADS[:1]) or tuple(heads) != active_heads:
        raise ValueError("Missing, repeated or out-of-order heads")
    if type(classes) is not int or classes < 2 or labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("One-dimensional integer labels and fixed class count required")
    if bool(((labels < 1) | (labels > classes)).any()) or not len(labels):
        raise ValueError("Historical selected labels must be one-based and nonempty")
    result = {}
    target = labels.cpu() - 1
    for name, score in heads.items():
        if (score.shape != (len(labels), classes) or not score.is_floating_point()
                or not bool(torch.isfinite(score).all())):
            raise ValueError("Head scores do not match the frozen label roster")
        predicted = score.cpu().argmax(1)
        old = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(old, (target.numpy(), predicted.numpy()), 1)
        observed = torch.bincount(target*classes+predicted, minlength=classes**2).reshape(classes, classes)
        if observed.tolist() != old.tolist() or (
                crosscheck is not None and not torch.equal(crosscheck(predicted,target,classes), observed)):
            raise ValueError("Independent confusion calculations disagree")
        result[name] = dict(confusion=old.tolist(), empty_classes=(np.flatnonzero(old.sum(axis=1)==0)+1).tolist())
    return result


def validate_indices(indices, expected):
    if (indices.ndim != 1 or indices.dtype != torch.long or not torch.equal(indices, expected)
            or len(indices.unique()) != len(indices) or bool((indices < 0).any())):
        raise ValueError("Changed, missing, repeated or out-of-order historical row identities")


def depgraph_from(root):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != DEPGRAPH or git("status", "--porcelain"):
        raise ValueError("Clean fixed Torch-Pruning source required")
    if "torch_pruning" in sys.modules:
        raise ValueError("Use a fresh audit process for the fixed Torch-Pruning source")
    init = root/"torch_pruning/__init__.py"
    spec = importlib.util.spec_from_file_location("torch_pruning", init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["torch_pruning"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("torch_pruning", None)
        raise
    return module


def structural(model, witness, channels, *, upstream_root=None):
    if upstream_root is None:
        from compression.g4_cnn_stage1 import _mapped_geometry
        _, kept, axes = _mapped_geometry(model, channels)
        return dict(backend="project", retained=kept, axes=axes, upstream_verified=False)
    _, receipt = compare_stage1(model,witness,channels,depgraph=depgraph_from(upstream_root))
    return dict(backend="torch_pruning",upstream_verified=True,**receipt)


def qualification(component, *, test_execution_passed, failed_tensors=None, numeric_scope):
    failure = failed_tensors or {}
    qualified = "not_verified" if not test_execution_passed else ("failed" if failure else "passed")
    return dict(schema="oss-c-qualification-v1", component=component,
                test_execution_passed=bool(test_execution_passed), numerical_qualification=qualified,
                numeric_scope=numeric_scope, failed_tensors=failure, production_authorized=False)


def require_production(receipt):
    raise ValueError(f"Research qualification receipts cannot authorize production: {receipt.get('component','unknown')}")
