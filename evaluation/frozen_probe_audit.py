"""Independent authorized feature, selected probability and class-metric checks."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.classification_metrics import metrics_from_confusion
from evaluation import frozen_probe_registry as registry
from evaluation.linear_probe import fingerprint, prepare
from g4_support_context import check_files
from model.research_config import digest
from research_runtime import tensor_hash


FEATURE_RTOL, FEATURE_ATOL = 1e-5, 1e-6
PROB_RTOL, PROB_ATOL = 1e-12, 1e-12
ROLES = ("train", "val")


def sparse_rows(q, regions, ids):
    q = q.to_sparse_coo().coalesce()
    rows, columns = q.indices().numpy()
    slots = np.full(q.shape[0], -1, dtype=np.int64)
    slots[ids] = np.arange(len(ids))
    take = slots[rows] >= 0
    out = np.zeros((len(ids), regions.shape[1]), dtype=regions.dtype)
    np.add.at(out, slots[rows[take]], q.values().numpy()[take, None]*regions[columns[take]])
    return out


def audit_pack(pack, bound, scope, case, local):
    """Reconstruct selected CNN rows and Q broadcast independently of the pack writer."""
    path, receipt, record, family, phases, overlay, selected, _ = bound
    registry.require(set(pack) == {"features", "labels", "row_ids", "identity"}
            and set(pack["features"]) == set(registry.LAYERS)
            and set(pack["labels"]) == set(pack["row_ids"]) == set(ROLES)
            and pack["identity"]["policy"] == "authenticated_full54_train_val_probe_pack_v1"
            and pack["identity"]["query_labels_accessed"] is False
            and pack["identity"]["label_authorization"] == "accepted_case_family_K10_V5_phases"
            and (pack["identity"]["case"], pack["identity"]["selected_model_sha256"], pack["identity"]["selected_step"],
                 pack["identity"]["family_hash"], pack["identity"]["phase_hashes"],
                 pack["identity"]["capture_receipt_sha256"], pack["identity"]["freeze_sha256"],
                 pack["identity"]["analysis_sha256"]) == (case["name"], selected["primary_sha256"],
                 selected["selected_step"], family["family_hash"], case["phase_hashes"], case["capture_receipt_sha256"],
                 scope["freeze_sha256"], scope["analysis_sha256"]),
            "Stored pack does not bind the actual authorized historical model and releases")
    classes = scope["probe_plan"]["classes"]
    support = {}
    for phase, role, count in (("adapt", "train", 10), ("validation", "val", 5)):
        ids, labels = pack["row_ids"][role], pack["labels"][role]
        registry.require(ids.dtype == labels.dtype == torch.int64
            and ids.tolist() == phases[phase]["indices"] and labels.tolist() == [i-1 for i in phases[phase]["labels"]]
            and torch.equal(ids, overlay[role+"_indices"]) and torch.equal(labels, overlay[role+"_labels"]),
            "Stored pack has changed, reordered or swapped authorized labels/rows")
        support[role] = torch.bincount(labels, minlength=classes).tolist()
        registry.require(support[role] == [count]*classes, "Full scene class support required")
    check_files(local)
    cells, maximum = 0, 0.0
    for stage, width in (("stage1", 128), ("stage2", 64)):
        values = torch.load(path.parent / f"{stage}-features.pt", map_location="cpu", weights_only=True)
        registry.require(set(values) == {"cnn", "graph", "Q"}, "Exact historical stage archive required")
        q, cnn, graph = (values[k] for k in ("Q", "cnn", "graph"))
        registry.require(q.shape[0] == math.prod(family["shape"]) and tensor_hash(q) == record["prepared_fields"]["Q"]
            and cnn.shape == (q.shape[0], width) and graph.shape == (q.shape[1], width)
            and cnn.dtype == graph.dtype == q.dtype, "Stage tensor/Q identity differs in independent audit")
        for role in ROLES:
            ids = pack["row_ids"][role].numpy()
            g = graph.numpy()
            expected = {"cnn": cnn.numpy()[ids],
                        "graph": q.numpy()[ids] @ g if q.layout == torch.strided else sparse_rows(q, g, ids)}
            for branch, reference in expected.items():
                value = pack["features"][f"{stage}_{branch}"][role].numpy()
                registry.require(value.shape == reference.shape and value.dtype == reference.dtype
                    and (np.array_equal(value, reference) if branch == "cnn" else
                         np.allclose(value, reference, rtol=FEATURE_RTOL, atol=FEATURE_ATOL)),
                    f"Authorized stage features differ at {stage}/{branch}/{role}")
                cells += value.size
                maximum = max(maximum, float(np.max(np.abs(value-reference))))
    check_files(local)
    return {"feature_cells_checked": cells, "maximum_feature_absolute_error": maximum,
            "authorized_train_support": support["train"], "authorized_val_support": support["val"]}


def endpoint_logp(pack, plan, state):
    """Serialize the unchanged PyTorch probe decision reference for independent audit."""
    layers, _, _ = prepare(pack, plan)
    return {name: {role: F.log_softmax(rows[role] @ best["weight"].T + best["bias"], dim=1)
                   for role in ROLES} for name, rows in layers.items()
            for best in (state["layers"][name]["best"],)}


def numpy_logp(x, weight, bias):
    logits = x @ weight.T + bias
    shifted = logits-logits.max(1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(1, keepdims=True))


def class_metrics(logp, labels):
    classes = logp.shape[1]
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (labels, logp.argmax(1)), 1)
    metric = metrics_from_confusion(cm, range(1, classes+1))
    return {"confusion": cm.tolist(), "oa": metric["oa"], "aa": metric["aa"], "kappa": metric["kappa"],
            "macro_f1": metric["macro_f1"], "recall": [item["recall"] for item in metric["per_class"]],
            "f1": [item["f1"] for item in metric["per_class"]], "support": cm.sum(1).tolist(),
            "zero_recall_classes": [item["class_id"] for item in metric["per_class"] if item["recall"] == 0],
            "nll": float(-logp[np.arange(len(labels)), labels].mean())}


def comparison(reference, other):
    left, right = reference.argmax(1), other.argmax(1)
    delta = np.abs(np.exp(reference)-np.exp(other))
    diff = np.flatnonzero(left != right)
    best = np.sort(reference, axis=1)[:, -2:]
    return {"strict_decisions": not len(diff), "probabilities_close":
            bool(np.allclose(np.exp(reference), np.exp(other), rtol=PROB_RTOL, atol=PROB_ATOL)),
            "maximum_absolute_probability_error": float(delta.max()), "decision_disagreements": len(diff),
            "different_rows": [{"ordinal": int(i), "torch_class": int(left[i])+1, "numpy_class": int(right[i])+1,
                                "torch_log_probability_margin": float(best[i, 1]-best[i, 0])} for i in diff]}


def audit_probes(pack, scope, state, report, recorded):
    plan = scope["probe_plan"]
    layers, labels, metadata = prepare(pack, plan)
    expected_identity = digest({"policy": plan["policy"], "plan": plan, "input": fingerprint(pack)})
    registry.require(state["identity"] == expected_identity and state["cursor"] == plan["steps"]
        and state["checksum"] == fingerprint({k: v for k, v in state.items() if k != "checksum"})
        and set(layers) == set(state["layers"]) == set(recorded) == set(report["layers"]) == set(registry.LAYERS),
        "Complete probe state and identity required for independent review")
    result, failures = {}, []
    for name, layer in state["layers"].items():
        history, best = layer["history"], layer["best"]
        registry.require([row["step"] for row in history] == list(range(1, plan["steps"]+1))
            and all(math.isfinite(row["val_nll"]) and math.isfinite(row["train_objective_before_update"])
                    and math.isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0 for row in history),
            "Invalid probe history sequence")
        selected = min(history, key=lambda row: row["val_nll"])
        saved = report["layers"][name]
        columns = scope["columns"][name.split("_")[0]]
        registry.require(best["step"] == selected["step"] == saved["selected_step"]
            and best["nll"] == selected["val_nll"] == saved["metrics"]["val"]["nll"]
            and saved["normalization"]["parameters"] == plan["classes"]*(plan["width"]+1)
            and saved["normalization"]["columns"] == metadata[name]["columns"] == columns,
            "Probe width/columns or first-minimum selected step differs")
        raw = {role: pack["features"][name][role].numpy().astype(np.float64)[:, columns] for role in ROLES}
        mean, sd = raw["train"].mean(0), raw["train"].std(0)
        scale = np.where(sd > plan["scale_floor"], sd, 1)
        registry.require(np.allclose(mean, saved["normalization"]["mean"], rtol=PROB_RTOL, atol=PROB_ATOL)
            and np.allclose(scale, saved["normalization"]["scale"], rtol=PROB_RTOL, atol=PROB_ATOL),
            "Independent train-only centering or scale differs")
        result[name] = {"selected_step": best["step"], "selected_at_budget_end": best["step"] == plan["steps"],
                        "parameters": saved["normalization"]["parameters"],
                        "near_constant_columns": saved["normalization"]["near_constant_columns"], "roles": {}}
        for role in ROLES:
            x, y = layers[name][role], labels[role].numpy()
            authority = F.log_softmax(x @ best["weight"].T + best["bias"], dim=1)
            registry.require(torch.equal(authority, recorded[name][role]), "Saved authoritative probabilities differ")
            authority = authority.numpy()
            alternate = numpy_logp((raw[role]-mean)/scale, best["weight"].numpy(), best["bias"].numpy())
            cmp = comparison(authority, alternate)
            observed = class_metrics(authority, y)
            independent = class_metrics(alternate, y)
            old = saved["metrics"][role]
            registry.require(all(observed[key] == old[key] for key in ("oa", "confusion", "recall", "support"))
                and math.isclose(observed["aa"], old["aa"], rel_tol=0, abs_tol=PROB_ATOL)
                and math.isclose(observed["nll"], old["nll"], rel_tol=0, abs_tol=PROB_ATOL),
                "Independent metric counts from saved decision reference differ")
            if not (cmp["strict_decisions"] and cmp["probabilities_close"]):
                failures.append({"layer": name, "role": role, **cmp})
            terminal = F.log_softmax(x @ layer["weight"].T + layer["bias"], dim=1).numpy()
            result[name]["roles"][role] = {"selected": observed, "independent_selected": independent,
                "terminal": class_metrics(terminal, y), "cross_backend": cmp,
                "roundoff_sensitive_rows": old["roundoff_sensitive_rows"], "exact_tie_rows": old["exact_tie_rows"]}
    return {"layers": result, "numerical_failures": failures, "independent_selected_probability_roles": 8,
            "history_selection_roles": 4, "nonselected_event_probability_replays": 0,
            "probability_rtol": PROB_RTOL, "probability_atol": PROB_ATOL}
