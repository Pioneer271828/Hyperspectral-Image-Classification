"""Supplemental OpenOOD audit for the authorized E4c development artifact chain."""

from collections import Counter
import json
from math import ceil, isclose
from pathlib import Path

import numpy as np

from evaluation.known_source_admission import _verified_source
from evaluation.known_development_authority import read_authority
from evaluation.open_set import OpenSetScorer, _labels, detection_metrics
from evaluation.open_set_producer import produce_open_set
from evaluation.registered_open_set import HEADS, _code, _deferred, _load, _rows, export_open_set, freeze_open_set
from evaluation.wp09_upstream import OpenOODReference
from model.research_config import digest
from research_runtime import external_directory, require_frozen_code
from transfer.registered_development import _json_new, _read


def prepare(authority, source_receipt, root, *, out, expected_commit, temperature, coverage):
    code = _code(expected_commit)
    scope, context, _ = read_authority(authority)
    identity, _, _, protected = _verified_source(authority, source_receipt)
    source = json.loads(_read(source_receipt))
    trial = json.loads(_read(source["registration"]))
    prepared = json.loads(_read(trial["prepared"]))
    if scope["scope"] != "synthetic":
        raise ValueError("This OpenOOD review is registered only on the authorized synthetic fixture")
    upstream = OpenOODReference(root)
    out = external_directory(out)
    if any(out == Path(p["path"]).parent or out in Path(p["path"]).parents for p in protected):
        raise ValueError("Output must be outside authorized inputs")
    out.mkdir()
    req_dir = out / "registration"
    req_dir.mkdir()
    label = context["labels"]
    registration = _json_new(req_dir / "produce.json", dict(version=1, kind="m8_open_set_development_production_v1",
        known=context["known"], unknown=context["unknown"], dataset=context["dataset"],
        checkpoint=identity["source"]["weights"], sidecar=identity["source"]["sidecar"],
        training_declaration=source["training_declaration"], observations=prepared["observations"],
        labels=label, temperature=temperature, known_coverage=coverage))
    producer = produce_open_set(registration, out=out / "production", expected_commit=code["commit"])
    production = json.loads(_read(producer))
    frozen = freeze_open_set(production["arrays_registration"], out=out / "frozen", expected_commit=code["commit"])
    f = json.loads(_read(frozen))
    v, ev, support, cal, target, pins = _load(production["arrays_registration"])
    score_receipts = []
    columns = [row["id"] for row in context["known"]]
    for row in f["rows"]:
        if row["method"] == "prototype_squared_distance":
            continue
        scorer = OpenSetScorer(row["scorer"], row["scorer_sha256"])
        key = row["array_key"]
        cal_ref, _ = upstream.compare_logits(scorer, cal[key], input_kind="raw_logits", column_ids=columns)
        score_ref, _ = upstream.compare_logits(scorer, target[key], input_kind="raw_logits", column_ids=columns)
        native, _ = scorer.score(target[key], input_kind="raw_logits")
        threshold = row["calibration"]["threshold"]
        if not np.array_equal(score_ref > threshold, native > threshold):
            raise ValueError("OpenOOD scores alter original reject-at-threshold decision")
        score_receipts.append(dict(head=row["head"], method=row["method"], temperature=scorer.as_dict()["temperature"],
            source_threshold=threshold, calibration_oracle_threshold=float(np.sort(cal_ref)[ceil(coverage*len(cal_ref))-1]),
            absolute_score_difference=float(np.abs(score_ref-native).max()), paired_decisions=len(native)))
    require_frozen_code(code)
    return _json_new(out / "approval.json", dict(authority=authority, source_receipt=source_receipt, producer=producer,
        frozen=frozen, released_labels=label, code=code, provider=upstream.evidence, columns=columns,
        comparison=score_receipts, label_reads=0, query=0, optimizer_updates=0))


def independent(scorer, row, labels, values):
    # Rebuild the three scoring rules independently from the frozen source prototypes and raw logits.
    spec = scorer.as_dict()
    known, unknown = ([item["id"] for item in spec[key]] for key in ("known", "unknown"))
    a = np.asarray(values, dtype=np.float64)
    if row["method"] == "prototype_squared_distance":
        distances = np.stack([((a-p)**2).sum(axis=1) for p in spec["prototype"]], axis=1)
        score, indices = distances.min(1), distances.argmin(1)
    else:
        t = spec["temperature"]
        log_norm = np.logaddexp.reduce(a/t, axis=1)
        score = 1-np.exp(a.max(1)/t-log_norm) if row["method"]=="msp" else -t*log_norm
        indices = a.argmax(1)
    native, _ = scorer.score(a, input_kind=row["input_kind"])
    if not np.allclose(score,native,atol=1e-6,rtol=1e-5):
        raise ValueError("Independent score differs")
    # Preserve the exact deployed score at ties so finite precision never changes the frozen comparator.
    keep = native <= row["calibration"]["threshold"]
    predicted = np.where(keep,np.asarray(known)[indices],-1)
    counts = Counter((y if y in known else -1,p) for y,p in zip(labels,predicted,strict=True) if y)
    known_recall, known_f1, urecall = {}, {}, {}
    for k in known:
        support = int((labels==k).sum())
        tp = counts[k,k]
        fp = sum(counts[z,k] for z in known+[-1] if z!=k)
        known_recall[k] = tp/support if support else None
        known_f1[k] = 2*tp/(support+tp+fp) if support else None
    for k in unknown:
        n = int((labels==k).sum())
        urecall[k] = dict(id=k,support=n,recall=float(np.mean(~keep[labels==k])) if n else None)
    macro = lambda vals: float(np.mean(list(vals))) if all(v is not None for v in vals) else None
    osstar, f1 = macro(known_recall.values()), macro(known_f1.values())
    umask = np.isin(labels,unknown)
    ur = float(np.mean(~keep[umask])) if umask.any() else None
    hos = None if osstar is None or ur is None else (2*osstar*ur/(osstar+ur) if osstar+ur else 0.)
    pos, neg = native[umask], native[np.isin(labels,known)]
    if len(pos) and len(neg):
        auc = sum(float((v>neg).sum()+.5*(v==neg).sum()) for v in pos)/(len(pos)*len(neg))
        fpr = float(np.mean(neg >= sorted(pos,reverse=True)[ceil(.95*len(pos))-1]))
    else:
        auc = fpr = None
    return dict(confusion=[[counts[i,j] for j in known+[-1]] for i in known+[-1]],
                unknown_per_class=list(urecall.values()), osstar_class_macro_recall=osstar, known_macro_f1=f1,
                hos=hos, auroc=auc, fpr95=fpr, unlabeled_excluded=int((labels==0).sum()))


def evaluate_approval(approval, authority, source_receipt, root, *, out, expected_commit):
    code = _code(expected_commit)
    a = json.loads(_read(approval))
    upstream = OpenOODReference(root)
    if (a["authority"] != authority or a["source_receipt"] != source_receipt or a["code"] != code
            or a["provider"] != upstream.evidence or a["label_reads"] != 0):
        raise ValueError("Pinned approval and producer identity differ")
    identity, _, _, protected = _verified_source(authority, source_receipt)
    _, context, _ = read_authority(authority)
    src = json.loads(_read(source_receipt))
    trial = json.loads(_read(src["registration"]))
    prepared = json.loads(_read(trial["prepared"]))
    producer = json.loads(_read(a["producer"]))
    request = json.loads(_read(producer["registration"]))
    f = json.loads(_read(a["frozen"]))
    v, evidence, support, cal, target, pins = _load(f["registration"])
    if (identity["scope"] != "synthetic" or producer["arrays_registration"] != f["registration"]
            or producer["code"] != code or producer["network_output_verified"] is not True
            or producer["evaluation_labels_read"] is not False or producer["query"] or producer["optimizer_updates"]
            or request["observations"] != prepared["observations"] or request["training_declaration"] != src["training_declaration"]
            or request["sidecar"] != identity["source"]["sidecar"] or request["checkpoint"] != identity["source"]["weights"]
            or request["temperature"] != v["temperature"] or request["known_coverage"] != v["known_coverage"]
            or v["model"] != identity["source"]["weights"] or v["labels"] != context["labels"]
            or v["labels"] != a["released_labels"] or v["known"] != context["known"] or v["unknown"] != context["unknown"]
            or producer["outputs"] != {k: v[k] for k in ("source", "calibration", "evaluation")}
            or a["columns"] != [x["id"] for x in v["known"]] or f["rows"] != _rows(v,evidence,support,cal)):
        raise ValueError("Frozen producer, columns, temperature, labels or scorer differs")
    _deferred(a["released_labels"])
    out = external_directory(out)
    # Original E4c routine owns pre-release validation, export, and failure custody.
    export = export_open_set(a["frozen"],out=out,expected_commit=code["commit"])
    release = json.loads(_read(a["released_labels"]))
    labels = _labels(release["labels"],len(target["sample_ids"]))
    result = []
    for row, report in zip(f["rows"],json.loads(_read(export))["reports"],strict=True):
        actual = json.loads(_read(report["report"]))
        scorer = OpenSetScorer(row["scorer"],row["scorer_sha256"])
        values = target[row["array_key"]]
        recomputed = independent(scorer,row,labels,values)
        for key,wanted in recomputed.items():
            got = actual["detection"][key] if key in ("auroc","fpr95") else actual[key]
            if got != wanted and (not isinstance(wanted,float) or got is None or not isclose(got,wanted,abs_tol=1e-14)):
                raise ValueError(f"Independent metric differs: {key}")
        entry = dict(head=row["head"],method=row["method"],metrics=recomputed)
        if row["method"] != "prototype_squared_distance":
            scores, predicted = upstream.compare_logits(scorer,values,input_kind="raw_logits",column_ids=a["columns"])
            spec = scorer.as_dict()
            positive = np.isin(labels,[x["id"] for x in spec["unknown"]])
            mask = labels!=0
            encoded = np.where(positive[mask],-1,labels[mask])
            if np.any(encoded==-1) and np.any(encoded!=-1):
                confidence = -scores[mask]
                auroc, ap_in, ap_out, fpr95 = upstream.metrics_module.auc_and_fpr_recall(confidence,encoded,.95)
                reverse, _ = upstream.metrics_module.fpr_recall(confidence,encoded,.95)
                legacy = detection_metrics(scores[mask],positive[mask])
                if not isclose(auroc,legacy["auroc"],abs_tol=1e-14) or not isclose(fpr95,legacy["fpr95"],abs_tol=1e-14):
                    raise ValueError("Aligned unknown-positive ROC differs")
                entry["openood"] = dict(auroc=float(auroc), fpr95_unknown_positive=float(fpr95),
                    fpr95_known_positive=float(reverse), aupr_in=float(ap_in), aupr_out=float(ap_out))
            else:
                entry["openood"] = None
        result.append(entry)
    for pin in protected:
        if pin != a["released_labels"]:
            _read(pin)
    require_frozen_code(code)
    return _json_new(out / "openood-audit.json", dict(status="passed", authority=authority,
         original_export=export, approvals=approval, methods=result, optimizer_updates=0, raw_gt=0, query=0))
