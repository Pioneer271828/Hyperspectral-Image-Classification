"""Seal and publish paired WP09 development evidence through the WP12 entry."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.classification_metrics import classification_metrics
from evaluation.online_tta import _labels as read_closed_labels, _runs, _score_verified_runs
from evaluation.open_set import OpenSetScorer, _array_hash, _calibration, _labels, _matrix, digest, evaluate
from evaluation.registered_open_set import HEADS, METHODS, _code, _load
from evaluate_tta import MODES
from research_runtime import external_directory, hash_file, require_frozen_code
from transfer.online_tta_stream import ORDERS, fields_are, optional_pin, parse_json, read_pin, validate_pin


KIND = "wp09_synthetic_paired_workflow_v1"


def _reserve(pin, roles):
    validate_pin(pin)
    current = Path(pin["path"])
    for role in roles:
        if current.is_relative_to(role.parent):
            raise ValueError("A deferred label directory cannot supply pre-scoring evidence")
        try:
            if current.samefile(role):
                raise ValueError("A deferred label file cannot alias pre-scoring evidence")
        except FileNotFoundError:
            pass  # The independent label release can still be withheld.


def _directory(path, roles, label_inodes):
    base = external_directory(path)
    if any(base.is_relative_to(role.parent) or role.parent.is_relative_to(base) for role in roles):
        raise ValueError("Deferred label custody must be disjoint from run and replay directories")
    for entry in base.rglob("*"):
        meta = entry.lstat()
        if entry.is_symlink() or (meta.st_dev,meta.st_ino) in label_inodes:
            raise ValueError("Run/replay directory contains a link or deferred label alias")


def _open_rows(frozen, v, e, source, calibration):
    """Validate committed source identities and frozen parameters without fitting."""
    known = [r["id"] for r in v["known"]]
    for pack in (source, calibration):
        if set(pack["labels"].tolist()) != set(known):
            raise ValueError("Every known class and only known classes required in source evidence")
    expected = [(h,m) for h in HEADS for m in METHODS]
    if [(r["head"],r["method"]) for r in frozen["rows"]] != expected:
        raise ValueError("Expected the frozen nine head/scorer combinations")
    for row in frozen["rows"]:
        fields_are(row, "head method input_kind array_key scorer scorer_sha256 calibration calibration_sha256")
        head, method = row["head"], row["method"]
        proto = method == METHODS[-1]
        hi = HEADS.index(head)
        kind = "features" if proto else "raw_logits"
        key = ("features_" if proto else "logits_")+head
        scorer = OpenSetScorer(row["scorer"], row["scorer_sha256"])
        spec = scorer.as_dict()
        cal = row["calibration"]
        _calibration(scorer, cal, row["calibration_sha256"])
        ids = calibration["sample_ids"].tolist()
        data = digest([_array_hash(_matrix(calibration[key])), _array_hash(_labels(calibration["labels"],len(ids))), ids])
        if (spec["method"] != method or spec["known"] != v["known"] or spec["unknown"] != v["unknown"]
                or spec["model_sha256"] != v["model"]["sha256"] or spec["representation"] != e["representations"][hi]
                or spec["temperature"] != (1 if proto else v["temperature"][hi])
                or row["input_kind"] != kind or row["array_key"] != key or cal["input_kind"] != kind
                or cal["role"] != "source_validation" or cal["known_coverage"] != v["known_coverage"]
                or cal["provenance_sha256"] != v["calibration"]["sha256"] or cal["sample_ids"] != ids
                or cal["data_sha256"] != data):
            raise ValueError("Open-set scorer or fixed source validation identity differs")
        if proto:
            ids = source["sample_ids"].tolist()
            data = digest([_array_hash(_matrix(source[key])), _array_hash(_labels(source["labels"],len(ids))),ids])
            if (spec["source_ids"] != ids or spec["source_data_sha256"] != data
                    or spec["source_provenance"] != v["source"]["sha256"]):
                raise ValueError("Prototype support is not the admitted known source split")
            values = _matrix(source[key])
            labels = _labels(source["labels"],len(ids))
            by_class = [values[labels==c] for c in known]
            if spec["prototype"] != [np.sum(x/len(x),axis=0).tolist() for x in by_class]:
                raise ValueError("Frozen prototype is not derived from the declared source split")
        scores,_ = scorer.score(calibration[key],input_kind=kind)
        coverage = cal["known_coverage"]
        if type(coverage) not in (int,float) or not 0 < coverage <= 1:
            raise ValueError("Source known coverage must be fixed before scoring")
        kth = math.ceil(coverage*len(scores))-1
        if (float(np.partition(scores,kth)[kth]) != cal["threshold"]
                or float(np.mean(scores<=cal["threshold"])) != cal["empirical_known_coverage"]):
            raise ValueError("Frozen threshold differs from the source-only order statistic")


def _verify(plan_pin, code, *, extra_pins=()):
    plan = parse_json(read_pin(plan_pin))
    fields_are(plan, "kind closed_plan open_freeze closed_labels closed_label_files open_labels")
    if plan["kind"] != KIND:
        raise ValueError("Only the explicit synthetic WP09 joint workflow is admitted")
    if not isinstance(plan["closed_label_files"],list) or not plan["closed_label_files"]:
        raise ValueError("Declare the deferred closed rasters before opening scoring inputs")
    deferred = (plan["closed_labels"],plan["open_labels"],*plan["closed_label_files"])
    for pin in deferred:
        validate_pin(pin)
    closed_path = Path(plan["closed_labels"]["path"])
    open_path = Path(plan["open_labels"]["path"])
    if (closed_path.parent.is_relative_to(open_path.parent) or open_path.parent.is_relative_to(closed_path.parent)
            or len({p["path"] for p in deferred}) != len(deferred)
            or any(not Path(p["path"]).is_relative_to(closed_path.parent) for p in plan["closed_label_files"])):
        raise ValueError("Declare separate closed/open custody and distinct closed rasters")
    roles = tuple(Path(p["path"]) for p in deferred)
    label_inodes = set()
    for role in roles:
        try:
            metadata = role.stat()
        except FileNotFoundError:
            continue
        label_inodes.add((metadata.st_dev,metadata.st_ino))
    for pin in (plan_pin,plan["closed_plan"],plan["open_freeze"],*extra_pins):
        _reserve(pin,roles)
    closed = parse_json(read_pin(plan["closed_plan"]))
    fields_are(closed, "version runs label_release")
    if closed["version"] != 1 or closed["label_release"] != plan["closed_labels"]:
        raise ValueError("Use a synthetic closed label release")
    frozen = parse_json(read_pin(plan["open_freeze"]))
    fields_are(frozen, "version code registration inputs rows development_ids deferred_labels "
                      "evaluation_labels_read query optimizer_updates producer_truth_verified freeze_hash")
    if (frozen["version"] != 1 or frozen["code"] != code or frozen["evaluation_labels_read"] is not False
            or frozen["query"] or frozen["optimizer_updates"] or frozen["producer_truth_verified"] is not False
            or frozen["freeze_hash"] != digest({k:v for k,v in frozen.items() if k!="freeze_hash"})):
        raise ValueError("Original clean-code, label-blind open-set freeze required")
    if frozen["deferred_labels"] != plan["open_labels"]:
        raise ValueError("Frozen open release differs from the deferred input")
    _reserve(frozen["registration"],roles)
    registration = parse_json(read_pin(frozen["registration"]))
    if registration.get("labels") != frozen["deferred_labels"]:
        raise ValueError("The deferred open label identity changed")
    for key in ("model","model_evidence","source","calibration","evaluation"):
        _reserve(registration[key], roles)
    for item in closed["runs"]:
        fields_are(item, "registration directory replay")
        for pin in (item["registration"],item["replay"]):
            _reserve(pin, roles)
        for path in (item["directory"],Path(item["replay"]["path"]).parent):
            _directory(path,roles,label_inodes)
        reg = parse_json(read_pin(item["registration"]))
        for pin in (reg["provenance"], *reg["source"].values(), *reg["observations"].values()):
            _reserve(pin, roles)
        provenance = parse_json(read_pin(reg["provenance"]))
        if (provenance.get("kind") != "synthetic" or reg["source"]["weights"] != registration["model"]
                or reg["class_ids"] != [r["id"] for r in registration["known"]]
                or (reg["commit"],reg["source_hash"]) != (code["commit"],code["source_hash"])):
            raise ValueError("The two analyses must bind the same synthetic source and ordered known classes")
    v,e,source,cal,target,pins = _load(frozen["registration"], source_validation_only=True)
    if pins != frozen["inputs"] or target["sample_ids"].tolist() != frozen["development_ids"]:
        raise ValueError("Frozen open inputs or evaluation IDs differ")
    _open_rows(frozen,v,e,source,cal)
    runs, proofs = _runs(closed, None, str(closed_path))
    resets = {key[1] for key in runs}
    required = {(mode,reset,order) for reset in resets for mode in MODES for order in ORDERS}
    if not {"scene","continual"} <= resets or set(runs) != required:
        raise ValueError("Each declared reset needs the four modes and three registered orders")
    return plan, closed, frozen, v, target, runs, proofs


def _publish_new(out, value):
    path = external_directory(out)
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent) as folder:
        incoming = Path(folder)/"report.json"
        with incoming.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(incoming, path)
    return {"path":str(path),"sha256":hash_file(path)}


def _seal_identity(plan_pin, plan, closed, frozen, proofs, code):
    return dict(kind=KIND, plan=plan_pin, input_code=code["commit"], input_source_hash=code["source_hash"],
                open_freeze=plan["open_freeze"], closed_runs=proofs,
                deferred=[closed["label_release"],frozen["deferred_labels"],*plan["closed_label_files"]],
                label_reads=0, query_reads=0, optimizer_updates=0)


def seal(plan_pin, out, expected_commit):
    code = _code(expected_commit)
    plan, closed, frozen, _, _, _, proofs = _verify(plan_pin,code)
    roles = [Path(x["path"]) for x in (closed["label_release"],frozen["deferred_labels"])]
    out = external_directory(out)
    if any(out.is_relative_to(r.parent) for r in roles):
        raise ValueError("Use an evidence destination outside label custody")
    require_frozen_code(code)
    return _publish_new(out,_seal_identity(plan_pin,plan,closed,frozen,proofs,code))


def _counts(reg, labels, classes):
    unique = {}
    for frame in reg.value["frames"]:
        key = reg.observation_identity(frame)
        value = labels[frame["frame_id"]]
        if key in unique and not np.array_equal(unique[key],value):
            raise ValueError("Repeated observation cannot change its offline ground truth")
        unique[key] = value
    all_labels = np.concatenate(list(labels.values()))
    different = np.concatenate(list(unique.values()))
    return dict(frame_exposures=len(labels), observed_hashes=len(unique),
                per_class=[dict(class_id=c, frame_support=int((all_labels==c).sum()),
                                distinct_observation_offset_support=int((different==c).sum())) for c in classes],
                excluded_zero_exposures=int((all_labels==0).sum()), excluded_zero_unique=int((different==0).sum()))


def publish(seal_pin, out, expected_commit):
    code = _code(expected_commit)
    saved = parse_json(read_pin(seal_pin))
    plan,closed,frozen,v,target,runs,proofs = _verify(saved["plan"],code,extra_pins=(seal_pin,))
    if saved != _seal_identity(saved["plan"],plan,closed,frozen,proofs,code):
        raise ValueError("Seal does not cover this exact completed workflow")
    out = external_directory(out)
    if any(out.is_relative_to(Path(r["path"]).parent) for r in saved["deferred"]):
        raise ValueError("Use an evidence destination outside label custody")
    if out.exists():
        previous = parse_json(out.read_bytes())
        if (previous.get("seal") != seal_pin or previous.get("digest") !=
                digest({k:v for k,v in previous.items() if k!="digest"})):
            raise ValueError("Existing publication belongs to a different seal or was altered")
        return {"path":str(out),"sha256":hash_file(out),"reopened":True,"label_reads":0,"optimizer_updates":0}
    reg = next(iter(runs.values()))[0]
    labels, small = read_closed_labels(closed["label_release"],reg,
                        reserved_directory=Path(closed["label_release"]["path"]).parent,
                        expected_files={tuple(sorted(p.items())) for p in plan["closed_label_files"]})
    classes = reg.value["class_ids"]
    closed_scores = _score_verified_runs(1,runs,reg,labels,small,"synthetic")
    packet = parse_json(read_pin(frozen["deferred_labels"]))
    fields_are(packet, "version role sample_ids labels")
    if (packet["version"] != 1 or packet["role"] != "development_evaluation"
            or packet["sample_ids"] != frozen["development_ids"]):
        raise ValueError("Independent open-set label role or sample order differs")
    truth = _labels(packet["labels"],len(packet["sample_ids"]))
    reports = {}
    for row in frozen["rows"]:
        scorer = OpenSetScorer(row["scorer"],row["scorer_sha256"])
        report = evaluate(scorer,row["calibration"],row["calibration_sha256"],target[row["array_key"]],
                          truth,packet["sample_ids"],input_kind=row["input_kind"])
        reports[row["head"]+"/"+row["method"]] = report
    known = np.isin(truth,classes)
    closed_controls = {h:classification_metrics(truth[known],
                       np.asarray(classes)[target["logits_"+h][known].argmax(1)],classes) for h in HEADS}
    result = dict(kind=KIND, seal=seal_pin, real_claim=False, closed_set=closed_scores,
                  closed_support=_counts(reg,labels,classes), open_set=reports,
                  open_known_closed_control=closed_controls,
                  fixed_source_calibration=True, calibration_updates=0, query_reads=0, optimizer_updates=0)
    result["digest"] = digest(result)
    require_frozen_code(code)
    return _publish_new(out,result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit",required=True)
    commands = parser.add_subparsers(dest="command",required=True)
    for action, name in (("seal","plan"),("publish","seal")):
        sub = commands.add_parser(action)
        sub.add_argument("--"+name,type=Path,required=True)
        sub.add_argument("--"+name+"-sha256",required=True)
        sub.add_argument("--out",type=Path,required=True)
    args = parser.parse_args(argv)
    name = "plan" if args.command == "seal" else "seal"
    fn = seal if name == "plan" else publish
    print(json.dumps(fn(optional_pin(getattr(args,name),getattr(args,name+"_sha256")),
                        args.out,args.expected_commit),sort_keys=True))
