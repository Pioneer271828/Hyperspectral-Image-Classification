"""Run a pinned WP07 degradation and multiscale evidence campaign in the LXC."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from joint_training import _run_lock
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code
from transfer import development_bank as bank, development_bank_custody as custody
from transfer import development_bank_training as fixtures, registered_development as transport
from transfer import spatial_multiscale as multi, spatial_multiscale_training as training
from transfer.container_runtime import training_container_kind
from transfer.development_blocks import read_pin, scene_name
from transfer.xuzhou_multiscale_audit import _exact


BANK_PRODUCER = "0f4006e9b817fdcce50a86507e3c6d554b3f01f3"
FIXTURES = "b9646c08489d60f338f886342f78378a0f4237016e6f175b3105f7663e8c7945"
REAL = {
    "xuzhou": "ab332574cc68dc9407deaace246617dc746f8b38c019b74961a9aeae8df0bd3f",
    "whu-honghu": "244e378a41eeef7b7468a1ec39b0812d258dbc6c401bd4e31cedf2c67e01f64a",
    "paviaU": "8e899993313d293402b68f6fe2373bf766ec78d215f4a7acca9ff4533530a679",
}
NICK = {"xuzhou":"xuzhou", "whu-honghu":"honghu", "paviaU":"pavia"}


def read(pin, root):
    return json.loads(read_pin(pin, root))


def synthetic(root, pin, count):
    release, _ = fixtures.fixture(count)
    return bank.Bank(read(pin, root), root, release)


def register(root, out, code):
    old = dict(path=str(root / "wp02-bank-fixtures06/matrix.json"), sha256=FIXTURES)
    previous = read(old, root)
    out.mkdir()
    cases, physical = {}, {}
    for count in (9,22):
        opened = synthetic(root, previous["banks"][str(count)], count)
        for seed in training.SEEDS:
            for arm in multi.ARMS:
                name = f"{count}-{seed}-{arm}"
                cases[name] = transport._json_new(out / f"{name}.json", training.contract(opened, seed, arm, code))
        physical[str(count)] = multi.condition_audit(opened, fixtures._baseline(opened.release, 64), code)
    result = transport._json_new(out / "condition-audit.json", physical)
    return transport._json_new(out / "matrix.json", dict(code=code, cases=cases, banks=previous["banks"],
        original_fixture=old, conditions=result, engineering_maximum_updates=48*48, real_updates=0))


def _packet(pin, root):
    return torch.load(io.BytesIO(read_pin(pin, root)), map_location="cpu", weights_only=True)


def stage(root, matrix, case, phase, code):
    reg = read(matrix, root)
    if reg["code"] != code or case not in reg["cases"]:
        raise ValueError("Spatial stage is outside the preregistered matrix")
    spec = read(reg["cases"][case], root)
    opened = synthetic(root, reg["banks"][case.split("-")[0]], int(case.split("-")[0]))
    out = Path(matrix["path"]).parent / case
    out.mkdir(exist_ok=True)
    with _run_lock(out):
        previous = {"resume":"prefix", "completed":"resume"}.get(phase)
        checkpoint = None
        if previous:
            parent = read(transport._pin(out / f"{previous}.json"), root)
            if parent["stage"] != previous or parent["registration"] != reg["cases"][case]:
                raise ValueError("Spatial stage predecessor differs")
            checkpoint = _packet(parent["checkpoint"], out)
        result = training.execute(opened, spec, phase, out / f"{phase}.pt", code, checkpoint)
        result.update(case=case, registration=reg["cases"][case], checkpoint=transport._pin(result["checkpoint"]))
        return transport._json_new(out / f"{phase}.json", result)


def audit(root, matrix, out, code):
    reg = read(matrix, root)
    if reg["code"] != code or len(reg["cases"]) != 48:
        raise ValueError("Wrong spatial audit registration")
    result = {}
    for case, pin in reg["cases"].items():
        spec = read(pin, root)
        folder = Path(matrix["path"]).parent / case
        summaries = [read(transport._pin(folder / f"{phase}.json"), root) for phase in
                     ("full","prefix","resume","completed")]
        if any(row["case"] != case or row["registration"] != pin for row in summaries):
            raise ValueError("Spatial stage audit identity differs")
        full, prefix, resumed, complete = [_packet(row["checkpoint"], folder) for row in summaries]
        if (not _exact(full,resumed) or not _exact(full,complete)
                or not _exact(full["history"][:7],prefix["history"])
                or not _exact(full["validation"][:2],prefix["validation"])
                or [row["updates"] for row in summaries] != [24,7,17,0]):
            raise ValueError(f"Exact spatial continuation failed for {case}")
        opened = synthetic(root, reg["banks"][case.split("-")[0]], int(case.split("-")[0]))
        first = min(full["validation"],key=lambda row: row["fused_nll"])
        if full["selected"]["step"] != first["step"]:
            raise ValueError("Spatial first-minimum selection differs")
        endpoints = {}
        for name,state,event in (("selected",full["selected"]["state"],first),
                                 ("last",full["model"],full["validation"][-1])):
            run = training.Run(opened,spec,code)
            run.model.load_state_dict(state,strict=True)
            run._evaluate(event["step"])
            if not _exact(run.validation[0],event):
                raise ValueError("Independent endpoint reconstruction differs")
            endpoints[name] = event
        result[case] = dict(exact=True, stages=summaries, history=full["history"], validation=full["validation"],
                            endpoints=endpoints)
    return transport._json_new(out,dict(code=code,matrix=matrix,cases=result,synthetic_updates=2304,real_updates=0))


def real_scan(root, scene, out, code):
    original = dict(path=str(root / f"wp02-bank-review-fix01/{NICK[scene]}-replay06.json"), sha256=REAL[scene])
    previous = read(original, root)
    if previous["producer"] != BANK_PRODUCER or previous["scene"] != scene or previous["real_updates"]:
        raise ValueError("Original bank receipt differs")
    intake = custody.PROVIDERS[scene](root)
    opened = bank.Bank(previous["bank"], root, intake.release)
    started = perf_counter()
    result = multi.scan(opened, intake.baseline, previous, code)
    opened.verify_authority(previous["bank"])
    if custody.protect(intake) < 1 or opened.release.coverage() != previous["coverage"]:
        raise ValueError("Original observations or coverage differ")
    return transport._json_new(out,dict(scene=scene,bank=opened.pin,bank_producer=BANK_PRODUCER,code=code,prior=original,
        coverage=previous["coverage"],legacy_heads=previous["previous_probability_hashes"]+previous["previous_pavia_heads"],
        elapsed=perf_counter()-started,**result))


def supplement(root, scans, audit_pin, out, code):
    from transfer.development_block_custody import historical, _protect
    old = read(historical(root,"wp12"),root)
    tables = {**old["preserved"],**old["tables"]}
    _protect(root,tables.values())
    rows = [read(pin,root) for pin in scans]
    synthetic = read(audit_pin,root)
    if (len(tables)!=10 or set(row["scene"] for row in rows)!=set(REAL) or len(rows)!=3
            or any(row["code"]!=code or any(row[key] for key in ("real_updates","raw_gt","query","new_labels"))
                   for row in rows) or synthetic["code"]!=code or synthetic["synthetic_updates"]!=2304):
        raise ValueError("Spatial scope or historical evidence differs")
    out.mkdir()
    with (out / "spatial-scope.csv").open("x",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene","scope","blocks","views","prediction_forwards","synthetic_updates","real_updates"))
        for row in rows:
            writer.writerow((row["scene"],"frozen_observation",row["blocks"],len(row["rows"]),sum(row["counts"].values()),0,0))
        writer.writerow(("9/22 classes","synthetic_recovery",16,"",0,2304,0))
    return transport._json_new(out / "supplement.json",dict(code=code,scans=scans,synthetic=audit_pin,old_tables=tables,
        old_models=65,old_heads=193,new_real_models=0,real_updates=0,scope=transport._pin(out/"spatial-scope.csv")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--runs",required=True,type=Path)
    modes = p.add_subparsers(dest="command",required=True)
    for name in ("register","stage","audit","scan","supplement"):
        sub = modes.add_parser(name)
        if name != "stage":
            sub.add_argument("--out",required=True,type=Path)
        if name in ("stage","audit"):
            sub.add_argument("--matrix",required=True,nargs=2)
        if name == "stage":
            sub.add_argument("--case",required=True)
            sub.add_argument("--phase",required=True,choices=("full","prefix","resume","completed"))
        if name == "scan":
            sub.add_argument("--scene",required=True)
        if name == "supplement":
            sub.add_argument("--scan",action="append",nargs=2,required=True)
            sub.add_argument("--audit",nargs=2,required=True)
    args = p.parse_args()
    if not training_container_kind() or not __debug__:
        p.error("Authorized LXC with assertions required")
    torch.set_num_threads(1)
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        p.error("Exact source SHA required")
    require_frozen_code(code)
    root = external_directory(args.runs.resolve(strict=True))
    out = external_directory(args.out) if hasattr(args,"out") else None
    if root.name != "runs" or (out is not None and (out==root or not out.is_relative_to(root))):
        p.error("Require authorized runs root and bounded output")
    sha = code["commit"]
    pin = lambda pair: dict(zip(("path","sha256"),pair,strict=True))
    if args.command == "register":
        result = register(root,out,sha)
    elif args.command == "stage":
        result = stage(root,pin(args.matrix),args.case,args.phase,sha)
    elif args.command == "audit":
        result = audit(root,pin(args.matrix),out,sha)
    elif args.command == "scan":
        result = real_scan(root,scene_name(args.scene),out,sha)
    else:
        result = supplement(root,[pin(a) for a in args.scan],pin(args.audit),out,sha)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
