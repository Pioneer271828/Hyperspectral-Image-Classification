"""Rebind the original label-free registrations, then compare each candidate event."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from evaluate_tta import HEADS
from evaluation.wp09_upstream import tent_reference
from model.research_config import digest
from research_runtime import ROOT, code_snapshot, external_directory, hash_file, require_frozen_code, tensor_hash, write_json
from transfer.online_tta_run import committed_history, load_checkpoint, read_predictions, runtime_identity
from transfer.online_tta_stream import StreamRegistration, parse_json, read_pin
from transfer.wp09_tent import execute_tent


CAMPAIGN = "36b42c6377357f7b93057adcbc25dc6ce5335e938eebf8ace8da79df1c90688d"
EXECUTION = "0e8bf626f51293fef16551fc54390a4c3781177557555d60c1478bc1131a0441"
AUDIT = "8c4f945fbc637fd97aa92cdb45875b02d0aeeef95e0ef57216699f912a0e8154"
REAL_CODE = "34a5721c751ca22bccfacdf5adc3aec7e832a809"


def pin(path):
    return dict(path=str(path), sha256=hash_file(path))


def write(path, value):
    write_json(path, value)
    return pin(path)


def read(value):
    return parse_json(read_pin(value))


def assumptions(runs, code):
    original = runs / "wp09-online-tta-preparation01/registration/campaign.json"
    old_root = runs / "wp09-online-tta-real01"
    pins = [dict(path=str(path), sha256=checksum) for path, checksum in (
        (original, CAMPAIGN), (old_root / "execution.json", EXECUTION), (old_root / "audit/audit.json", AUDIT))]
    campaign, completed, audit = map(read, pins)
    if (campaign["commit"] != REAL_CODE or campaign["budget"] != dict(predictions=288, optimizer_updates=144, statistics_updates=72)
            or completed["campaign_sha256"] != CAMPAIGN or completed["actual"] != campaign["budget"]
            or audit["execution_sha256"] != EXECUTION or audit["raw_gt"] or audit["query"]):
        raise ValueError("Exact original label-free campaign required")
    subprocess.run(["git", "diff", "--exit-code", REAL_CODE, code["commit"], "--",
                    "evaluate_tta.py", "model", "model/checkpoint_io.py"], cwd=ROOT, check=True)
    pins += [campaign["provenance"], campaign["legacy"], *campaign["registrations"].values()]
    legacy = read(campaign["legacy"])
    pins += legacy["protected"] if isinstance(legacy["protected"], list) else list(legacy["protected"].values())
    for old in campaign["registrations"].values():
        reg = StreamRegistration(old)
        if (reg.value["runtime"] != runtime_identity() or reg.provenance["kind"] != "e4b_pavia_development"
                or reg.provenance["selected_step"] != 7 or len(reg.value["observations"]) != 2):
            raise ValueError("Original seventh-step PaviaU development scope required")
        pins += [*reg.value["source"].values(), *reg.value["observations"].values()]
    return campaign, old_root, list({p["path"]: p for p in pins}.values())


def arrays(a, b, name):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid {name} array")
    diff = np.abs(b.astype(np.float64)-a.astype(np.float64))
    bound = 1e-6+1e-5*np.abs(a.astype(np.float64))
    return dict(elements=a.size, exceeded=int((diff > bound).sum()), max_abs=float(diff.max()),
                max_ratio=float((diff/bound).max()))


def compare(name, old_path, new_path, old_pin, new_pin, provider):
    old, new = StreamRegistration(old_pin), StreamRegistration(new_pin)
    if provider:
        new.adaptation_provider = provider
    a, _ = committed_history(old, old_path, complete=True)
    b, _ = committed_history(new, new_path, complete=True)
    result = []
    for i, (before, after) in enumerate(zip(a, b, strict=True), 1):
        for field in ("frame", "reset", "cursor", "observation_sha256", "counts", "optimizer", "rng"):
            if before[field] != after[field]:
                raise ValueError(f"{name} changed {field}")
        dirs = [path / f"frame-{i:06d}" for path in (old_path, new_path)]
        logits = [read_predictions(path / "prediction.npz") for path in dirs]
        scores = {head: arrays(logits[0][head], logits[1][head], head) for head in HEADS}
        flips = {head: int((logits[0][head].argmax(1)!=logits[1][head].argmax(1)).sum()) for head in HEADS}
        if any(flips.values()) or any(v["exceeded"] for v in scores.values()):
            raise ValueError("Three-head pre-update equivalence failed")
        states = [load_checkpoint(path / "checkpoint.pt", event["checkpoint"])
                  for path, event in zip(dirs, (before, after), strict=True)]
        gradients = [torch.load(path / "gradients.pt", weights_only=True) if event["gradients"] else {}
                     for path, event in zip(dirs, (before, after), strict=True)]
        if gradients[0].keys() != gradients[1].keys():
            raise ValueError("Allowed gradients changed")
        grads = {}
        for key, g in gradients[0].items():
            h = gradients[1][key]
            if (g is None) != (h is None):
                raise ValueError("Gradient presence changed")
            if g is not None:
                grads[key] = arrays(g.numpy(), h.numpy(), key)
        values = {}
        for field in ("model", "adapter"):
            ref, got = (one[field] for one in states)
            if (ref is None) != (got is None) or (ref is not None and ref.keys() != got.keys()):
                raise ValueError("Restored parameter structure changed")
            for key, x in (ref or {}).items():
                ident = ("adapter." if field=="adapter" else "")+key
                y = got[key]
                if tensor_hash(x) != tensor_hash(y):
                    if ident not in (set(before["changed"]) | set(after["changed"])) and ident not in gradients[0]:
                        raise ValueError(f"Frozen parameter or statistic differs: {ident}")
                    values[ident] = arrays(x.numpy(), y.numpy(), ident)
        if any(v["exceeded"] for v in (*values.values(), *grads.values())):
            raise ValueError("State or gradient tolerance exceeded")
        result.append(dict(cursor=i, probabilities=scores, changes=flips, gradient=grads, state_differences=values,
                           actual_changed=after["changed"], native_changed=before["changed"], entropy_before=before["entropy"],
                           entropy_after=after["entropy"], pre_state=after["before"], post_state=after["after"]))
    return result


def worker(value):
    if value["provider"] is None:
        from transfer.online_tta_run import execute
        return execute(value["registration"], value["out"], to_frame=value["stop"], admission=value.get("admission"))
    return execute_tent(value["registration"], value["out"], value["provider"], to_frame=value["stop"],
                        admission=value.get("admission"))


def run_child(reg, dest, stop, tent, log, admission=None):
    args = dict(registration=reg, out=str(dest), stop=stop, provider=str(tent) if tent else None, admission=admission)
    start = perf_counter()
    proc = subprocess.run([sys.executable, "-m", "scripts.review_wp09_tent", "--worker", json.dumps(args)],
                          cwd=ROOT, capture_output=True, text=True)
    log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"Worker failed. Evidence at {log}")
    return dict(result=json.loads(proc.stdout), seconds=perf_counter()-start)


def real(runs, tent, out, code):
    c, old, protected = assumptions(runs, code)
    _, identity = tent_reference(tent)
    provider = digest(dict(identity, component="softmax_entropy", native_clamped_floor=1e-8))
    for item in protected:
        read_pin(item)
    out.mkdir()
    specs = {}
    for name, p in c["registrations"].items():
        spec = read(p)
        specs[name] = write(out / (name + ".json"), dict(spec, commit=code["commit"], source_hash=code["source_hash"]))
    write(out / "scope.json", dict(commit=code["commit"], registrations=specs, references=c["registrations"],
        start_utc=datetime.now(timezone.utc).isoformat(),
        original=CAMPAIGN, provider=identity, budget=c["budget"], estimated_minutes=[10,20], raw_gt=0, query=0))

    def one(name):
        reg = specs[name]
        full, split = out / (name+"-full"), out / (name+"-split")
        runs = [run_child(reg, dest, stop, tent, out / f"{name}-{tag}.log")
                for tag, dest, stop in (("full",full,None),("prefix",split,2),("resume",split,None),("complete",split,None))]
        view = StreamRegistration(reg)
        view.adaptation_provider = provider
        a, atip = committed_history(view, full, complete=True)
        b, btip = committed_history(view, split, complete=True)
        if a != b or atip != btip or any(runs[-1]["result"]["actual"].values()):
            raise ValueError("Continuous and split candidate evidence differs")
        check = compare(name, old / (name+"-full"), full, c["registrations"][name], reg, provider)
        return name, dict(executions=runs, event_comparison=check)

    start = perf_counter()
    rows = dict(one(name) for name in specs)
    actual = Counter()
    for row in rows.values():
        for stage in row["executions"]:
            actual.update(stage["result"]["actual"])
    if actual != c["budget"]:
        raise ValueError("Executed campaign escaped its registered budget")
    for item in protected:
        read_pin(item)
    require_frozen_code(code)
    write(out / "receipt.json", dict(status="passed", commit=code["commit"], protected=len(protected),
         actual=actual, seconds=perf_counter()-start, cases=rows, raw_gt=0, query=0,
         independent_optimizer_updates=0, independent_statistics_updates=0))


def main():
    if len(sys.argv)==3 and sys.argv[1]=="--worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
        return
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    code = code_snapshot()
    if code["commit"] != args.expected_commit:
        raise ValueError("Exact source required")
    require_frozen_code(code)
    if args.preflight_only:
        campaign, _, pins = assumptions(args.runs.resolve(), code)
        _, upstream = tent_reference(args.tent)
        for item in pins:
            read_pin(item)
        print(json.dumps(dict(cases=len(campaign["registrations"]), budgets=campaign["budget"],
                             protected=len(pins), provider=upstream)))
    else:
        real(args.runs.resolve(), args.tent.resolve(), external_directory(args.out), code)


if __name__ == "__main__":
    main()
