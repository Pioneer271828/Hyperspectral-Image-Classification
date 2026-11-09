"""CPU probes of explicitly pinned train/val packs; no dataset or query loader.

File pins authenticate supplied bytes, not the upstream model/split legitimacy.
Real use additionally requires the study's accepted frozen capture boundary.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.linear_probe import advance
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json


def run(pack, pack_sha256, plan, plan_sha256, out, code, *, resume=None, resume_sha256=None, stop_after=None):
    started = time.perf_counter()
    require_frozen_code(code)
    out = external_directory(out)
    if out.exists(): raise FileExistsError("Probe output must be a fresh external directory")
    pins = {str(Path(pack).resolve()):pack_sha256, str(Path(plan).resolve()):plan_sha256}
    if (resume is None) != (resume_sha256 is None): raise ValueError("Paired resume path and digest required")
    if resume is not None: pins[str(Path(resume).resolve())] = resume_sha256
    if len(pins) != (3 if resume is not None else 2): raise ValueError("Distinct input files required")
    for path, sha in pins.items():
        source = Path(path)
        if (out == source or out in source.parents or source.parent == out or source.parent in out.parents):
            raise ValueError("Outputs must be outside every protected input directory")
        if not isinstance(sha,str) or len(sha) != 64 or hash_file(source) != sha:
            raise ValueError("Probe input SHA256 mismatch")
    settings = json.loads(Path(plan).read_text())
    features = torch.load(pack, map_location="cpu", weights_only=True)
    previous = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    state, report = advance(features, settings, state=previous, stop_after=stop_after)
    require_frozen_code(code)
    if any(hash_file(path) != sha for path,sha in pins.items()): raise ValueError("Probe inputs changed")
    out.mkdir(parents=True, exist_ok=False)
    torch.save(state, out / "state.pt")
    write_json(out / "report.json", report)
    receipt = {"status":report["status"], "code":code, "protected_files":pins,
        "outputs":{n:hash_file(out/n) for n in ("state.pt","report.json")},
        "elapsed_seconds":time.perf_counter()-started, "probe_updates_this_call":report["probe_updates_this_call"],
        "backbone_updates":0, "query_labels_accessed":False, "input_files_unchanged":True,
        "upstream_authorization_independently_verified":False,
        "scope":"Pinned caller-supplied train/val feature engineering; not real study acceptance"}
    write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "pack", "pack-sha256", "plan", "plan-sha256", "out"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--resume"); parser.add_argument("--resume-sha256")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(argv); code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux probe checkout required")
    result = run(args.pack,args.pack_sha256,args.plan,args.plan_sha256,args.out,code,
                 resume=args.resume,resume_sha256=args.resume_sha256,stop_after=args.stop_after)
    print(json.dumps({k:result[k] for k in ("status","probe_updates_this_call","elapsed_seconds")}))


if __name__ == "__main__": main()
