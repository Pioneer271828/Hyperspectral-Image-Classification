"""Ingest frozen54 accuracy and pinned hardware evidence without rescoring."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp12_evidence import accuracy_records, join_hardware, legacy_cpu_cost
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.summarize_unified_cohort import read_pinned


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--positive-control-case" in argv or "--positive-control-publish" in argv:
        from evaluation.wp12_positive_cli import main as positive_control

        return positive_control(argv)
    if "--native-pathways-review" in argv:
        return native_pathways(argv)
    if "--real-attribution-review" in argv:
        return real_attribution(argv)
    if "--external-paired-history" in argv:
        return external_paired_history(argv)
    if "--v1-spectral-audit" in argv:
        return v1_spectral_audit(argv)
    if "--long-stage-case" in argv or "--long-stage-publish" in argv:
        from evaluation.wp12_long_cli import main as long_coverage

        return long_coverage(argv)
    if "--historical-stage-run" in argv:
        from evaluation.frozen_stage_cli import historical_cli

        return historical_cli(argv)
    if "--frozen-stages" in argv:
        from evaluation.frozen_stage_cli import publish_cli

        return publish_cli(argv)
    if "--native-real" in argv:
        native_real(argv)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux evidence checkout required")
    out = external_directory(args.out)
    registration = read_pinned(args.registration, args.registration_sha256)
    if (not isinstance(registration, dict) or set(registration) != {"accuracy", "legacy_cpu_cost", "hardware"}
            or set(registration["accuracy"]) != {"analysis", "receipt", "freeze"}):
        raise ValueError("Explicit accuracy and optional cost evidence groups required")
    pins = {}

    def read(spec):
        if set(spec) != {"path", "sha256"} or not Path(spec["path"]).is_absolute():
            raise ValueError("Explicit absolute evidence path and SHA-256 required")
        path = str(Path(spec["path"]).resolve(strict=True))
        value = read_pinned(path, spec["sha256"])
        if pins.setdefault(path, spec["sha256"]) != spec["sha256"]:
            raise ValueError("Conflicting pins for the same artifact")
        return value

    source = registration["accuracy"]
    accuracy = accuracy_records(read(source["analysis"]), read(source["receipt"]), read(source["freeze"]),
                                analysis_sha=source["analysis"]["sha256"], freeze_sha=source["freeze"]["sha256"])
    diagnostic, combined = None, None
    legacy = registration["legacy_cpu_cost"]
    if legacy is not None:
        if set(legacy) != {"plan", "receipt", "execution"}:
            raise ValueError("Legacy benchmark requires plan, receipt and execution evidence")
        diagnostic = legacy_cpu_cost(read(legacy["plan"]), read(legacy["receipt"]), read(legacy["execution"]),
                                     plan_sha=legacy["plan"]["sha256"], receipt_sha=legacy["receipt"]["sha256"])
        measured = {row["checkpoint_sha256"] for row in diagnostic["rows"]}
        diagnostic["matching_frozen_cases"] = sum(row["checkpoint_sha256"] in measured for row in accuracy["cases"])
    hardware = registration["hardware"]
    if hardware is not None:
        if set(hardware) != {"plan", "receipt", "baseline"}:
            raise ValueError("Hardware join requires a registered plan, receipt and baseline")
        combined = join_hardware(accuracy, read(hardware["plan"]), read(hardware["receipt"]),
                                 cost_plan_sha=hardware["plan"]["sha256"], baseline=hardware["baseline"])
    report = dict(status="combined" if combined is not None else "accuracy_ready_hardware_pending",
                  registration_sha256=args.registration_sha256, inputs=pins, consumer_commit=code["commit"],
                  accuracy=accuracy, legacy_cpu_cost=diagnostic, combined=combined)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    require_frozen_code(code)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def native_pathways(argv):
    from evaluation.frozen_training_stages import require
    from evaluation.wp12_pathways_review import review

    parser = argparse.ArgumentParser(description="Verify native path measurements and released role receipts")
    parser.add_argument("--native-pathways-review",action="store_true",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    names = ("real_root","branch_root","controls_cpu","controls_gpu","gpu_real","external","readouts")
    for name in names:
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args = parser.parse_args(argv)
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    code = code_snapshot()
    sources = {name:getattr(args,name).resolve(strict=True) for name in names}
    require(root in out.parents and code["commit"]==args.expected_commit and not out.exists()
            and all(root in p.parents for p in sources.values()),"Inputs or output escaped the authorized runs")
    require_frozen_code(code)
    result = review(root,out,code,real_root=sources["real_root"],branch_root=sources["branch_root"],
                    cpu_path=sources["controls_cpu"],gpu_path=sources["controls_gpu"],gpu_real=sources["gpu_real"],
                    external_path=sources["external"],readout_path=sources["readouts"])
    require_frozen_code(code)
    print(json.dumps(result),flush=True)


def real_attribution(argv):
    from evaluation.frozen_training_stages import require
    from evaluation.wp12_attribution_review import review
    from evaluation.wp12_real_attribution import SHARD_GROUPS

    parser = argparse.ArgumentParser(description="Independently verify frozen development attribution")
    parser.add_argument("--real-attribution-review",action="store_true",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--source-shard",nargs=3,action="append",required=True,metavar=("GROUP","DIRECTORY","SHA"))
    parser.add_argument("--floor",nargs=2,required=True,metavar=("FILE","SHA"))
    parser.add_argument("--locked-rivals",nargs=2,required=True,metavar=("FILE","SHA"))
    args = parser.parse_args(argv)
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and code["commit"]==args.expected_commit and not out.exists(),
            "Use the exact source commit and a fresh external result")
    require_frozen_code(code)
    sources = {group:Path(directory).resolve(strict=True) for group,directory,_ in args.source_shard}
    producers = {group:commit for group,_,commit in args.source_shard}
    require(set(sources)==set(SHARD_GROUPS) and len(args.source_shard)==len(SHARD_GROUPS)
            and all(root in folder.parents for folder in sources.values()),"Each released group needs its exact pinned source")
    floor_pin = dict(zip(("path","sha256"),args.floor,strict=True))
    rival_pin = dict(zip(("path","sha256"),args.locked_rivals,strict=True))
    result = review(root,out,code,sources,producers,floor_pin,rival_pin)
    require_frozen_code(code)
    print(json.dumps(result),flush=True)


def external_paired_history(argv):
    from evaluation.frozen_training_stages import require
    from evaluation.wp12_paired_history import index, review_attachment
    from scripts.run_frozen_band_region import _write

    parser = argparse.ArgumentParser(description="Index historical paired development evidence without inference")
    parser.add_argument("--external-paired-history",action="store_true",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--external-review",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and args.expected_commit==code["commit"],
            "Use the exact clean source checkout and authorized output root")
    require_frozen_code(code)
    row = index(root)
    if args.external_review:
        link = dict(zip(("path","sha256"),args.external_review,strict=True))
        row["external_paired_review"] = review_attachment(root,link,code["commit"])
    row["producer"] = code["commit"]
    require_frozen_code(code)
    out.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(_write(out,row)),flush=True)


def v1_spectral_audit(argv):
    import torch

    from evaluation.wp06_observation_audit import run

    parser = argparse.ArgumentParser(description="Read only the fixed v1 synthetic trajectory archive")
    parser.add_argument("--v1-spectral-audit",action="store_true",required=True)
    parser.add_argument("--expected-commit",required=True)
    for key in ("runs","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("freeze","review"):
        parser.add_argument("--"+key,nargs=2,required=True,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    source = code_snapshot()
    require_frozen_code(source)
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    if sys.platform != "linux" or not __debug__ or source["commit"] != args.expected_commit or root not in out.parents:
        raise ValueError("Clean expected LXC checkout and external output required")
    if out.exists():
        raise ValueError("Use a new retrospective receipt path")
    torch.set_num_threads(1)
    pins = [dict(zip(("path","sha256"),row,strict=True)) for row in (args.freeze,args.review)]
    print(json.dumps(run(root,out,source,*pins)),flush=True)


def native_real(argv):
    from evaluation.wp12_release_results import publish

    parser = argparse.ArgumentParser(description="Publish admitted real native and frozen54 results")
    parser.add_argument("--native-real", action="store_true", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux evidence checkout required")
    root, out = external_directory(args.runs), external_directory(args.out)
    if not root.is_dir() or root not in out.parents:
        raise ValueError("Real evidence source and output must be in the authorized runs")
    print(json.dumps(publish(root,out,code)),flush=True)


if __name__ == "__main__":
    main()
