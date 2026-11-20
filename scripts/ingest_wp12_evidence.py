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
    if "--paired-structure-review" in argv:
        from scripts.run_wp_paired_structure import main as paired_structure

        return paired_structure(["review",*(arg for arg in argv if arg!="--paired-structure-review")])
    if "--cross-backend-review" in argv:
        return cross_backend_review(argv)
    if "--shared-edge-interaction-review" in argv:
        return shared_edge_interaction_review(argv)
    if "--native-pathways-review" in argv:
        return native_pathways(argv)
    if "--real-attribution-review" in argv:
        return real_attribution(argv)
    if "--real-readiness-review" in argv:
        return real_readiness(argv)
    if "--m1-e1-history" in argv:
        return m1_e1_history(argv)
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
    if "--classification-cost-review" in argv:
        return classification_cost_review(argv)
    if "--wp10-deployment-review" in argv:
        from evaluation.wp12_wp10_deployment import main as wp10_deployment

        return wp10_deployment([arg for arg in argv if arg != "--wp10-deployment-review"])
    if "--wp09-review" in argv:
        from evaluation.wp09_stream_review import main as wp09_review

        return wp09_review([arg for arg in argv if arg != "--wp09-review"])
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


def cross_backend_review(argv):
    from evaluation.frozen_training_stages import require
    from evaluation.wp12_cross_backend import review_case
    from scripts.run_native_released_points import arithmetic

    parser = argparse.ArgumentParser(description="Independently review paired native, scale and message models")
    parser.add_argument("--cross-backend-review", action="store_true", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--capture", nargs=2, metavar=("PATH", "SHA256"), required=True)
    parser.add_argument("--reference-capture", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and code["commit"] == args.expected_commit and not out.exists(),
            "Expected clean review checkout and fresh authorized result path")
    require_frozen_code(code)
    reference = dict(zip(("path","sha256"),args.reference_capture)) if args.reference_capture else None
    with arithmetic(root,None):
        result = review_case(root, dict(path=args.capture[0], sha256=args.capture[1]), out, code,
                             test_history=[] if args.synthetic else None, reference=reference)
    print(json.dumps(result), flush=True)


def shared_edge_interaction_review(argv):
    from evaluation import wp12_shared_edge_interaction as shared
    from evaluation.frozen_training_stages import json_pin, require
    from scripts.run_frozen_band_region import _write

    parser = argparse.ArgumentParser(description="Recompute frozen same-mask band and edge contrasts")
    parser.add_argument("--shared-edge-interaction-review", action="store_true", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--cpu", action="append", nargs=3, required=True, metavar=("CASE", "PATH", "SHA256"))
    parser.add_argument("--cuda", action="append", nargs=3, default=[], metavar=("CASE", "PATH", "SHA256"))
    args = parser.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    code = code_snapshot()
    require(sys.platform == "linux" and root in out.parents and not out.exists()
            and code["commit"] == args.expected_commit, "Expected a clean review checkout and fresh authorized result")
    require_frozen_code(code)

    def open_pins(items, device):
        pins, sources = {}, {}
        for case, path, sha in items:
            require(case in shared.CASES and case not in pins, "Expected distinct registered focal cases")
            pins[case] = dict(path=path, sha256=sha)
            sources[case] = json_pin(root, pins[case])
            require(sources[case]["case"] == case and sources[case]["device"] == device,
                    "Receipt device or case is foreign")
        return pins, sources

    cpu_pins, cpu = open_pins(args.cpu, "cpu")
    cuda_pins, cuda = open_pins(args.cuda, "cuda")
    require(cuda.keys() <= cpu.keys(), "Every GPU receipt needs its exact CPU baseline")
    parity = {}
    for case, row in cuda.items():
        require(row["reference"] == [cpu_pins[case]["path"], cpu_pins[case]["sha256"]],
                "The GPU receipt did not reuse the pinned CPU plan")
        parity[case] = shared.device_parity(cpu[case], row, code["commit"])
    groups = {case:shared.reduction(row, code["commit"]) for case,row in cpu.items()}
    require_frozen_code(code)
    result = _write(out, dict(policy=shared.SCHEMA, producer=code["commit"], cpu=cpu_pins, cuda=cuda_pins,
                             results=groups, device_parity=parity, real_optimizer_updates=0,
                             raw_gt_loaded=False, query_accessed=False, independent_confirmation=False))
    print(json.dumps(result), flush=True)


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


def real_readiness(argv):
    import re
    import subprocess

    from evaluation.frozen_training_stages import require
    from evaluation.wp12_real_readiness import index
    from research_runtime import write_json

    parser = argparse.ArgumentParser(description="Verify released real readiness and equal-capacity evidence")
    parser.add_argument("--real-readiness-review", action="store_true", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--capacity-case", action="append", nargs=3, required=True,
                        metavar=("CASE", "PATH", "SHA256"))
    args = parser.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    incoming = root.parent / "datasets" / "incoming"
    code = code_snapshot()
    require(sys.platform == "linux" and code["commit"] == args.expected_commit and root in out.parents
            and incoming.resolve(strict=True) == incoming and incoming.is_dir()
            and re.fullmatch("[0-9a-f]{40}", args.producer) and not out.exists(),
            "Use the expected Linux checkout, the registered runs and intake roots, and a fresh output")
    require_frozen_code(code)
    subprocess.run(["git", "merge-base", "--is-ancestor", args.producer, code["commit"]],
                   cwd=Path(__file__).resolve().parents[1], check=True)
    pins = {case: dict(path=path, sha256=sha) for case, path, sha in args.capacity_case}
    require(len(pins) == len(args.capacity_case), "Repeated capacity endpoint")
    result = index(root, incoming, pins, args.producer)
    require_frozen_code(code)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(**result, reviewer=code["commit"], capacity_producer=args.producer))
    print(out, flush=True)


def m1_e1_history(argv):
    from evaluation.frozen_training_stages import require
    from evaluation.wp12_m1_e1_history import index
    from scripts.run_frozen_band_region import _write

    parser = argparse.ArgumentParser(description="Read frozen M1/E1 integer counts, budgets and protocol provenance")
    parser.add_argument("--m1-e1-history",action="store_true",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    code = code_snapshot()
    require(sys.platform=="linux" and code["commit"]==args.expected_commit and root in out.parents
            and not out.exists(),"Historical intake needs the expected Linux checkout and a fresh authorized output")
    require_frozen_code(code)
    result = index(root)
    require_frozen_code(code)
    print(json.dumps(_write(out,dict(**result,producer=code["commit"]))),flush=True)


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


def classification_cost_review(argv):
    from evaluation.wp12_classification_review import publish, require

    parser = argparse.ArgumentParser(description="Review protocol-scoped WP12 accuracy and cost archives")
    parser.add_argument("--classification-cost-review",action="store_true",required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--combined-acceptance",nargs=2,metavar=("PATH","SHA256"))
    parser.add_argument("--support-review",nargs=2,action="append",default=[],metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    code = code_snapshot()
    require_frozen_code(code)
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    require(sys.platform=="linux" and code["commit"]==args.expected_commit
            and root in out.parents and not out.exists() and out.parent.is_dir(),
            "Use a clean expected Linux checkout and a fresh output below the authorized runs root")
    accepted = dict(zip(("path","sha256"),args.combined_acceptance,strict=True)) if args.combined_acceptance else None
    supports = [dict(path=path,sha256=sha) for path,sha in args.support_review]
    print(json.dumps(publish(root,out,code,accepted,supports),ensure_ascii=False),flush=True)


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
