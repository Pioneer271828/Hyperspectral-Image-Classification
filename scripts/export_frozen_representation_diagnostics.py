"""Explicit post-analysis all54-boundary feature diagnostics; no label reader."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.frozen_representation_capture import export_batch, export_case
from research_runtime import code_snapshot, require_frozen_code


def repair(argv):
    from evaluation.wp12_representation import publish
    from research_runtime import external_directory
    from transfer.representation_campaign import execute, register
    parser = argparse.ArgumentParser(description="Registered synthetic repair and WP12 frozen localization")
    parser.add_argument("--repair-action",required=True,choices=("register","run","wp12"))
    parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--runs",required=True,type=Path)
    parser.add_argument("--directory",required=True,type=Path)
    parser.add_argument("--case",default="all")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--registration-sha256")
    parser.add_argument("--real",nargs=2,metavar=("PATH","SHA256"))
    args = parser.parse_args(argv)
    code = code_snapshot()
    if sys.platform!="linux" or not __debug__ or code["dirty"] or code["commit"]!=args.expected_commit:
        raise RuntimeError("Clean expected Linux representation checkout required")
    require_frozen_code(code)
    root,place = external_directory(args.runs),external_directory(args.directory)
    if root not in place.parents or not root.is_dir():
        raise ValueError("Registered run directory must be under the authorized runs")
    if args.repair_action=="register":
        print(json.dumps(register(place,code)),flush=True)
    elif args.repair_action=="run":
        execute(place,code,args.case)
    else:
        if args.out is None or args.real is None or args.registration_sha256 is None:
            parser.error("WP12 requires --out, --real PIN and --registration-sha256")
        out = external_directory(args.out)
        bridge = Path(args.real[0]).resolve(strict=True)
        if root not in out.parents or root not in bridge.parents:
            raise ValueError("WP12 output and historical proof must stay in the authorized run directory")
        print(json.dumps(publish(place,out,code,args.registration_sha256,
                                  {"path":str(bridge),"sha256":args.real[1]})),flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(value=="--repair-action" or value.startswith("--repair-action=") for value in argv):
        repair(argv)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "freeze", "freeze-sha256", "analysis", "analysis-sha256", "out"):
        parser.add_argument("--" + key, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--case")
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-regions", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variance-floor", required=True, type=float)
    args = parser.parse_args(argv); code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected Linux frozen capture checkout required")
    require_frozen_code(code)
    run = export_batch if args.all else export_case
    names = () if args.all else (args.case,)
    result = run(args.freeze, args.freeze_sha256, args.analysis, args.analysis_sha256, *names,
        args.out, code, device=args.device, maximum_regions=args.maximum_regions,
        seed=args.seed, variance_floor=args.variance_floor)
    identity = {"model_forwards": result["model_forwards"]} if args.all else {"case": result["case"]}
    print(json.dumps({**identity, "status": result["status"], "out": args.out}))


if __name__ == "__main__":
    main()
