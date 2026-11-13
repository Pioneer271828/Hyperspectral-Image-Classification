"""Drive the fixed five-arm sparse released-point comparison."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_runtime import code_snapshot, external_directory
from scripts import run_native_released_points as cli
from transfer import native_released_campaign as campaign, native_released_capacity as capacity
from transfer import registered_development as dev
from transfer.native_scale_campaign import read


def pin(pair):
    return {"path": str(Path(pair[0]).resolve(strict=True)), "sha256": pair[1]} if pair else None


def main(argv=None, *, intake=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    modes = parser.add_subparsers(dest="action", required=True)
    control = modes.add_parser("capacity")
    control.add_argument("--classes", type=int, choices=(9, 22), required=True)
    control.add_argument("--seed", type=int, choices=(64, 512, 2048), required=True)
    control.add_argument("--factor", type=int, choices=(2, 4), required=True)
    control.add_argument("--intervention", choices=(*capacity.CONDITIONS, capacity.CONSTANT), required=True)
    control.add_argument("--out", type=Path, required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--preflight", nargs=2, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--execution", type=Path, required=True)
    for name in ("run", "audit"):
        action = modes.add_parser(name)
        action.add_argument("--matrix", nargs=2, required=True)
        action.add_argument("--release", nargs=2)
        if name == "audit":
            action.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root, code = args.runs.resolve(strict=True), code_snapshot()
    campaign.require(code["commit"] == args.expected_commit, "Unexpected comparison source commit")
    if args.action == "prepare":
        out = external_directory(args.out)
        result = campaign.prepare(root, out, external_directory(args.execution),
                                  pin(args.preflight), code, engineering=intake is not None)
        dev._json_new(out / "separate-capacity-draft.json", capacity.draft(read(result, root)))
    elif args.action == "run":
        result = campaign.run(root, pin(args.matrix), pin(args.release), code, intake=intake, cli=cli)
    elif args.action == "capacity":
        from transfer import native_released_training as data, native_scale_training as training
        out = external_directory(args.out)
        campaign.require(root in out.parents and not out.exists(), "Use a fresh external synthetic control directory")
        arm = capacity.CONSTANT if args.intervention == capacity.CONSTANT else "joint"
        spec = capacity.experiment(args.classes, args.seed, arm, args.factor, args.intervention, code["commit"])
        out.mkdir()
        contract = dev._json_new(out / "registration.json", spec)
        stage = training.execute(spec, "full", out / "model.pt", code["commit"], released=data)
        result = dev._json_new(out / "receipt.json", {**stage, "registration": contract,
                               "checkpoint": dev._pin(out / "model.pt"), "synthetic": True, "real_updates": 0})
    else:
        out = external_directory(args.out)
        campaign.require(root in out.parents, "Audit must stay within the authorized results root")
        result = campaign.audit(root, pin(args.matrix), pin(args.release), out, code, intake=intake, cli=cli)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
