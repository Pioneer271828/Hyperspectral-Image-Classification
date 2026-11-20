"""Replay pinned WP05/WP07/WP08 models through the admitted comparison protocol."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation import wp12_cross_backend as comparison
from evaluation.frozen_training_stages import require
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_native_released_points import arithmetic
from transfer.native_paired_head_fixture import VIEWS


def pin(pair):
    require(pair is not None, "Provide the input path and its original SHA-256")
    return dict(path=pair[0], sha256=pair[1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fixture", "capture", "review", "history"))
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--input", nargs=2, metavar=("PATH", "SHA256"))
    p.add_argument("--paviau-replay", nargs=3, metavar=("FAMILY", "PATH", "SHA256"), action="append")
    p.add_argument("--gpu")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--kind", choices=("xuzhou", "honghu"))
    p.add_argument("--layout", choices=("a", "b"), default="a")
    p.add_argument("--seed", type=int, action="append")
    p.add_argument("--full-only", action="store_true")
    p.add_argument("--cost-only", action="store_true")
    args = p.parse_args(argv)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    code = code_snapshot()
    require(code["commit"] == args.expected_commit and root in out.parents and not out.exists(),
            "Use a clean matching commit and a new result path under the authorized runs")
    require_frozen_code(code)
    torch.set_num_threads(1)
    require(args.paviau_replay is None or args.action == "history", "Only history can consume PaviaU archive replays")
    if args.action == "fixture":
        from evaluation.wp12_cross_backend_fixture import fixture

        require(args.synthetic and args.kind and not args.input and not args.gpu,
                "Fixture generation is explicitly synthetic")
        value = fixture(out, args.kind, args.layout, args.seed or [64,512],
                        views=("full",) if args.full_only else VIEWS, cost_only=args.cost_only)
    else:
        require(args.kind is args.seed is None and not args.full_only and not args.cost_only,
                "Fixture flags cannot change an already-authorized comparison")
        history = [] if args.synthetic else None
        if args.action == "capture":
            with arithmetic(root, args.gpu) as execution:
                value = comparison.run_case(root, pin(args.input), out, code, execution, test_history=history)
        elif args.action == "review":
            require(args.gpu is None, "Independent WP12 receipt review is CPU only")
            value = comparison.review_case(root, pin(args.input), out, code, test_history=history)
        else:
            from evaluation.wp12_cross_backend_history import review_history

            require(args.gpu is None and not args.synthetic and not args.input,
                    "Historical compatibility replay uses the original admitted archive protocols")
            links = args.paviau_replay or []
            require(len({item[0] for item in links})==len(links),"Repeated PaviaU replay family")
            value = review_history(root, out, code, paviau_replays={key:pin([path,sha]) for key,path,sha in links})
    require_frozen_code(code)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
