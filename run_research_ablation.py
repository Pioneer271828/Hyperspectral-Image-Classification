"""Plan and execute C0 experiments using the unchanged legacy training algorithm."""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback

from model.research_config import ResearchConfig, digest
from research_ablation_presets import build_research_experiments
from research_runtime import (AlreadyCompleted, RunRecord, code_snapshot, external_directory, require_frozen_code, write_json)


def plan_experiments(configs, root, code, resume=False):
    root = external_directory(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": 1, "source_hash": code["source_hash"], "entries": [
        {"run_id": RunRecord(root, config, code).run_id, "config": config.to_dict(),
         "config_hash": config.config_hash, "budget": config.budget} for config in configs]}
    plan["plan_hash"] = digest(plan)
    path = root / "plan.json"
    # Exclusive creation also prevents racing planners from overwriting each other.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        write_json(root / "planning_code.json", code)
    except FileExistsError:
        if not resume or json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Output root has a different/existing plan; use a new root or --resume with the same plan")
    return plan


def execute_one(config, root, code, resume=False, trainer=None):
    if trainer is None:
        from train_vca_multidataset import train as trainer
    record = RunRecord(root, config, code, resume=resume)
    with record:
        # Logs close before artifact hashing in RunRecord.__exit__.
        with (record.attempt / "training.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    require_frozen_code(code)
                    try:
                        trainer(config.trainer_args(record.save_path, record.metrics_path), run_record=record)
                    finally:
                        require_frozen_code(code)
                except AlreadyCompleted:
                    print("Previous success verified; no optimizer steps executed")
                    raise
                except BaseException:
                    traceback.print_exc()
                    raise
    return record.manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--experiments", default="configured", help="configured, all, or comma-separated existing preset names")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; default is config seed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify success or create a new attempt after failure")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    config = ResearchConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    names = [s.strip() for s in args.experiments.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds is not None else [config.training.seed]
    configs = build_research_experiments(config, names, seeds)
    code = code_snapshot()
    if code["dirty"]:
        parser.error("Commit and synchronize current code before planning/running")
    plan = plan_experiments(configs, args.out_root, code, resume=args.resume)
    print(f"plan={plan['plan_hash']} runs={len(configs)} output={args.out_root}", flush=True)
    if args.dry_run:
        print("DRY_RUN: no datasets loaded, no model constructed, no training")
        return 0
    failures = 0
    for config in configs:
        try:
            result = execute_one(config, args.out_root, code, resume=args.resume)
            print(f"{result['run_id']}: {result['status']}", flush=True)
        except Exception as error:
            failures += 1
            print(f"{config.name}: failed: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
