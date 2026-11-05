"""Eight-update real source precheck for the explicit ordered validation v2."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint_baseline import NAMES
from joint_training import load_plan, run_training
from loadData.dataset_registry import file_hash
from m4_ordered_source import MODES, short_settings, M4SourceOrderedShortRuntime
from m4_transfer import graph_binding
from research_runtime import code_snapshot, require_frozen_code
from source_data_context import load_source_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("expected-commit", "prepared-dir", "data-root", "out-dir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--held-out", choices=NAMES, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote commit required")
    context = load_source_context(args.prepared_dir, args.data_root, held_out=args.held_out,
        consumer=code, manifest_sha256=load_plan("configs/experiments/base97_training.json")["prepared_manifest_sha256"])
    settings = short_settings(args.held_out, 64, args.mode)
    provenance = copy.deepcopy(context.record)
    provenance.update(m4_mode=args.mode, m4_graph_binding=graph_binding(context.items))
    def guard():
        require_frozen_code(code)
        if any(file_hash(p) != h for p, h in context.record["protected_files"].items()):
            raise ValueError("Source input changed")
    result = run_training(settings, context.items, args.out_dir, provenance, code,
        resume=args.resume, stop_after=args.stop_after, runtime_adapter=M4SourceOrderedShortRuntime(),
        finish_guard=guard)
    print(result["operation"], result["updates_this_call"], flush=True)


if __name__ == "__main__":
    main()

