"""Bounded ordered engineering consumer of accepted sanitized support/K contexts.

This is a twelve-update engineering profile, not a full experiment registration.
The context catalogue stays execution-disabled and never releases query labels.
"""
import argparse
import copy
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from g4_support_context import POLICY as CONTEXT_POLICY, check_files, context_name, load_context, read
from joint_baseline import NAMES, _item_fingerprint
from joint_training import run_training
from loadData.dataset_registry import DATASETS, file_hash
from m4_full_target_runtime import FullSourceTargetRuntime, TARGET_ARITHMETIC
from m4_transfer import MODES, target_options
from model.research_config import digest
from none_full_target_runtime import NoneFullSourceTargetRuntime
from research_runtime import code_snapshot, external_directory, require_frozen_code
from target_initialization import METHODS

POLICY = "g4_support_target_ordered_engineering_v1"


def settings(target, method, mode, *, model_seed, support_seed, per_class, binding, source_spec=None, device="cpu"):
    if method not in METHODS[:2] or mode not in MODES:
        raise ValueError("Only bounded scratch/full-source finetune engineering roles allowed")
    value = target_options(target, method, mode, seed=model_seed, support_seed=support_seed,
                           per_class=per_class, device=device)
    context = binding["context"]
    if (set(binding) != {"context", "item_sha256"} or context["policy"] != CONTEXT_POLICY
            or context["context"] != context_name(target, support_seed, per_class)
            or any(context[k] != v for k, v in {"support_seed": support_seed, "support_per_class": per_class,
                "validation_per_class": 5, "model_seed": model_seed, "head_seed": model_seed + 1000,
                "total_annotation_labels": DATASETS[target][2] * (per_class + 5),
                "full_target_scene_transductive": True, "query_labels_accessed": False,
                "execution_enabled": False, "optimizer_updates": 0}.items())):
        raise ValueError("Authenticated engineering-only context and explicit seed roles required")
    import re
    if any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) for h in
           (binding["item_sha256"], context["manifest_sha256"], context["family_hash"], context["graph_identity"])):
        raise ValueError("Pinned context, family, graph and item identities required")
    if (method == METHODS[0]) != (not source_spec):
        raise ValueError("Scratch/full-source role differs")
    value.update(protocol=POLICY, support_context_binding=copy.deepcopy(binding),
        source_spec_sha256=digest(source_spec or {}),
        execution_policy={"phase": "engineering_only", "maximum_optimizer_updates": 12,
                          "query_labels_accessed": False, "full_experiment": False},
        full_source_initialization="g4_none_full_source_binding_v2" if mode == "none" else "g4_m4_full_source_binding_v2",
        target_validation_arithmetic=copy.deepcopy(TARGET_ARITHMETIC))
    return value


class _BoundContext:
    def validate(self, options):
        mode = options["config"]["model"]["region_residual_mode"]
        if (mode == "none") != isinstance(self, NoneFullSourceTargetRuntime):
            raise ValueError("Adapter class and explicit architecture mode differ")
        expected = settings(options["target"], options["method"], options["config"]["model"]["region_residual_mode"],
            model_seed=options["model_seed"], support_seed=options["support_seed"],
            per_class=options["support_per_class"], binding=options["support_context_binding"],
            source_spec=self.source_spec, device=options["config"]["device"])
        if {k: v for k, v in options.items() if k not in ("executed_optimizer_steps", "selected_step")} != expected:
            raise ValueError("Bounded support target profile/configuration differs")
        return True

    def validate_inputs(self, items):
        super().validate_inputs(items)
        if digest(_item_fingerprint(items[self.settings["target"]])) != self.settings["support_context_binding"]["item_sha256"]:
            raise ValueError("Bound support labels/observation differ")

    def checkpoint_metadata(self, model):
        return {**super().checkpoint_metadata(model),
                "support_context_binding": copy.deepcopy(self.settings["support_context_binding"])}

    def load_selected(self, path, items, raw_source_path, *, device="cpu"):
        self.validate_inputs(items)
        model = super().load_selected(path, items, raw_source_path, device=device)
        if model.training_config.get("support_context_binding") != self.settings["support_context_binding"]:
            raise ValueError("Selected support/K context binding differs")
        return model


class NoneSupportTargetRuntime(_BoundContext, NoneFullSourceTargetRuntime):
    pass


class ActiveSupportTargetRuntime(_BoundContext, FullSourceTargetRuntime):
    pass


def source_pins(spec):
    pins = {}
    for key, path in spec.items():
        if key.endswith("_sha256"):
            continue
        hash_key = {"source_path": "source_weight_sha256"}.get(key, key + "_sha256")
        if hash_key not in spec:
            raise ValueError("Every full-source file needs an explicit hash")
        pins[str(Path(path).resolve(strict=True))] = spec[hash_key]
        if key == "source_path":
            pins[str(Path(str(path) + ".json").resolve(strict=True))] = spec["source_sidecar_sha256"]
    check_files(pins)
    return pins


def prepare(manifest, manifest_sha256, out, *, target, method, mode, model_seed, support_seed,
            per_class, source_spec=None, source_files=None, device="cpu"):
    """Authenticate data/code before returning the bounded adapter and immutable guard."""
    source = code_snapshot()
    require_frozen_code(source)
    items, context, guard = load_context(manifest, manifest_sha256, target=target,
        support_seed=support_seed, per_class=per_class, model_seed=model_seed)
    spec = copy.deepcopy(source_spec or {})
    if method not in METHODS[:2] or (method == METHODS[0]) != (not spec):
        raise ValueError("Explicit scratch/full-source engineering role required")
    pins = {**source_pins(spec), **(source_files or {})}
    check_files(pins)
    out = external_directory(out)
    catalogue = Path(manifest).resolve(strict=True).parent
    roots = {catalogue, *(Path(p).resolve().parent for p in read(manifest)["protected_files"]),
             *(Path(p).resolve().parent for p in pins)}
    if any(out == root or out in root.parents or root in out.parents for root in roots):
        raise ValueError("Engineering outputs must be separate from all context/source input directories")
    binding = {"context": context, "item_sha256": digest(_item_fingerprint(items[target]))}
    options = settings(target, method, mode, model_seed=model_seed, support_seed=support_seed,
        per_class=per_class, binding=binding, source_spec=spec, device=device)
    runtime_type = NoneSupportTargetRuntime if mode == "none" else ActiveSupportTargetRuntime
    runtime = runtime_type(options, items, out, source_spec=spec)
    provenance = {"policy": POLICY, "support_context": context, "query_labels_accessed": False,
                  "engineering_only": True, "source_spec_sha256": digest(spec)}
    def finish():
        require_frozen_code(source)
        guard()
        check_files(pins)
    finish()
    return options, items, provenance, source, runtime, finish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", required=True, choices=NAMES)
    parser.add_argument("--method", required=True, choices=METHODS[:2])
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--model-seed", type=int, required=True, choices=(64, 512, 2048))
    parser.add_argument("--support-seed", type=int, required=True, choices=(64, 512, 2048))
    parser.add_argument("--per-class", type=int, required=True, choices=(1, 5, 10))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-spec")
    parser.add_argument("--source-spec-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, choices=(2,))
    args = parser.parse_args(argv)
    source = code_snapshot()
    if source["commit"] != args.expected_commit:
        raise ValueError("Expected complete engineering commit differs")
    require_frozen_code(source)
    if bool(args.source_spec) != bool(args.source_spec_sha256):
        raise ValueError("Source specification file and hash must be paired")
    pins = {}
    if args.source_spec:
        pins[str(Path(args.source_spec).resolve(strict=True))] = args.source_spec_sha256
        check_files(pins)
    options, items, provenance, source, runtime, guard = prepare(args.manifest, args.manifest_sha256, args.out,
        target=args.target, method=args.method, mode=args.mode, model_seed=args.model_seed,
        support_seed=args.support_seed, per_class=args.per_class,
        source_spec=read(args.source_spec) if args.source_spec else None, source_files=pins, device=args.device)
    with redirect_stdout(sys.stderr):
        result = run_training(options, items, args.out, provenance, source, runtime_adapter=runtime,
            finish_guard=guard, resume=args.resume, stop_after=args.stop_after, device=args.device)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
