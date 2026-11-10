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


if __name__ == "__main__":
    main()
