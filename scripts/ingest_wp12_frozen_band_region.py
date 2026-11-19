"""Authenticate WP05/WP06 released paired evidence and export the WP12 supplement."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path

import numpy as np

from evaluation.frozen_band_region import HEADS, paired, summarize
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import POLICY, _read, _write, check, pin
from transfer.development_bank import _atomic_new


def _audit_pair(head):
    source, target, labels = [], [], []
    for row in head["classes"]:
        cls = row["class_id"]
        for before, after, n in row["transitions"]:
            check(type(n) is int and n > 0, "Transition counts must be positive integers")
            source.extend([before] * n)
            target.extend([after] * n)
            labels.extend([cls] * n)
    reconstructed = paired(target, source, labels, len(head["classes"]))
    check(digest(reconstructed) == digest(head), "Class flips, confusion or metrics fail reconstruction")


def _csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_new(path, stream.getvalue().encode())
    return pin(path)["sha256"]


def ingest(root, scope_pin, summary_pin, case_dir, output, code):
    for path in (scope_pin["path"], summary_pin["path"], case_dir, output):
        check(Path(path).resolve().is_relative_to(root), "WP12 input/output escapes authorized runs")
    scope, summary = _read(scope_pin), _read(summary_pin)
    check(scope["policy"] == summary["schema"] == POLICY and summary["status"] == "passed"
          and summary["scope"] == scope_pin["sha256"] and summary["commit"] == scope["commit"]
          and summary["raw_gt_loaded"] is summary["query_accessed"] is False
          and summary["optimizer_updates"] == scope["optimizer_updates"] == 0
          and summary["focus"] == scope["focus"] and summary["rare_policy"] == scope["rare_policy"],
          "WP12 frozen source, role or update scope changed")
    expected = {f"{lane}__{case}.json": (lane, case)
                for lane, cases in scope["cases"].items() for case in cases}
    check(set(expected) == set(summary["artifacts"]) and len(expected) == summary["cases"],
          "WP12 cannot admit a partial matrix")
    cases, metrics, recalls, risks = [], [], [], []
    counts = Counter()
    for name, (lane, case) in expected.items():
        path = case_dir / name
        row = _read(pin(path, summary["artifacts"][name]))
        current = row["result"]
        check(row["scope"] == summary["scope"] and (row["lane"], row["case"]) == (lane, case)
              and row["result_hash"] == digest(current), "WP12 case identity changed")
        cases.append(current)
        scene = current["scene"]
        counts[scene] += 1
        metadata = dict(scene=scene, case=case, cohort=current["cohort"], model_seed=current["model_seed"],
                        scope="released_devval", zero_updates=True,
                        evidence="archived_prediction" if lane == "archive" else "new_frozen_forward")
        if lane != "archive":
            check(current["deleted_negative"]["deleted_poison_equal"]
                  and current["deleted_negative"]["leaky_vca_input_detected"], "Missing deleted-input controls")
            risks.append(dict(scene=scene, case=case, historical_confusion=current["historical_confusion_check"],
                              spatial_groups=current["group_counts"], physical_forwards=current["physical_forwards"]))
        for contrast, heads in current["paired"].items():
            check(set(heads) == set(HEADS), "WP12 requires three head records")
            for head, values in heads.items():
                _audit_pair(values)
                metrics.append(dict(**metadata, contrast=contrast, head=head, n=values["denominator"],
                                    source_oa=values["source_oa"], target_oa=values["target_oa"],
                                    source_aa=values["source_aa"], target_aa=values["target_aa"],
                                    delta_oa=values["delta_oa"], delta_aa=values["delta_aa"],
                                    flips=values["flips"], correct_to_error=values["correct_to_error"],
                                    error_to_correct=values["error_to_correct"]))
                for item in values["classes"]:
                    recalls.append(dict(**metadata, contrast=contrast, head=head,
                                        focused=item["class_id"] in scope["focus"][scene],
                                        **{key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                                           if isinstance(value, (dict, list)) else value
                                           for key, value in item.items()}))
    check(sum(x["physical_forwards"] for x in cases) == summary["physical_forwards"]
          and summary["archived_cases"] == len(scope["cases"]["archive"])
          and digest(summarize(cases)) == digest(summary["groups"]), "WP12 seed aggregate failed replay")
    output.mkdir(exist_ok=False)
    artifacts = {"paired-metrics.csv": _csv(output / "paired-metrics.csv", metrics),
                 "paired-classes.csv": _csv(output / "paired-classes.csv", recalls)}
    return _write(output / "receipt.json", dict(
        schema="wp12_wp05_wp06_frozen_band_region_v1", source=summary_pin["sha256"],
        scope=scope_pin["sha256"], producer_commit=summary["commit"], intake_commit=code["commit"],
        status="accepted_development_diagnostic", counts=dict(counts), cases=len(cases),
        comparison_rows=len(metrics), per_class_rows=len(recalls), zero_optimizer_updates=True,
        query_accessed=False, raw_gt_loaded=False, artifacts=artifacts, native_checks=risks,
        sample_sd_policy="independent_model_seeds_within_same_frozen_cohort",
        scope_limit="development_sensitivity_only", physical_metadata="nominal_or_approximate",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "scope", "summary", "cases", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("scope-sha256", "summary-sha256", "expected-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected WP12 checkout")
    result = ingest(external_directory(args.root), pin(args.scope, args.scope_sha256),
                    pin(args.summary, args.summary_sha256), args.cases, external_directory(args.output), code)
    require_frozen_code(code)
    print(result)


if __name__ == "__main__":
    main()
