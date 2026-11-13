"""Append a status-only table while verifying every historical WP12 table."""

import csv
import io
import json
from pathlib import Path

from research_runtime import external_directory
from transfer import development_block_custody as custody, registered_development as transport
from transfer.development_blocks import SCENES, read_pin


def supplement(root, admission, out):
    root = external_directory(Path(root).resolve(strict=True))
    report = json.loads(read_pin(admission, root))
    if (report["scope"] != "authorized_spatial_development_zero_update" or set(report["scenes"]) != set(SCENES)
            or any(report[key] for key in ("updates", "query", "raw_gt", "labels_added"))):
        raise ValueError("Expected complete zero-update development admission")
    prior = custody.historical(root, "wp12")
    previous = json.loads(read_pin(prior, root))
    original = {**previous["preserved"], **previous["tables"]}
    if len(original) != 10 or (previous["old_real_models"], previous["old_real_heads"]) != (65, 193):
        raise ValueError("Historical WP12 production scope differs")
    custody._protect(root, original.values())
    out = external_directory(out)
    if not out.is_relative_to(root):
        raise ValueError("WP12 evidence must remain in the authorized runs root")
    out.mkdir(exist_ok=False)
    table = out / "real-development-admission.csv"
    with table.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene", "status", "profile", "blocks", "forwards", "new_models", "updates", "reason"))
        for name, row in report["scenes"].items():
            writer.writerow((name, row["status"], row.get("profile", ""), len(row.get("roles", [])),
                             row["actual_forwards"], 0, 0, row.get("reason_code", "")))
    return transport._json_new(out / "supplement.json", dict(admission=admission, prior=prior,
        preserved=original, old_real_models=65, old_real_heads=193, new_models=0, updates=0,
        status_table=transport._pin(table)))
