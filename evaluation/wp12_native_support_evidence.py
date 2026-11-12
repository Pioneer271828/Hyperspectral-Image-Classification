"""Add native evidence to WP12 without rewriting the reviewed production tables."""

from collections import Counter
import csv
import json
from pathlib import Path
from shutil import copyfile

from research_runtime import external_directory
from transfer import registered_development as dev


OLD_TABLES = {
    "controlled-graph-synthetic.csv": "2d7078ba5c32fb45838155f6fc8f1eaff54d85051caf03c061b8a1901572ee66",
    "observation-condition-table.csv": "d2ddb610f8ef2960cc16f2c33f76273998c65467bd6fd6691876bed8ce63372e",
    "production-table.csv": "ed719c2d9eb8db6fe2ca871cabdbc33066bd375a782e6731fb9d05246924dc76",
    "real-factorial-conditions.csv": "151a006f9e100a39cbc25ddbe702391bfae7587ab142643e2a9e4bef64f38b82",
    "real-factorial-effects.csv": "c209b10800512d91344b88aa5b6424623de9494caa45a80a86aef308144a4db0",
}


def _read(pin):
    return json.loads(dev._read(pin))


def adapt(out, previous, registration, freeze, audit, cost, real, adapter_commit):
    reg, frozen, assessed, latency, probe = map(_read, (registration, freeze, audit, cost, real))
    if (reg["scope"] != "synthetic" or reg["maximum_updates"] != 6624 or frozen["updates"] != 6624
            or assessed["updates"] != 6624 or assessed["audit_updates"] or assessed["query"]
            or assessed["registration"] != frozen["registration"] or assessed["registration"] != registration
            or assessed["freeze"] != freeze or latency["registration"] != registration or latency["freeze"] != freeze
            or assessed["consumer"] != adapter_commit or latency["consumer"] != adapter_commit
            or assessed["producer"] != reg["code"] or latency["producer"] != reg["code"]
            or assessed["synthetic_recall_floor"]["candidate_admission"] is not False
            or latency["optimization"] or latency["query"] or latency["roles"] != ["devval"]
            or set(latency["repeats"]) != {"off", "local", "joint", "fixed"}
            or probe["scope"] != "authorized_development_zero_update" or any(
                probe[key] for key in ("updates", "raw_gt", "query", "labels_added"))):
        raise ValueError("Native registration, frozen audit, cost or real custody differs")
    if assessed["results"].keys() != frozen["rows"].keys() or set(probe["scenes"]) != {"xuzhou", "paviaU", "whu-honghu"}:
        raise ValueError("Incomplete result or real scene roster")
    scopes = Counter()
    for key, row in assessed["results"].items():
        scopes[row["work_package"]] += 1
        for endpoint, value in row["endpoints"].items():
            package = frozen["rows"][key]["endpoints"][endpoint]
            if value["package"] != package or dev._pin(package["path"]) != package:
                raise ValueError("Endpoint package identity differs from the audited publication")
    if scopes != {"WP08": 63, "WP07": 12}:
        raise ValueError("Unregistered native experiment scope")
    previous = external_directory(previous)
    original_tables = {name: dict(path=str(previous / name), sha256=checksum) for name, checksum in OLD_TABLES.items()}
    old_rows = list(csv.DictReader(dev._read(original_tables["production-table.csv"]).decode().splitlines()))
    models = {(row["module"], row["case_id"], row["arm"], row["state_sha256"]) for row in old_rows}
    if (len(old_rows) != 193 or len(models) != 65
            or len(dev._read(original_tables["controlled-graph-synthetic.csv"]).decode().splitlines()) != 55):
        raise ValueError("Old production or controlled evidence cardinality differs")
    out = external_directory(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, pin in original_tables.items():
        original = dev._read(pin)
        copyfile(pin["path"], out / name)
        if (out / name).read_bytes() != original or dev._read(pin) != original:
            raise ValueError("Old CSV bytes changed")
    with (out / "native-synthetic.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "arm", "work_package", "endpoint", "step", "head", "nll", "oa", "aa", "count"))
        for identity, row in assessed["results"].items():
            key, arm = identity.split("/")
            for endpoint, summary in row["endpoints"].items():
                for head, values in summary["metrics"].items():
                    writer.writerow((key, arm, row["work_package"], endpoint, summary["step"], head,
                                     values["nll"], values["oa"], values["aa"], values["count"]))
    with (out / "native-real-zero-update.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene", "scope", "status", "role_arrays", "actual_forwards", "optimizer_updates", "reason"))
        for scene, row in probe["scenes"].items():
            writer.writerow((scene, probe["scope"], row["status"], len(row["role_arrays"]), row["actual_forwards"],
                             0, row.get("reason")))
    return dev._json_new(out / "native-supplement.json", dict(version=2, producer=reg["code"], adapter=adapter_commit,
        preserved=dict(real_models=len(models), real_heads=len(old_rows), controlled_rows=54, tables=original_tables),
        synthetic=dict(scope="native_cpu_engineering", registration=registration, freeze=freeze, audit=audit, cost=cost,
                       cases=dict(scopes), updates=assessed["updates"], gpu="NOT_RUN"),
        real=dict(scope=probe["scope"], receipt=real, updates=0, query=0, scenes=probe["scenes"],
                  research_training="NOT_RUN"),
        new_tables={name: dev._pin(out / name) for name in ("native-synthetic.csv", "native-real-zero-update.csv")}))
