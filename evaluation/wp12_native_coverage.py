"""Independently audit native coverage checkpoints, class outcomes and measured costs."""

from collections import defaultdict
import csv
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluation.representation_localization import HEADS, STAGES
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import check, pin, _read, _write
from transfer import native_coverage_queue as queue
from transfer.native_coverage_replay import EXPECTED_ARCHIVES, SCHEMA as REPLAY_SCHEMA
from transfer.spatial_coverage_training import load
from transfer.xuzhou_multiscale_audit import _exact


def readout(rows, classes):
    confusion = np.zeros((classes, classes), dtype=np.int64)
    losses = []
    for labels, probabilities in rows:
        for truth, p in zip(labels, probabilities, strict=True):
            check(1 <= truth <= classes and len(p) == classes and all(math.isfinite(q) and 0 <= q <= 1 for q in p)
                  and abs(math.fsum(p)-1.0) < 2e-6, "Overflow or foreign released class probabilities")
            assigned = max(range(classes), key=p.__getitem__)
            confusion[truth-1, assigned] += 1
            losses.append(-math.log(max(p[truth-1], 1e-8)))
    return confusion, math.fsum(losses)


def audit_case(place, cfg, code):
    case_pin = pin(place/"receipt.json")
    item = _read(case_pin)
    reg = item["registration_file"]
    spec = _read(reg)
    check(reg == pin(place/"registration.json") and spec["producer"] == code["commit"]
          and digest(spec) == item["registration"] and spec["case"] == cfg
          and item["inputs"] == digest(spec["inputs"]) and item["bank"] == spec["bank"]["sha256"]
          and item["numeric"] == spec["numeric"] and item["real_updates"] == 0 and item["query_accessed"] is False,
          "Foreign queue case, data or process registration")
    packets, cost = {}, {}
    for phase, updates in zip(queue.PHASES, (12, 4, 8, 0), strict=True):
        stage = item["stages"][phase]
        path = place/f"{phase}.pt"
        parent = queue.PREDECESSOR.get(phase)
        cursor = 4 if phase == "prefix" else 12
        check(stage == _read(pin(place/f"{phase}.json")) and stage["phase"] == phase
              and stage["checkpoint"] == pin(path) and stage["registration"] == item["registration"]
              and stage["parent"] == (None if parent is None else item["stages"][parent]["checkpoint"])
              and stage["updates"] == updates, "Checkpoint path, history or registered budget differs")
        packet = load(path)
        packets[phase] = packet
        check(packet["cursor"] == len(packet["history"]) == cursor and
              packet["registration"] == item["registration"] and packet["format"] == queue.SCHEMA and
              [record["step"] for record in packet["validation"]] == [t for t in spec["events"] if t <= cursor]
              and all(int(v["step"]) == cursor for v in packet["optimizer"]["state"].values()),
              "On-disk optimizer, validation or event cursor differs")
        times = stage["timing"]
        check(all(math.isfinite(t) and t >= 0 for t in times.values()) and
              (all(t == 0 for t in times.values()) if not updates else times["train_update_seconds"] > 0),
              "Missing or invalid phase cost")
        cost[phase] = dict(updates=updates, cursor=cursor, selected=stage["selected"], timings=times,
                           checkpoint_sha256=stage["checkpoint"]["sha256"])
    full, prefix, resume, completed = (packets[phase] for phase in queue.PHASES)
    check(item["optimizer_updates"] == 24 and _exact(full, resume) and _exact(full, completed)
          and _exact(full["history"][:4], prefix["history"]), "Resumed packet state or fit budget differs")
    selected = min(full["validation"], key=lambda row: row["fused_nll"])
    check(selected["step"] == item["selected_step"] == full["selected"]["step"]
          and selected["fused_nll"] == item["selection_nll"] == full["selected"]["fused_nll"],
          "First-minimum selection differs from the saved weights")
    classes = queue.SCENES[cfg["scene"]]
    evidence = item["predictions"]
    check([row["observation"] for row in evidence] == [row["observation"] for row in selected["rows"]],
          "Selected probabilities have foreign role, band or space")
    result = {}
    nll = []
    for band in queue.VALIDATION_BANDS:
        result[band] = {}
        for head in HEADS:
            confusion, total = readout(((row["truth"], row["probabilities"][head]) for row in evidence
                                         if row["band"] == band), classes)
            stored = item["validation"][band]["heads"][head]
            support = confusion.sum(axis=1).tolist()
            check(confusion.tolist() == stored["confusion"] and sum(support) == stored["n"]
                  and support == item["roster"]["devval"] and abs(total-stored["nll"]) < 1e-9,
                  "Independent per-class counts or released likelihood disagree")
            if head == "fused":
                nll.append(total)
            result[band][head] = [dict(n=n, correct=int(confusion[c,c]), recall=float(confusion[c,c])/n if n else None)
                                  for c, n in enumerate(support)]
    expected = math.fsum(nll) / (sum(item["roster"]["devval"])*len(queue.VALIDATION_BANDS))
    check(abs(expected-item["selection_nll"]) < 1e-10, "Independent per-point selection objective differs")
    return case_pin["sha256"], item, result, cost


def replay_table(root, code, writer):
    link = pin(root/"replay.json")
    record = _read(link)
    check(record["schema"] == REPLAY_SCHEMA and record["producer"] == code["commit"]
          and set(record["scenes"]) == set(queue.SCENES) and record["real_updates"] == 0
          and record["raw_gt_loaded"] is record["query_accessed"] is record["formal_research_claim_allowed"] is False,
          "WP12 expects a compatible frozen development replay from this checkout")
    rows = {}
    for scene, item in record["scenes"].items():
        archive = "paviaU-64" if scene == "paviaU" else scene
        check(item["archive_sha256"] == EXPECTED_ARCHIVES[archive] and
              [row["band"] for row in item["rows"]] == ["full", "central_gap"], "Archived provenance differs")
        rows[scene] = {}
        for row in item["rows"]:
            check(row["role"] == "devval" and row["factor"] == 1 and row["untouched_by_probes"]
                  and row["probabilities_fp32_equal"] and row["graph_equal"]
                  and set(row["probes"]) == set(STAGES) and all(v == 0 for v in row["max_probability_error"].values()),
                  "Archived predictor, graph or tap has drifted")
            rows[scene][row["band"]] = {}
            for head in HEADS:
                confusion, nll = readout([(row["labels"],row["probabilities"][head])], queue.SCENES[scene])
                counts = confusion.sum(axis=1).tolist()
                rows[scene][row["band"]][head] = dict(n=sum(counts), class_support=counts, nll=nll)
                for c, n in enumerate(counts, 1):
                    writer.writerow((scene,row["band"],head,c,n,int(confusion[c-1,c-1]),
                                     float(confusion[c-1,c-1])/n if n else None))
    return link["sha256"], rows


def publish(root, code):
    registration = queue.load_queue(root, code)
    out = root/"wp12"
    check(not out.exists(), "WP12 acceptance must be published once into a fresh output")
    with TemporaryDirectory(dir=root, prefix=".native-coverage-wp12-") as temporary:
        stage = Path(temporary)/"publication"
        stage.mkdir()
        with (stage/"real-per-class.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("scene","band","head","class","n","correct","recall"))
            real_source, real_metrics = replay_table(root, code, writer)
        inventory, outcomes, costs = {}, {}, {}
        partitions = defaultdict(set)
        with (stage/"synthetic-per-class.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("case","scene","support_seed","k","split","arm","band","head","class","n","correct","recall"))
            for case, cfg in registration["cases"].items():
                place = root/"cases"/"cpu"/case
                identity, item, outcomes[case], phase_cost = audit_case(place, cfg, code)
                inventory[case] = identity
                origin = tuple(cfg[key] for key in ("scene","support_seed","k","split"))
                partitions[origin].add((item["inputs"], item["bank"], item["source"]["source"], item["source"]["target"]))
                costs[case] = dict(phases=phase_cost, selected=item["selected_step"], parameters=item["parameters"],
                                   validation_seconds=item["endpoint_validation_seconds"], elapsed_seconds=item["elapsed_seconds"])
                for band, result in outcomes[case].items():
                    for head, rows in result.items():
                        for c, row in enumerate(rows, 1):
                            writer.writerow((case,*origin,cfg["spectral_arm"],band,head,c,row["n"],row["correct"],row["recall"]))
        check(len(partitions) == 24 and all(len(v) == 1 for v in partitions.values())
              and len({next(iter(v))[1] for v in partitions.values()}) == 24,
              "Matched arms, support budgets or spatial Bank identity differ")
        with (stage/"synthetic-vs-none.csv").open("w",encoding="utf-8",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("case","band","head","class","n","recall_delta"))
            for case, bands in outcomes.items():
                arm = registration["cases"][case]["spectral_arm"]
                if arm == "none":
                    continue
                control = outcomes[case.removesuffix(arm)+"none"]
                for band, heads in bands.items():
                    for head, rows in heads.items():
                        for c, (row, basis) in enumerate(zip(rows,control[band][head],strict=True),1):
                            check(row["n"] == basis["n"], "Paired spectral arms have a different denominator")
                            delta = row["recall"] - basis["recall"] if row["n"] else None
                            writer.writerow((case,band,head,c,row["n"],delta))
        device_runs, reloads = {}, {}
        for folder in sorted((root/"cases").glob("GPU-*")):
            for path in sorted(folder.iterdir()):
                if not (path/"receipt.json").is_file():
                    continue
                identity, item, _, phase_cost = audit_case(path, registration["cases"][path.name], code)
                check(item["numeric"]["physical_uuid"] == folder.name, "Wrong physical device in qualification receipt")
                device_runs[f"{folder.name}/{path.name}"] = dict(sha256=identity, phases=phase_cost, selected=item["selected_step"])
        for path in sorted((root/"device-reloads").glob("*.json")):
            identity = pin(path)
            item = _read(identity)
            check(item["producer"] == code["commit"] and item["source_device"] != item["target_device"]
                  and item["optimizer_updates"] == item["real_updates"] == 0
                  and max(item["probability_max_error"].values()) <= 2e-4,
                  "Foreign cross-device read-only evaluation")
            reloads[path.stem] = dict(sha256=identity["sha256"], result=item)
        _write(stage/"costs.json", costs)
        _write(stage/"devices.json", dict(fits=device_runs, frozen_reloads=reloads))
        require_frozen_code(code)
        inputs = pin(root/"queue.json")
        files = {path.name:pin(path)["sha256"] for path in stage.iterdir()}
        output = _write(stage/"receipt.json", dict(status="accepted_native_coverage_engineering", producer=code["commit"],
            queue_sha256=inputs["sha256"], previous_real_replay=real_source, real=real_metrics,
            cases=inventory, files=files, fits=len(inventory), banks=len(partitions), synthetic_updates=24*len(inventory),
            completed_reentry_updates=0, cuda_fits=len(device_runs), cuda_updates=24*len(device_runs),
            frozen_device_reloads=len(reloads), real_updates=0, raw_gt_loaded=False, query_accessed=False,
            formal_research_claim_allowed=False))
        os.rename(stage, out)
    return dict(path=str(out/"receipt.json"), sha256=output["sha256"])
