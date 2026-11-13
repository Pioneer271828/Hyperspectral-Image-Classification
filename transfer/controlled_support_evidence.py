"""Audit and measure registered interventions on the frozen native endpoint population."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.controlled_support_metrics import paired_metrics
from evaluation.controlled_support_oracle import audit as support_oracle, population
from evaluation.wp12_native_support_evidence import OLD_TABLES
from inference.native_support_checkpoint import load_endpoint, predict_registered_tile
from research_runtime import external_directory, tensor_hash
from transfer import controlled_support_campaign as campaign, native_support_campaign as native
from transfer import registered_development as dev, xuzhou_multiscale as stages
from transfer.controlled_regional_support import ARMS
from transfer.g3_training import _forward, state_hash
from transfer.native_support import NativeSupportModel, capture_context, native_forward
from transfer.native_support_intervention import BoundIntervention, forward_intervention, prepare_block
from transfer.native_support_real import _two_blocks
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact


FROZEN_ARMS = ("off", "joint", "fixed")
ENDPOINTS = ("selected", "last")
REVIEWED_TABLES = {**OLD_TABLES,
    "native-synthetic.csv": "8092030f8f879a1ca68bb2304b27603eb905d8572871a399a81c5b3757b21a3a",
    "native-real-zero-update.csv": "63e7e1e2f493f79fe3022ce4e38666ff7c25957b76cb3dcb00ab15b2edc3883f"}
HEAD_LABELS = ("fused", "graph", "cnn")


def _tensor_inventory(model):
    return dict(state=state_hash(model), config=model.export_config(),
                training=[m.training for m in model.modules()], cache=model.backbone.adapter.cache_attention)


def _bound(entry, case, receipt, arm, owner):
    return BoundIntervention(entry, campaign.trial(case), campaign.random_trial(case), receipt, arm, owner,
                             expected_sha256=receipt["sha256"])


@torch.no_grad()
def _comparison(model, scene, record, role, entry, case, owner, step):
    tile = scene.plan["tiles"][role["id"]]
    receipt, cost = prepare_block(entry, role["box"], campaign.trial(case), campaign.random_trial(case),
                                  step, campaign.intervention_seed(case, role["id"]), owner)
    validated_subsets = support_oracle(dict(receipt, mandatory=entry["mandatory"] + [[i, i] for i in range(len(receipt["points"]))],
                                            partitions=entry["context"]["partitions"]))
    outputs = {}
    timings = {}
    traces = {}
    for arm in ARMS:
        if arm == "matched" and receipt["match"]["status"] != "MATCHED":
            continue
        bound = _bound(entry, case, receipt, arm, owner)
        timings[arm] = {}
        out, frame = forward_intervention(model, scene, record, tile, bound, detail=True, timing=timings[arm])
        if not torch.equal(frame["stage2"].indices().T, torch.tensor(bound._pairs)):
            raise ValueError("Executed COO differs from intervention receipt")
        outputs[arm] = out
        traces[arm] = dict(frame["trace"], probabilities=[tensor_hash(value) for value in out[:3]],
                           actual_values=tensor_hash(frame["stage2"].values()))
    comparisons = {arm: paired_metrics(record, role, outputs["reference"], value)
                   for arm, value in outputs.items() if arm != "reference"}
    return dict(receipt=receipt, search_cost=cost, oracle_enumerated=validated_subsets, timings=timings, traces=traces,
                comparison=comparisons, old_random_invariants=population(receipt["old_random"]["selected"], receipt["points"]),
                admission="NOT_ASSESSED"), outputs["reference"]


@torch.no_grad()
def intervene(pin, code, output, *, producer_commit=None):
    _, reg, old, frozen, accepted_audit, real = campaign.load(pin, code, producer_commit=producer_commit)
    started = perf_counter()
    synthetic = {}
    for key in reg["cases"]:
        case = old["cases"][key]
        data = native._case_data(case)
        baseline, scene, record, _, bank = data
        source = _tensor_inventory(baseline)
        for arm in FROZEN_ARMS:
            identity = f"{key}/{arm}"
            for end in ENDPOINTS:
                package = frozen["rows"][identity]["endpoints"][end]
                model, payload = load_endpoint(package, reg["original"]["registration"], case, data)
                inventory = _tensor_inventory(model)
                rows = []
                for block, entry in bank.items():
                    tile_id = int(block)
                    role = record["roles"][tile_id]
                    tile = scene.plan["tiles"][tile_id]
                    if arm == "off":
                        if role["role"] == "devval":
                            probabilities, _ = predict_registered_tile(model, payload, scene, record, tile_id, bank)
                            expected = accepted_audit["results"][identity]["endpoints"][end]["probability_hashes"][0]
                            if [tensor_hash(p) for p in probabilities[:3]] != expected:
                                raise ValueError("Original off endpoint probabilities changed")
                        else:
                            probabilities = _forward(model, scene, tile)
                        rows.append(dict(role=role["role"], block=block, status="NOT_APPLICABLE_OFF",
                                         original=paired_metrics(record, role, probabilities, probabilities)))
                        continue
                    step = 0 if role["role"] == "devval" else campaign.BUDGET if arm == "fixed" else payload["step"]
                    owner = dict(endpoint=package, state=payload["state_hash"], bank=case["bank"], key=identity, end=end)
                    result, reference = _comparison(model, scene, record, role, entry, case, owner, step)
                    if role["role"] == "devval":
                        original, _ = predict_registered_tile(model, payload, scene, record, tile_id, bank)
                        if any(not torch.equal(a, b) for a, b in zip(original[:3], reference[:3], strict=True)):
                            raise ValueError("Original fixed step-zero validation differs")
                    elif arm == "fixed" and result["receipt"]["reference"] != payload["donor"]["snapshots"][block]:
                        raise ValueError("Original fixed training support differs from the donor")
                    rows.append(dict(role=role["role"], block=block, step=step, **result))
                if _tensor_inventory(model) != inventory or _tensor_inventory(baseline) != source:
                    raise ValueError("Frozen intervention changed model or shared source")
                synthetic[f"{identity}/{end}"] = dict(endpoint=package, rows=rows, model_unchanged=True)
        print(f"intervened {key}", flush=True)
    real_rows = _real_probe(real)
    return dev._json_new(external_directory(output), dict(registration=pin, code=code["commit"],
        synthetic=synthetic, real=real_rows, updates=0, query=0, labels_added=0,
        candidate_admission="NOT_ASSESSED", elapsed_seconds=perf_counter() - started))


def _real_probe(real):
    prior = real["scenes"]["xuzhou"]
    manifest = prior["released_role_manifest"]
    scene, roles, pins = _two_blocks(manifest)
    if pins != prior["role_arrays"]:
        raise ValueError("Real probe differs from the two previously admitted development arrays")
    record = campaign.read(manifest)["record"]
    baseline = dev._load_source(prior["source"], prior["sidecar"])
    if state_hash(baseline) != prior["source_state"]:
        raise ValueError("Real source state differs")
    model = NativeSupportModel.initialize(baseline, 7321)
    inventory = _tensor_inventory(model)
    case = {"support_seed": 19064}
    results = {}
    for role in roles:
        entry = capture_context(model, scene, role, manifest["sha256"], prior["source_state"])
        owner = dict(source=prior["source"], manifest=manifest, purpose="zero_update_applicability")
        step = 0
        result, _ = _comparison(model, scene, record, role, entry, case, owner, step)
        results[role["role"]] = result
    if _tensor_inventory(model) != inventory:
        raise ValueError("Real applicability probe changed frozen source")
    return dict(xuzhou=dict(status="PROBED", blocks=results, custody=pins, updates=0),
                paviaU=dict(status="BLOCKED", reason=real["scenes"]["paviaU"]["reason"]),
                honghu=dict(status="BLOCKED", reason=real["scenes"]["whu-honghu"]["reason"]))


def audit(pin, code, output, *, producer_commit=None):
    root, reg, old, frozen, _, _ = campaign.load(pin, code, producer_commit=producer_commit)
    rows = {}
    observed_updates = 0
    for key, package in reg["cases"].items():
        case = old["cases"][key]
        baseline, scene, record, plan, bank = native._case_data(case)
        oracle = {}
        for name, block in package["blocks"].items():
            context = bank[name]
            oracle[name] = support_oracle(dict(block, mandatory=context["mandatory"] +
                [[i, i] for i in range(len(block["points"]))], partitions=context["context"]["partitions"]))
        if not package["eligible"]:
            rows[key] = dict(status=package["status"], oracle_subsets=oracle, arms={}, updates=0)
            continue
        by_arm = {}
        starts = {}
        for arm in ARMS:
            attempt = {}
            packets = {}
            update_counts = {}
            arm_cost = {}
            stage_map = native._stages(campaign.BUDGET)
            for name, (_, _, parent) in stage_map.items():
                status, receipt = stages.read_stage(root / "matrix" / key, pin, arm, name, attempt.get(parent),
                                                     arms=ARMS, stages=stage_map)
                packet = native._load_pt(status["checkpoint"])
                if packet["contract_hash"] != package["contracts"][arm]:
                    raise ValueError("Recovery checkpoint contract differs")
                attempt[name], packets[name], update_counts[name] = receipt, packet, status["new_updates"]
                arm_cost[name] = {k: v for k, v in status.items() if k.endswith("_seconds") or k == "timing"}
            full = packets["full"]
            for name in ("resume", "completed"):
                if any(not _exact(full[k], packets[name][k]) for k in SCIENCE):
                    raise ValueError("Recovery scientific state differs")
            if not _exact(full["history"][:7], packets["prefix"]["history"]):
                raise ValueError("Recovery prefix history differs")
            run = campaign._factory(case, (baseline, scene, record, plan, bank), package, arm)[0]()
            starts[arm] = run.spec["source_state"]
            endpoints = {}
            for end, weights, index in (("selected", "selected_state", full["selected_step"]),
                                        ("last", "state", campaign.BUDGET)):
                run.model.load_state_dict(full[weights], strict=True)
                if state_hash(run.model) != full["history"][index - 1]["state_hash"]:
                    raise ValueError("Controlled endpoint tensor identity differs")
                with torch.no_grad():
                    run.model.eval()
                    for role in record["roles"]:
                        if role["role"] == "devval":
                            outputs, frame = native_forward(run.model, scene, scene.plan["tiles"][role["id"]],
                                selector=run.devval[str(role["id"])])
                            endpoints[end] = dict(step=index, sha256=state_hash(run.model),
                                heads=[tensor_hash(v) for v in outputs[:3]], metrics=paired_metrics(record, role, outputs, outputs),
                                validation_support=frame["trace"]["snapshot"])
                if run._evaluate(run.model) != full["history"][index - 1]["validation"]:
                    raise ValueError("Recovery endpoint validation differs")
            if min(full["history"], key=lambda row: row["validation"]["nll"])["step"] != full["selected_step"]:
                raise ValueError("Recovery selected step differs")
            observed_updates += sum(update_counts.values())
            by_arm[arm] = dict(attempts=attempt, updates=update_counts, timings=arm_cost, endpoints=endpoints,
                first_last_loss=[full["history"][0]["loss"], full["history"][-1]["loss"]],
                source_state=starts[arm], exact_fields=list(SCIENCE))
        if len(set(starts.values())) != 1:
            raise ValueError("Controlled arms do not share initial weights")
        old_row = frozen["rows"][f"{key}/fixed"]
        old_receipt = campaign.read(old_row["attempts"]["full"])
        old_full = native._load_pt(old_receipt["checkpoint"])
        ref_receipt = campaign.read(by_arm["reference"]["attempts"]["full"])
        ref_full = native._load_pt(ref_receipt["checkpoint"])
        shared = tuple(field for field in SCIENCE if field != "history")
        if any(not _exact(old_full[field], ref_full[field]) for field in shared):
            raise ValueError("New reference differs from the reviewed Issue447 fixed donor training")
        fields = ("step", "tile_id", "anchor", "support_ids", "loss", "state_hash", "validation", "train_probabilities", "scorers")
        if any(any(not _exact(a[field], b[field]) for field in fields)
               for a, b in zip(old_full["history"], ref_full["history"], strict=True)):
            raise ValueError("Reference update history differs from the reviewed Issue447 fixed arm")
        rows[key] = dict(status="PASSED", arms=by_arm, oracle_subsets=oracle,
                         historical_reference_exact=dict(fields=shared, history_fields=fields, attempt=old_row["attempts"]["full"]))
    if observed_updates != reg["registered_updates"]:
        raise ValueError("Controlled optimization accounting differs")
    return dev._json_new(external_directory(output), dict(registration=pin, code=code["commit"], rows=rows,
                                                         observed_updates=observed_updates, audit_updates=0, query=0))


def supplement(pin, code, output, previous, intervention, recovery_audit, cost_evidence, *, producer_commit=None):
    _, reg, _, _, _, _ = campaign.load(pin, code, producer_commit=producer_commit)
    frozen, recovery, costs = (campaign.read(item) for item in (intervention, recovery_audit, cost_evidence))
    if (any(row["registration"] != pin or row["query"] for row in (frozen, recovery, costs))
            or costs["intervention"] != intervention or frozen["updates"] or costs["updates"]
            or recovery["observed_updates"] != reg["registered_updates"] or recovery["audit_updates"]):
        raise ValueError("Controlled evidence registration, budget, or custody differs")
    root = external_directory(output)
    root.mkdir(parents=True, exist_ok=False)
    previous = external_directory(previous)
    table_pins = {}
    for name, checksum in REVIEWED_TABLES.items():
        original = dict(path=str(previous / name), sha256=checksum)
        blob = dev._read(original)
        with (root / name).open("xb") as stream:
            stream.write(blob)
        if (root / name).read_bytes() != blob or dev._read(original) != blob:
            raise ValueError("An earlier WP12 table changed")
        table_pins[name] = original
    old = list(csv.DictReader((root / "production-table.csv").read_text().splitlines()))
    models = {(row["module"], row["case_id"], row["arm"], row["state_sha256"]) for row in old}
    if len(old) != 193 or len(models) != 65:
        raise ValueError("Earlier WP12 real table cardinality changed")
    with (root / "controlled-intervention-synthetic.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "block", "role", "arm", "match_status", "head", "oa", "aa", "kappa",
                         "new_errors", "corrected", "least_represented_new_errors", "admission"))
        for identity, result in frozen["synthetic"].items():
            for row in result["rows"]:
                if row.get("status") == "NOT_APPLICABLE_OFF":
                    writer.writerow((identity, row["block"], row["role"], "off", row["status"],
                                     "", "", "", "", "", "", "", "NOT_ASSESSED"))
                    continue
                status = row["receipt"]["match"]["status"]
                for arm, comparison in row["comparison"].items():
                    for head, scores in comparison["heads"].items():
                        metrics = scores["intervention"]
                        writer.writerow((identity, row["block"], row["role"], arm, status, head,
                            metrics["oa"], metrics["aa"], metrics["kappa"], scores["counts"].get("new_error", 0),
                            scores["counts"].get("corrected", 0), scores["least_represented_new_errors"], "NOT_ASSESSED"))
    with (root / "controlled-recovery-synthetic.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "status", "arm", "endpoint", "head", "step", "oa", "aa", "kappa", "updates"))
        for key, row in recovery["rows"].items():
            if not row["arms"]:
                writer.writerow((key, row["status"], "", "", "", "", "", "", "", 0))
            for arm, arm_row in row["arms"].items():
                for endpoint, metrics in arm_row["endpoints"].items():
                    for head, pair in metrics["metrics"]["heads"].items():
                        values = pair["intervention"]
                        writer.writerow((key, row["status"], arm, endpoint, head, metrics["step"],
                            values["oa"], values["aa"], values["kappa"], sum(arm_row["updates"].values())))
    with (root / "controlled-intervention-real-zero-update.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scene", "role", "status", "matched_status", "updates"))
        for scene, row in frozen["real"].items():
            if "blocks" in row:
                for role, block in row["blocks"].items():
                    writer.writerow((scene, role, row["status"], block["receipt"]["match"]["status"], 0))
            else:
                writer.writerow((scene, "", row["status"], "", 0))
    return dev._json_new(root / "controlled-supplement.json", dict(registration=pin, producer=reg["code"],
        consumer=code["commit"], preserved=table_pins, old_real_models=len(models), old_real_heads=len(old),
        evidence=dict(frozen_intervention=intervention, fixed_recovery=recovery_audit, cpu_cost=cost_evidence),
        tables={item.name: dev._pin(item) for item in root.glob("controlled-*.csv")}, real_updates=0,
        query=0, candidate_admission="NOT_ASSESSED"))
