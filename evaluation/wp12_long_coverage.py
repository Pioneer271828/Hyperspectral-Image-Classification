"""Independent sealed-checkpoint replay and paired estimates for WP12."""

from collections import defaultdict
from itertools import product
from pathlib import Path
import statistics
from time import perf_counter

import torch

from evaluation.frozen_training_stages import HEADS, VIEWS, file_pin, json_pin, paired, require, scores
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_scale_training as native
from transfer.native_long_campaign import ARMS, CLOCK, DESIGN, SCENES, SEEDS, STEPS, MatchedProvider
from transfer.native_long_run import REFERENCE, campaign, checkpoints
from transfer.spatial_coverage_protocol import bank_view, retained_positions


def frozen(root, place, production, freeze_pin):
    plan, pin = campaign(root, place, production)
    freeze = json_pin(root, freeze_pin)
    require(freeze_pin == file_pin(place/"freeze.json") and freeze["campaign"] == pin
            and freeze["scope"] == plan["scope"] and freeze["updates"] == 12*STEPS
            and set(freeze["cases"]) == set(plan["cases"])
            and freeze["query_accessed"] is freeze["raw_gt_loaded"] is False,
            "Full frozen campaign required before opening confirmation data")
    return plan, freeze


def add_metrics(row):
    matrix = row["confusion"]
    row["aa"] = statistics.mean(n for n in row["recall"] if n is not None)
    require(row["count"] == sum(row["support"]) == sum(sum(line) for line in matrix), "Class denominator differs")
    return row


def read_confirmation(run, provider, blocks, step):
    events = [dict(tile=row["id"], band=band, factor=1, profile=None, condition="missing")
              for row, band in product(blocks, VIEWS)]
    by_tile = {row["id"]: row for row in blocks}
    rows = {view: [] for view in VIEWS}

    def materialize(event):
        return bank_view(provider.bank, by_tile[event["tile"]], run.producer, event["band"])

    def capture(event, actual, output, targets):
        probs = {head: tensor.detach().cpu().numpy()[targets.cells].tolist()
                 for head, tensor in zip(HEADS, output[:3], strict=True)}
        for pos, (pixel, label) in enumerate(targets.points):
            rows[event["band"]].append(dict(id=[event["tile"], pixel], truth=label,
                                            probabilities={h: probs[h][pos] for h in HEADS}))

    started = perf_counter()
    if run.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(run.device)
    evaluated = native.evaluate_validation(run.model, events, step, materialize, run._forward,
                                           released=provider, detailed=True, observe=capture)
    if run.device.type == "cuda":
        torch.cuda.synchronize(run.device)
    peak = torch.cuda.max_memory_allocated(run.device) if run.device.type == "cuda" else None
    metrics = {}
    for band in VIEWS:
        rows_for_band = [row for row in evaluated["rows"] if row["band"] == band]
        master = provider.bank.release.bands
        band_ids = [master.band_ids[i] for i in retained_positions(master, band)]
        require(all(row["band"] == band and row["graph"]["q_nnz"] > 0
                    and row["graph"]["a0_nnz"] > 0 and row["band_ids"] == band_ids for row in rows_for_band),
                "Missing native learned graph trace or foreign band IDs")
        scores_by_head = {head: add_metrics(value) for head, value in scores(rows[band], provider.classes).items()}
        for head, value in scores_by_head.items():
            native_counts = [sum(row["scores"][head]["point_confusion"][t][p] for row in rows_for_band)
                             for t,p in product(range(provider.classes), repeat=2)]
            require(native_counts == [n for line in value["confusion"] for n in line] and
                    abs(sum(row["scores"][head]["nll_sum"] for row in rows_for_band)/value["count"]-value["nll"]) < 1e-10,
                    "Native released readout differs from independent confusion/NLL")
        comparisons = paired(rows["full"], rows[band], provider.classes, {1:2, 2:1, 3:4, 4:3})
        for head in HEADS:
            move = comparisons[head]["transitions"]
            comparisons[head]["correct_to_error"] = sum(n for t,a,b,n in move if a == t and b != t)
            comparisons[head]["error_to_correct"] = sum(n for t,a,b,n in move if a != t and b == t)
        metrics[band] = dict(heads=scores_by_head, band_ids=band_ids,
                             fit_statistics=[row["fit_stats"] for row in rows_for_band],
                             graph_sha256=digest([row["graph"] for row in rows_for_band]),
                             point_roster=digest([(row["id"],row["truth"]) for row in rows[band]]),
                             comparison=comparisons)
    return dict(step=step, heads_by_view=metrics, replay_seconds=perf_counter()-started, peak_cuda_bytes=peak)


def independent_case(root, place, out, authority, producer, freeze_pin, name, execution):
    plan, freeze = frozen(root, place, producer, freeze_pin)
    require(name in plan["cases"], "Unknown audit case")
    original = freeze["cases"][name]
    if out.exists():
        receipt = json_pin(root, file_pin(out))
        require(receipt["freeze"] == freeze_pin and receipt["source"] == original
                and receipt["consumer"] == authority["commit"], "Existing audit is foreign")
        return file_pin(out)
    start = perf_counter()
    receipt = json_pin(root, original)
    spec = json_pin(root, receipt["spec"])
    require(receipt["device"] == spec["numeric"] == execution.numeric, "Independent replay arithmetic differs")
    contract = digest(spec)
    require(receipt["registration"] == contract and spec["selection"] == DESIGN["selection"]
            and all(row["band"] == "full" for row in spec["validation"]), "Altered views cannot choose a checkpoint")
    journals = checkpoints(root, place/"cases"/name, contract, read_all=True)
    require([row["pin"] for row in journals] == receipt["checkpoints"] and journals[-1]["pin"] == receipt["checkpoint"],
            "Frozen source checkpoint roster changed")
    provider = MatchedProvider(root, producer, plan, name)
    run = native.Run(spec, producer["commit"], execution=execution, released=provider)
    final = journals[-1]["item"]["training"]
    run.restore(final, STEPS)
    require([item["step"] for item in final["validation"]] == list(CLOCK) and len(final["history"]) == STEPS
            and all(row["learning_rate"] == spec["learning_rate"] for row in journals), "Validation clock, LR or updates differ")
    clock = {row["step"]: row["item"]["training"] for row in journals}
    require(set(CLOCK) <= clock.keys(), "Accepted selection events require original saved state")
    full_nll = {}
    for step in CLOCK:
        actual_state = clock[step]
        run.restore(actual_state, step)
        record = final["validation"][CLOCK.index(step)]
        checked = run.evaluate(step)
        require(checked["step"] == record["step"] and checked["rows"] == record["rows"]
                and checked["fused_nll"] == record["fused_nll"]
                and native.state_hash(run.model) == record["state_sha256"],
                "Full-band production call or calibrated selection differs from the archived event")
        full_nll[step] = checked["fused_nll"]
    selection = min(CLOCK, key=full_nll.__getitem__)
    require(selection == final["selected"]["step"] == receipt["selected"]
            and native._exact(final["selected"]["state"], clock[selection]["model"]),
            "First-minimum validation selection or frozen selected weights differ")

    if name == REFERENCE and plan["scope"] == "synthetic_only":
        ref = place/"reference"/name
        reference = checkpoints(root, ref, contract, read_all=True)
        reference_receipt = json_pin(root, file_pin(ref/"receipt.json"))
        require(reference_receipt["checkpoint"] == reference[-1]["pin"]
                and reference_receipt["registration"] == contract
                and reference_receipt["updates"] == STEPS and native._exact(reference[-1]["item"]["training"], final)
                and any(row["reason"] == "SIGTERM" for row in journals),
                "Interrupted/restarted trajectory differs from original continuous reference")
        reference_proof = dict(state_optimizer_rng_history_exact=True, checkpoint=reference_receipt["checkpoint"]["sha256"],
                               extra_updates=STEPS, drained_at=[row["step"] for row in journals if row["reason"] == "SIGTERM"],
                               seconds=sum(row["wall_seconds"] for row in reference))
    else:
        reference_proof = None

    blocks = provider.verify_confirmation()
    views = {}
    for stage, step in (("initial",0), ("middle",750), ("selected",selection), ("last",1500)):
        run.model.load_state_dict(clock[step]["model"], strict=True)
        before = native.state_hash(run.model)
        views[stage] = read_confirmation(run, provider, blocks, step)
        require(native.state_hash(run.model) == before, "Frozen confirmation changed a model")
    provider.bank.verify_authority(provider.bank.pin)
    require_frozen_code(authority)
    journal = [{key: row[key] for key in ("step","start","updates","reason","wall_seconds","timing")} for row in journals]
    totals = {key: sum(row["timing"][key] for row in journal) for key in journal[0]["timing"]}
    first_durable, last_durable = (Path(row["pin"]["path"]).stat().st_mtime_ns for row in (journals[0], journals[-1]))
    require(last_durable >= first_durable, "Checkpoint durable timestamps are out of order")
    result = dict(policy=plan["schema"], status="passed", name=name, profile=plan["cases"][name],
                  freeze=freeze_pin, source=original, consumer=authority["commit"], updates=0,
                  selected=selection, full_validation_nll=full_nll, full_validation_bitwise=True,
                  proposal=plan["design"], exposure=digest(spec["training"]), released_inputs=digest(spec["inputs"]),
                  initial_shared=spec["initialization"]["source"], initial_state=spec["initialization"]["state"],
                  raw_gt_loaded=False, query_accessed=False, phases=views, device=execution.numeric,
                  cost=dict(training_exposure=totals, first_validation=journal[0]["timing"]["validation_seconds"],
                            checkpointed_fit_seconds=sum(row["wall_seconds"] for row in journal),
                            first_to_last_durable_seconds=(last_durable-first_durable)/1e9,
                            checkpoint_bytes=sum(Path(row["pin"]["path"]).stat().st_size for row in journals),
                            verification_wall_seconds=perf_counter()-start),
                  journal=journal, reference=reference_proof)
    return _write(out, result)


def blocked_history(root, pin):
    previous = json_pin(root, pin)
    require(previous["policy"] == "wp12_frozen_development_stages_v1"
            and previous["original_archive"]["checked_cases"] == 54
            and previous["query_accessed"] is previous["raw_gt_loaded"] is False,
            "Only the trusted prior WP12 development scope may be reused")
    proof = previous["honghu"]
    require(len(proof["accepted"]) == len(proof["missing"]) == 10 and proof["added"] == 0
            and len(set(proof["accepted"]) | proof["missing"].keys()) == 20
            and all(row["status"] == "not_started" and row["stage_files"] == row["case_files"] == 0
                    for row in proof["missing"].values()), "Unweighted historical cells cannot become long evidence")
    return dict(prior_receipt=pin["sha256"], independent_accepted=10, without_checkpoint=sorted(proof["missing"]),
                new_replays=0, new_updates=0)


def publish(root, place, out, authority, producer, freeze_pin, receipts, history_pin=None):
    plan, frozen_value = frozen(root, place, producer, freeze_pin)
    all_cases = {}
    for name, expected in plan["cases"].items():
        artifact = receipts/f"{name}.json"
        row = json_pin(root, file_pin(artifact))
        require(row["profile"] == expected and row["name"] == name and row["status"] == "passed"
                and row["freeze"] == freeze_pin and row["source"] == frozen_value["cases"][name]
                and row["consumer"] == authority["commit"] and row["full_validation_bitwise"]
                and row["updates"] == 0 and row["raw_gt_loaded"] is row["query_accessed"] is False,
                "Incomplete or foreign independent audit")
        all_cases[name] = row
    effects = {}
    arithmetic_pairs = {}
    cohorts = defaultdict(list)
    for scene, seed in product(SCENES, SEEDS):
        base, active = (all_cases[f"{scene}-m{seed}-{arm}"] for arm in ARMS)
        require(all(base[key] == active[key] for key in ("exposure", "released_inputs", "initial_shared")),
                "Coverage arm changed shared initialization, physical observation or opportunity")
        arithmetic_key = lambda row: {k: v for k,v in row["device"].items() if k != "physical_uuid"}
        arithmetic_pairs[f"{scene}/m{seed}"] = arithmetic_key(base) == arithmetic_key(active)
        pair = {}
        for stage, view, head in product(("initial","middle","selected","last"), VIEWS, HEADS):
            a,b = (row["phases"][stage]["heads_by_view"][view] for row in (base,active))
            require(a["point_roster"] == b["point_roster"] and a["band_ids"] == b["band_ids"],
                    "Coverage arm changed confirmation role or physical band IDs")
            x,y = a["heads"][head],b["heads"][head]
            require(x["support"] == y["support"], "Coverage arm changed class support")
            delta = [after-before for before,after in zip(x["recall"],y["recall"],strict=True)]
            pair[f"{stage}/{view}/{head}"] = dict(oa_delta=y["oa"]-x["oa"], aa_delta=y["aa"]-x["aa"],
                      class_recall_delta=delta, rare_pass=min(delta[-2:]) >= -DESIGN["rare_recall_max_loss"])
            for arm, value in zip(ARMS,(x,y),strict=True):
                cohorts[(scene,arm,stage,view,head)].append((value["oa"],value["aa"],value["recall"]))
        effects[f"{scene}/m{seed}"] = pair
    summaries = {}
    for key, values in cohorts.items():
        require(len(values) == 3, "Three independent model seeds are required for SD")
        metrics = dict(zip(("oa","aa","class_recall"), zip(*values,strict=True),strict=True))
        summary = {k: dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for k,v in metrics.items() if k != "class_recall"}
        summary["class_recall"] = [dict(mean=statistics.mean(col),sample_sd=statistics.stdev(col))
                                   for col in zip(*metrics["class_recall"],strict=True)]
        summaries["/".join(key)] = summary
    synthetic = None
    if plan["scope"] == "synthetic_only":
        selected = [row["phases"]["selected"]["heads_by_view"]["central_gap"]["heads"]["fused"]["recall"]
                    for row in all_cases.values()]
        deleted = [statistics.mean(row[:2]) for row in selected]
        redundant = [statistics.mean(row[2:4]) for row in selected]
        threshold = DESIGN["synthetic_redundant_min_aa"]
        ceiling = DESIGN["synthetic_deleted_max_aa"]
        synthetic = dict(deleted_max_aa=ceiling, redundant_min_aa=threshold,
                         deleted_within_threshold=sum(x <= ceiling for x in deleted),
                         redundant_meets_threshold=sum(x >= threshold for x in redundant),
                         total=len(selected), extra_reference_updates=STEPS)
        require(all_cases[REFERENCE]["reference"]["state_optimizer_rng_history_exact"], "Required recovery proof missing")
    historical = None if history_pin is None else blocked_history(root, history_pin)
    result = dict(status="passed", scope=plan["scope"], policy=plan["schema"], producer=producer["commit"],
                  consumer=authority["commit"], freeze=freeze_pin, optimizer_updates=12*STEPS,
                  new_audit_optimizer_updates=0, cases=all_cases, sample_sd=summaries,
                  matched_arms=effects, matched_arithmetic=arithmetic_pairs,
                  synthetic_control=synthetic, historical_missing=historical,
                  query_accessed=False, raw_gt_loaded=False)
    require_frozen_code(authority)
    return _write(out, result)
