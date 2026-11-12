"""Freeze completed intervals and independently replay support and endpoint evidence."""

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.native_support_reference import sparse_forward, support_trace
from inference.native_support_checkpoint import load_endpoint, predict_registered_tile
from model.research_config import digest
from research_runtime import external_directory, tensor_hash
from transfer import native_support_campaign as campaign, registered_development as dev
from transfer.curriculum_training import CurriculumRun
from transfer.g3_training import _forward, state_hash
from transfer.native_support import make_selector, native_forward
from transfer.xuzhou_multiscale_audit import HEADS, SCIENCE, _exact


def _case_arms(case):
    return [*case["contracts"], *(("fixed",) if case["work_package"] == "WP08" else ())]


def _attempts(folder, pin, case, arm):
    receipts, packets = {}, {}
    for name, (_, stop, previous) in campaign._stages(case["budget"]).items():
        status, receipt = campaign._read_stage(folder, pin, case, arm, name, receipts.get(previous))
        packet = campaign._load_pt(status["checkpoint"])
        if (packet["contract_hash"] != case["contracts"][arm] or status["contract"] != packet["contract_hash"]
                or digest(packet["spec"]) != packet["contract_hash"] or len(packet["history"]) != stop):
            raise ValueError("Stage packet does not match frozen contract")
        receipts[name], packets[name] = receipt, packet
    full = packets["full"]
    for name in ("resume", "completed"):
        if any(not _exact(full[key], packets[name][key]) for key in SCIENCE):
            raise ValueError("Cross-process optimizer, model, support history or RNG differ")
    if not _exact(full["history"][:campaign.RECOVERY_PREFIX], packets["prefix"]["history"]):
        raise ValueError("Independent prefix history differs")
    return receipts, full


def freeze(pin, code):
    root, reg = campaign.read_registration(pin, code)
    rows = {}
    count = 0
    for key, entry in reg["cases"].items():
        for arm in _case_arms(entry):
            folder, case, _, donor = campaign.open_case(root, pin, reg, key, arm)
            receipts, packet = _attempts(folder, pin, case, arm)
            spec = packet["spec"]
            published = {}
            for name, field, step in (("selected", "selected_state", packet["selected_step"]),
                                      ("last", "state", case["budget"])):
                path = folder / "published" / arm / (name + ".pt")
                payload = dict(version=2, registration=pin, case=key, arm=arm, endpoint=name, step=step,
                    development=case["development"], work_package=case["work_package"], config=spec["architecture"],
                    bank=case.get("bank_hash"), trial=spec.get("regional_support", {}).get("trial"),
                    conditions=spec.get("multiscale", {}).get("synthetic_conditions"), state=packet[field],
                    training_config=spec["source_training_config"], state_hash=packet["history"][step - 1]["state_hash"],
                    donor=donor, attempts=receipts)
                CurriculumRun._write(payload, path)
                published[name] = dev._pin(path)
            rows[f"{key}/{arm}"] = dict(attempts=receipts, endpoints=published, updates=2 * case["budget"])
            count += 2 * case["budget"]
    if count != reg["maximum_updates"]:
        raise ValueError("Frozen attempts do not exhaust the registered envelope")
    return dev._json_new(root / "freeze.json", dict(version=2, registration=pin, rows=rows, updates=count,
                                                   query=0, real_updates=0))


def _reference(bank, settings, budget):
    result = {}
    for key, entry in bank.items():
        c = entry["context"]
        result[key] = support_trace(entry["adjacency"].to_dense().numpy(), entry["spectra"].numpy(),
            entry["graph_features"].numpy(), node_ids=c["node_ids"], partitions=c["partitions"],
            mandatory=entry["mandatory"], schedule=settings["schedule"], mode=settings["mode"],
            seed=settings["support_seed"], steps=budget if c["role"] == "devtrain" else 0)
    return result


def _sample_stream(scene, record, case):
    width = scene.plan["shape"][1]
    released = record["train_support"]
    members = {}
    for tile in record["roles"]:
        if tile["role"] != "devtrain":
            continue
        y0, y1, x0, x1 = tile["box"]
        members[tile["id"]] = [(i, label) for i, label in zip(released["indices"], released["labels"], strict=True)
                                 if y0 <= i // width < y1 and x0 <= i % width < x1]
    classes = record["classes"]
    choices = {c: [tile for tile, items in members.items() if any(label == c for _, label in items)] for c in classes}
    rng = np.random.Generator(np.random.PCG64(case["sampler_seed"]))
    for step in range(1, case["budget"] + 1):
        anchor = classes[int(rng.integers(len(classes)))]
        pool = choices[anchor]
        tile = pool[int(rng.integers(len(pool)))]
        yield dict(step=step, anchor=anchor, tile_id=tile, support_ids=[i for i, _ in members[tile]])


def _history(packet, data, case, arm):
    _, scene, record, _, bank = data
    history = packet["history"]
    selectors = {}
    support_counts = Counter()
    range_rows = []
    activity = Counter()
    settings = packet["spec"].get("regional_support", {}).get("trial")
    reference = _reference(bank, settings, case["budget"]) if settings is not None else {}
    for key, history_reference in reference.items():
        selector = make_selector(bank[key], settings, settings["support_seed"])
        for step, row in enumerate(history_reference):
            selector.advance(step)
            actual = selector.snapshot()
            if any(actual[name] != value for name, value in row.items()):
                raise ValueError(f"Independent global support schedule differs at {key}:{step}")
            support_counts["schedule_states"] += 1
    previous = {}
    for event, row in zip(_sample_stream(scene, record, case), history, strict=True):
        if any(row[key] != value for key, value in event.items()):
            raise ValueError("Independent released support sample stream differs")
        activity.update(row["gradient_parameters"])
        if settings is None:
            continue
        key = str(event["tile_id"])
        step = case["budget"] if arm == "fixed" else event["step"]
        state = row["support"]["snapshot"]
        if any(state[name] != value for name, value in reference[key][step].items()):
            raise ValueError(f"Independent pool, support, ranking or RNG differs at {key}:{step}")
        context = bank[key]
        ids = context["context"]["node_ids"]
        original = set(map(tuple, context["adjacency"].indices().T.tolist()))
        pairs = set(map(tuple, state["selected"]))
        in_deg = Counter(i for i, _ in pairs)
        out_deg = Counter(j for _, j in pairs)
        distance = [abs(ids[i] // 256 - ids[j] // 256) + abs(ids[i] % 256 - ids[j] % 256) for i, j in pairs]
        range_rows.append(dict(step=event["step"], tile=event["tile_id"], nnz=len(pairs),
            remote=len(pairs - original), receive_degree=[in_deg[i] for i in range(len(ids))],
            send_degree=[out_deg[i] for i in range(len(ids))], distances=dict(Counter(map(str, distance))),
            support_sha256=digest(state["selected"]), pool_sha256=digest(state["pool"]),
            changed_since_visit=None if key not in previous else pairs != previous[key],
            refresh_step=state["refresh_step"], budget=state["budget"], pool=len(state["pool"])))
        previous[key] = pairs
        support_counts["rows"] += 1
        for name, evidence in row["scorers"].items():
            selectors.setdefault(name, Counter())
            selectors[name]["positive_gradients"] += (evidence["gradient_l1"] or 0.) > 0.
            selectors[name]["updates"] += evidence["changed"] > 0
    order = packet["spec"]["parameters"]
    optimizer = packet["optimizer"]
    names = dict(zip(optimizer["param_groups"][0]["params"], order, strict=True))
    if {names[i]: int(value["step"]) for i, value in optimizer["state"].items()} != activity:
        raise ValueError("Adam optimizer state differs from actual gradient history")
    if any(min(v.values()) <= 0 for v in selectors.values()):
        raise ValueError("A native scorer did not receive positive gradient and update")
    step = min(history, key=lambda row: row["validation"]["nll"])["step"]
    if step != packet["selected_step"]:
        raise ValueError("Independent first-min selection differs")
    return dict(support_steps=support_counts["rows"], schedule_states_checked=support_counts["schedule_states"],
                scorer_activity=selectors, support_geometry=range_rows,
                source_groups=packet["spec"].get("regional_support", {}).get("parameters"),
                scales=[row["scales"] for row in history] if case["work_package"] == "WP07" else None,
                parameter_order=order, exposure=packet["progress"]["exposure"])


def _endpoint(package, pin, case, data, training, *, original=None):
    _, scene, record, _, bank = data
    model, value = load_endpoint(package, pin, case, data)
    release = record["validation_support"]
    labels = dict(zip(release["indices"], release["labels"], strict=True))
    width = scene.plan["shape"][1]
    matrices = np.zeros((3, scene.classes, scene.classes), dtype=np.int64)
    losses = np.zeros(3)
    probabilities = []
    check = []
    checked_labels = set()
    with torch.no_grad():
        for tile in record["roles"]:
            if tile["role"] != "devval":
                continue
            y0, y1, x0, x1 = tile["box"]
            ids = [i for i in labels if y0 <= i // width < y1 and x0 <= i % width < x1]
            if not ids:
                continue
            checked_labels.update(ids)
            position = [(i // width - y0) * (x1 - x0) + i % width - x0 for i in ids]
            targets = np.asarray([labels[i] - 1 for i in ids])
            output, detail = predict_registered_tile(model, value, scene, record, tile["id"], bank, detail=True)
            probabilities.append([tensor_hash(p) for p in output[:3]])
            if original is not None:
                if value["trial"] is not None:
                    replay, _ = native_forward(original.model, scene, scene.plan["tiles"][tile["id"]],
                                               selector=original.devval[str(tile["id"])])
                elif case["work_package"] == "WP07":
                    with original._conditions(original.model):
                        replay = _forward(original.model, scene, scene.plan["tiles"][tile["id"]],
                                          multiscale_condition_only=True)
                else:
                    replay = _forward(original.model, scene, scene.plan["tiles"][tile["id"]])
                if any(not torch.equal(a, b) for a, b in zip(output[:3], replay[:3], strict=True)):
                    raise ValueError("Published three-head probabilities differ from the original runner")
            for i, pred in enumerate(output[:3]):
                probs = pred.numpy()[position]
                np.add.at(matrices[i], (targets, probs.argmax(1)), 1)
                losses[i] += float(np.log(np.maximum(probs[np.arange(len(ids)), targets], 1e-8)).sum())
            if detail:
                conv = model.backbone.HGNN_Branch.hgc2
                expected = sparse_forward(detail["A0"].to_dense().numpy(), detail["score"].numpy(),
                    detail["trace"]["snapshot"]["selected"], detail["stage1"].numpy(), conv.weight.numpy(),
                    0. if conv.bias is None else conv.bias.numpy(), detail["Q"].to_dense().numpy())
                actual = [detail["stage2"].values().numpy(), detail["region_output"].numpy(), detail["pixel_graph"].numpy()]
                errors = [float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)]
                if any(not np.allclose(a, b, atol=3e-6, rtol=1e-5) for a, b in zip(expected, actual, strict=True)):
                    raise ValueError("NumPy value/aggregation oracle differs")
                check.append(dict(tile=tile["id"], errors=errors, values=sum(a.size for a in actual)))
    if checked_labels != labels.keys() or not (matrices.sum(axis=2) == 5).all():
        raise ValueError("Independent endpoint omitted released labels")
    metrics = {head: dict(metrics_from_confusion(matrix, record["classes"]), nll=-float(loss) / len(labels))
               for head, matrix, loss in zip(HEADS, matrices, losses, strict=True)}
    if (training["validation"]["confusion"] != metrics["fused"]["confusion_matrix"]
            or training["validation"]["nll"] != metrics["fused"]["nll"]):
        raise ValueError("Independent endpoint differs from actual production validation")
    return dict(step=value["step"], probability_hashes=probabilities, metrics=metrics, sparse_oracle=check,
                package=package, bytes=Path(package["path"]).stat().st_size,
                runner_probability_replay=original is not None)


def load_frozen(pin, frozen, code):
    root, reg = campaign.read_registration(pin, code)
    value = json.loads(dev._read(frozen))
    if value["registration"] != pin or value["updates"] != reg["maximum_updates"] or value["query"] or value["real_updates"]:
        raise ValueError("Frozen publication differs")
    expected = {f"{key}/{arm}" for key, case in reg["cases"].items() for arm in _case_arms(case)}
    if value["rows"].keys() != expected:
        raise ValueError("Frozen publication omits registered arms")
    return root, reg, value


def audit(pin, frozen, code, producer=None, *, output=None):
    started = perf_counter()
    root, reg, value = load_frozen(pin, frozen, code if producer is None else producer)
    results = {}
    for identity, row in value["rows"].items():
        key, arm = identity.split("/")
        folder, case, data, donor = campaign.open_case(root, pin, reg, key, arm)
        receipts, packet = _attempts(folder, pin, case, arm)
        if receipts != row["attempts"]:
            raise ValueError("Audit stage pins differ from publication")
        endpoints = {}
        original = campaign._factory(case, data, arm, donor=donor)[0]()
        for name, package in row["endpoints"].items():
            step = packet["selected_step"] if name == "selected" else case["budget"]
            state = packet["selected_state" if name == "selected" else "state"]
            original.model.load_state_dict(state, strict=True)
            original.model.eval()
            if state_hash(original.model) != packet["history"][step - 1]["state_hash"]:
                raise ValueError("Original runner endpoint tensors differ")
            endpoints[name] = _endpoint(package, pin, case, data, packet["history"][step - 1], original=original)
        results[identity] = dict(work_package=case["work_package"], endpoints=endpoints,
            history=_history(packet, data, case, arm), recovery_exact_fields=list(SCIENCE), updates=row["updates"])
        print(f"audited {identity}", flush=True)
    for identity, row in results.items():
        key, arm = identity.split("/")
        if row["work_package"] != "WP08":
            continue
        control = results[f"{key}/off"]["endpoints"]
        row["synthetic_recall_delta_vs_off"] = {
            endpoint: {head: dict(
                delta_pp_by_class={str(a["class_id"]): 100 * (a["recall"] - b["recall"]) for a, b in zip(
                    metrics["per_class"], control[endpoint]["metrics"][head]["per_class"], strict=True)},
                meets_descriptive_floor=all(a["recall"] - b["recall"] >= -.02 for a, b in zip(
                    metrics["per_class"], control[endpoint]["metrics"][head]["per_class"], strict=True)))
                for head, metrics in data["metrics"].items()} for endpoint, data in row["endpoints"].items()}
    target = root / "audit.json" if output is None else external_directory(output)
    return dev._json_new(target, dict(version=2, registration=pin, freeze=frozen, results=results,
        producer=reg["code"], consumer=code["commit"], updates=value["updates"], audit_updates=0, query=0,
        synthetic_recall_floor=dict(delta_pp=-2., source="Issue411/PR412 per-class development recall floor used descriptively",
                                    scope="per_class_recall_only", candidate_admission=False), wall_seconds=perf_counter() - started))
