"""One frozen WP11 manifest for static, fallback and admitted routed execution."""

from contextlib import ExitStack
import resource
from time import perf_counter_ns
from unittest.mock import patch

import torch

from evaluation.wp11_confirmation import admit, load_held_aside_tiles
from evaluation.wp11_model_pair import LIGHT_PINS, MODEL_FILES, forbid_aliases, load_pair
from evaluation.wp11_static_cost import PATHS, elapsed_ms, environment, load_static, output_hashes, prepare_requests, scan
from inference import tile_cascade
from inference.route_controls import _output_hash, random_cascade, request_hash
from inference.trained_light import trained_cascade
from model.research_config import digest
from research_runtime import tensor_hash, write_json
from scripts.prepare_wp11_confirmation import resolve_confirmation
from transfer.g3_training import state_hash
from transfer.observation_policy import scene_identity
from transfer.registered_development import load_development
from transfer.registered_wp11_learning import REAL, RELEASE, read, verify


POLICY = "wp11-unified-execution-v1"
MODES = (*PATHS, "all_fallback", "adaptive", "random")
ROUTED = {"adaptive", "random"}


def model_spec(release, pin):
    return {**{key: release[key] for key in (*MODEL_FILES, *LIGHT_PINS)}, "matched_release": pin}


def release_metadata(pin):
    release = read(pin)
    if release["policy"] != RELEASE:
        raise ValueError("Use an audited matched release")
    original = read(release["registration"])
    request = read(original["request"])
    if (original["commit"] != release["commit"] or original["source_state"] != release["source_state"]
            or request["inputs"]["development"] != release["development"]):
        raise ValueError("Matched release and original registration differ")
    scope = request["scope"]
    if scope not in ("synthetic", REAL):
        raise ValueError("Unrecognized release scope")
    return release, original, scope, [pin, release["registration"], original["request"]]


def release_context(pin):
    release, original, scope, metadata = release_metadata(pin)
    scene, record, files = load_development(release["development"])
    full, light, history, models = load_pair(model_spec(release, pin), scene, record)
    return release, original, scope, scene, record, full, light, history, [*metadata, *files, *models]


def verify_cost(pin, release_pin, release, input_hash):
    cost = read(pin)
    if (cost["status"] != "passed" or cost["release"] != release_pin or cost["input_hash"] != input_hash
            or cost["commit"] != release["commit"] or cost["scope"] != "existing_devval_image_only"
            or cost["environment"] != environment() or cost["query"] != 0 or cost["optimizer_updates"] != 0
            or set(cost["summary"]) != set(PATHS)):
        raise ValueError("Existing cost evidence is for different code, models, inputs or timing")
    if (cost["summary"]["full_source"]["weights"] != release["source_state"]
            or cost["summary"]["full_matched"]["weights"] != release["full_state"]):
        raise ValueError("Existing cost evidence uses a different frozen model")
    return dict(file=pin, scope=cost["environment"]["scope"], blocks=cost["blocks"],
                pixels=cost["pixels"], repeats=len(cost["resident"]),
                startup_repeats={k: len(v) for k, v in cost["startup"].items()})


def confirmed_routes(spec, release_pin, full, light, scene, record, history):
    """Replay frozen authorization metadata, with no access to held-aside labels."""
    if set(spec) != {"registration", "receipt"}:
        raise ValueError("Exact confirmation registration and execution receipt required")
    reg = read(spec["registration"])
    if reg.get("matched_release") != release_pin:
        raise ValueError("Confirmation is for a different matched model pair")
    labels = reg["confirmation_labels"]
    inputs = [spec["registration"], spec["receipt"], reg["admission"], reg["gate_protocol"],
              reg["gate_freeze"], reg["fixed"], reg["curve_protocol"], reg["comparison"]]
    tiles = {row["tile_id"]: row["file"] for row in reg["tiles"]}
    inputs.extend(tiles.values())
    forbid_aliases(inputs, [labels])
    receipt = read(spec["receipt"])
    resolved = resolve_confirmation(dict(reg["confirmation_binding"], freeze=reg["gate_freeze"],
                                         confirmation_labels=labels), registration=spec["registration"])
    if (resolved["result"]["status"] != "registered" or reg["fixed"] != resolved["fixed"]
            or read(reg["curve_protocol"]) != resolved["curve"] or read(reg["comparison"]) != resolved["comparison"]
            or any(reg[k] != v for k, v in resolved["inputs"].items())):
        raise ValueError("The admitted comparison differs from its calibration freeze")
    admission = read(reg["admission"])
    gate, frozen = read(reg["gate_protocol"]), read(reg["gate_freeze"])
    result = receipt["confirmation"]
    if (receipt["status"] != "passed" or receipt["query"] != 0 or receipt["optimizer_updates"] != 0
            or receipt["registration_sha256"] != digest(admission) or receipt["full_history_sha256"] != digest(history)
            or result["protocol_hash"] != digest(gate) or result["freeze_hash"] != digest(frozen)
            or result["status"] != "empirical_pass" or result["confirmation_hash"] != admission["labels_sha256"]
            or result["selected"] != frozen["selected"] or result["inference_kwargs"].get("mode") != "adaptive"):
        raise ValueError("A successful frozen independent confirmation is required for routing")
    if len(tiles) != len(reg["tiles"]) or set(tiles) != {b["tile_id"] for b in admission["blocks"]}:
        raise ValueError("Exact held-aside observations required")
    load_held_aside_tiles(scene, tiles)
    requests = admit(full, light, scene, record, history, admission, expected_registration_hash=digest(admission))
    confirmed = {b["id"] for b in admission["blocks"] if b["role"] == "confirmation"}
    requests = [req for req in requests if req["id"] in confirmed]
    timing = receipt["timing"]
    if (timing["input_hash"] != request_hash(requests) or timing["source_state"] != state_hash(full)
            or timing["light_state"] != state_hash(light.model)
            or timing["adaptive_kwargs"] != result["inference_kwargs"]
            or timing["random_plan_hash"] != digest(timing["random_plan"])):
        raise ValueError("Confirmed runtime requests, random control or threshold differ")
    value = dict(admission=digest(admission), protocol=digest(gate), freeze=digest(frozen),
                 input_hash=request_hash(requests), options=result["inference_kwargs"],
                 random_hash=timing["random_plan_hash"], random=timing["random_plan"])
    return value, requests, [*inputs, *resolved["pins"]]


def build(request, code, out):
    spec = read(request)
    if set(spec) != {"release", "static_cost", "confirmed_routes"}:
        raise ValueError("Exact model release, cost and optional route evidence required")
    release_pin = spec["release"]
    release, origin, scope, scene, record, full, light, history, pins = release_context(release_pin)
    static = prepare_requests(scene, record)
    static_hash = request_hash(static)
    cost = None
    if spec["static_cost"] is not None:
        cost = verify_cost(spec["static_cost"], release_pin, release, static_hash)
        pins.append(spec["static_cost"])
    routing = None
    if spec["confirmed_routes"] is not None:
        routing, _, route_pins = confirmed_routes(spec["confirmed_routes"], release_pin, full, light, scene, record, history)
        pins.extend(route_pins)
    verify([request, *pins], code, out)
    if out.exists():
        raise ValueError("New external execution manifest required")
    inherited = read(release["source_evidence"])
    models = {
        "light": dict(package=release["light"], state=state_hash(light.model), heads=["light"],
                      precision="fp32", training=release["light_training_sha256"],
                      selection=light.specification["training_spec"]["selection"],
                      selected_step=light.specification["selected_step"], ancestry=release["source_history"]),
        "full_source": dict(package=release["source_weights"], metadata=release["source_metadata"],
                           state=release["source_state"], heads=["fused", "graph", "cnn"], precision="fp32",
                           training=release["source_evidence"], selection="original_min_fused_NLL_then_earliest",
                           selected_step=inherited.get("selected", {}).get("step"),
                           producer_commit=inherited.get("producer_commit"),
                           ancestry=release["source_history"]),
        "full_matched": dict(package=release["full_weights"], metadata=release["full_metadata"],
                            state=release["full_state"], heads=["fused", "graph", "cnn"], precision="fp32",
                            training=release["registration"], selection="first_min_full_devval_fused_NLL_each_update",
                            selected_step=full.training_config["wp11_matched"]["selected_step"],
                            ancestry=release["full_history"]),
    }
    value = dict(policy=POLICY, commit=code["commit"], source_hash=code["source_hash"], request=request,
                 producer=release["commit"], release=release_pin, scope=scope, models=models,
                 development=release["development"], preprocessing=history["scene_identity"],
                 classes=record["classes"], role_hash=digest(record["roles"]), exposure=origin["plan_hash"],
                 observation_role="devval_image_only", static_input=static_hash, timing=environment(),
                 archived_cost=cost, routing=routing,
                 default="full_source", original_gt_read=False, query=0)
    out.mkdir()
    write_json(out / "manifest.json", value)
    return value


def instrumentation(stack, full, stats):
    """Collect actual inclusive graph and gate intervals inside the end-to-end scan."""
    def timed(function, name):
        def invoke(*args, **kwargs):
            start = perf_counter_ns()
            result = function(*args, **kwargs)
            stats[f"{name}_calls"] += 1
            stats[f"{name}_ms"] += elapsed_ms(start)
            return result
        return invoke
    if full is not None:
        graph = full.backbone.learnable_graph
        stack.enter_context(patch.object(graph, "build_graph", new=timed(graph.build_graph, "graph")))
    stack.enter_context(patch.object(tile_cascade, "gate_score", new=timed(tile_cascade.gate_score, "gate")))


def execute(pin, code, mode, out):
    manifest = read(pin)
    if (manifest["policy"] != POLICY or manifest["commit"] != code["commit"] or manifest["source_hash"] != code["source_hash"]
            or manifest["timing"] != environment() or mode not in MODES or out.exists()):
        raise ValueError("Exact execution code, timing, registered mode and fresh output required")
    if mode in ROUTED and manifest["routing"] is None:
        raise ValueError("Adaptive and random control require the admitted independent confirmation")
    begin = perf_counter_ns()
    request = read(manifest["request"])
    if manifest["release"] != request["release"]:
        raise ValueError("Execution model manifest was substituted")
    release, _, scope, pins = release_metadata(manifest["release"])
    ready_ms = elapsed_ms(begin)
    setup = perf_counter_ns()
    scene, record, dev_files = load_development(release["development"])
    pins.extend(dev_files)
    if (scope != manifest["scope"] or scene_identity(scene, record) != manifest["preprocessing"]
            or record["classes"] != manifest["classes"] or digest(record["roles"]) != manifest["role_hash"]):
        raise ValueError("Registered preprocessing, class order or spatial roles changed")
    if mode not in ROUTED:
        requests = prepare_requests(scene, record)
        if request_hash(requests) != manifest["static_input"]:
            raise ValueError("Registered static input bytes differ")
    prepare_ms = elapsed_ms(setup)
    load_begin = perf_counter_ns()
    if mode in PATHS:
        model = load_static(release, mode)
        if state_hash(model) != manifest["models"][mode]["state"]:
            raise ValueError("Static model state differs from manifest")
        pins.extend(release[key] for key in (("source_weights", "source_metadata", "light") if mode == "light"
                    else ("source_weights", "source_metadata") if mode == "full_source"
                    else ("full_weights", "full_metadata")))
        graph_model = None if mode == "light" else model
    else:
        full, light, history, pair_pins = load_pair(model_spec(release, manifest["release"]), scene, record)
        pins.extend(pair_pins)
        if (state_hash(full) != manifest["models"]["full_matched"]["state"]
                or state_hash(light.model) != manifest["models"]["light"]["state"]):
            raise ValueError("Frozen pair states differ from manifest")
        graph_model = full
    load_ms = elapsed_ms(load_begin)
    routing_prepare_ms = None
    if mode in ROUTED:
        routing_begin = perf_counter_ns()
        routing, requests, extras = confirmed_routes(request["confirmed_routes"], manifest["release"], full, light,
                                                      scene, record, history)
        pins.extend(extras)
        if routing != manifest["routing"]:
            raise ValueError("Admitted routing and input identity differ from the manifest")
        routing_prepare_ms = elapsed_ms(routing_begin)
    identity = request_hash(requests)
    stats = dict(graph_calls=0, graph_ms=0., gate_calls=0, gate_ms=0.)
    with ExitStack() as stack:
        instrumentation(stack, graph_model, stats)
        start = perf_counter_ns()
        if mode in PATHS:
            result = scan(model, mode, requests)
        elif mode == "all_fallback":
            observed = [dict(row, coverage_known=False) for row in requests]
            result = trained_cascade(full, light, observed, mode="fast")
        elif mode == "adaptive":
            result = trained_cascade(full, light, requests, **routing["options"])
        else:
            result = random_cascade(full, light, requests, routing["random"], expected_plan_hash=routing["random_hash"])
        forward_ms = elapsed_ms(start)
    sync_begin = perf_counter_ns()
    if mode in PATHS:
        output = output_hashes(mode, result)
        counts = dict(light_calls=len(requests) if mode == "light" else 0, full_calls=stats["graph_calls"],
                      accepted_tiles=len(requests) if mode == "light" else 0, total_tiles=len(requests))
    else:
        output = dict(routed=_output_hash(result),
                      probabilities=tensor_hash([item["probabilities"] for item in result["outputs"]]))
        counts = result["counts"]
    sync_ms = elapsed_ms(sync_begin)
    total_ms = elapsed_ms(begin)
    if mode == "all_fallback":
        reference = scan(full, "full_matched", requests)
        if output["probabilities"] != tensor_hash([row[0] for row in reference]):
            raise ValueError("Full fallback output differs from the static full route")
    if stats["graph_calls"] != counts["full_calls"] or request_hash(requests) != identity:
        raise ValueError("Graph executions or registered input changed")
    verify([pin, manifest["request"], *pins], code, out)
    out.mkdir()
    result = dict(status="passed", policy=POLICY, commit=code["commit"], manifest=pin, mode=mode,
                  scope=scope, input_hash=identity, output_hashes=output, counts=counts, instrumentation=stats,
                  blocks=len(requests), pixels=sum(int(r["x"].shape[0] * r["x"].shape[1]) for r in requests),
                  graph_skips=len(requests)-stats["graph_calls"],
                  fallback_blocks=counts["full_calls"] if mode not in PATHS else 0,
                  fallback_light_already_run=counts["light_calls"]-counts["accepted_tiles"],
                  timing=dict(manifest_metadata_ms=ready_ms, development_input_prepare_ms=prepare_ms,
                              model_load_authentication_ms=load_ms, routing_prepare_ms=routing_prepare_ms,
                              resident_scan_ms=forward_ms, cpu_output_materialization_ms=sync_ms,
                              execute_to_materialized_output_ms=total_ms),
                  environment=manifest["timing"], archived_cost=manifest["archived_cost"],
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  original_gt_read=False, query=0, optimizer_updates=0)
    write_json(out / "execution.json", result)
    return result


def validate_result(manifest, expected_pin, receipt, mode):
    identity = manifest["routing"]["input_hash"] if mode in ROUTED and manifest["routing"] else manifest["static_input"]
    if (receipt.get("status") != "passed" or receipt.get("commit") != manifest["commit"]
            or receipt.get("manifest") != expected_pin or receipt.get("mode") != mode
            or receipt.get("input_hash") != identity or receipt.get("environment") != manifest["timing"]):
        raise ValueError("Existing result is bound to different code, input, mode or timing")
    return receipt
