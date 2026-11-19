"""Load frozen WP05, WP07 and WP08 models without changing their graph semantics."""

from contextlib import nullcontext
from time import perf_counter

import torch

from evaluation.frozen_training_stages import json_pin, pinned, require
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from transfer.conditional_multiscale import ConditionalMultiscaleVCASAHGCN
from transfer.multiscale_training import MultiscaleTrial
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.region_message_attention import RegionMessageVCASAHGCN, profile_region_messages

from transfer import wp_region_graph as regional


BACKENDS = ("native", "multiscale", "aligned", "region_message")
TRAIN_BACKENDS = (*BACKENDS,regional.BACKEND)


def make_model(architecture, backend):
    if backend == regional.BACKEND:
        require(set(architecture) in ({"format_version","class_counts","backbone_kwargs","regional_support"},
                                      {"format_version","class_counts","backbone_kwargs","regional_support","training_config"})
                and architecture["format_version"]==2,
                "Exact native regional archive architecture required")
        support = architecture["regional_support"]
        require(set(support)=={"version","scorer_seed","device"} and support["version"]==2
                and support["device"]=="cpu_fp32","The frozen native scorer architecture differs")
        model = regional.RegionSupportModel(architecture["class_counts"],scorer_seed=support["scorer_seed"],
                                             **architecture["backbone_kwargs"]).eval()
        model.training_config = architecture.get("training_config",{})
        model.backbone.adapter.cache_attention = False
        return model
    require(set(architecture) in ({"format_version", "class_counts", "backbone_kwargs"},
                                 {"format_version", "class_counts", "backbone_kwargs", "training_config"})
            and architecture["format_version"] == 1 and backend in BACKENDS,
            "An exact registered architecture and supported backend are required")
    counts, kwargs = architecture["class_counts"], architecture["backbone_kwargs"]
    require(counts and all(type(value) is int and value > 1 for value in counts.values()),
            "Model head must have explicit positive class identities")
    multiscale = {"cnn_multiscale", "cnn_dilation", "graph_multiscale", "graph_radius"} & kwargs.keys()
    require(("region_attention" in kwargs) == (backend == "region_message")
            and bool(multiscale) == (backend == "multiscale")
            and (backend != "region_message" or isinstance(kwargs["region_attention"], dict))
            and (not multiscale or {"cnn_multiscale", "cnn_dilation"} <= multiscale),
            "Architecture disagrees with the declared backend")
    factory = {"native": MultiDatasetVCASAHGCN, "aligned": MultiDatasetVCASAHGCN,
               "multiscale": ConditionalMultiscaleVCASAHGCN, "region_message": RegionMessageVCASAHGCN}[backend]
    with torch.random.fork_rng(devices=[]):
        model = factory(counts, **kwargs).eval()
    b = model.backbone
    exchanged = backend in ("aligned", "region_message")
    require(b.graph_mode == b.graph_assignment_mode == b.graph_adjacency_mode == "learned"
            and (b.interaction is not None) == exchanged
            and (not exchanged or b.alignment.mode == "region")
            and b.graph_router is None and b.edge_updater is None,
            "Checkpoint uses an incompatible graph or interaction contract")
    model.training_config = architecture.get("training_config", {})
    model.backbone.adapter.cache_attention = False
    return model


def restore(architecture, backend, state, expected_state, *, device="cpu"):
    model = make_model(architecture, backend)
    reference = model.state_dict()
    require(state.keys() == reference.keys() and all(
        torch.is_tensor(value) and value.shape == reference[key].shape and value.dtype == reference[key].dtype
        and (not value.is_floating_point() or torch.isfinite(value).all().item())
        for key, value in state.items()), "Frozen tensors disagree with registered architecture or precision")
    model.load_state_dict(state, strict=True)
    require(state_hash(model) == expected_state, "Frozen state fingerprint differs")
    return model.to(device).eval().requires_grad_(False)


def load_export(root, export, *, device="cpu"):
    fields = {"backend", "weights", "metadata", "state"}
    region = export["backend"]==regional.BACKEND
    require(set(export)==(fields|{"graph_program"} if region else fields), "Exact checkpoint export pins required")
    weights = pinned(root, export["weights"])
    metadata = pinned(root, export["metadata"])
    require(metadata == weights.with_suffix(weights.suffix + ".json"), "Weights and architecture are not sidecars")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    architecture = json_pin(root, export["metadata"])
    model = restore(architecture,export["backend"],state,export["state"],device=device)
    if region:
        graph = export["graph_program"]
        fields = {"policy","seed","budget","weights","state"}
        controlled = graph["policy"] in regional.CAUSAL_POLICIES
        require(set(graph)==(fields|{"causal_design"} if controlled else fields),
                "Regional export lacks its frozen source or graph witness")
        anchor = torch.load(pinned(root,graph["weights"]),map_location="cpu",weights_only=True)
        source = restore(architecture,regional.BACKEND,anchor,graph["state"])
        causal = json_pin(root,graph["causal_design"]) if controlled else None
        model.graph_program = regional.program_for(source,graph["policy"],graph["seed"],graph["budget"],causal)
    return model


def semantic_identity(model, backend):
    bone = model.backbone
    first = bone.CNN_Branch[0]
    identity = dict(backend=backend, assignment=bone.graph_assignment_mode, adjacency=bone.graph_adjacency_mode,
                cnn_scales=dict(mode=first.mode, dilation=first.dilation) if backend=="multiscale" else None,
                range_mode=getattr(model, "graph_multiscale", "off"), graph_radius=getattr(model, "graph_radius", 1),
                interaction=bone.interaction_config.interaction_mode,
                direction=bone.interaction_config.interaction_direction if bone.interaction is not None else None,
                attention=getattr(model, "region_attention", None))
    if backend==regional.BACKEND:
        program = model.graph_program
        identity["regional_support"] = dict(version=regional.VERSION,policy=program.policy,seed=program.seed,
                                             budget=program.budget,anchor=state_hash(program.anchor),scale=regional.SCALE)
    return identity


DIRECTIONS = ("both", "cnn_to_graph", "graph_to_cnn")
GATES = dict(alignment_mode="region", interaction_mode="gated", interaction_strength=.1,
             interaction_hidden_dim=32, interaction_chunk_pixels=16384, graph_update_mode="none")
ATTENTION = dict(width=32, landmark_grid=(2,2), top_k=3, query_chunk=16)


def train_variant(base, spec, seed):
    """Map an identical shared start to each registered scale or directional architecture."""
    backend = spec["backend"]
    if backend == regional.BACKEND:
        require(set(spec)=={"backend","policy"} and spec["policy"] in regional.POLICIES,
                "Explicit and bounded regional stage-two policy required")
        return regional.RegionSupportModel.initialize(base,seed)
    if backend == "native":
        require(set(spec) == {"backend"}, "Unexpected native baseline switches")
        return base
    if backend == "multiscale":
        require(set(spec) == {"backend","trial"}, "Expected one explicit WP07 scale trial")
        model = MultiscaleTrial(**spec["trial"]).initialize(base,seed).eval()
        model.backbone.adapter.cache_attention = False
        return model
    require((backend == "aligned" and set(spec) == {"backend"}) or
            (backend == "region_message" and set(spec) == {"backend","mode","direction"}
             and spec["mode"] in ("local","combined") and spec["direction"] in DIRECTIONS),
            "Expected an aligned gate or a registered paired directional backend")
    cfg = base.export_config()
    shared = base.state_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cfg["backbone_kwargs"].update(GATES, interaction_direction="both")
        aligned = make_model(cfg,"aligned")
    initial = aligned.state_dict()
    require(shared.keys() <= initial.keys() and all(k.startswith("backbone.interaction.") for k in initial.keys()-shared.keys()),
            "Alignment must extend the original shared inventory")
    aligned.load_state_dict({**initial,**shared},strict=True)
    if backend == "aligned":
        return aligned
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cfg["backbone_kwargs"]["region_attention"] = dict(mode=spec["mode"],**ATTENTION)
        model = make_model(cfg,"region_message")
    outer = "backbone.interaction."
    mapped = {outer+"exchange."+k.removeprefix(outer) if k.startswith(outer) else k:v
              for k,v in aligned.state_dict().items()}
    complete = model.state_dict()
    fresh = {f"{outer}{branch}.{projection}.weight" for branch in ("to_pixels","to_regions")
             for projection in ("query","key","value","output")}
    require(mapped.keys() <= complete.keys() and complete.keys()-mapped.keys()==fresh
            and all(complete[k].shape==v.shape and complete[k].dtype==v.dtype for k,v in mapped.items()),
            "Region projections and renamed outer gate no longer match the archived layout")
    model.load_state_dict({**complete,**mapped},strict=True)
    direction = spec["direction"]
    if direction != "both":
        cfg["backbone_kwargs"]["interaction_direction"] = direction
        active = make_model(cfg,"region_message")
        obsolete = ("graph_to_cnn","to_pixels") if direction=="cnn_to_graph" else ("cnn_to_graph","to_regions")
        reject = (outer+"exchange."+obsolete[0]+".",outer+obsolete[1]+".")
        active.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith(reject)},strict=True)
        model = active
    return model


@torch.no_grad()
def run_block(model, actual, producer, role, execution):
    """Run the existing native observation forward with exact bands, no truth or optimizer."""
    timing, message = {}, {}
    manager = profile_region_messages(message) if isinstance(model, RegionMessageVCASAHGCN) else nullcontext()
    execution.sync()
    start = perf_counter()
    with manager:
        if isinstance(model,regional.RegionSupportModel):
            output, graph, timing = regional.forward_region(model,actual,producer,role,execution,model.graph_program)
        else:
            output, graph = forward(model, actual, producer, role, execution=execution, timed=timing)
    execution.sync()
    stats = dict(elapsed=perf_counter()-start, graph_seconds=sum(timing.get("graph_seconds", [])),
                 message_seconds={name: sum(values) for name, values in message.items()},
                 message_trace=output[3].get("interaction", {}).get("region_message", {}))
    require(len(output) >= 4 and all(value.shape[0] == actual.receipt["pixels"] for value in output[:3]),
            "Backend did not emit the original three point heads")
    if "regional" in timing:
        stats["regional_seconds"] = timing["regional"]
    return output[:3], graph, stats
