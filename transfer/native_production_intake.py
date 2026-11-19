"""Read-only preregistration for a separately approved future real production comparison."""

from dataclasses import asdict

from evaluation.frozen_training_stages import json_pin, pinned, require
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from transfer.native_independent_release import open_independent
from transfer.native_production_fixture import SEEDS, VIEWS
from transfer.native_production_model import ARMS, Switches
from transfer.native_production_provider import ProductionProvider

REQUEST = "wp05_wp06_real_production_intake_v1"
APPROVAL = "wp05_wp06_real_production_scope_approval_v1"


def open_legacy_weights(path, device, *, architecture, switches, axis):
    """Use the existing strict weights-only checkpoint import with an explicit contract."""
    require(switches.arm == "native" and switches.spectral == switches.residual == "none"
            and axis and tuple(axis) == tuple(sorted(set(axis))), "Legacy restore needs an exact unchanged native topology")
    model = load_model_checkpoint(path,device,config=architecture)
    require(model.backbone.graph_mode == "learned" and model.backbone.coverage_mode == "none"
            and model.backbone.region_residual_mode == "none", "Legacy native topology differs from the declaration")
    model.backbone.adapter.cache_attention = False
    return model


def real_preflight(root, request_pin, authority_pin, producer):
    request, authority = (json_pin(root,pin) for pin in (request_pin,authority_pin))
    require(set(request) == {"schema","independent_manifest","independent_authority","source",
                             "axis","model_seeds","arms","spectral","residual","views","updates_per_arm","graph_scale"}
            and request["schema"]==REQUEST and request["model_seeds"]==list(SEEDS)
            and request["arms"]==list(ARMS) and request["views"]==list(VIEWS)
            and type(request["updates_per_arm"]) is int and request["updates_per_arm"]>0,
            "Real comparison request must bind all unchanged paired controls and a fixed positive budget")
    expected = dict(schema=APPROVAL,approved_request_sha256=request_pin["sha256"],producer=producer,
                    use="read_only_preflight",train_authorized=False)
    require(authority==expected, "Scope requires a separate trusted preflight approval")
    bank,result,roles = open_independent(root,request["independent_manifest"],request["independent_authority"])
    require(bank.release.profile!="synthetic" and result["uses"]["independent_accuracy"]["eligible"]
            and result["uses"]["cost"]["eligible"], "Real intake needs independent protected roles and cost authorization")
    source = request["source"]
    require(set(source) == {"weights","metadata","state"}, "Prior checkpoint needs weights, verified metadata and state hash")
    for item in (source["weights"],source["metadata"]):
        pinned(root,item)
    providers = {}
    for arm in ARMS:
        switches = Switches(arm=arm,spectral=request["spectral"],
                            residual="none" if arm=="pixel" else request["residual"],
                            graph_scale=request["graph_scale"])
        # Pass approved case identity through the shared production provider. No optimizer is instantiated.
        provider = ProductionProvider(root,bank,roles,axis=request["axis"],config=switches,
                    source=source,seed=SEEDS[0],budget=request["updates_per_arm"],producer=producer,
                    approval=authority_pin["sha256"])
        model = provider.model_factory(provider.baseline(SEEDS[0]),arm,SEEDS[0]+7321)
        providers[arm] = dict(switches=asdict(switches),parameters=sum(p.numel() for p in model.parameters()))
    require(len({providers[arm]["parameters"] for arm in ARMS[:3]})==1,
            "Real arms must use the same native parameter inventory")
    return dict(schema=REQUEST,scope="read_only_preflight",request=request_pin["sha256"],authority=authority_pin["sha256"],
                producer=producer,bank=bank.pin["sha256"],roles=digest(roles),qualification=result["uses"],
                providers=providers,real_updates=0,independent_real_accuracy=None,query_accessed=False,raw_gt_loaded=False)
