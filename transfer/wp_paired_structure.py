"""One released-point optimizer for the registered scale and directional backends."""

from copy import deepcopy
from itertools import product
import re

import numpy as np
import torch

from evaluation.frozen_training_stages import file_pin, json_pin, pinned, require
from evaluation.wp12_cross_backend import SELECTION
from evaluation.wp12_scientific_claim import COLUMNS, PAVIA, PHYSICAL
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent, wp_cross_backend as backend, wp_support_match as support_match
from transfer import wp_region_graph as regional, wp_causal_graph_design as causal
from transfer import wp_coverage_release as coverage
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.native_production_model import Switches
from transfer.native_production_provider import ProductionProvider
from transfer.native_released_points import LOSS, MODE
from transfer.native_scale_observation import forward
from transfer.p4_runtime_device import UUID
from transfer.spatial_coverage_protocol import bank_view, model_from_source, retained_positions

SCHEMA = "wp05_wp08_paired_structure_training_v1"
SYNTHETIC = "test_external_format_only"
REAL = "external_real_execution"
AUTHORITY = "wp05_wp08_paired_structure_training_approval_v1"
VIEWS = ("full","central_gap","matched_dispersed")
MATCHES = ("equal_parameters","architecture_with_capacity_change")


def named(key):
    return isinstance(key,str) and re.fullmatch("[a-z0-9_-]+",key) is not None


def validate_group(group, releases, *, test):
    fields = {"release","seed","factor","steps","clock","arms","pairs","device","purpose"}
    require(fields<=group.keys()<=fields|{"recovery_arms"} and group["release"] in releases and type(group["seed"]) is int
            and 0<=group["seed"]<2**32 and type(group["factor"]) is int and group["factor"] in (1,2,4)
            and type(group["steps"]) is int and 1<=group["steps"]<=9999
            and group["purpose"] in (("science","recovery") if test else ("science",))
            and isinstance(group["device"],str) and (UUID.fullmatch(group["device"]) or test and group["device"]=="cpu"),
            "Invalid paired experiment scope, lattice, seed, budget or physical device")
    clock = group["clock"]
    require(isinstance(clock,list) and len(clock)>=2 and all(type(v) is int for v in clock)
            and clock==sorted(set(clock)) and clock[0]==0 and clock[-1]==group["steps"],
            "Shared calibration clock must start at zero and finish at the full budget")
    arms,pairs = group["arms"],group["pairs"]
    require(isinstance(arms,dict) and len(arms)>=2 and all(named(k) and isinstance(v,dict)
            and v.get("backend") in backend.TRAIN_BACKENDS for k,v in arms.items()), "Unknown paired model inventory")
    if "recovery_arms" in group:
        references = group["recovery_arms"]
        require(test and group["purpose"]=="recovery" and isinstance(references,list) and references
                and all(isinstance(arm,str) and arm in arms for arm in references)
                and references==sorted(set(references)), "Explicit engineering continuous reference roster required")
    require(isinstance(pairs,list) and len(pairs)>0 and all(
        isinstance(p,dict) and set(p)=={"baseline","candidate","comparison"}
        and p["baseline"]!=p["candidate"] and p["comparison"] in MATCHES
        and p["baseline"] in arms and p["candidate"] in arms for p in pairs),
        "Explicit baseline, candidate and capacity relation required")
    spectral = [spec for spec in arms.values() if spec["backend"]==coverage.BACKEND]
    require(bool(spectral)==("coverage" in releases[group["release"]]),
            "The five E1 arms need an explicit observation-exposure release")
    if spectral:
        modes = {spec.get("condition") for spec in spectral}
        required = set(coverage.MODES) if group["purpose"]=="science" else {"none","continuous"}
        require(len(spectral)==len(arms)==len(modes) and modes==required
                and all(set(s)=={"backend","condition"} for s in spectral),
                "Expected the registered E1 five-mode roster or the bounded recovery pair")
    region = [spec for spec in arms.values() if spec["backend"]==regional.BACKEND]
    if region:
        require(test and group["factor"]==1 and len(region)==len(arms)
                and all(p["comparison"]=="equal_parameters" for p in pairs)
                and all(set(s)=={"backend","policy"} and s["policy"] in regional.POLICIES for s in region)
                and len({s["policy"] for s in region})==len(region)
                and releases[group["release"]].get("matched_exposure")==support_match.MODE
                and 8<=group["steps"]<=200,
                "Regional structural comparison requires same-capacity engineering arms and matched exposure")
    controlled = [s for s in region if s["policy"] in causal.POLICIES]
    has_witness = "causal_graph" in releases[group["release"]]
    require(bool(controlled)==has_witness and (not controlled or len(controlled)==len(arms)),
            "Controlled and native graph policies require distinct released cohorts")
    links = [(p["baseline"],p["candidate"]) for p in pairs]
    require(len(set(links))==len(links) and {name for pair in links for name in pair}==arms.keys(),
            "The pair plan repeats a comparison or omits a trained arm")


def request_spec(root, request_pin, *, test):
    request = json_pin(root,request_pin)
    required = {"schema","scope","releases","groups"}
    require(required <= request.keys() <= required|{"support_contrasts"} and request["schema"]==SCHEMA
            and request["scope"]==(SYNTHETIC if test else REAL)
            and request["releases"] and request["groups"]
            and all(named(key) for key in (*request["releases"],*request["groups"])),
            "Explicit registered paired training request required")
    for release in request["releases"].values():
        needed = {"manifest","authority","source","k"}
        require(needed <= release.keys() <= needed|{"observation_protocol","claim_policy","matched_exposure","causal_graph","coverage"}
                and release.get("observation_protocol", PHYSICAL) in (PHYSICAL, COLUMNS)
                and release.get("claim_policy") in (None, PAVIA)
                and release.get("matched_exposure") in (None, support_match.MODE)
                and ("coverage" not in release or release["coverage"]==coverage.GRAPH)
                and type(release["k"]) is int and release["k"]>0,
                "Release must bind observed Bank roles and support budget")
        require(release["source"] is None or set(release["source"])=={"weights","metadata","state"},
                "Source must use a frozen checkpoint with its original sidecar")
    for group in request["groups"].values():
        validate_group(group,request["releases"],test=test)
    require({g["release"] for g in request["groups"].values()}==request["releases"].keys(),
            "Request contains a release with no registered experiment")
    if "support_contrasts" in request:
        contrasts = request["support_contrasts"]
        require(isinstance(contrasts, dict) and contrasts and all(
            named(name) and isinstance(c, dict) and set(c) == {"baseline", "comparison", "factor"}
            and c["baseline"] != c["comparison"] and c["factor"] in support_match.FACTORS
            and all(g in request["groups"] and
                    request["releases"][request["groups"][g]["release"]].get("matched_exposure") == support_match.MODE
                    for g in (c["baseline"], c["comparison"]))
            for name, c in contrasts.items()), "Invalid preregistered support contrasts")
    return request


def expected_approval(request, pin, out, code, *, test):
    require(not test or request["scope"]==SYNTHETIC,"Engineering approval cannot authorize real data")
    return dict(schema=AUTHORITY,decision="approved",scope=request["scope"],request=pin,
                producer=code["commit"],source_hash=code["source_hash"],destination=str(out),
                optimizer_updates=sum(g["steps"]*len(g["arms"]) for g in request["groups"].values()),
                devices={name:g["device"] for name,g in request["groups"].items()},
                query_authorized=False,raw_gt_authorized=False)


def admit(root, out, record, *, test):
    bank,intake,roles = independent.open_independent(root,record["manifest"],record["authority"],
                                                     test_history=[] if test else None)
    require(bank.root!=out and bank.root not in out.parents and out not in bank.root.parents,
            "Training destination overlaps a published Bank")
    columns = record.get("observation_protocol", PHYSICAL)==COLUMNS
    require(columns == ("delivered_columns" in intake),
            "The approved columns and physical protocols are separate registrations")
    views = independent.COLUMN_VIEWS if columns else VIEWS
    exposure = intake.get("observation_exposure")
    require(("coverage" in record)==(exposure is not None),
            "Coverage and other paired arms require separately authorized view rosters")
    approved_views = tuple(intake.get("observation_views",[]))
    require(intake["uses"]["independent_accuracy"]["eligible"]
            and (columns or intake["uses"]["cost"]["eligible"])
            and (approved_views==views or exposure is not None and set(views)<=set(approved_views))
            and intake["role_reports"]["support"]["per_class"]==[record["k"]]*bank.release.classes
            and roles["support"] and roles["calibration"] and roles["evaluation"],
            "Paired training needs eligible three-role custody, agreed bands and covered support budget")
    removed = [len(bank.release.bands.band_ids)-len(retained_positions(bank.release.bands,v)) for v in VIEWS]
    require(removed[0]==0 and removed[1]==removed[2]>0,"Observed drop views must remove the same number of physical bands")
    if not test:
        require(bank.release.profile!="synthetic" and (columns or intake["physical_wavelengths_verified"]),
                "Real matched spectral training needs verified physical band metadata")
    if "causal_graph" in record:
        require(test and columns,"Synthetic controlled geometry cannot authorize native real inference")
        causal.verify(json_pin(root,record["causal_graph"]),bank=bank.pin["sha256"],
                      scene=bank.release.scene,roles=roles,bands=bank.release.bands)
    require(record.get("claim_policy") is None or
            (bank.release.scene == "paviaU" and not columns),
            "The prospective Pavia claim is specific to the physical registered views")
    return bank,intake,roles


def open_source(root, bank, entry):
    if entry is None:
        return None
    weights,meta = (pinned(root,entry[k]) for k in ("weights","metadata"))
    require(meta==weights.with_suffix(".pt.json"),"Wrong shared checkpoint sidecar")
    with torch.random.fork_rng(devices=[]):
        model = load_model_checkpoint(weights,"cpu")
    require(state_hash(model)==entry["state"]==bank.release.source.get("shared_state")
            and model.backbone.graph_mode in ("static","learned"),"Foreign released shared state")
    return model


def initialization(model, base, source):
    params = dict(model.named_parameters())
    original = base.state_dict()
    state = model.state_dict()
    require(original.keys() <= state.keys() and all(state[k].shape==v.shape and state[k].dtype==v.dtype
            and tensor_hash(state[k])==tensor_hash(v) for k,v in original.items()), "Variant changed a shared starting tensor")
    extra = {"alignment":[],"messages":[],"selectors":[]}
    if isinstance(model,regional.RegionSupportModel):
        extra["regional_scoring"] = []
    if model.backbone.coverage_modulation is not None:
        extra["coverage"] = []
    for name in sorted(params.keys()-original.keys()):
        if name in {"backbone.spectrum_score.weight","backbone.graph_score.weight"} and "regional_scoring" in extra:
            slot = "regional_scoring"
        elif name.startswith("backbone.coverage_modulation.") and "coverage" in extra:
            slot = "coverage"
        elif name.startswith("backbone.interaction."):
            slot = "messages" if name.startswith(("backbone.interaction.to_pixels.","backbone.interaction.to_regions.")) else "alignment"
        else:
            require("selector" in name or "range_" in name or "condition_" in name,"Unknown fresh variant parameter")
            slot = "selectors"
        extra[slot].append(name)
    groups = {**extra,"target_graph":[k for k in params if k.startswith("backbone.learnable_graph.")],
                      "target_heads":[k for k in params if not k.startswith("backbone.")]}
    fresh = {key:tensor_hash(params[key]) for names in groups.values() for key in names}
    return dict(state=state_hash(model),base=state_hash(base),source=source,groups=groups,fresh=fresh,
                counts={name:sum(params[k].numel() for k in names) for name,names in groups.items()},
                parameters=sum(v.numel() for v in params.values()),
                inventory={name:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_hash(value))
                           for name,value in params.items()})


def base_model(bank, seed, source):
    model = _baseline(bank.release,seed)
    report = None if source is None else model_from_source(
        source,model,bank.release.scene,native_source=source.backbone.graph_mode=="learned")
    return model,report


def start_arms(bank, group, source):
    base, shared = base_model(bank,group["seed"],source)
    result = {}
    for arm, config in group["arms"].items():
        model = backend.train_variant(base,config,group["seed"]+7321)
        result[arm] = initialization(model,base,shared)
    for route in group["pairs"]:
        first,last = (result[route[key]] for key in ("baseline","candidate"))
        equal = first["parameters"]==last["parameters"]
        require(equal==(route["comparison"]=="equal_parameters"), "Claimed capacity relation differs from initialized model")
        a,b = (group["arms"][route[key]] for key in ("baseline","candidate"))
        if a["backend"]==b["backend"]==regional.BACKEND:
            require(first["inventory"]==last["inventory"], "Regional policies must share every initial parameter")
        if equal and a["backend"]==b["backend"]==coverage.BACKEND:
            require(first["inventory"]==last["inventory"],"Equal-capacity E1 controls must share initial tensors")
        if equal and a["backend"]==b["backend"]=="region_message":
            require(a["direction"]==b["direction"] and first["inventory"]==last["inventory"],
                    "Equal-capacity directional comparison changed projection initialization or direction")
    return result


def register(root, out, code, request_pin, approval_pin, *, test):
    request = request_spec(root,request_pin,test=test)
    approval = json_pin(root,approval_pin)
    require(approval==expected_approval(request,request_pin,out,code,test=test),
            "Separate training approval does not bind scope, budget, devices, code and destination")
    inputs, started, matched = {},{},{}
    for key,entry in request["releases"].items():
        bank,intake,roles = admit(root,out,entry,test=test)
        source = open_source(root,bank,entry["source"])
        inputs[key] = dict(bank=bank.pin,role_assignment=digest(roles),authority=intake,scene=bank.release.scene)
        if entry.get("matched_exposure"):
            matched[key] = (bank, roles, support_match.inventory(bank, roles, entry["k"]))
            inputs[key]["matched_support"] = digest(matched[key][2])
        if "causal_graph" in entry:
            controlled = json_pin(root,entry["causal_graph"])
            probe = causal.search_controls(controlled["layout"])
            inputs[key]["causal_preflight"] = dict(witness=digest(controlled["witness"]),
                status=controlled["witness"]["status"],
                negatives={k:{f:v for f,v in item.items() if f!="seconds"} for k,item in probe.items()})
        require(bank.reads==0,"Registration opened observation data")
        for name,group in request["groups"].items():
            if group["release"]==key:
                require(group["factor"] in intake.get("observation_factors",[1])
                        and group["steps"]>=len(roles["support"])*len(
                            intake.get("observation_exposure",{}).get("support",intake["observation_views"])),
                        "Lattice not approved or budget cannot expose each support block to all registered views")
                if key in matched:
                    blocks = len(roles["support"])
                    require(group["steps"] % blocks == 0 and group["steps"] // blocks >= entry["k"],
                            "Matched support requires equal complete visits and coverage of every released point")
                started[name] = start_arms(bank,group,source)
    require(len({i["bank"]["sha256"] for i in inputs.values()})==len(inputs),"Duplicate release identity")
    require_frozen_code(code)
    out.mkdir(parents=True,exist_ok=True)
    path = out/"registration.json"
    record = dict(schema=SCHEMA,producer=code["commit"],source_hash=code["source_hash"],scope=request["scope"],
                  request=request_pin,approval=approval_pin,inputs=inputs,initializations=started,
                  groups=request["groups"],releases=request["releases"],optimizer_updates=approval["optimizer_updates"])
    if "support_contrasts" in request:
        record["support_contrasts"] = {}
        for name, contrast in request["support_contrasts"].items():
            a,b = (request["groups"][contrast[key]] for key in ("baseline", "comparison"))
            ra,rb = a["release"],b["release"]
            proof = support_match.validate_contrast(contrast["factor"], matched[ra], matched[rb],
                a, b, request["releases"][ra], request["releases"][rb])
            record["support_contrasts"][name] = {**contrast, **proof}
    if path.exists():
        require(json_pin(root,file_pin(path))==record,"Existing registration differs")
        return file_pin(path)
    return _write(path,record)


def open_plan(root, out, code, *, test):
    link = file_pin(out/"registration.json")
    plan = json_pin(root,link)
    request = request_spec(root,plan["request"],test=test)
    require(plan["schema"]==SCHEMA and plan["producer"]==code["commit"] and plan["source_hash"]==code["source_hash"]
            and plan["groups"]==request["groups"] and plan["releases"]==request["releases"] and plan["scope"]==request["scope"]
            and json_pin(root,plan["approval"])==expected_approval(request,plan["request"],out,code,test=test)
            and plan["optimizer_updates"]==expected_approval(request,plan["request"],out,code,test=test)["optimizer_updates"],
            "A different code checkout, roster, scope or physical assignment cannot continue this campaign")
    require({k:{field:v[field] for field in ("baseline","comparison","factor")}
             for k,v in plan.get("support_contrasts", {}).items()} == request.get("support_contrasts", {}),
            "Registered support contrasts differ from the request")
    return plan,link


class StructuralProvider(ProductionProvider):
    """Only the release, registered model and exposure change around native Run."""

    def __init__(self, root, out, plan, group, arm, code, *, test):
        spec = plan["groups"][group]
        entry = plan["releases"][spec["release"]]
        bank,intake,roles = admit(root,out,entry,test=test)
        original = plan["inputs"][spec["release"]]
        require(bank.pin==original["bank"] and digest(roles)==original["role_assignment"] and intake==original["authority"],
                "Bank, authority or ordered roles differ from registration")
        region = spec["arms"][arm]["backend"]==regional.BACKEND
        config = Switches(graph_scale=regional.SCALE if region else 50 if bank.release.scene=="paviaU" else 100)
        super().__init__(root,bank,roles,axis=bank.release.bands.band_ids,config=config,source=entry["source"],
                         seed=spec["seed"],budget=spec["steps"],producer=code["commit"],
                         approval=None if test else plan["approval"]["sha256"],
                         delivered_axis="delivered_columns" in intake)
        self.spec,self.arm,self.test = spec,arm,test
        self.intake = intake
        self.views = tuple(intake["observation_views"])
        self.exposure = intake.get("observation_exposure")
        self.causal = json_pin(root,entry["causal_graph"]) if "causal_graph" in entry else None
        if self.causal is not None:
            preflight = original["causal_preflight"]
            require(preflight["status"]=="MATCHED" and preflight["witness"]==digest(self.causal["witness"])
                    and preflight["negatives"]["exhaustive"]["status"]=="INFEASIBLE"
                    and preflight["negatives"]["limited"]["status"]=="NOT_FOUND_WITHIN_BUDGET",
                    "Queue cannot consume an unresolved or changed graph preflight")
        self.matched_exposure = entry.get("matched_exposure")
        if self.matched_exposure:
            require(digest(support_match.inventory(bank, roles, entry["k"])) == original["matched_support"],
                    "Actual support inventory changed after registration")
        self.init = plan["initializations"][group][arm]
        self.release = open_source(root,bank,self.source)
        self._digest = None
        self.event_step = 0

    def baseline(self, seed):
        return base_model(self.bank,seed,self.release)[0]

    def model_factory(self, source, arm, seed):
        require(arm==self.arm and seed==self.seed+7321,"A different architecture or initialization was requested")
        model = backend.train_variant(source,self.spec["arms"][arm],seed)
        require(state_hash(model)==self.init["state"],"Initialized variant differs from the approved shared start")
        if self.exposure is not None:
            model.coverage_program = coverage.GraphProgram(deepcopy(source),self.config.graph_scale)
        if isinstance(model,regional.RegionSupportModel):
            anchor = deepcopy(model).eval().requires_grad_(False)
            config = self.spec["arms"][arm]
            model.graph_program = regional.program_for(anchor,config["policy"],self.seed+19000,self.budget,self.causal)
        return model

    def make_observation(self, role, event):
        row = self.bank.role_row(role, event["tile"])
        view, metadata = independent.observation_view(event["band"], self.intake)
        actual, labels, targets = bank_view(self.bank, row, self.producer, view, factor=event["factor"],
                                            metadata=metadata)
        require(actual.receipt["source"]["band_ids"] ==
                independent.observation_bands(self.bank.release.bands, event["band"], self.intake),
                "Materialized spectral stream differs from the approved deletions")
        return actual, labels, targets

    @staticmethod
    def key(role, event):
        key = ProductionProvider.key(role, event)
        return f'{key}/{event["visit"]}' if "visit" in event else key

    def materialize(self, role, event, producer):
        purpose = {"devtrain":"support","devval":"calibration"}[role]
        require(producer==self.producer and event["tile"] in self.roles[purpose]
                and event["factor"]==self.spec["factor"] and event["band"] in (self.views if self.exposure is None else self.exposure[purpose]),
                "Optimization or selection used an unauthorized role, factor or band")
        self.event_step = event.get("step",0)
        key = ProductionProvider.key(role, event)
        if key not in self._cache:
            self._cache[key] = self.make_observation(role,event)
        actual, labels, targets = self._cache[key]
        if self.matched_exposure and role == "devtrain":
            targets = targets.select(support_match.choose(targets.points, event["visit"]))
            labels = np.asarray(targets.points, dtype=np.int64)
        return actual, labels, targets

    def registered_forward(self, model, actual, role, execution):
        if isinstance(model,regional.RegionSupportModel):
            return regional.forward_region(model,actual,self.producer,role,execution,model.graph_program,step=self.event_step)
        if self.exposure is not None:
            return coverage.run(model,actual,self.producer,role,execution,model.coverage_program)
        timing = {}
        result,graph = forward(model,actual,self.producer,role,execution=execution,timed=timing,
                               graph_scale=self.config.graph_scale)
        return result,graph,timing

    def validate_history(self, spec, packet):
        super().validate_history(spec,packet)
        if spec["architecture"]["backend"]==regional.BACKEND:
            for row in [*packet["history"],*(r for v in packet["validation"] for r in v["rows"])]:
                proof = regional.verify_trace(row["graph"]["regional_support"],row["observation"],
                                              require_policy=spec["architecture"]["policy"])
                if self.causal is not None:
                    from transfer.wp_causal_graph import verify_trace

                    verify_trace(proof,self.causal,row["tile"],"devtrain" if row.get("step") else "devval",spec["steps"])

    def contract(self, classes,seed,arm,factor,producer,*,numeric=None):
        require((classes,seed,arm,factor,producer)==(self.classes,self.seed,self.arm,self.spec["factor"],self.producer)
                and numeric is not None and (numeric["physical_uuid"] or "cpu")==self.spec["device"],
                "A paired arm cannot change its cohort, optimization lattice or physical device")
        if self._digest is not None:
            require(self._digest["numeric"]==numeric,"Numeric protocol changed within a case")
            return deepcopy(self._digest)
        support = self.roles["support"]
        rng = np.random.default_rng(seed)
        schedule = []
        fit_bands = self.views if self.exposure is None else self.exposure["support"]
        calibration_bands = self.views if self.exposure is None else self.exposure["calibration"]
        epoch = 0
        while len(schedule)<self.budget:
            for tile in rng.permutation(support).tolist():
                event = dict(step=len(schedule)+1,tile=tile,factor=factor,
                    band=fit_bands[(epoch+support.index(tile))%len(fit_bands)],condition="registered")
                if self.matched_exposure:
                    event["visit"] = epoch
                schedule.append(event)
            epoch += 1
        schedule = schedule[:self.budget]
        val = [dict(tile=tile,factor=factor,band=band,condition="registered")
               for tile,band in product(self.roles["calibration"],calibration_bands)]
        roster = {}
        for role,events in (("devtrain",schedule),("devval",val)):
            for event in events:
                key = self.key(role,event)
                if key not in roster:
                    roster[key] = self.evidence(*self.materialize(role,event,producer))
        self._digest = dict(format=SCHEMA,producer=producer,seed=seed,arm=arm,classes=classes,factor=factor,
            module_seed=seed+7321,rng_seed=seed+15000,steps=self.budget,learning_rate=.002,
            optimizer_group="adam_single_all_parameters",objective=LOSS,supervision=MODE,numeric=numeric,
            bank=self.bank.pin,roles=digest(self.roles),band_axis=self.bank.release.bands.fingerprint,
            graph_scale=self.config.graph_scale,selection=SELECTION,events=self.spec["clock"],
            training=schedule,validation=val,inputs=roster,initialization=self.init,scope=self.spec["purpose"],
            architecture=self.spec["arms"][arm],frozen_stage_protocol=SCHEMA,
            query_accessed=False,raw_gt_loaded=False,real_updates=0 if self.test else self.budget)
        return deepcopy(self._digest)
