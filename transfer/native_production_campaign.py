"""Bounded multi-source production verification through the established native runner."""

from dataclasses import asdict
from itertools import product
from pathlib import Path

from evaluation.frozen_training_stages import file_pin, json_pin, require
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_production_fixture as fixture, native_production_model as candidate, native_scale_training as native
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import case_lock
from transfer.native_long_run import packet
from transfer.native_production_provider import ProductionProvider
from transfer.xuzhou_multiscale_audit import _exact

SCHEMA = "wp05_wp06_production_confirmation_v1"
REFERENCE = "s64-k211-center"
SMOKE = "residual-continuous"
MATCHED_COST_VIEWS = ((1,"full"),(1,"central_gap"))
SHIFTED_COST_VIEWS = (*MATCHED_COST_VIEWS,(2,"missing"))


def design():
    rows = {"source-a":dict(input="source-a", parent=None, seed=64, budget=fixture.SOURCE_STEPS,
                            config=asdict(candidate.Switches())),
            "source-b":dict(input="source-b", parent="source-a", seed=64, budget=fixture.SOURCE_STEPS,
                            config=asdict(candidate.Switches())),
            SMOKE:dict(input="k211", parent="source-b", seed=64, budget=6,
                       config=asdict(candidate.Switches(arm="center",spectral="continuous",residual="x_residual")))}
    for seed, support, arm in product(fixture.SEEDS, fixture.SUPPORT, candidate.ARMS):
        rows[f"s{seed}-k{support}-{arm}"] = dict(input=f"k{support}", parent="source-b", seed=seed,
                                               budget=fixture.STEPS, config=asdict(candidate.Switches(arm=arm)))
    return rows


def register(root, out, code):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/"registration.json"
    if manifest.exists():
        return plan_for(root,out,code)[1]
    input_dir = out/"inputs"
    input_dir.mkdir(exist_ok=True)
    inputs = {}
    for name, kind, support in (("source-a","source-a",fixture.SUPPORT[0]),
                                ("source-b","source-b",fixture.SUPPORT[0]),
                                *((f"k{k}","target",k) for k in fixture.SUPPORT)):
        _, proof = fixture.publish(input_dir/name, kind, support)
        inputs[name] = dict(kind=kind,support_seed=support,**proof)
    spec = dict(schema=SCHEMA, scope="synthetic_only", producer=code["commit"], design=design(), inputs=inputs,
                axis=list(fixture.BANDS), gates=fixture.GATES, restore_reference=REFERENCE,
                matched_confirmation=list(map(list,MATCHED_COST_VIEWS)),
                shifted_confirmation=list(map(list,SHIFTED_COST_VIEWS)), minimum_paired_seeds=3,
                confirmation_frozen_before_training=True, independent_real_accuracy=None,
                query_accessed=False, raw_gt_loaded=False, real_updates=0)
    require_frozen_code(code)
    return _write(manifest,spec)


def plan_for(root,out,code):
    link = file_pin(out/"registration.json")
    value = json_pin(root,link)
    require(value["schema"] == SCHEMA and value["producer"] == code["commit"] and value["scope"] == "synthetic_only"
            and value["design"] == design() and value["gates"] == fixture.GATES
            and value["axis"] == list(fixture.BANDS) and value["restore_reference"] == REFERENCE
            and value["confirmation_frozen_before_training"] is True and value["independent_real_accuracy"] is None
            and value["matched_confirmation"] == list(map(list,MATCHED_COST_VIEWS))
            and value["shifted_confirmation"] == list(map(list,SHIFTED_COST_VIEWS))
            and value["minimum_paired_seeds"] == 3 and value["real_updates"] == 0
            and value["query_accessed"] is value["raw_gt_loaded"] is False,
            "Production confirmation registration or custody differs")
    return value,link


def provider(root,out,plan,name,code):
    case = plan["design"][name]
    record = plan["inputs"][case["input"]]
    directory = out/"inputs"/case["input"]
    require((directory/"bank.json").is_file(), "Registered Bank is missing")
    bank, archived = fixture.publish(directory, record["kind"], record["support_seed"])
    require(archived == {k:v for k,v in record.items() if k not in ("kind","support_seed")},
            "Independent fixture bytes or role registration differs")
    parent = case["parent"]
    source = None
    if parent is not None:
        previous = json_pin(root,file_pin(out/"cases"/parent/"receipt.json"))
        require(previous["scope"] == SCHEMA and previous["case"] == parent and previous["updates"] == fixture.SOURCE_STEPS
                and previous["producer"] == code["commit"] and previous["real_updates"] == 0,
                "Expected complete native source pretraining")
        source = previous["selected_export"]
    return ProductionProvider(root,bank,record["roles"],axis=plan["axis"],
        config=candidate.Switches(**case["config"]),source=source,seed=case["seed"],
        budget=case["budget"],producer=code["commit"])


def export_selected(p, place, state):
    if p.config.arm == "pixel":
        return None
    model = p.model_factory(p.baseline(p.seed),p.config.arm,p.seed+7321).eval()
    model.load_state_dict(state,strict=True)
    path = place/"selected.pt"
    sidecar = path.with_suffix(".pt.json")
    if not path.exists() and not sidecar.exists():
        candidate.export_production(model,path,p.config,p.axis)
    restored, config, registered = candidate.open_production(path,"cpu")
    require(config == p.config and registered == p.axis and state_hash(restored) == state_hash(model),
            "Frozen deployment checkpoint differs from the selected native state")
    return dict(weights=file_pin(path),metadata=file_pin(sidecar),state=state_hash(model))


def execute_case(root,out,code,name,execution):
    plan, plan_pin = plan_for(root,out,code)
    require(name in plan["design"], "Unknown preregistered production cell")
    folder = out/"cases"
    folder.mkdir(exist_ok=True)
    place = folder/name
    with case_lock(place):
        receipt = place/"receipt.json"
        if receipt.exists():
            old = json_pin(root,file_pin(receipt))
            require(old["plan"] == plan_pin["sha256"] and old["case"] == name and old["producer"] == code["commit"],
                    "Completed receipt differs from the registered case")
            return dict(case=name,updates_this_invocation=0,receipt=file_pin(receipt))
        p = provider(root,out,plan,name,code)
        spec = p.contract(p.classes,p.seed,p.config.arm,1,p.producer,numeric=execution.numeric)
        registration = place/"registration.json"
        if registration.exists():
            require(json_pin(root,file_pin(registration)) == spec, "Resume changes the approved physical device or scope")
        else:
            _write(registration,spec)
        phases = ("prefix","resume","completed","full") if name == REFERENCE else ("full",)
        previous = None
        phase_receipts = {}
        launched = 0
        for phase in phases:
            output = place/f"{phase}.pt"
            if not output.exists():
                row = native.execute(spec,phase,output,p.producer,prior=previous if phase in ("resume","completed") else None,
                                     execution=execution,released=p)
                _write(place/f"{phase}.json",row)
                launched += row["updates"]
            end = packet(root,file_pin(output))
            require(end["registration"] == digest(spec), "Committed native optimizer state has a foreign registration")
            if phase in ("prefix","resume"):
                previous = end
            phase_receipts[phase] = dict(pin=file_pin(output),cursor=end["cursor"],selected=end["selected"]["step"],
                timing=json_pin(root,file_pin(place/f"{phase}.json"))["timing"] if (place/f"{phase}.json").exists() else None)
        if name == REFERENCE:
            resumed = packet(root,phase_receipts["resume"]["pin"])
            reentry = packet(root,phase_receipts["completed"]["pin"])
            require(_exact(end,resumed) and _exact(resumed,reentry), "Native continued/continuous/reentry state differs")
        require(end["cursor"] == len(end["history"]) == p.budget and
                all(item["supervised_area"] > 0 for item in end["history"]),
                "Matched effective update count differs")
        p.validate_history(spec,end)
        selected = export_selected(p,place,end["selected"]["state"])
        require_frozen_code(code)
        result = dict(scope=SCHEMA,plan=plan_pin["sha256"],case=name,producer=code["commit"],updates=p.budget,
            full=phase_receipts["full"]["pin"],phases=phase_receipts,spec=file_pin(registration),device=execution.numeric,
            selected=end["selected"]["step"],selected_export=selected,checkpoint_equal=name==REFERENCE,
            real_updates=0,query_accessed=False,raw_gt_loaded=False)
        return dict(case=name,updates_this_invocation=launched,receipt=_write(receipt,result))
