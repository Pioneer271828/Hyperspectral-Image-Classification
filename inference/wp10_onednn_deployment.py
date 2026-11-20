"""End-to-end CPU control for the already accepted single-layer oneDNN package."""
from pathlib import Path

from compression.int8_torchao_onednn import load_backend
from compression.int8_torchao_reference import TorchAOAffine
from compression.precision_fallback import LAYER_ORDER
from compression.precision_recovery_audit import load_package
from inference import wp10_cross_device as cross, wp10_low_rank_execution as frozen
from inference.native_deployment import Runtime
from research_runtime import require_frozen_code
from transfer import matched_route_admission as admission, registered_development as dev


QUALITY = "18887f7a1a7b3bfb3aacced82e020b9298e90562017afb1b2926ab44edac751b"
OBSERVATION = "9d6abec79c2ca03296806c8969b030a6641f2b1cdf2ae84292422f0e3fb1c27e"
BACKEND = "d22c6a33824c0cdfd58a10467b0b6bd1671238741fcc66b01e6c9a9a16beed8e"
PLAN = "wp10_frozen_onednn_pair_v1"
CONTROL, CANDIDATE, REJECTED = "fp32", "qat-first", "qat-both"


def preparation(pin, observation, code, out):
    require_frozen_code(code)
    if pin["sha256"] != QUALITY or observation["sha256"] != OBSERVATION:
        raise ValueError("Use the approved frozen QAT receipt and image-only packet")
    archive = admission.read(pin)
    chosen = archive["candidate"][CANDIDATE]
    guards = chosen["versus_same_budget_fp32_guards"]
    failed = archive["candidate"][REJECTED]["versus_same_budget_fp32_guards"]
    backend = archive["backend_pins"][CANDIDATE]
    if (not chosen["eligible"] or backend["sha256"] != BACKEND
            or cross.accepted_arm(CONTROL, CANDIDATE, guards) != CANDIDATE
            or cross.accepted_arm(CONTROL, REJECTED, failed) != CONTROL
            or any(archive[key] != 0 for key in ("query", "gt", "optimizer_updates", "calibration_new_calls"))):
        raise ValueError("Original precision guards, fallback or frozen backend identity differs")
    hashes = {CONTROL: archive["native"][CONTROL]["probability_hashes"], CANDIDATE: chosen["probability_hashes"]}
    packages = {name: archive["packaged_endpoints"][name] for name in hashes}
    out = cross.result_dir(out, [pin, observation, backend])
    reg = dict(commit=code["commit"], source_hash=code["source_hash"], kind=PLAN, dataset="xuzhou",
               class_ids=list(range(1, 10)), tiles=archive["evaluation_tiles"], resident_inputs=observation,
               backend=backend, packages=packages, hashes={name: [dict(zip(frozen.HEADS, row, strict=True)) for row in values]
                                                           for name, values in hashes.items()},
               guards=guards, rejected=failed, quality=pin, runtime=frozen.runtime())
    tiles = frozen.load_tiles(reg)
    if (len(tiles), sum(row["pixels"] for row in reg["tiles"])) != (5, 3920):
        raise ValueError("Original five-block QAT observation required")
    require_frozen_code(code)
    out.mkdir()
    return dev._json_new(out / "plan.json", reg)


def measure(pin, code, out, provider):
    require_frozen_code(code)
    reg = admission.read(pin)
    if (reg["kind"] != PLAN or reg["commit"] != code["commit"] or reg["source_hash"] != code["source_hash"]
            or reg["runtime"] != frozen.runtime() or reg["quality"]["sha256"] != QUALITY
            or reg["backend"]["sha256"] != BACKEND or reg["resident_inputs"]["sha256"] != OBSERVATION):
        raise ValueError("Current frozen oneDNN deployment plan required")
    protected = [pin, reg["backend"], reg["resident_inputs"], *[
        entry[key] for entry in reg["packages"].values() for key in ("weights", "sidecar") if entry[key] is not None]]
    out = cross.result_dir(out, protected)
    oracle = TorchAOAffine(provider)
    out.mkdir()

    def make(name):
        model = load_package(reg["packages"][name])
        if name == CANDIDATE:
            model = load_backend(reg["backend"], oracle, enabled=True, native=model)
        model.backbone.adapter.cache_attention = False
        return model

    def prove(name, row, values):
        if row["hashes"] != reg["hashes"][name]:
            raise ValueError("Full heads differ from the corresponding approved native/backend result")

    expected = {name: dict(state=entry["state"],
                           bytes=entry["bytes"] + (Path(reg["backend"]["path"]).stat().st_size if name == CANDIDATE else 0),
                           paths=[LAYER_ORDER[0]], int8=name == CANDIDATE,
                           guards=reg["guards"] if name == CANDIDATE else {})
                for name, entry in reg["packages"].items()}
    measured = cross.pair_run(reg, {"qat": (CONTROL, CANDIDATE)}, make, expected, prove, "cpu", out)
    fallback = cross.accepted_arm(CONTROL, REJECTED, reg["rejected"])
    row, values = cross.sample(Runtime(device="cpu", full=make(fallback)), reg)
    prove(fallback, row, values)
    admission.check_pins(protected)
    require_frozen_code(code)
    return dev._json_new(out / "receipt.json", dict(
        **measured, kind=PLAN, schema=cross.SCHEMA, commit=code["commit"], plan=pin,
        quality=reg["quality"], device={"kind": "cpu", "runtime": reg["runtime"]},
        resident_input_sha256=reg["resident_inputs"]["sha256"],
        rejected_route=dict(requested=REJECTED, executed=fallback, sample=row),
        optimizer_updates=0, calibration_updates=0, gt=0, query=0, predictive_label_reads=0))
