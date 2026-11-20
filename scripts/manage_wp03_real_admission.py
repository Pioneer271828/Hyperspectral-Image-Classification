"""Prepare and review isolated WP03 diagnostics and a separately released full54 cohort."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_light_duration import forecast
from g4_light_engineering_registration import pin, pinned
import g4_light_probe_pair as probe
import g4_light_real_campaign as campaign
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from transfer.container_runtime import training_container_kind


def item(args, name):
    path, sha = getattr(args, name)
    return {"path": str(Path(path).resolve(strict=True)), "sha256": sha}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--expected-commit", required=True)
    stages = root.add_subparsers(dest="stage", required=True)
    pinned_args = {
        "device-request": ("original", "engineering"),
        "intent": ("original",),
        "diagnose": ("intent",),
        "review-probe": ("intent",),
        "forecast": ("original", "inventory", "gpu-matrix"),
        "prepare": ("request",),
        "activate": ("plan", "authority"),
        "review-campaign": ("registration",),
        "verify-isolation": (),
    }
    for name, pins in pinned_args.items():
        stage = stages.add_parser(name)
        for key in pins:
            stage.add_argument("--" + key, nargs=2, required=True, metavar=("PATH", "SHA256"))
        if name != "verify-isolation":
            stage.add_argument("--out", required=True)
        if name in ("device-request", "intent"):
            stage.add_argument("--device", required=True, nargs=2, metavar=("INDEX", "UUID"))
        if name in ("verify-isolation", "diagnose"):
            stage.add_argument("--producer-checkout", required=True)
        if name == "intent":
            stage.add_argument("--migration", nargs=2, metavar=("PATH","SHA256"))
            stage.add_argument("--authority", nargs=2, metavar=("PATH","SHA256"))
        if name == "review-probe":
            stage.add_argument("--samples", required=True)
    return root


def main():
    args = parser().parse_args()
    code = code_snapshot()
    if not training_container_kind() or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean exact authorized container checkout required")
    if args.stage == "verify-isolation":
        root = external_directory(args.producer_checkout)
        subprocess.run([sys.executable, "-I", "-B", "-c", probe.ISOLATED_PAIR, str(probe.PROBE_ROOT),
                        json.dumps({"producer_checkout": str(root), "verify_source_only": True}), code["source_hash"]],
                       cwd=root, check=True)
        return
    out = external_directory(args.out)
    if out.exists():
        raise ValueError("New independent output required")
    if args.stage in ("device-request", "intent"):
        device = {"slot": int(args.device[0]), "uuid": args.device[1]}
    if args.stage == "device-request":
        request = probe.device_request(item(args, "original"), device, item(args, "engineering"), code)
        out.mkdir()
        write_json(out / "device-request.json", request)
        write_json(out / "approval-draft.json", probe.approval_for(pin(out / "device-request.json"),
                   request, decision="review_required"))
    elif args.stage == "intent":
        value = probe.probe_intent(item(args, "original"), device, code,
                                  migration=item(args,"migration") if args.migration else None,
                                  authority=item(args,"authority") if args.authority else None)
        write_json(out, value)
    elif args.stage == "diagnose":
        intent = pinned(item(args, "intent"))
        if intent != probe.probe_intent(intent["registration"], intent["device"], code,
                                       migration=intent["migration"], authority=intent["authority"]):
            raise ValueError("Approved zero-update intent changed")
        probe.execute(intent, args.producer_checkout, out, code)
    elif args.stage == "review-probe":
        samples = external_directory(args.samples)
        if campaign.contained({samples}, out):
            raise ValueError("Keep independent comparison outside the sample tree")
        result = probe.review_pair(item(args, "intent"), {k: pin(samples / k / "receipt.json") for k in probe.SAMPLES})
        write_json(out, result)
    elif args.stage == "forecast":
        write_json(out, forecast(item(args,"original"), item(args,"inventory"), item(args,"gpu_matrix")))
    elif args.stage == "prepare":
        plan = campaign.prepare(item(args,"request"), code, verify_files=True)
        if campaign.contained({Path(plan["output"])}, out):
            raise ValueError("Plan must remain separate from the execution tree")
        out.mkdir()
        write_json(out / "plan.json", plan)
        write_json(out / "release-draft.json", campaign.release_for(pin(out / "plan.json"), plan, decision="review_required"))
    elif args.stage == "activate":
        registration = campaign.activate(item(args,"plan"), item(args,"authority"), code)
        plan = pinned(registration["plan"])
        if campaign.contained({Path(plan["output"])}, out):
            raise ValueError("Active registration must remain separate from its execution")
        write_json(out, registration)
    else:
        result = campaign.independent_campaign(item(args,"registration"), code)
        if campaign.contained({Path(pinned(pinned(result["registration"])["plan"])["output"])}, out):
            raise ValueError("Independent review must remain outside the owned execution")
        write_json(out, result)
    require_frozen_code(code)


if __name__ == "__main__":
    main()
