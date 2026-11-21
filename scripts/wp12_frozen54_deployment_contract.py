"""Wire contract shared by WP12 and the preserved-runtime benchmark worker."""

import hashlib
import json
from pathlib import Path
import re


PLAN = "wp12_frozen54_end_to_end_plan_v2"
APPROVAL = "wp12_frozen54_end_to_end_approval_v2"
COLD, WARM = "cold_process_first_prediction", "warm_end_to_end"
COUNT = {COLD: 3, WARM: 6}
WARMUPS = {COLD: 0, WARM: 2}
TIMEOUT = 120
PHASES = ("observation_read_and_verify", "observation_decode", "input_and_graph_transfer",
          "synchronized_forward", "three_head_host_decisions")
SETUP = ("weights_and_sidecar_cpu_load", "model_transfer")
GPU = re.compile(r"GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
EXCLUDED = ["coordinator_admission_and_lock_wait", "upstream_raw_observation_preparation",
            "construction_of_sealed_Q_A", "post_prediction_validation_and_persistence"]
TIMING = {
    COLD: dict(scope=COLD, start="coordinator_before_worker_spawn", stop="synchronized_three_head_host_materialization",
               excluded=EXCLUDED),
    WARM: dict(scope=WARM, start="before_observation_read_with_model_resident",
               stop="synchronized_three_head_host_materialization",
               excluded=[*EXCLUDED, "worker_startup_and_model_loading", "warmup_requests"]),
}


def worker_binding(plan_sha, case, scope, device, attempt, invocation):
    return dict(plan_sha256=plan_sha, case=case, scope=scope, device=device,
                attempt=attempt, invocation=invocation)


def digest_text(value):
    return hashlib.sha256(value).hexdigest()


def source(root, pin):
    if set(pin) != {"path", "sha256"} or not re.fullmatch("[a-f0-9]{64}", pin["sha256"]):
        raise ValueError("An explicit content pin is required")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve(strict=True) != path or root not in path.parents:
        raise ValueError("Pinned artifact must be inside the approved runs root")
    return path


def read(root, pin):
    content = source(root, pin).read_bytes()
    if digest_text(content) != pin["sha256"]:
        raise ValueError("Pinned artifact content differs")
    return content


def approval(root, plan, plan_pin, permit, device):
    if plan["engineering"]:
        if permit is not None:
            raise ValueError("Engineering runs do not consume a real approval")
        return
    if permit is None:
        raise ValueError("Independent approval for this exact real end-to-end plan is required")
    if not GPU.fullmatch(device) or Path(plan_pin["path"]).parent in Path(permit["path"]).parents:
        raise ValueError("Approval must come from outside the plan and name a physical GPU")
    expected = dict(schema=APPROVAL, decision="approved", plan_sha256=plan_pin["sha256"],
                    execution=plan["execution"], producer=plan["producer"], physical_uuid=device,
                    scopes=[COLD, WARM], data_access="sealed_observations_no_labels")
    value = json.loads(read(root, permit))
    ref = value.pop("approval_reference", None)
    if value != expected or not isinstance(ref, str) or not ref.strip():
        raise ValueError("Approval does not bind this use, plan, output and physical GPU")
