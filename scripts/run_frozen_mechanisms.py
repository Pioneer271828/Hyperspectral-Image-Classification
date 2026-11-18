"""Register and resume the released WP05/WP06 mechanism experiment."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import io
import json
from pathlib import Path
import re
import resource
from secrets import token_hex
from time import perf_counter

import numpy as np
import torch

from model.research_config import canonical_json, digest
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import PINS as BAND_REGION_PINS
from scripts.run_frozen_band_region import _freeze_bank, _honghu_context, _read, _write, check, pin, source_documents
from transfer import frozen_mechanism_honghu as honghu, frozen_mechanism_plan as plan
from transfer import frozen_mechanism_runner as native, p4_runtime_device as gpu
from transfer import registered_development as transport
from transfer.development_bank import _atomic_new


PREVIOUS = "wp05-wp06-band-region-20261119.tTSvTKQf"
PREVIOUS_SCOPE = "6a86e73c2ecde192df0d7a9da75dd6661e2f8282f95142b1bf230dae7fbbfde2"
PREVIOUS_SUMMARY = "33a2e61e74e17c3a3d7ff050c58c44066780eac175de54bd6e2bc84a3736ecfa"
SOURCES = {
    "short_scope": BAND_REGION_PINS["hh_scope"],
    "short_predictions": BAND_REGION_PINS["hh_predictions"],
    "short_independent": BAND_REGION_PINS["hh_audit"],
    "long_scope": ("wp05-v4-conditional-real-admission01/scope.json", honghu.LONG_SCOPE),
    "long_evidence": ("wp05-honghu-fit-existing-ten02/evidence.json",
                      "27123c497344d74e2b214d78654d6249d8a0e75adbcb514412d46a89e3c37e22"),
    "long_erratum": ("wp05-honghu-fit-existing-ten-erratum01/erratum.json",
                     "c1bc393aadf9e80a89765a046d8ed79e834c68fdf8a56b445e2cf38f1d79ce50"),
    "long_preflight": ("wp05-three-arm-fit-real-preflight04/scope.json",
                       "0676d13736c0c6be4106171205e873fde9d81e9124245a9d0fe5dbb6244b2ca8"),
}
LANES = ("xuzhou", "paviaU", "honghu24", "honghu60")


def now():
    return datetime.now(timezone.utc).isoformat()


def prior(root):
    folder = root / PREVIOUS
    links = dict(scope=pin(folder / "registration-v3/scope.json", PREVIOUS_SCOPE),
                 summary=pin(folder / "summary.json", PREVIOUS_SUMMARY))
    scope, summary = (_read(links[key]) for key in ("scope", "summary"))
    check(summary["scope"] == PREVIOUS_SCOPE and summary["status"] == "passed" and summary["cases"] == 48
          and scope["raw_gt_loaded"] is scope["query_accessed"] is False
          and summary["raw_gt_loaded"] is summary["query_accessed"] is False
          and scope["optimizer_updates"] == summary["optimizer_updates"] == 0,
          "Expected the exact accepted 48-case readonly parent")
    return links, scope, summary


def previous_case(root, summary, scene, name):
    filename = f"{scene}__{name}.json"
    return _read(pin(root / PREVIOUS / "cases-v3" / filename, summary["artifacts"][filename]))["result"]


def _case_form(scene, name):
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None or scene not in LANES:
        raise ValueError("Unexpected registered case identifier")
    return f"{scene}__{name}.npz"


def prepare(root, out, code):
    start = perf_counter()
    parents, old, summary = prior(root)
    sources = {name: pin(root / rel, sha) for name, (rel, sha) in SOURCES.items()}
    short, long, eligibility = honghu.accepted_catalog(root, sources, parents["scope"]["sha256"])
    individual = {}
    for scene, names in plan.ARCHITECTURES.items():
        bank, _ = _freeze_bank(root, scene, old)
        rows = [previous_case(root, summary, scene, name) for name in names]
        check(list(old["cases"][scene]) == list(names) and len({row["population"] for row in rows}) == 1
              and {row["bank"] for row in rows} == {bank.pin["sha256"]}, "Parent bank, cohort or IDs changed")
        individual[scene] = {name: dict(seed=row["model_seed"], model=row["model"],
                                      step=row["selected"], checkpoint=row["checkpoint"])
                             for name, row in zip(names, rows, strict=True)}
    scope = dict(
        schema=plan.POLICY, code=dict(commit=code["commit"], source_hash=code["source_hash"]),
        parents=parents, previous={k: old[k] for k in ("sources", "guards", "focus", "rare_policy")},
        sources=sources, alias_salt=token_hex(32), focus=plan.FOCUS, competitors=plan.EXPECTED_CONFUSIONS,
        cases={**{scene: list(names) for scene, names in plan.ARCHITECTURES.items()},
               "honghu24": [row["id"] for row in short], "honghu60": [row["id"] for row in long]},
        registered_honghu={"honghu24": short, "honghu60": long}, eligible_honghu=eligibility,
        windows={"xuzhou": plan.windows(436), "paviaU": plan.windows(103)}, selected=individual,
        controls=dict(random_window_seeds=plan.RANDOM_SEEDS, random_region_seed=plan.REGION_SEED,
                      graph_cells=plan.ALL_CELLS, full_information_cells=plan.FULL_INFORMATION,
                      full_anchor="baseline_features_and_full_graph",
                      prior_middle_window="excluded_from_position_sweep",
                      calibration="one_train_only_empirical_log_odds_temperature_no_weight_updates",
                      cross_device_abs_tolerance=1e-5),
        optimizer_updates=0, query_accessed=False, raw_gt_loaded=False,
        model_selection=False, registered_at=now(), setup_seconds=perf_counter()-start,
    )
    out.mkdir(exist_ok=False)
    return _write(out / "scope.json", scope)


def read_bundle(path_pin):
    with np.load(io.BytesIO(transport._read(path_pin)), allow_pickle=False) as stored:
        meta = json.loads(bytes(stored["metadata"]).decode())
        arrays = {key: np.asarray(stored[key]) for key in stored.files if key != "metadata"}
    return meta, arrays


def write_bundle(path, meta, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, metadata=np.frombuffer(canonical_json(meta).encode(), dtype=np.uint8), **arrays)
    _atomic_new(path, stream.getvalue())


@contextmanager
def case_lock(path):
    with path.open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def scope_check(root, scope_link, code, *, producer=True):
    scope = _read(scope_link)
    parents, old, summary = prior(root)
    check(scope["schema"] == plan.POLICY and scope["parents"] == parents
          and scope["previous"] == {k: old[k] for k in ("sources", "guards", "focus", "rare_policy")}
          and scope["raw_gt_loaded"] is scope["query_accessed"] is scope["model_selection"] is False
          and scope["optimizer_updates"] == 0, "Wrong mechanism scope, source or access policy")
    if producer:
        check(scope["code"] == dict(commit=code["commit"], source_hash=code["source_hash"]),
              "Worker checkout differs from the exact registered runtime")
    return scope, old, summary


def _execute(root, scope, parent_docs, old, summary, lane, name, state):
    if lane in plan.ARCHITECTURES:
        return native.run_windows(root, scope, parent_docs, lane, name, previous_case(root, summary, lane, name))
    if lane == "honghu24":
        selected = next(row for row in scope["registered_honghu"][lane] if row["id"] == name)
        return honghu.short_scores(root, scope["sources"], state["honghu_context"], selected, scope["alias_salt"])
    selected = next(row for row in scope["registered_honghu"][lane] if row["id"] == name)
    return honghu.long_scores(state["honghu_bank"], state["honghu_context"], selected, scope["alias_salt"])


def run(root, scope_link, output, code, partition, case):
    start, started = perf_counter(), now()
    scope, old, summary = scope_check(root, scope_link, code)
    output.mkdir(exist_ok=True)
    jobs = [(lane, name) for lane in LANES for name in scope["cases"][lane]
            if (partition == "all" or partition == lane) and (case is None or case == name)]
    check(bool(jobs), "No registered cases match the request")
    _, parent_docs = source_documents(root)
    state, completed, new = {}, [], 0
    for lane, name in jobs:
        filename = _case_form(lane, name)
        path = output / filename
        with case_lock(output / f".{filename}.lock"):
            if path.exists():
                meta, arrays = read_bundle(pin(path))
                check((meta["scope"], meta["lane"], meta["name"]) == (scope_link["sha256"], lane, name)
                      and meta["raw_gt_loaded"] is meta["query_accessed"] is False
                      and meta["optimizer_updates"] == 0 and set(meta["array_keys"]) == set(arrays),
                      "Resume found an incompatible case artifact")
            else:
                if lane.startswith("honghu") and "honghu_context" not in state:
                    state["honghu_context"] = _honghu_context(_read(scope["sources"]["short_scope"]))
                if lane == "honghu60" and "honghu_bank" not in state:
                    state["honghu_bank"], _ = _freeze_bank(root, "whu-honghu", old)
                tick = perf_counter()
                result, arrays = _execute(root, scope, parent_docs, old, summary, lane, name, state)
                meta = dict(scope=scope_link["sha256"], lane=lane, name=name, producer=code["commit"],
                            optimizer_updates=0, query_accessed=False, raw_gt_loaded=False, result=result,
                            finished=now(), elapsed=perf_counter()-tick, array_keys=sorted(arrays),
                            rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                write_bundle(path, meta, arrays)
                new += 1
                print(f"completed {lane}/{name} with {result.get('forwards', result.get('physical_forwards'))} forwards",
                      flush=True)
            completed.append(dict(case=f"{lane}/{name}", artifact=filename, sha256=pin(path)["sha256"]))
    require_frozen_code(code)
    print(json.dumps(dict(scope=scope_link["sha256"], completed=completed, new_cases=new, start=started,
                          finish=now(), elapsed_seconds=perf_counter()-start), sort_keys=True))


def collect(root, scope_link, cases, output, code):
    scope, _, _ = scope_check(root, scope_link, code)
    artifacts = {}
    for lane in LANES:
        for name in scope["cases"][lane]:
            filename = _case_form(lane, name)
            link = pin(cases / filename)
            meta, arrays = read_bundle(link)
            check(meta["scope"] == scope_link["sha256"] and meta["producer"] == code["commit"]
                  and (meta["lane"], meta["name"]) == (lane, name) and meta["optimizer_updates"] == 0
                  and meta["raw_gt_loaded"] is meta["query_accessed"] is False
                  and set(meta["array_keys"]) == set(arrays), "Cannot collect partial or mixed-protocol artifacts")
            artifacts[filename] = dict(sha256=link["sha256"], forwards=meta["result"].get(
                "physical_forwards", meta["result"].get("forwards")), elapsed=meta["elapsed"])
    return _write(output, dict(schema=plan.POLICY, scope=scope_link["sha256"], commit=code["commit"],
                               status="complete", cases=len(artifacts), artifacts=artifacts,
                               optimizer_updates=0, raw_gt_loaded=False, query_accessed=False, finished=now()))


def compare_devices(root, scope_link, output, code, physical):
    scope, _, summary = scope_check(root, scope_link, code)
    if re.fullmatch(r"GPU-[0-9a-f-]{36}", physical) is None:
        raise ValueError("Physical GPU UUID required")
    _, docs = source_documents(root)
    rows = []
    with gpu.lease(root / ".p4_gpu_leases" / f"{physical}.lock"):
        before = gpu.gate(physical)
        check(torch.cuda.device_count() == 1, "Expose only the admitted physical GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for scene, names in plan.ARCHITECTURES.items():
            for name in names:
                previous = previous_case(root, summary, scene, name)
                cpu, a = native.run_windows(root, scope, docs, scene, name, previous, limited=True)
                device, b = native.run_windows(root, scope, docs, scene, name, previous, limited=True, device="cuda")
                check(cpu["aliases"] == device["aliases"] and set(a) == set(b) and cpu["hashes"] == device["hashes"],
                      "Cross-device validation roster or observation identity differs")
                per_view = [{"view": k, "max_abs": float(np.max(np.abs(v - b[k]))),
                             "argmax_flips": int(np.sum(v.argmax(1) != b[k].argmax(1)))} for k, v in a.items()]
                max_abs = max(row["max_abs"] for row in per_view)
                flips = sum(row["argmax_flips"] for row in per_view)
                rows.append(dict(scene=scene, name=name, points=len(cpu["aliases"]), max_abs=max_abs,
                                 argmax_flips=flips, within_tolerance=max_abs <= scope["controls"]["cross_device_abs_tolerance"],
                                 views=per_view, cpu_seconds=cpu["runtime_seconds"], gpu_seconds=device["runtime_seconds"],
                                 cuda_peaks=device["cuda_peak_bytes"], label_scope="released_devval"))
        after = gpu.snapshot(physical)
    require_frozen_code(code)
    return _write(output, dict(scope=scope_link["sha256"], status="measured", physical_uuid=physical,
                              device_before=before, device_after=after, rows=rows, optimizer_updates=0,
                              query_accessed=False, raw_gt_loaded=False, finished=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "collect", "compare-devices"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--partition", choices=("all", *LANES), default="all")
    parser.add_argument("--case")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--gpu")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    torch.set_num_threads(1)
    root, out = external_directory(args.root), external_directory(args.output)
    check(out.is_relative_to(root) and root != out, "Output must be inside the authorized runs directory")
    if args.phase == "prepare":
        print(prepare(root, out, code))
        return
    check(args.scope and Path(args.scope[0]).resolve().is_relative_to(root), "Pinned private scope is required")
    source = pin(*args.scope)
    if args.phase == "run":
        run(root, source, out, code, args.partition, args.case)
    elif args.phase == "collect":
        check(args.cases is not None and args.cases.resolve().is_relative_to(root), "Authorized case directory required")
        print(collect(root, source, args.cases, out, code))
    else:
        check(args.gpu is not None, "Compare-devices requires a physical GPU UUID")
        print(compare_devices(root, source, out, code, args.gpu))


if __name__ == "__main__":
    main()
