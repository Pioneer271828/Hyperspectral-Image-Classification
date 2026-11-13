"""Run the frozen WP10 forward comparison on one physically leased CUDA device."""

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess

import torch

from compression.low_rank_recovery import HEADS
from compression.precision_cost import ResidentPrecisionTile
from evaluation.wp12_low_rank_evidence import verify_guard
from inference.wp10_low_rank_execution import _scan, package
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from scripts.review_wp10_tensorly import load_scope, locations, measure, paired, resident_tiles, score
from transfer import matched_route_admission as admission


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("runs", "legacy-checkout", "cpu-initial", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-commit", "gpu", "cpu-initial-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if (args.expected_commit != code["commit"] or torch.get_num_threads() != 1
            or not args.gpu.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu
            or torch.cuda.device_count() != 1):
        raise ValueError("Exact commit and one explicitly leased physical GPU required")
    require_frozen_code(code)
    sample = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, free, utilization = (part.strip() for part in sample.split(","))
    if uuid != args.gpu or int(free) < 12288 or int(utilization) > 60:
        raise ValueError("CUDA device no longer satisfies the shared resource threshold")

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = external_directory(args.out)
    out.mkdir()
    initial_pin = dict(path=str(args.cpu_initial.resolve()), sha256=args.cpu_initial_sha256)
    initial = admission.read(initial_pin)
    if initial["status"] != "passed" or initial["optimizer_updates"] != 0:
        raise ValueError("Successful native-initializer comparison required")
    pins, values, protected, _, _, dense, scene, record = load_scope(args.runs.resolve(),
                                                                     args.legacy_checkout.resolve(), code)
    reg = values["manifest"]
    tiles = resident_tiles(reg)
    selection = locations(reg, scene, record)
    dense.backbone.adapter.cache_attention = False
    torch.cuda.synchronize()
    # Time the device copy and synchronization separately from forwards.
    def prepare():
        result = [ResidentPrecisionTile(tile.tile_id, tile.image.to("cuda"), tile.wavelengths.to("cuda"))
                  for tile in tiles]
        torch.cuda.synchronize()
        return result

    cuda_tiles, synchronized_transfer = measure(prepare)
    candidates = {name: {k: row[k] for k in ("case", "initial", "package")}
                  for name, row in initial["candidates"].items()}
    entries = dict(reg["cases"], **candidates)
    results, cpu_rows, gpu_models = {}, {}, {}
    outputs = {}
    for name, entry in entries.items():
        cpu = package(entry)
        cpu_rows[name], cpu_output = score(cpu, reg, tiles, selection)
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        def load_cuda():
            cpu.to("cuda")
            torch.cuda.synchronize()
            return cpu
        device_model, device_load = measure(load_cuda)
        torch.cuda.synchronize()
        (row, gpu_output), first = measure(lambda: score(device_model, reg, cuda_tiles, selection))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        deltas = {head: max(float((left[i]-right[i].cpu()).abs().max())
                            for left, right in zip(cpu_output, gpu_output, strict=True))
                  for i, head in enumerate(HEADS)}
        changed = {head: sum(a != b for a, b in zip(cpu_rows[name]["predictions"][head],
                                                   row["predictions"][head], strict=True)) for head in HEADS}
        results[name] = dict(row, cpu_cuda_max_probability_delta=deltas, changed_predictions=changed,
                             host_to_device_model=device_load, first_instrumented_sweep=first,
                             peak_allocated_delta=peak-start, state=entry["package"]["state"])
        if name in candidates:
            gpu_models[name] = device_model
            outputs[name] = gpu_output
        print("cuda", name, flush=True)

    def launch(model):
        torch.cuda.synchronize()
        result = _scan(model, reg, cuda_tiles)
        torch.cuda.synchronize()
        return result

    limits = values["registration"]["protection"]["limits"]
    small = values["audit"]["small_classes"]
    comparisons = {}
    for arm in initial["comparisons"]:
        native, tensorly = f"native-{arm}", f"tensorly-{arm}"
        comparisons[arm] = dict(
            cpu_cuda_guard={name: verify_guard(cpu_rows[name], results[name], reg["class_ids"], small, limits)
                            for name in (native, tensorly)},
            native_tensorly_guard=verify_guard(results[native], results[tensorly], reg["class_ids"], small, limits),
            max_probability_delta={head: max(float((a[i]-b[i]).abs().max()) for a,b in
                zip(outputs[native], outputs[tensorly], strict=True)) for i,head in enumerate(HEADS)},
            resident=paired({name: lambda name=name: launch(gpu_models[name]) for name in (native, tensorly)}, 3, 12))
    for name, model in gpu_models.items():
        row, _ = score(model, reg, cuda_tiles, selection)
        if row["output_hashes"] != results[name]["output_hashes"]:
            raise ValueError("Strict CUDA repeat changed candidate output hashes")
    admission.check_pins([*pins.values(), *protected, initial_pin])
    require_frozen_code(code)
    write_json(out / "cuda.json", dict(status="completed_strict_forward", commit=code["commit"], sample=sample,
        physical_uuid=uuid, runtime_device=torch.cuda.get_device_name(0),
        strict=True, tf32=False, torch=str(torch.__version__), cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        resident_input_copy=synchronized_transfer, cases=results, comparisons=comparisons,
        original_cases=len(reg["cases"]), new_cases=len(candidates), cpu_initial=initial_pin,
        query=0, gt=0, optimizer_updates=0))


if __name__ == "__main__":
    main()
