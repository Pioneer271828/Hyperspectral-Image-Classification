"""Verify frozen train/val metadata and legacy VCA outputs without any optimizer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import whosmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_baseline import _item_fingerprint
from loadData.data_read import read_band_metadata
from loadData.observation_sampling import sample
from loadData.spatial_protocol import PartitionObservation
from model.observation_coverage import describe
from model.variable_channel_adapter import VariableChannelAdapter
from research_runtime import code_snapshot, hash_file, require_frozen_code, tensor_hash
from transfer.container_runtime import training_container_kind


MANIFEST = "414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b"
PREPARED = "base97a-honghu-issue97-20261103-prepared01"
SCENES = ("paviaU", "xuzhou", "whu-honghu")
ROLES = ("train", "val")
SNAPSHOT = "snapshots/joint-honghu-20261103-issue97/dataset"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(prepared, image_root, device):
    manifest_path = prepared / "manifest.json"
    require(hash_file(manifest_path) == MANIFEST, "Frozen train/validation manifest differs")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["test_scoring"] is False and tuple(manifest["datasets"]) == SCENES,
            "Only the frozen train/validation release is permitted")
    torch.manual_seed(64)
    adapter = VariableChannelAdapter(latent_channels=128, embed_dim=64).to(device).eval()
    rows = {}
    for scene, entry in manifest["datasets"].items():
        raw = entry["raw"]
        image = image_root / raw["image_path"]
        require(image.resolve(strict=True).is_relative_to(image_root)
                and hash_file(image) == raw["image_sha256"], "Frozen image bytes differ")
        headers = whosmat(image)
        require(len(headers) == 1 and tuple(headers[0][1]) == tuple(raw["shape"]), "Unexpected image MAT header")
        path = prepared / f"{scene}-prepared.pt"
        require(entry["prepared_file"] == path.name and hash_file(path) == entry["prepared_sha256"],
                "Frozen prepared bytes differ")
        item = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        require(not any(key.startswith(("test_", "query_")) for key in item)
                and _item_fingerprint(item) == entry["prepared_fields"], "Frozen prepared fields differ")
        metadata = read_band_metadata(scene, item["bands"])
        require(metadata.wavelength_quality == "approximate" and metadata.fwhm is None and item["fwhm"] is None,
                "Historical metadata quality or FWHM availability differs")
        torch.testing.assert_close(torch.tensor(metadata.wavelengths), item["wavelengths"], rtol=0, atol=0)
        result = dict(image_sha256=raw["image_sha256"], image_mat_variables=headers,
                      prepared_sha256=entry["prepared_sha256"], quality=metadata.wavelength_quality,
                      metadata_sha256=metadata.fingerprint, wavelength_tensor_sha256=tensor_hash(item["wavelengths"]),
                      bands=len(metadata.band_ids), limits_nm=[min(metadata.wavelengths), max(metadata.wavelengths)],
                      fwhm=None, approved_calibration=False, probes={})
        spectral = item["net_input"].reshape(-1, item["net_input"].shape[-1])
        wave = item["wavelengths"].to(device)
        for role in ROLES:
            labels = item[role + "_samples_gt"]
            ids = (labels > 0).nonzero().flatten()[:64]
            source = spectral[ids, :item["bands"]].unsqueeze(0).numpy()
            obs = PartitionObservation(role, source, np.zeros((1, len(ids), 2), dtype=np.float32),
                                       ids.numpy().reshape(1, -1), MANIFEST)
            legacy = adapter(torch.as_tensor(source, device=device), wave)
            opt_in, info = adapter(torch.as_tensor(source, device=device), band_metadata=metadata, return_attention=True)
            torch.testing.assert_close(legacy, opt_in, rtol=0, atol=0)
            probe = {}
            for name, config in (("full", {"mode": "none"}),
                                 ("stride2", {"mode": "uniform_stride", "stride": 2, "offset": 0})):
                view, record = sample(obs, metadata, config)
                values = torch.tensor(view.observation.spectral, device=device)
                observed, diag = adapter(values, band_metadata=view.bands, return_attention=True)
                expected = adapter(values, torch.tensor(view.bands.wavelengths, device=device))
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                coverage = describe(view.bands, diag["anchors"])
                require(diag["metadata_hash"] == record["selected_metadata_hash"]
                        == coverage.diagnostics["metadata_hash"], "VCA/E1 metadata binding differs")
                probe[name] = dict(pixels=len(ids), retained=len(view.bands.band_ids),
                                  active_vca_anchors=int(diag["anchor_valid"].sum()),
                                  usable_e1_anchors=int(coverage.usable.sum()),
                                  out_of_reference=coverage.diagnostics["out_of_reference_count"],
                                  same_device_legacy_max_abs_error=float((observed-expected).abs().max()))
            result["probes"][role] = probe
        rows[scene] = result
    path = image_root / "Xuzhou/Xuzhou_Readme.txt"
    require(path.resolve(strict=True).is_relative_to(image_root), "Xuzhou release readme escapes frozen image root")
    return dict(scenes=rows, prepared_manifest_sha256=MANIFEST, prepared_commit=manifest["code"]["commit"],
                xuzhou_readme_sha256=hash_file(path), xuzhou_image_hdr_present=(path.parent / "image.hdr").is_file(),
                optimizer_updates=0, heldout_reads=0, raw_gt_reads=0, trained_predictions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = code_snapshot()
    require(training_container_kind() == "lxc" and source["commit"] == args.expected_commit,
            "Authorized LXC and exact code required")
    require_frozen_code(source)
    root = args.base.resolve(strict=True)
    runs = root / "runs"
    out = args.out.resolve()
    require(out.is_relative_to(runs) and not out.exists(), "New output under the external runs directory required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        value = audit(runs / PREPARED, root / "datasets" / SNAPSHOT, args.device)
    require_frozen_code(source)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(dict(status="passed", commit=source["commit"], device=args.device, **value), stream,
                  indent=2, allow_nan=False)
    print("Three frozen image headers and train/validation metadata verified. Zero optimizer updates.")


if __name__ == "__main__":
    main()
