"""Audit actual frozen VCA settings and train/validation spectral response."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from loadData.data_read import read_band_metadata
from model.variable_channel_adapter import VariableChannelAdapter
from scripts.audit_oss_a_history import CAMPAIGNS, PREPARATION, SCENES, PHASES, digest, read


MANIFEST = "414084cb22aef1c1866bb6512fa1f99c4397425de625c6fc2608ce0fb3eb1b6b"


def frozen_inputs(runs):
    directory = runs/PREPARATION
    path = directory/"manifest.json"
    if digest(path)!=MANIFEST:
        raise ValueError("Unexpected historical train/validation manifest")
    entries = read(path)["datasets"]
    result = {}
    for scene in SCENES:
        file = directory/(scene+"-prepared.pt")
        entry = entries[scene]
        if entry["prepared_file"]!=file.name or digest(file)!=entry["prepared_sha256"]:
            raise ValueError("Changed historical train/validation payload")
        item = torch.load(file,map_location="cpu",weights_only=True,mmap=True)
        if any(key.startswith("test_") for key in item):
            raise ValueError("Forbidden held-out fields")
        bands = item["bands"]
        wave = torch.as_tensor(item["wavelengths"],dtype=torch.float32)
        metadata = read_band_metadata(scene,bands)
        torch.testing.assert_close(wave,torch.tensor(metadata.wavelengths),rtol=0,atol=0)
        if item["fwhm"] is not None:
            raise ValueError("Unexpected historical FWHM")
        flat = item["net_input"].reshape(-1,item["net_input"].shape[-1])
        result[scene] = (wave,metadata.wavelength_quality,
            {role:flat[(item[role+"_samples_gt"]>0).nonzero().flatten()[:64],:bands].clone() for role in PHASES})
    return result


def probe(adapter,scene_inputs):
    wave,quality,roles = scene_inputs
    rows = []
    for role,pixels in roles.items():
        for view,positions in (("full",torch.arange(len(wave))),("even",torch.arange(0,len(wave),2))):
            measured = wave[positions]
            x = pixels[:,positions].unsqueeze(0)
            with torch.no_grad():
                y,info = adapter(x,measured,return_attention=True,wavelength_unit="nm")
                if adapter.attention_mode=="wavelength":
                    historical,_ = adapter(x,measured,return_attention=True)
                    torch.testing.assert_close(y,historical,rtol=0,atol=0)
                active = info["anchor_valid"].bool()
                attention = info["attention"]
                energy = y.double().square().mean(dim=(0,1))
                rows.append(dict(role=role,view=view,pixels=pixels.shape[0],bands=len(measured),quality=quality,
                    active_anchors=int(active.sum()),anchors=len(active),
                    below=int((measured<adapter.wavelength_min).sum()),above=int((measured>adapter.wavelength_max).sum()),
                    minimum_nm=float(measured.min()),maximum_nm=float(measured.max()),
                    active_anchor_energy=float(energy[active].sum()),inactive_anchor_energy=float(energy[~active].sum()),
                    entropy=float((-(attention*attention.clamp_min(1e-30).log()).sum(1))[active].mean()),
                    explicit_nm_matches_historical_auto=True if adapter.attention_mode=="wavelength" else None))
    return rows


def audit(runs):
    inputs = frozen_inputs(runs)
    rows = []
    for campaign,(count,commit) in CAMPAIGNS.items():
        cases = sorted((runs/campaign/"cases").iterdir())
        if len(cases)!=count:
            raise ValueError("Frozen study case roster differs")
        for case in cases:
            result = read(case/"result.json")
            if result["start_code"]["commit"]!=commit or result["test_scoring"] is not False:
                raise ValueError("Frozen study identity differs")
            config = result["settings"]["config"]["model"]
            step = result["selected_step"]
            file = case/f"state-{step:06d}.pt"
            if digest(file)!=result["artifacts"][file.name]:
                raise ValueError("Changed selected training state")
            packet = torch.load(file,map_location="cpu",weights_only=True,mmap=True)
            if packet["cursor"]!=step:
                raise ValueError("Frozen cursor differs")
            prefix = "backbone.adapter."
            state = {k[len(prefix):]:v for k,v in packet["model"].items() if k.startswith(prefix)}
            adapter = VariableChannelAdapter(
                latent_channels=config["latent_channels"],embed_dim=config["adapter_embed_dim"],
                attention_mode=config["adapter_attention_mode"],chunk_pixels=config["adapter_chunk_pixels"],
                checkpoint_chunks=config["adapter_checkpoint_chunks"]).eval()
            adapter.load_state_dict(state,strict=True)
            rows.append(dict(campaign=campaign,case=case.name,selected_step=step,source_commit=commit,
                adapter_mode=adapter.attention_mode,latent_channels=adapter.latent_channels,embed_dim=adapter.embed_dim,
                residual=config["region_residual_mode"],aux_weights=[result["settings"]["config"]["loss"][key]
                    for key in ("graph_branch_loss_weight","cnn_loss_weight")],
                input_scope="up_to_64_existing_ordered_pixels_per_train_or_validation_role",
                scenes={scene:probe(adapter,values) for scene,values in inputs.items()}))
    return dict(cases=rows,optimizer_updates=0,heldout_reads=0,manifest=MANIFEST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.runs)
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(f"{len(result['cases'])} frozen models and {len(result['cases'])*12} authorized spectral probes")


if __name__=="__main__":
    main()
