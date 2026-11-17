"""Use the fixed DOFA position helper only inside an isolated RNG scope."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.variable_channel_adapter import VariableChannelAdapter
from research_runtime import hash_file
from transfer.oss_a_spectral import SCENES, diagnostic


DOFA_BLOB_SHA256 = "f3ab4b3af85b73765bc4a48e39b12d2ed6c81d2c06936b3e9ca16b1b10e25a28"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dofa-source",required=True,type=Path)
    cli.add_argument("--out",required=True,type=Path)
    args = cli.parse_args()
    if hash_file(args.dofa_source)!=DOFA_BLOB_SHA256:
        cli.error("Fixed, inspected DOFA source required")
    spec = importlib.util.spec_from_file_location("_oss_a_dofa_wavelength_reference",args.dofa_source)
    module = importlib.util.module_from_spec(spec)
    before = torch.random.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        spec.loader.exec_module(module)
        reseeded = not torch.equal(before,torch.random.get_rng_state())
        torch.manual_seed(64)
        adapter = VariableChannelAdapter(latent_channels=64,embed_dim=32)
        rows = [row for scene in SCENES for row in diagnostic(
            adapter,scene,dofa_position_encoder=module.get_1d_sincos_pos_embed_from_grid_torch)]
    if not torch.equal(before,torch.random.get_rng_state()):
        raise RuntimeError("DOFA diagnostic changed caller RNG")
    output = dict(rows=rows,source_sha256=DOFA_BLOB_SHA256,import_seed_side_effect=reseeded,
                  rng_restored=True,optimizer_updates=0,real_labels=0)
    with args.out.open("x") as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print("Verified 12 wavelength/index and full/missing-band diagnostics")


if __name__=="__main__":
    main()
