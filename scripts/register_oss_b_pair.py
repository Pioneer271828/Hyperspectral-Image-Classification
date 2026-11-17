"""Validate a read-only registration draft for the next real joint comparison."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_oss_a_history import PREPARATION, SCENES, digest, read
from scripts.audit_oss_b_spectral import MANIFEST


SEEDS = (64,512,2048)
VIEWS = ("original_global","covered_range","same_anchor_count_global")
SUPPORT_ROOT = 64
STEPS = (3,6,9,12)


def build(runs):
    if runs.name!="runs":
        raise ValueError("Authorized runs directory required")
    directory = runs/PREPARATION
    manifest_path = directory/"manifest.json"
    if digest(manifest_path)!=MANIFEST:
        raise ValueError("Frozen historical preparation manifest changed")
    manifest = read(manifest_path)
    if manifest["test_scoring"] is not False or tuple(manifest["datasets"])!=SCENES:
        raise ValueError("Fixed train/validation-only scene roster required")
    sources = {}
    for scene in SCENES:
        row = manifest["datasets"][scene]
        name = scene+"-prepared.pt"
        if row["prepared_file"]!=name:
            raise ValueError("Unexpected frozen payload name")
        payload = directory/name
        if payload.is_symlink() or digest(payload)!=row["prepared_sha256"]:
            raise ValueError("Fixed train/validation payload changed")
        sources[scene] = dict(path=name,sha256=row["prepared_sha256"])
    cells = [dict(view=view,model_seed=seed,support_seed=SUPPORT_ROOT,updates=12)
             for view,seed in product(VIEWS,SEEDS)]
    return dict(schema="oss-b-real-pair-draft-v1",issue=462,
        data=dict(manifest_sha256=MANIFEST,source=PREPARATION,roles=["train","validation"],sources=sources,
                  graph="original fixed Q/A and original normalization and label supports"),
        fixed=dict(adapter="wavelength",latent_channels=128,embed_dim=64,residual="x_residual",
                   supervision="legacy_three_heads",outputs=["fused","graph","cnn"],
                   validation_steps=STEPS,selection="calibrated_first_maximum_macro_oa",
                   optimization="existing joint optimizer, task schedule and scheduler",
                   gates=dict(probability_rtol=1e-5,probability_atol=1e-6,aa_max_drop_pp=.5,
                              honghu_small_class_max_drop_pp=2.0,
                              honghu_guard_classes=[15,21],primary="paired_fused_validation_macro_oa")),
        hypothesis="Move 128 anchor coordinates into the observed nm range without changing tensors or their trainable capacity; test against the original and a matched-active-anchor global control.",
        cells=cells,requested_real_optimizer_updates=108,real_optimizer_updates=0,
        time_estimate_minutes=[15,30],gpu="one fixed admitted RTX 4090 UUID, no role crossing",
        additional_authority_required=["candidate version and control freeze","LXC worker and prior notification of budget and estimate"],
        state="draft_no_real_launch",real_training_authorized=False,heldout_reads=0,
        output="new runs directory outside the checkout, with schedule, exact checkpoint, train/validation results, and failed cases")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args = cli.parse_args()
    result = build(args.runs.resolve(strict=True))
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2)
    print("Verified the 108-update real draft registration; zero optimizer updates executed")


if __name__=="__main__":
    main()
