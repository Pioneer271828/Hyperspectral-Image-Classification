"""Guard the interpretations and input boundaries of production confirmation."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from evaluation import wp12_locked_support as locked, wp12_native_production as proof
from evaluation.frozen_training_stages import VIEWS, file_pin
from evaluation.retained_spectral_evidence import SHARED_LAYERS
from transfer import development_bank as custody, native_production_fixture as fixture
from transfer.development_bank_training import _baseline
from transfer.native_production_intake import open_legacy_weights, real_preflight
from transfer.native_production_model import Switches, axis_slots
from transfer.native_production_provider import ProductionProvider
from transfer.spatial_coverage_protocol import retained_positions


def test_view_refit_is_rejected_when_labeled_fixed():
    base = dict(layers=dict.fromkeys(SHARED_LAYERS), fit_class_ids=[1,2],fit_sha256="full")
    rows = {view:dict(fixed_full_support=deepcopy(base),readout=deepcopy(base)) for view in VIEWS}
    rows["central_gap"]["readout"]["fit_sha256"] = "gap"
    locked.validate_fit_modes(rows)
    rows["central_gap"]["fixed_full_support"]["fit_sha256"] = "gap"
    with pytest.raises(ValueError,match="mislabeled"):
        locked.validate_fit_modes(rows)


def test_real_diagnostics_cannot_be_promoted_in_confirmation_schema():
    base = dict(schema=proof.RAW,scope="independent_synthetic_confirmation",independent_real_accuracy=None,
                real_updates=0,query_accessed=False,raw_gt_loaded=False)
    proof.verify_scope(base)
    with pytest.raises(ValueError,match="independent synthetic"):
        proof.verify_scope({**base,"scope":"released_development_diagnostics_only"})
    with pytest.raises(ValueError,match="independent synthetic"):
        proof.verify_scope({**base,"independent_real_accuracy":1.0})


def test_equal_aa_cannot_cancel_a_rare_class_loss():
    truth = list(range(1,fixture.CLASSES+1))
    before = [1,1,*truth[2:]]
    after = [*truth[:-1],1]

    def score(decisions):
        row = dict(heads={head:np.eye(fixture.CLASSES)[np.asarray(decisions)-1].tolist() for head in proof.HEADS},
                   truth=truth,repeats=[dict(total_seconds=1,prepare_seconds=0,forward_seconds=1,graph_seconds=0,
                             peak_cuda_bytes=None,process_peak_rss_kib=0)]*3)
        row.update(dict.fromkeys(("tile","factor","band","domain","mode","grid","original_pixels","retained_ids",
                                  "retained_sha256","released","graph")))
        return proof.scored(row)["heads"]["fused"]

    reference,current=score(before),score(after)
    guard=proof.paired_guard(reference,current,truth)
    assert guard["aa_delta"]==0 and guard["pass_aggregate"] and not guard["eligible"]
    assert guard["by_class"][str(fixture.RARE)]["losses"]==1
    with pytest.raises(ValueError,match="Average AA"):
        proof.paired_guard(reference,{**current,"per_class":[]},truth)


def test_removed_columns_neither_reappear_nor_affect_retained_values(tmp_path):
    release,arrays,roles,blocks=fixture.definition("target",fixture.SUPPORT[0])
    removed=sorted(set(range(release.shape[-1]))-set(retained_positions(release.bands,"central_gap")))
    tile=roles["evaluation"][0]

    def provider(name,spec,values):
        final=custody.publish(tmp_path/name,tmp_path,spec,lambda row:values[row["id"]])
        pin={k:final[k] for k in ("root","sha256","registration")}
        bank=custody.Bank(pin,tmp_path,spec)
        return ProductionProvider(tmp_path,bank,roles,axis=fixture.BANDS,config=Switches(),
                                  source=None,seed=64,budget=6,producer="fixture")

    first=provider("baseline",release,arrays)
    corrupted={**arrays,tile:arrays[tile].copy()}
    corrupted[tile][...,removed]+=400.
    rebound=deepcopy(release.rows)
    for row in rebound:
        if row["id"]==tile:
            row["origin"]["array_sha256"]=custody.array_hash(corrupted[tile])
    second=provider("poisoned",replace(release,rows=rebound),corrupted)
    actual,_,_=first.confirmation("fixture",tile,1,"central_gap")
    amended,_,_=second.confirmation("fixture",tile,1,"central_gap")
    assert actual.receipt["spectral"]==amended.receipt["spectral"]
    assert axis_slots(fixture.BANDS,actual,"fixture","devval")==axis_slots(fixture.BANDS,amended,"fixture","devval")
    full,_,_=first.confirmation("fixture",tile,1,"full")
    forged=dict(tile=tile,factor=1,band="central_gap",**{k:blocks[str(tile)][k] for k in ("domain","mode")},
                retained_ids=list(full.view.bands.band_ids))
    with pytest.raises(ValueError,match="full-band information"):
        proof.validate_row(first,forged,(tile,1,"central_gap",blocks[str(tile)]),"fixture")


def test_legacy_strict_weights_and_preflight_rejects_unsigned_request(tmp_path):
    release,_,_,_=fixture.definition("target",fixture.SUPPORT[0])
    model=_baseline(release,64)
    target=tmp_path/"legacy.pt"
    torch.save(model.state_dict(),target)
    source=model.export_config()["backbone_kwargs"]
    restored=open_legacy_weights(target,"cpu",architecture=source,switches=Switches(),axis=fixture.BANDS)
    for left,right in zip(model.state_dict().values(),restored.state_dict().values(),strict=True):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    path=tmp_path/"empty.json"
    path.write_text(json.dumps({}),encoding="utf-8")
    with pytest.raises(ValueError,match="Real comparison request"):
        real_preflight(tmp_path,file_pin(path),file_pin(path),"fixture")
