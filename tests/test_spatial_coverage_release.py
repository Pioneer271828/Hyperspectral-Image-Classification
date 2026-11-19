from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from transfer import development_bank as custody, frozen_mechanism_graph as graphs
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import apply_view, bank_view, model_from_source, parent_folds
from transfer.spatial_coverage_training import generated, shared_source


def make_bank(root, classes, split):
    release, arrays = generated(classes, split)
    item = custody.publish(root/f"{classes}-{split}",root,release,lambda row: arrays[row["id"]])
    return custody.Bank({k:item[k] for k in ("root","sha256","registration")},root,release)


def test_folds_exclude_nearby_and_common_parent_without_relabeling():
    rows = [dict(role="devtrain",blocked=None,id=i,parent_tile=p,box=box,samples=[[i,1 if i<3 else 2]])
            for i,p,box in ((0,5,[0,4,0,4]),(1,5,[0,4,6,10]),(2,11,[4,8,0,4]),(3,42,[15,18,0,4]))]
    for fold in parent_folds(rows,3,2):
        assert not set(fold["fit"]) & set(fold["held"])
        if 0 in fold["held"]:
            assert 1 in fold["held"] and 2 not in fold["fit"]
        assert fold["held_counts"][2] == 0
        assert fold["min_gap"] is None or fold["min_gap"] >= 2


def test_missing_is_real_absence_and_index_preserves_original_positions(tmp_path):
    torch.set_num_threads(1)
    bank = make_bank(tmp_path,9,"a")
    row=bank.role_rows("devval")[0]
    model = _baseline(bank.release,64)
    witness=model_from_source(shared_source("xuzhou",81271),model,"xuzhou")
    assert len(witness["copied"]) == 29 and len(witness["graph"]) == 6
    assert model.backbone.adapter.cache_attention is False
    wavelengths={}
    callback=model.backbone.adapter.forward
    for view in ("missing","nominal_index","dispersed"):
        bound,_,_ = bank_view(bank,row,"fixture",view)
        assert bound.receipt["source"]["band_ids"] == list(bound.view.bands.band_ids)
        with patch.object(model.backbone.adapter,"forward",wraps=callback) as observed:
            forward(model,bound,"fixture","devval")
            wavelengths[view]=observed.call_args.kwargs["wavelengths"]
    assert wavelengths["missing"] is None
    assert wavelengths["nominal_index"] is not None
    assert wavelengths["dispersed"] is not None
    assert wavelengths["nominal_index"].max() <= 1
    assert wavelengths["dispersed"].min() > 100


def test_deleted_column_and_foreign_split_q_are_isolated(tmp_path):
    torch.set_num_threads(1)
    a,b = (make_bank(tmp_path,9,name) for name in ("a","b"))
    assert a.release.statistics["statistics_hash"] != b.release.statistics["statistics_hash"]
    model = _baseline(a.release,64)
    key = state_hash(model)
    row = a.role_rows("devtrain")[0]
    raw = a.materialize(row,{"mode":"none"})[2]
    retained,positions,_=apply_view(raw,a.release.bands,"continuous")
    poison=raw.observation.spectral.copy()
    poison[..., list(set(range(poison.shape[-1]))-set(positions))] = 1.e9
    dirty=replace(raw,observation=replace(raw.observation,spectral=poison))
    perturbed,_,_=apply_view(dirty,a.release.bands,"continuous")
    q = graphs.execute(model,"xuzhou",retained,[0],9,100,role="devtrain",model_state=key)
    p = graphs.execute(model,"xuzhou",perturbed,[0],9,100,role="devtrain",model_state=key)
    graphs.assert_same_view(q,p)
    with pytest.raises(ValueError,match="Foreign"):
        wrong = replace(retained,observation=replace(retained.observation,plan_hash=b.release.plan_hash))
        graphs.execute(model,"xuzhou",wrong,[0],9,100,role="devtrain",model_state=key,frozen=q.graph)
    with pytest.raises(ValueError):
        b.materialize(row,{"mode":"none"})
    with pytest.raises(ValueError,match="Band view"):
        apply_view(raw, replace(a.release.bands, band_ids=tuple(reversed(a.release.bands.band_ids))),"continuous")
