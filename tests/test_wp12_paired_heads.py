"""Guard retained-only metadata, strict role isolation and per-class adoption decisions."""

from dataclasses import replace

import numpy as np
import pytest

from evaluation.wp12_paired_head_review import RAW, checked_scope, pair_scores, matched_points
from transfer import native_paired_head_fixture as fixture
from transfer.native_mechanism_edges import publish
from transfer.native_ordinal_deployment import DeliveredAxis, bind_retained_ordinal
from transfer.native_paired_head import freeze_for_head
from transfer.development_bank_training import _baseline
from transfer.spatial_coverage_protocol import bank_view


def test_retained_only_ordinal_matches_registered_route_at_factor_two(tmp_path):
    bank,roles,_=publish(tmp_path/"known-delivery","remote",64)
    row=bank.role_row("devval",roles["evaluation"][0])
    missing,_,_=bank_view(bank,row,"fixture","central_gap",factor=2,metadata="missing")
    expected,_,_=bank_view(bank,row,"fixture","central_gap",factor=2,metadata="ordinal")
    original_reads=bank.reads
    ready=DeliveredAxis.from_bank(bank)
    remapped=bind_retained_ordinal(missing,"fixture","devval",ready)
    assert bank.reads==original_reads
    assert remapped.receipt==expected.receipt
    assert remapped.view.bands.fwhm is None and remapped.view.bands.wavelength_unit=="normalized"
    assert actual_arrays(remapped,missing)
    assert bind_retained_ordinal(missing,"fixture","devval",None) is missing
    with pytest.raises(ValueError,match="foreign"):
        bind_retained_ordinal(missing,"fixture","devval",replace(ready,bank_sha256="0"*64))


def actual_arrays(a,b):
    return all(np.array_equal(getattr(a.view.observation,key),getattr(b.view.observation,key))
               for key in ("spectral","coordinates","global_ids"))


def test_head_only_optimizer_does_not_acquire_batch_norm_or_graph_params():
    metadata=fixture.SPECS["honghu"]
    from types import SimpleNamespace

    model=_baseline(SimpleNamespace(scene=metadata[0],classes=metadata[1]),64)
    model=freeze_for_head(model,metadata[0])
    admitted={name for name,value in model.named_parameters() if value.requires_grad}
    assert admitted=={"heads.whu-honghu.weight","heads.whu-honghu.bias"}
    assert sum(param.numel() for param in model.parameters() if param.requires_grad)==65*22


def test_aggregate_score_cannot_promote_small_class_or_insufficient_population():
    truth=list(range(1,23))*5
    original=truth.copy()
    changed=truth.copy()
    # All improvements must not cancel a newly introduced error on class 21.
    original[0]=2
    changed[20]=2
    result=pair_scores(truth,original,changed,22,"whu-honghu",head_kind="fused",guard_population=5)
    assert result["delta_aa"]==0 and not result["protected"]
    assert result["ordinary_and_rare"]["21"]["introduced"]==1
    sparse=pair_scores(truth[:22],truth[:22],truth[:22],22,"whu-honghu",head_kind="fused",guard_population=5)
    assert not sparse["accepted"] and not sparse["adequate"]


def test_reject_development_scope_or_foreign_labels():
    good=dict(schema=RAW,scope="independent_synthetic_confirmation",independent_real_accuracy=None,
              query_accessed=False,raw_gt_loaded=False,real_updates=0)
    checked_scope(good)
    with pytest.raises(ValueError,match="Development evidence"):
        checked_scope(good|dict(scope="released_development_offline_only"))
    record=dict.fromkeys(("source","tile","band","metadata","layout","mechanism","truth","points","spectral",
                         "geometry","fit","shape","band_ids"))
    with pytest.raises(ValueError,match="truth"):
        matched_points(record,record|dict(truth=[15,21]))


def test_synthetic_two_draws_and_k1_are_role_safe():
    for kind,view in (("xuzhou",10),("honghu",5)):
        a=fixture.definition(kind,fixture.SUPPORT[0])
        b=fixture.definition(kind,fixture.SUPPORT[1])
        assert a[-1]["counts"]["support"]==[view]*a[0].classes
        assert a[-1]["minimum_buffer"]>=8
        sets=[{idx for row in r[0].rows if row["role"]=="devtrain" for idx,_ in row["samples"]} for r in (a,b)]
        assert sets[0].isdisjoint(sets[1])
    rare=fixture.definition("honghu",fixture.SUPPORT[0],override_k=1)
    assert rare[-1]["counts"]["support"]==[1]*22
