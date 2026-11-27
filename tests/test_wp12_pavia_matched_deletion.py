from argparse import Namespace

import pytest

from evaluation import wp12_pavia_matched_deletion as review


def test_partial_and_unmatched_support_cannot_clear_the_recall_gate():
    template = dict(seed="64", arm="low_weight", head="fused")
    rows = [dict(template, class_id="1", support="3", delta_recall_pp="0",
                 interpretation="strict_context_intersection"),
            dict(template, class_id="2", support="0", delta_recall_pp="",
                 interpretation="strict_context_intersection")]
    result, = review._protection(rows, certified=True)
    assert (result["status"], result["reference"], result["unsupported_classes"]) == ("partial_support", "important", "2")
    rows[0]["delta_recall_pp"] = "-100"
    result, = review._protection(rows, certified=True)
    assert (result["status"], result["classes_beyond_drop_limit"]) == ("exceeds_limit", "1")
    result, = review._protection(rows, certified=False)
    assert result["status"] == "diagnostic_only" and result["reference"] == "none"


def test_wrong_capture_pin_is_rejected_before_any_artifact_is_opened(monkeypatch, tmp_path):
    commit = "1" * 40
    monkeypatch.setattr(review, "code_snapshot", lambda: {"commit": commit})
    monkeypatch.setattr(review, "require_frozen_code", lambda code: None)
    monkeypatch.setattr(review.audit, "read", lambda *args: pytest.fail("unexpected artifact read"))
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="approved archived capture digest"):
        review.publish(Namespace(expected_commit=commit, source_sha256="0" * 64, out=out))
    assert not out.exists()
