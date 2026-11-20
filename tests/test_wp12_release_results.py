from copy import deepcopy

import pytest

from evaluation import wp12_release_results as release
from evaluation.wp12_release_results import HEADS, cohorts, cost_missing, metrics, select
from scripts.wp12_frozen54_deployment_contract import COLD, WARM, TIMING


def sample(seed, device, workload, matrix):
    return dict(source="fixture",protocol="released",scene="xuzhou",seed=seed,support=512,
                method="native",arm="none",role="devval",band="full",factor=1,k=5,endpoint="selected",
                model=str(seed),device=device,workload=workload,measurement_scope=None,
                cost=cost_missing("not timed"),heads={head:metrics(matrix,2) for head in HEADS})


def test_aggregate_requires_matched_scope_and_keeps_undefined_recall():
    rows=[sample(64,"GPU-A","resident",[[5,0],[0,0]]),sample(512,"GPU-A","resident",[[4,1],[0,0]]),
          sample(2048,"GPU-B","cold",[[1,4],[0,0]])]
    groups=cohorts(rows)
    supports=[row for row in groups if row["head"]=="fused"]
    assert [row["model_seeds"] for row in supports]==[[64,512],[2048]]
    assert supports[0]["oa"]["mean"]==.9 and supports[0]["oa"]["sample_sd"]==pytest.approx(.141421356)
    assert supports[1]["oa"]["sample_sd"] is None
    assert supports[0]["per_class"][1]["recall"]=={"mean":None,"sample_sd":None}
    assert select(rows,scene="xuzhou",device="GPU-B")==[rows[-1]]
    with pytest.raises(ValueError,match="filter"):
        select(rows,protocol_family="foreign")
    with pytest.raises(ValueError,match="duplicates"):
        cohorts([rows[0],rows[0]])


def test_end_to_end_publication_preserves_the_exact_historical_case(tmp_path, monkeypatch):
    historical = [sample(seed, "GPU-resident", "ordered-tensors", [[5, 0], [0, 0]]) for seed in range(54)]
    cases = {}
    for original in historical:
        seed = original["seed"]
        cases[str(seed)] = dict(scene=original["scene"], seed=seed, arm="native/none",
                                historical=deepcopy(original), checkpoint=original["model"],
                                evaluation=original["protocol"], workload=original["workload"],
                                scopes={scope: {"timing": TIMING[scope]} for scope in (COLD, WARM)})
    audit = dict(status="audited", engineering=False, query_accessed=False, new_optimizer_updates=0,
                 new_resident_measurements=0, source="measurement-sha", device="GPU-measured",
                 plan={"sha256": "plan-sha"}, approval={"sha256": "approval-sha"}, cases=cases)
    summary = dict(audit=release.proof(tmp_path, "frozen54_e2e_audit"), new_resident_measurements=0,
                   source=audit["source"], device=audit["device"], plan_sha256=audit["plan"]["sha256"],
                   approval_sha256=audit["approval"]["sha256"], legacy_probability_identity_all_54=True)
    documents = {summary["audit"]["path"]: audit, release.proof(tmp_path, "frozen54_e2e_summary")["path"]: summary}
    monkeypatch.setattr(release, "document", lambda root, pin: documents[pin["path"]])

    result, receipt = release.frozen54_end_to_end(tmp_path, historical)
    assert result == {"audit": audit, "scene_summary": summary}
    assert receipt["cases"] == 54 and receipt["scopes"] == [COLD, WARM]

    cases["0"]["historical"]["heads"]["fused"]["per_class"][0]["recall"] = 0.4
    with pytest.raises(ValueError, match="paired historical accuracy"):
        release.frozen54_end_to_end(tmp_path, historical)
