# Current baseline and delivery status before M4

The functional reference is **`6fa2d06747f79e90f95928b099ba2c3c0b18cbd0`**, on `codex/issue-68-g3-run-provenance`. Its final runtime-provenance acceptance tested code `aa243eef45f4d0bb0868b888c6cc3252ce703e0f`; the remaining change was documentation, with all 166 source/config fingerprints checked unchanged. It includes VCA/dense-graph cache fixes, dynamic-task registration, actual G3 run settings, data protocols and the English README.

Delivery navigation and its verifier are added on top in `codex/issue-70-delivery-baseline`. That layer changes documentation and its own validation utility/tests only. The exact delivery test SHA and final documentation SHA are recorded in the delivery report/PR. Main is still the old `018fe70fcbd3e902cd5fe0a05296bdd1d38d5bfb` snapshot; a default-branch clone does not provide the delivered modules. [Draft integration PR #71](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/71) is a separate review candidate, with no automatic merge.

## One current source for versions and status

[delivery-status.json](../architecture/delivery-status.json) is the machine-readable delivery snapshot. It separates implementation, declared engineering acceptance, empirical evidence and main integration. The [research checklist](../../research/当前搭建状态与审查整改清单_20261102.md) owns the current research queue; [module-map.json](../architecture/module-map.json) owns code/test navigation. Historical reports remain evidence for their exact scope and SHA. Update the delivery snapshot and related Issue when completing a task; do not reinterpret an old pending/running paragraph as current state.

The snapshot covers **33 open PRs, 36 non-PR Issues and 102 commits beyond main**, observed before Issue #70. All 33 PR heads are ancestors of the functional reference. Their actual base-to-head ranges leave **17 bridge commits** without an independent PR range: early pilot preparation and G3 validation/budget/LR work. The JSON lists each full SHA and evidence. These commits are included in integration review; they are not silently omitted or assigned invented PRs. PR #55 explicitly had no separately confirmed Issue; Issue #70 records that tracking gap. PR #34 remains Draft in GitHub despite its completed experiment report; later PR #40 fixes its resume-summary issue.

The versioned counts describe that snapshot, not the number of live PRs after the current delivery and integration candidates are opened. Issue/PR open state remains unchanged by this ledger: a delivered engineering module can still await review, and a negative or infeasible experiment can be a complete report without proving the proposed model/protocol works. Issue #62 completion is reconciled with PR #63; Issues #58–61 retain their requests plus links to the actual outcomes.

## Clone and verify in the training container

Use the authorized container and a new checkout, first inspecting any existing checkout for uncommitted work. Keep full history: several regression tests read earlier source commits. Do not use `--depth 1`, overwrite a running checkout or store outputs under the repository.

```bash
git clone --branch codex/issue-70-delivery-baseline \
  https://github.com/Pioneer271828/Hyperspectral-Image-Classification.git hsi-pre-m4-delivery
cd hsi-pre-m4-delivery
# Pin the full published delivery SHA recorded in the delivery report/PR.
git switch --detach PUBLISHED_DELIVERY_SHA
git status --short --branch
python scripts/verify_delivery_status.py \
  --expected-commit PUBLISHED_DELIVERY_SHA --out-dir /absolute/external/NEW_DELIVERY_RECEIPT
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py' -v
G3_DEVICE=cuda OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_run_provenance.py -v
python smoke_test.py --device cuda
```

Use the existing environment's absolute Python path, as recorded in the [remote guide](remote-validation.md). If Git transport fails, synchronize an already published full-history Git bundle and verify its ref/SHA before checkout. This is Git-object synchronization, not proof of direct GitHub fetching. Credentials are not copied into the container. Exact commands, receipt and retained failures belong in the run report.

For the functional reference without the navigation layer, clone its branch and pin `6fa2d06747f79e90f95928b099ba2c3c0b18cbd0`; run its existing suite (201 tests at acceptance) and smoke commands. `verify_delivery_status.py` is available only in the newer delivery layer. The verifier rejects missing history, misleading merge/inclusion claims, dependency cycles, unlisted bridge commits, missing evidence and algorithm/training changes beyond this reference. It verifies a dated snapshot offline; it does not query current GitHub approval status.

## Repository locations and execution boundaries

- `C:/Users/asus/.codex/worktrees/hsi-baseline/E2E_VCA` is the active Git worktree, currently carrying the published development branches.
- `F:/E2E_VCA` is the original checkout with separate old code, local modifications, data and materials. It shares Git objects with the worktree; its files/branch do not update automatically.
- The user-facing plans and current reports in `F:/E2E_VCA/research` receive the declared status updates. Their historical full text can differ from tracked copies; this does not make the original F-drive code the current implementation.
- The remote container holds isolated checkouts, datasets and run artifacts. Execute all code, dependency installation, static checks and tests there. Do not move/delete worktrees or copy current code over the original F-drive files as part of this delivery task.

## Integration review and the next mechanism

Review existing module PRs in dependency order using their actual base/head refs, then the two explicit bridge gaps before PRs #55 and #63. Check the three final repair PRs #65/#67/#69, include the accepted scope and preserve negative results. The separate draft main candidate aggregates the delivered baseline and this navigation layer for review. If a later merge introduces conflict resolutions or changes executable files, validate that final commit. Branch publication, a draft PR and a successful engineering check do not mean main has been updated.

M4 engineering can start from the verified delivery branch without waiting for main merge. Its first scope is region representation/pixel residual decomposition with an independent off switch, gradients and capacity controls. M6/M7 stay outside that initial experiment. Existing development validation has zero effective boundary labels under the declared rule, so the boundary-evaluation contract must be established before claiming boundary gains. Next come M1 training integration, independent E1 coverage tests, then MON-2/3 and pruning/quantization according to the research checklist.
