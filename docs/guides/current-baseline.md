# Development baseline and delivery

The primary workspace is `F:/E2E_VCA`, branch `codex/f-drive-primary`. Review and resolve integration changes in a separate worktree based on the fetched `github/main`; keep local datasets, model weights, logs and uncommitted files in place. See [migration and preservation](f-drive-primary.md) and the [project handoff](../../research/项目开发交接_20261105.md).

## Current delivery and branch roles

Use [current-status.json](../architecture/current-status.json) and its [generated checklist](../../research/当前搭建状态与审查整改清单_20261102.md) for research status. Check GitHub and the corresponding PR for the current merge state. The [pre-M4 guide](current-baseline-pre-m4-20261102.md) and [delivery ledger](../architecture/delivery-status.json) describe historical audits.

| Role | Branch or identifier | Scope |
|---|---|---|
| Published code | `github/main` after fetch | Reviewed integration; cite the exact SHA used for any verification |
| Primary development | `codex/f-drive-primary` | Project entry point and status projection |
| Review candidate | Dedicated integration branch and worktree | Small, dependency-ordered changes with evidence from the candidate SHA |
| Frozen experiments | Exact tested SHAs and immutable experiment tags | Preserve original implementation, failures, budgets and protocols |

Main merges require independent review. Use ancestry, patch equivalence and the reviewed file scope to identify superseded work. Keep the original branch and experiment identity for a frozen result. Close a superseded PR only after its replacement has been accepted.

## Integration contract

The [delivery ledger](../architecture/delivery-status.json) records the accepted historical boundaries. `scripts/check_repository.py` checks required ancestry, paths, the status projection and repository integrity. Use full history for both the checker and tests that reproduce old results. Changes to a frozen runtime contract require explicit review.

Keep default configuration, label access, update budgets and checkpoint compatibility under review when combining modules. Code integration and experiment acceptance are separate: record the exact trained producer, authorized data scope and independently verified results before making a research claim.

## Remote acceptance and automatic execution

Use the [authorized LXC environment](remote-validation.md) at `ssh -p 1222 lbx@10.69.37.243`. Check container identity, checkout cleanliness and the full SHA first. Use a separate checkout for a new candidate and keep data, logs and checkpoints outside it.

Use `python scripts/check_repository.py --expected-commit <full SHA>` for current integration acceptance. `scripts/validate_baseline.py` still checks the original frozen file inventory; the M8 validator and its E0a replay require the original M8 source identity and its accepted manifest. For current frozen prediction exports use [the standalone E0a exporter](evaluation-e0a.md#冻结预测导出入口).

The workflow `.github/workflows/remote-validation.yml` requires a dedicated authorized runner with labels `self-hosted`, `linux`, `hsi-training-container`. Its default interpreter is `/home/lbx/hsi-pioneer271828/envs/hsi/bin/python`; use `HSI_VALIDATION_PYTHON` to select a different existing interpreter on that runner. Report the workflow result only when an actual job has run. Manual SSH acceptance is recorded separately.

The runner uses an isolated Actions workspace, full history and existing dependencies. The CPU suite runs for matching events when the runner is available. Before requesting focused GPU checks by manual dispatch, confirm a free device and the required lease. Keep failed jobs and receipts. The [2026-11-05 acceptance report](../../research/reports/仓库审查整改与开发整合验收_20261105.md) records the historical commands and exact SHAs.

## Updating project status

Edit only `docs/architecture/current-status.json`, then remotely run:

```bash
python scripts/check_repository.py --write-status
```

Review and commit the generated current checklist with the JSON. The two main plans and mandatory scope point to that checklist and retain dated research designs/records. Experiment progress comes from each frozen campaign receipt; this delivery task neither reads live query labels nor changes another task's training status.

Sensor identity and wavelength quality, independent support/spatial repeats, compression effects, latency and all remaining research candidates require their own evidence. Keep them open instead of marking them resolved by engineering integration.
