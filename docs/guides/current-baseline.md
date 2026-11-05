# Current delivery and branch roles

Issue #167 resolves the 2026-11-05 branch audit. Read the [current status](../../research/当前搭建状态与审查整改清单_20261102.md) for research progress and [current-status.json](../architecture/current-status.json) for the single editable source. The [pre-M4 guide](current-baseline-pre-m4-20261102.md) and [delivery ledger](../architecture/delivery-status.json) are explicitly historical snapshots, not current branch inventories.

| Role | Branch or identifier | Scope |
|---|---|---|
| Published main snapshot | `4d2c7dbe4a081d671b1367bc4fb3eabf5e929ccc` | Initial baseline, includes the old VCA cache bug |
| Focused main repair candidate | `codex/issue-167-main-cache-repair` | Accepted VCA cache/autograd repair and compatibility tests |
| Development integration candidate | `codex/issue-167-development-integration` | Accepted G4I plus MON-2/3C, snapshot fix and NUM-1 history |
| Frozen experiments | Exact tested SHAs and immutable experiment tags | Preserve original implementation, failures, budgets and protocols |

Main merges require independent review. Branch publication is not a main update. Do not merge all historical PRs blindly: use commit inclusion and reviewed patch scope, retain unresolved scientific work, and close or archive superseded PRs only after an accepted integration. No branches or tags were deleted by this task.

## Integration contract

The JSON lists required ancestor SHAs, required module paths and frozen G4I model/training/data paths. `scripts/check_repository.py` rejects missing repairs, shallow history, stale generated status, changed tracked files and changes to those frozen runtime paths. New research changes require an explicit new integration contract, rather than weakening this one silently.

MON-2/3 remain opt-in and retain their original native FP32 and overhead limitations. Historical MON-3C rejected all three fixed channel candidates. Snapshot capture now owns storage before downstream in-place operations. No new pruning, quantization or speed result is asserted. M4-G4A is still outside this validated integration until its separate acceptance completes.

## Remote acceptance and automatic execution

Use the authorized training container and its existing environment. Check container identity, checkout cleanliness and full SHA first. [Remote synchronization](remote-validation.md) permits published Git bundles when GitHub transport is unavailable. Never reset/clean a shared checkout, update a running producer, or put datasets/logs/checkpoints in the source tree.

The workflow `.github/workflows/remote-validation.yml` targets labels `self-hosted`, `linux`, `hsi-training-container`. A repository administrator must register a dedicated runner inside the authorized training environment; no runner credentials are created or committed here. Set the repository variable `HSI_VALIDATION_PYTHON` to the existing interpreter if its path differs from `/home/rmvision/venvs/hsi-baseline/bin/python`. Until a runner is registered and an actual job succeeds, CI remains pending, even after manual SSH acceptance passes.

The runner must use an isolated Actions workspace, full history and the existing dependencies. CPU validation is automatic for matching events. Focused GPU validation is manual dispatch; it does not interrupt other workloads or start experiments. Workflow permissions are read-only and checkout credentials are not retained. Preserve failed jobs/receipts. The [remote acceptance report](../../research/reports/仓库审查整改与开发整合验收_20261105.md) records the exact tested commits and commands after acceptance.

## Updating project status

Edit only `docs/architecture/current-status.json`, then remotely run:

```bash
python scripts/check_repository.py --write-status
```

Review and commit the generated current checklist with the JSON. The two main plans and mandatory scope point to that checklist and retain dated research designs/records. Experiment progress comes from each frozen campaign receipt; this delivery task neither reads live query labels nor changes another task's training status.

Sensor identity and wavelength quality, independent support/spatial repeats, compression effects, latency and all remaining research candidates require their own evidence. Keep them open instead of marking them resolved by engineering integration.
