# Hyperspectral Image Classification

Research code for a VCA spectral adapter and CNN/HGCN dual-branch backbone.
The three research axes are region–pixel complementarity, cross-sensor adaptation,
and pruning/quantization, all with accuracy as the primary objective.

Primary Windows workspace: **`F:/E2E_VCA`**, branch **`codex/f-drive-primary`**. Read the [project handoff](research/项目开发交接_20261105.md) before continuing. Research training remains paused; the M4 full-source study failed its strict native reload gate and is incomplete. Local original files are preserved outside Git as described in the [migration record](docs/guides/f-drive-primary.md).

## Current entry points

- [Current status](research/当前搭建状态与审查整改清单_20261102.md): the single human-readable status table, generated from [current-status.json](docs/architecture/current-status.json).
- [Delivery and branch guide](docs/guides/current-baseline.md): exact branch roles, integration boundaries and remote validation.
- [Module map](docs/architecture/module-map.json): code and test navigation; historical notes are evidence for their named versions.
- [Research master plan](research/研究内容整合与详细搭建测试计划_2026-10-28.md), [implementation plan](research/模块化搭建与测试消融实施计划_2026-10-28.md), and [mandatory scope](research/全量研究模块必测清单_20261104.md): design and experiment requirements.
- [Contributing](CONTRIBUTING.md) and [execution rules](AGENTS.md).

## Obtain the development integration

```bash
git clone --branch codex/f-drive-primary https://github.com/Pioneer271828/Hyperspectral-Image-Classification.git hsi-development
cd hsi-development
git status --short --branch
git rev-parse HEAD
```

Use full history: several regression fixtures read immutable historical commits.
For a reported result, pin its tested SHA or immutable experiment tag instead of
using the latest branch tip. This integration is a review candidate; main has not
received these changes. The main-based VCA repair is separately reviewable on
`codex/issue-167-main-cache-repair`.

## Validate remotely

All Python, dependency installation, checks and training run in the authorized
Linux training container via `ssh rmvision`. Windows is used for editing, reading
files and Git. Reuse the documented environment; see the [remote guide](docs/guides/remote-validation.md).

In a clean remote checkout, using its existing Python environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_repository.py --expected-commit "$(git rev-parse HEAD)"
python -m unittest discover -s tests -v
MON3B_DEVICE=cuda python -m unittest discover -s tests -p test_monitoring_interventions.py -v
REPAIR_DEVICE=cuda python -m unittest discover -s tests -p test_vca_release_compatibility.py -v
```

[Automatic acceptance](.github/workflows/remote-validation.yml) targets a dedicated
self-hosted Linux training-container runner. Runner registration is pending;
workflow presence is not a successful CI run. It installs no dependencies, starts
no real-data training and uses no dataset or SSH secrets. Focused GPU acceptance
is an explicit manual dispatch option.

## Research evidence and boundaries

Results and remaining work live in the current status table. A module can be
implemented and tested while its accuracy hypothesis fails. Joint seen-scene
accuracy, few-shot transductive adaptation and strict unseen-sensor generalization
are separate claims. Dataset identities, label budgets, graph context, source
checkpoints, seeds, selection rules and numerical amendments belong to each
frozen experiment record.

This delivery integrates accepted G4I, MON-2/3C, the activation-snapshot repair and
NUM-1 provenance. It preserves the accepted G4I model/training/data configuration
files. Frozen running checkouts and experiment tags are not updated by this work.
The newer M4-G4A development branch is outside this acceptance scope.

Datasets, weights and raw run artifacts are stored outside checkouts and are not
shipped in Git. No compression or speed improvement follows merely from masks,
synthetic tests or theoretical operation counts.
