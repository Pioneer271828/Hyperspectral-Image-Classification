# Frozen G4G short-loop reproduction

The runtime tested and used on real data is `b29f7b9f1a99c1245d593dcd167404d3f953d450`, tag `module-e4g4g-target-short-v1`. `real.py` and `verify.py` reproduce the recorded LXC layout; they are orchestration/audit records added after runtime acceptance. Copy them **outside** the frozen checkout, keep that checkout clean, and inspect paths/output ownership before execution. Never overwrite a completed directory. A fresh independent rerun needs a new explicitly reviewed output root; do not substitute another source/data protocol.

`real.py` records every twelve-update CLI in `launch.json`, protects fifteen source/observation/family files, runs nine cases and performs immutable completed resume. Each frozen-head case interrupts at five and resumes seven updates. `verify.py` checks saved/raw calibrated state and graph binding, independently recomputes54 head metrics/720 recalls, and verifies training/input files unchanged. Query labels are not loaded or scored. All execution belongs in an authorized remote container, never Windows.

See the [acceptance report](../../reports/E4G4G_目标支持训练选模与冻结特征恢复验收_20261105.md) for exact commands, retained failures and numerical scope. These scripts are not full target training or a proof of transfer accuracy.
