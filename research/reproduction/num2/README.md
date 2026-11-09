# Frozen M4 decision evaluation amendment

This is the NUM-2 pre-test amendment (#143), following the descriptive NUM-1 audit (#141). It does **not** pass or erase the original unstable absolute1e-5 probability gate. The exact selected checkpoints, training budgets and accuracy-benefit gates remain unchanged. See the [full report](../../reports/M4S2_残差与辅助监督三种子完整配对及精度敏感性_20261104.md).

- Producer checkout stays at `b36f1c976120eb8abd7d74e7f44295227550e9ea` in `/home/lbx/hsi-pioneer271828/checkouts/m4s2-issue135`.
- Accepted evaluation consumer was `9d5318803d926eacebf8984bc4e6a7bed388bbc4` in a separate checkout. Later report/QA files are additive; they were not the producer or evaluated consumer.
- `evaluate.py` verifies all frozen state and train/validation decisions before test scoring; emits deterministic FP32 and exact-cast same-state FP64 readouts. The original graph conversion is matched before promotion. TF32 matrix multiplication is off, cuDNN TF32 remains allowed for FP32 tensors; this is not a pure-IEEE FP32 benchmark.
- `verify.py` independently reconstructs raw-label confusion matrices,144 head metrics/1920 recalls, paired effects, class guards and precision sensitivity. It does not choose a preferred precision/model by accuracy. Data/checkpoints/packets remain outside Git.
- Runtime4 unit tests include near-tie decision rejection and explicit retention of the original probability failure; original producer399 regressions are separate evidence.

Paths and output names describe this frozen user environment. Replaying in another environment requires isolated producer/consumer checkouts and adapting paths with new script hashes recorded. Never add verifier files to the frozen producer checkout, overwrite historical output, or update a running checkout. The original `analysis01` failure and all finalization retries are retained; `analysis02` is the amended evaluation. This supports decision metrics under two tested contexts, not bitwise reproducibility, probability calibration, unseen-sensor generalization or independent spatial confirmation.
