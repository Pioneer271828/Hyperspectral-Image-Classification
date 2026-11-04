# E0-B1 independent support verification

The verifier recomputes frozen support counts using SciPy neighborhood extrema and masked binary dilation, without importing the production boundary implementation. It checks per-class counts, partition conservation and immutable raw/prepared inputs. Run only in the remote training environment. Exact tested SHA, commands and external output paths are in the E0-B1 report. No model predictions or optimizer steps are involved.

Both label-access policies are descriptive: within-split geometry versus whole-annotation geometry followed by scoring-mask restriction. Whole-annotation results must never feed training, selection or pruning. Observed nonboundary pixels are not certified interiors.
