# C12: missing publication figures

Issue #535, sole writer C; base `3752b89c6c42807053f2aef94d0a4c3831a4abf5`.
Only this directory is changed. D's manuscript and prior figures are read-only.
All computation and PDF rendering run on the authorized LXC. No model, image
array, new prediction, optimizer, training or query scoring is used.

Audit D's frozen source archive before generating figures. Existing scene-effect,
class-mean, routing, architecture and cost diagrams are reused or left untouched.
Add only the missing same-protocol S2 four-cell view, the separately qualified
WP04 failed class-protection panel, and the updated C7/C9 established support
versus C10 prospective admission diagrams.

```
python research/paper_c_paper_figures/freeze.py --expected-commit FULL_SHA --run NEW_RUN
python research/paper_c_paper_figures/figures.py --expected-commit FULL_SHA --run RUN --out NEW_OUTPUT
python research/paper_c_paper_figures/audit.py --run RUN --output OUTPUT --out NEW_RECEIPT
```

The frozen inputs contain deidentified confusion/summary values only. Three
model seeds do not create three acquisitions. Historical S2, WP04 and C7 labels
must not be pooled into a unique-pixel total across releases. Prospective HyRANK
cells remain unmeasured, with no performance marks or zeros substituted for NA.
