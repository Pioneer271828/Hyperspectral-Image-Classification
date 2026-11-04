# E0-B2 frozen spatial diagnostics

`verify_spatial.py` is the independent raw-packet audit run at14e03f0d348617f0f8616908ecac58ea94e41521. It uses SciPy neighborhood morphology and direct confusion/branch arithmetic, without importing the production metric or boundary functions. Exact commands are in the E0-B2 report.

`summarize.py` is an exact archive of the external post-verification reporting script, with recorded dedicated LXC paths. It emits aggregate statistics only and never runs a model. Its output path is exclusive; adapting paths for another run requires recording the changed script identity. The summary contains all defined cases, including undefined metrics and negative findings.

The analysis does not constitute new training, untouched-test confirmation, or proof of general boundary improvement. Preserve first-run schema failure, all original weights/predictions, and numerical restoration limitations.
