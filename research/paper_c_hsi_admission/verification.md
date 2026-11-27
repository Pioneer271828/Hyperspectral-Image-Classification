# C6 verification and remaining research gate

The engineering result is a reusable input reader and explicit admission gates
for the original numeric development labels. Shared-class cross-device
classification is **not yet admitted**, and no model accuracy was measured.
Current blockers and exact external materials are in `pair-feasibility.json`.
The HyRANK alternative is an original-report lead, not an admitted new dataset.

All new code execution below ran on user-authorized rmvision. Windows held
source, Git bundles, the table-only crop and aggregate receipts. Existing LXC
materials were read/hashed/streamed for preservation; no new LXC training/test
replaced rmvision validation. No shared model/config/CLI/checkpoint or status
ledger was changed; C is the sole writer of this directory.

| Check | Actual source | Result |
| --- | --- | --- |
| Original metadata/portable release identity | `18e9931eaccb4b79c493a0a4d48ec52741ccfff8` | Both banks admitted; no spectral arrays opened |
| Independent NPY decoder, frozen normalization, explicit clipped patch reference | `ca27dce880fcd0f16b75858229fbaa5d4fcd7867` | All112 originally labeled blocks passed; original310 labels,110128928 payload bytes; no classifier |
| Strict parent subset feasibility | `9025711bd651cb63cb0d1a20b3e3e1b7a41b4d65` | Metadata only; XZ all9 classes feasible, HH21 lost and HH7 devval4; no role reassignment |
| Module negative and functional tests | `3ce951a920a561cf234f36c9d71310f5b7e58a6d` | 11 passed in0.024s |
| Numeric source/target development intake | same | Two directions/four single-block input probes passed; shared-head and confirmation requests rejected before payload reads |
| Repository integration checker and JSON parse | `5f8b6f83acf1122f4b01ab42160bdeec31ae326a` | Five JSON documents parsed; repository checker exit1, inherited stale current-status Markdown |
| Full private preservation and fresh restore | `7e24362e1f65e291d16f40b03c0ed309030f5a16` |260 source files restored and hashed,0 mismatch; source bundle restored to exact complete SHA/clean worktree; both original bank admissions restored |

Private archive `preservation01.tar.gz`:118312928bytes,
SHA256`cd5ab7ea5b20b5281b88e4edf37a85d2f5669bd7199965bdc728ab8b897727eb`.
This is a verified second private copy **on the same authorized rmvision host**,
not a separate-host disaster-recovery claim. Source/evidence live outside the
checkout under
`/home/train/hsi-artifact-backups/paper-c-hsi-admission-20261127-01/`.
The full restore is in `restore01/`; inputs remain read-only to scientific
consumers. Do not place the archive or restored spectral/label/PDF files on
Windows or public Git. The unrelated historical8.337GB backup remains incomplete.

Exact validation commands, from a clean checkout with the recorded full SHA:

```sh
python -m unittest discover -s research/paper_c_hsi_admission -p test_bank_reader.py -v
python research/paper_c_hsi_admission/audit_inputs.py --expected-commit ca27dce880fcd0f16b75858229fbaa5d4fcd7867 --root "$ROLES" --out "$RUN/runs/validation07/independent-inputs01.json"
python research/paper_c_hsi_admission/spatial_plan.py --expected-commit 9025711bd651cb63cb0d1a20b3e3e1b7a41b4d65 --root "$ROLES" --out "$RUN/runs/spatial09/strict-original-roles01.json"
python research/paper_c_hsi_admission/development_probe.py --expected-commit 3ce951a920a561cf234f36c9d71310f5b7e58a6d --root "$ROLES" --out "$RUN/runs/validation11/development-probe01.json"
python scripts/check_repository.py --expected-commit 5f8b6f83acf1122f4b01ab42160bdeec31ae326a
python research/paper_c_hsi_admission/preserve.py --expected-commit 7e24362e1f65e291d16f40b03c0ed309030f5a16 --root "$ROLES" --run-root "$RUN" --out "$RUN/preservation01.tar.gz" --restore "$RUN/restore01"
```

`ROLES` is A's hash-pinned original released input root documented in README;
`RUN` is the private C6 root above. These are recorded used output paths, so a
replay must choose fresh paths. Python3.12.3/NumPy1.26.4 were reused. For original
PDF review only, private environment `env-pdf` installed pypdf6.1.3 and
PyMuPDF1.26.5; no general environment upgrade or author implementation was run.

The initial reader failure requiring absent unlabeled-tile JSON, unsuccessful
primary-metadata retrievals and integration-check failure remain in private
logs. The correction admits A's original labeled-only preservation without
fetching new unlabeled contexts. Dedicated CI registration was not performed;
there is no CI success claim. Zero new real model calls, optimizer updates,
label memberships and raw-GT/query reads apply to the entire package.

No physical band grid, certified VNIR-only view, common semantic head,
independent confirmation or causal sensor effect follows from the passed input
checks. The proposed64/64-step baseline remains a draft with zero authorized
updates, pending semantic/product admission and concrete time/cost registration.
