# C11 recovery findings

Issue #533; base `ad663e9af23356568a0c28a2f508353887d8286a`.
Branch `codex/paper-c-paper-recovery-20261127`; sole edit scope is this directory.
Training remains paused. Windows receives source and aggregate receipts only.

## What was actually missing

The unfinished 8,337,564,720-byte baseline archive does **not** imply that all
selected models lacked an off-host copy. Later receipts verify 18 baseline and
15 transfer cases, each with initial/selected/current complete state envelopes,
plus all 1,173 routing output files. These historical receipts are reused, not
reported as a new connection to that host. The currently authorized LXC is the
only execution/storage host used in C11.

The full original gzip was rehashed and still matches
`b9cf9adc398d79214c23abd034e4178ff4e82c2fdd0fb2ba81fd5ac092fdb045`.
The last recorded off-host prefix is 535,822,336 bytes; no subsequent complete
transfer receipt was found. Its historical lossless XZ derivative is another
same-host representation, not an independent recovery host.

The two papers additionally depend on input banks, exact spatial/label roles,
frozen normalization, code versions and older paper-specific weights. C3 short
budgets do not substitute for Paper 1's historical S2 model family or the
forty-sweep PaviaU native/DSNet comparison. C11 therefore supplemented these
actual dependencies rather than copying the 8 GB archive again.

All 2,062 frozen dependency paths existed and matched their expected hashes.
After reusing existing objects, 1,605 unique objects / **3,722,836,254 bytes** were
newly copied inside the LXC, then independently copied out of that backup into a
fresh restoration store. Both byte counts and SHA-256 matched, with zero file
differences. The new objects include 358 previously unrepresented released bank
arrays, frozen historical S2 inputs (opaque copies), model/state files, manifests,
code and role metadata. No GT/query array was decoded or scored.

An actual source reconstruction detected that the native dependency commit
`395caef57fba8ab6a09223538f748179dab552f2` was absent from the previously collected
bundles. Its clean LXC checkout was bundled separately. The author's DSNet clone
is partial: a full-history bundle attempted an unnecessary promisor fetch and
failed. The recovery package instead preserves exactly the reused model and
license, authenticated by original commit/tree/blob proofs at
`e910795073b52373ce0771ac85139a17f250b236`. It does not claim a complete backup of
the author's Git history.

## Restored evidence

Five indexed archives were restored byte-for-byte: baseline 4,772 files, transfer
4,116, routes 1,173, PaviaU 14,447 and three-scene released roles 328. Two older
archives were also physically checked: pilot 61 files and routing 1,892 logical
files / 1,137 unique objects. Existing WP04 factorial objects and original case
references were resolved by exact SHA rather than retraining.

The complete rebuilt chain is PaviaU→HongHu, model seed64, native-source3/target3
development engineering. It starts from backed-up source and target states,
proves source-backbone inheritance with matched fresh target heads/optimizer/RNG,
reconstructs normalized requests from backed-up spectral arrays and spatial roles,
and recomputes stored predictions into the original result table. All 99 C3
initial/selected/current state references passed CPU envelope inspection. Ten
HongHu development blocks yielded 40 requests, 36 saved stages and 144 view-metric
rows, with zero differences. The full stride1 result matches the selected target
checkpoint's recorded selection readout. No model was constructed or called.

These 110 development points are reused across three stride views and an
approximate ≤1000nm view. Six rounds are timing repeats. They do not create new
spatial samples, independent acquisitions, authenticated SRF interventions or
new cross-device efficacy evidence. The old G3 source10000/target1000 result must
not be pooled with source3/target3 as a matched six-direction accuracy matrix.

Paper 1's archived S2 aggregate values reproduce 28 FP32 table rows (seed values,
means and sample SD). This rebuild does not reopen query arrays or rerun scoring.
The historical NUM-2 amendment and the original probability policy's failure to
pass stably remain explicit. Fixed splits with model seeds64/512/2048 do not
measure independent geographic uncertainty.

The PaviaU forty-sweep capsule was additionally compared with its live LXC source:
all 14,447 retained file hashes matched. All 7,192 omitted intermediate state
files still exist (13,795,133,296 bytes). Their recorded hashes are distinct;
their full current tensor bytes were not rehashed in this supplementary audit.
The prior off-host receipt's 19 CPU state references are historical evidence,
not a claim of fresh neural replay or exact cross-host probability identity.

## Remaining exposure

| Material | Inspected critical/incremental coverage | Remaining historical state exposure |
|---|---|---|
| C3 baseline | Every initial/selected/current plus metadata; original full gzip currently verified on LXC | 2,290 unique intermediate/profile hashes / 9,692,418,647 bytes absent from inspected small capsules; full gzip is a same-host copy |
| C3 transfer | All 15 initial/selected/current envelopes and output history | 1,991 unique intermediate hashes / 1,585,490,519 bytes absent from inspected backup stores |
| Paper 1 S2 | 12 cases; 25 retained state paths, selected/full final exports, JSON/validation history and original inputs | 587 intermediate state hashes / 1,796,619,239 bytes omitted; current hashes not rechecked for that omitted set |
| Paper 1 PaviaU40 | Complete retained capsule and input roles; historical off-host verification reused | 7,192 intermediate states remain source-side; existence/size and recorded identities checked |
| Code | Project/native Git recovery plus fixed author component proofs | Author full Git history not required for the reused component and not backed up |

The union of the C3 historical hashes absent from inspected small stores is
**4,281 hashes / 11,277,909,166 bytes**, after deduplication; this is not the
remaining compressed transfer length. S2 never saved a step-zero state: its
earliest saved state is step3. A missing original step-zero envelope cannot be
repaired by labeling a later state as initial.

The new private dependency store and its fresh restore are on the same LXC.
They improve file-level recovery but do not remove host-loss exposure. Full
off-host history, newly supplemented private inputs, and exact original-runtime
continuation remain unverified. The genuine external dependency is a reachable,
explicitly authorized independent private storage destination. No host fallback,
private export to Windows, new data application or training is attempted.

## Verification and limits

Inventory: published producer `75671680fb92c6392c50c993440a051dd375c3c3`,
`recover.py --expected-commit <producer> --reuse-restored <attempt02> --out <attempt03>`.
The reuse path was checked against original archive hashes and every retained
logical file; failed attempts remain in place. Inventory wall time was 532.337s.

Reconstruction: verifier `9b811174f8c4952e469c0edb1aedb37fee381da0`,
`rebuild.py --expected-commit <verifier> --backup <attempt03>`; passed with neural
execution disabled. `python -m unittest discover -s research/paper_c_paper_recovery
-p test_restore.py -v`: 5/5 passed on the authorized LXC. Tests cover archive pins,
direct/object restoration and rejection of corrupted/missing objects and unsafe
paths. Recovery controls and source are independently sealed/restored by
`seal.py`; the external receipts identify their final actual source and hashes.

`scripts/check_repository.py --expected-commit <verifier>` exited1 on the inherited
stale current-status Markdown. Shared delivery status was outside this package's
write scope. No registered self-hosted CI or complete fresh-environment install
is claimed. No model structure, loss, sampler, budget or selection policy changed.
