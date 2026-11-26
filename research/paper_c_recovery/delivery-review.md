# C package 3: recoverable engineering comparison

Issue [517](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/517),
Draft [518](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/pull/518).
Only `research/paper_c_recovery/` changes. The preceding C package, shared model
defaults, public interfaces and status ledger remain unchanged. No main merge.

## Completed work and exact producer identities

| Frozen producer | Actual work |
|---|---|
| `be72a03742e02f78caa8aa204506272db335a151` | 18 DSNet/native scratch models, 2286 real optimizer updates |
| `110c95d769a6ec88d8cf8c9ac14298302afb959f` | Five new ordered directions × three seeds, 2007 new target updates; no new source fits |
| `f4c5e11dbdc413ba922d3064e21deb4002726246` | 15 frozen transfer models × six arms × six rotated rounds = 540 routing stages, zero optimizer updates |

The unique new real optimizer total is **4293**. The nine native source models
are part of the first 2286 updates; transfer reuse does not add their fitting
cost again. Source cost remains additional to each target adaptation. Synthetic
transaction/integration/profile tests are separate and never counted as real
training. All scientific execution used the authorized LXC, CPU single thread.

The producer environment was Python3.10.20, Torch2.7.1+cu128, NumPy2.0.2 and
SciPy1.15.3, with CUDA disabled and one compute thread. No dependency installation
or shared training configuration was changed. Private artifact roots under the
authorized project root are `runs/paper-c-recovery-20261126`,
`runs/paper-c-transfer-engineering-20261126`,
`runs/paper-c-routes-engineering-20261126` and
`backups/paper-c-recovery-20261126-01`, all outside checkouts. The three producer
checkouts remain clean at the exact SHAs above. CPU off-host byte/state inspection
used its existing Python3.12.3/Torch2.7.0a0 build; this differing runtime was not
used for training or claimed to continue the optimizer identically.

The scratch registration SHA256 is
`8263398636b284a27ee5f14cec3494fe0f8dc943195cde7042315d658d8cad17`.
The transfer registration SHA256 is
`154de65e4941f5f8efe1265a54918006497be5206b9cc3b723b203d91b73a2f3`.
Both were frozen before their real execution. Baseline and first routing case
stopped after two durable commits and resumed explicitly; failed attempts and
receipts remain in the private run roots. No uncertain update was replayed.

## Recovery and independent verification

Each initial/update commit contains parameters, buffers, Adam state, all CPU RNG
states, module modes, exposure cursor and first-minimum development selection.
Atomic immutable directories plus record/state hashes detect corruption and
pending uncertain updates. A pending intent is rejected rather than silently
retrained. This is complete-boundary recovery; arbitrary mid-update continuation
and exact cross-host optimizer continuation have not been established.

Independent audits passed all 18 scratch models and all 15 transfers. Transfer
initial backbones exactly equal the selected source; target heads, initial RNG
and empty Adam exactly equal the corresponding scratch initial state. Target
orders, updates and selection events match. The source head/optimizer/RNG are
not inherited. All 15 routing cases independently reproduce raw input identities,
shape-aware matched random plans, full/light endpoints and aggregate metrics.
Custodian-array verification trapped neural calls at zero.

Actual completed scratch CLI reentry passed 18 cases with zero model/data
construction, deserialization, neural calls and optimizer updates. Selected
scratch checkpoints were also reloaded: exact readouts, 192 development neural
calls, zero optimizer updates. Completed routing CLI reentry passed all 15 cases
with zero model/data construction or new updates and unchanged case artifacts;
107190720 static endpoint probability elements match exactly across six rounds.
That audit ran at `9898e13c0b598a73efaf872a7ae5e535d03c8d2c` against the frozen
routing producer. Later input-identity hardening was not retroactively called
the original producer; independent input-byte audit already passed all cases.

Unit checks cover nine transaction cases, actual six-shape method recovery,
three admission negatives, transfer initialization/recovery, capsule retention,
shape-aware cohorts and three frame transport cases: **19 distinct unit cases**.
The original model integration requires pinned author/native dependencies and
passed with both provided; a dependency skip is not counted as that proof.
Transport checks at `b90dea40d8cf82e41893eb4575fdb70505ca247b` passed three tests.
The [execution-binding table](test-evidence.json) pairs each group with its actual
earlier source SHA and retained pass/failure log. Log text alone does not contain
every source identity. They were not all rerun at the documentation tip.

Tests use `python -m unittest discover -s research/paper_c_recovery -p <pattern> -v`
at each table entry's frozen SHA, with `PAPER_C_NATIVE` and `PAPER_C_AUTHOR` set to
the clean pinned checkouts. Readback entry points are `summarize.py`,
`audit_transfer.py`, `audit_routes.py`, `check_reentry.py`, `verify_selected.py`
and `check_route_reentry.py`; their retained logs and aggregate JSON files are in
the review export. Final source AST/JSON parsing and ZIP hash/JSON checks are
separate from these model integration tests.

The manual whole-repository command at
`9898e13c0b598a73efaf872a7ae5e535d03c8d2c` was:

```sh
python scripts/check_repository.py --expected-commit 9898e13c0b598a73efaf872a7ae5e535d03c8d2c
```

It **failed** on the inherited stale generated current-status Markdown. Its
pre-existing SHA256 remains
`306cc6e7661723ca0df62271b71fa7fa568bc3f2ecca3f25f607c08fc4f6984f`.
C did not change another writer's ledger/projection. No CI ran; no whole-project
acceptance is claimed.

## Descriptive results and boundaries

Both methods use the same K5 support and V5 development releases, 200 distinct
training and 200 separate development points across three scenes, exact paired
block order, Adam1e-4 and three label-exposure sweeps. Original native graph
losses and DSNet CE+SAD remain different, as do capacity/context (tile-local
native graph versus replicate-padded 7×7 DSNet patches). This common engineering
recipe is not the external author training protocol or shared project default.

External source review is inherited from the frozen
[C reference ledger](../paper1_c/source-review.json). DSNet dynamically loads
author `hanzhu97702/DSNet` at
`e910795073b52373ce0771ac85139a17f250b236`, whose root code license is GPL-3.0;
no author files, trainer, split or data protocol are vendored. The adapter
preserves the author's unclamped SAD/abundance path and its singular-input
failure behavior. Native runtime dependency is pinned to
`395caef57fba8ab6a09223538f748179dab552f2`; released bank custodian is pinned to
`49bce8c5fa2d8b79c0e58b2a70693c7b897d4eb7`. The complete external `demo.py`
trainer is not reproduced. Numerical adapter tests establish that specific
forward path, not a paper-level reproduction or architecture superiority.

| Scene | Native mean OA % | DSNet mean OA % | DSNet − native pp |
|---|---:|---:|---:|
| PaviaU | 11.852 | 25.926 | +14.074 |
| HongHu | 6.061 | 10.909 | +4.848 |
| Xuzhou | 17.778 | 17.037 | −0.741 |

These very short-budget selected readouts do not establish architecture
effectiveness or convergence. Measured baseline component seconds were forward
33.364, backward 44.219, optimizer 8.576, development selection 10.463 and state
checkpoint 34.017. Stored baseline state bytes were 9723854025. Cold verification,
loading, audit, IPC and backups are additional; these are shared-CPU measurements,
not isolated deployment or GPU performance. The pending forty-sweep proposal
estimates 129443310132 bytes and was not started or declared complete.

The 15 new target adaptations separately measured forward 48.941s, backward
65.100s, optimizer 5.626s, development selection 13.314s and checkpoint 24.138s;
their stored state files total 1602285328 bytes. These values come from the
existing independent lineage audit's committed per-update records, with the
same exclusions. All selected target steps are positive; one HongHu target
selected step132 rather than its final198, so selection is not replaced by a
last-step assumption.

The bounded historical inventory found zero matching candidates for the five
remaining directions among 49 sidecars/eight tasks across two hosts. This is a
bounded old inventory, not a global absence claim. Newly preregistered source3
→ target3 models now remove that engineering weight-lineage blocker for
PaviaU→HongHu, Xuzhou→PaviaU, Xuzhou→HongHu, HongHu→PaviaU and HongHu→Xuzhou.
Their full-model mean OA differences from paired target-scratch3 controls are
approximately −1.21, 0.00, −0.91, −1.48 and −5.93 percentage points respectively.
All five new directions are development engineering. The prior PaviaU→Xuzhou
G3 result has different source10000/target1000 budgets and K10 protocol; the six
available directions must not be pooled as one comparable efficacy matrix.

The cheap model is the exact trained target CNN branch, copied without extra
fitting. Full inference rebuilds Q/A for every heavy call, with zero graph reuse.
Including paid setup, all routes made 21948 light calls and 13098 full/graph calls,
13452 gate calls, zero coverage fallbacks and zero light failures. All five
direction means use exactly 50% of static full graph calls. Mean paired resident
p50 ratios are about 1.06, .94, 1.03, .94 and 1.07: no stable speed benefit claim.
The two PNG/SVG figures and 45-row source table were visually checked; clipped
cost-chart labels were repaired in `figures02`. Original `figures01` is retained.

Coverage dominates the whole-tile margin range and saturates all shape/coverage
matched cohorts in nine cases (HongHu/Xuzhou targets); the matched-random plan is
therefore forced for every random seed. In the six PaviaU-target cases cohorts
are not all saturated and the observed random plan differs. The earlier C2/G3
negative result remains frozen and is neither erased nor generalized beyond its
scope. Approximate ≤1000nm cuts are no-ops for PaviaU/HongHu; Xuzhou's cut retains
122/436 bands. None is a verified camera SRF intervention.

## Device, semantics and roles

The [primary WHU provider](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm)
supports HongHu 940×475×270, 2017-11-20, 17mm lens, Headwall Nano-Hyperspec and
published class21 Broad bean count1328. Hanchuan's 2016-06-17 date agrees with the
earlier frozen catalog; it is a recheck, not a correction. Metadata does not
authenticate local retained-band/SRF mapping, serial calibration or class order.
HongHu class21 geography still shares a parent. Xuzhou coal/cement naming conflict
remains unresolved; original numeric target IDs are used without remapping.
PaviaU wavelength conventions remain approximate. Three model seeds and six
timing rounds do not create independent geographic, sensor or evaluation roles.
All 30 ordered-pair rows remain versioned; available development profiles are
six, formal independent confirmation directions are **zero**. No new labels,
full GT, query access, calibration or confirmation was released.

## Timing, backup and remaining work

Package timing uses `clock.curr_time` only, starting 2026-11-26T08:28:01Z.
The final timing receipt records actual elapsed wall time. Effective work and
pure waiting were not separately instrumented and remain unknown; the four-hour
effective-work target is not asserted as achieved. A paired sample bracketed by
clock-tool 11:36:09–11:36:10Z showed source `date -u` 19:39:15Z. These clocks differ
by about8h03m, so source/GitHub display timestamps are not subtracted from clock
tool times. Committed per-component costs use monotonic timers. Background
backup overlaps analysis/documentation; no time is padded with repeated training.

Original source artifacts are retained in their authorized external LXC roots.
Critical capsules include every initial/selected/current complete state and the
full original file-hash history, but physically omit other intermediate states.
The full 8337564720-byte historical baseline archive is separately sealed at
SHA256 `b9cf9adc398d79214c23abd034e4178ff4e82c2fdd0fb2ba81fd5ac092fdb045`;
its off-host transfer is incomplete. Interrupted partials/journals are retained.

Transfer critical capsule: 17655194 bytes,
SHA256 `24c7a57188f2185f1f6eba7344702fecdb0653e638c7222ec511c08c69813101`.
Off-host byte restoration and all 15 initial/selected/current states passed CPU
inspection with zero model construction, neural calls and optimizer updates.
It omits 1991 historical states / 1585490519 bytes, which remain on the LXC.

Baseline critical capsule: 106360985 bytes,
SHA256 `dc3dd8c12eb3d9c9bfb083c11b7736f27fb0b6171837ff9b82bb1dd90d8c01b7`.
It omits 2291 intermediate/profile states / 9693195208 bytes. Off-host byte
restoration and all 18 initial/selected/current state envelopes passed CPU
inspection with zero model construction, neural calls and optimizer updates.
Together the two critical capsules verify 33 complete initial, 33 selected and
33 final states. Original raw-bank/runtime/path dependencies are still required
for execution; this inspection does not prove exact cross-host continuation.

All routing outputs are sealed with exact whole-file SHA deduplication:
161206780 bytes, SHA256
`4684a7ee159b400aa56322614b77089bfcf1e4ddfee15fca16365b4d4923d6d6`;
1173 logical files, 710 unique objects. Every original probability output remains
represented byte-identically; no raw input bank is included. Off-host restoration
verified all 1173 logical files / 1256931576 bytes and 710 unique objects, with
zero neural calls or optimizer updates. The final review exporter requires this
receipt plus both critical state receipts before accepting a completed matrix.
Later closing receipts/source/aggregate review copies are incremental.

Next useful research work requires separately admitted compute/storage and a
preregistered sufficient-budget design, or genuinely independent geographic and
sensor/semantic role evidence. Existing exposed V5 points cannot become
confirmation by renaming. No additional training is started to fill elapsed time.
