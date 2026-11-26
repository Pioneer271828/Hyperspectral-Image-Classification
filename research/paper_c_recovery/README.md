# Recoverable development comparison

Issue [517](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/517).
Sole writer C, branch `codex/paper-c-recovery-20261126`; changes stay in this
directory. Base is published C package `45f88cf22ba1de89bc48f2e03a8487ad17f34908`.
The previous package and its coverage/matched-random negative result remain frozen.

This package implements isolated update transactions, equal released-label
exposure, original method adapters, synthetic recovery tests, and cost/admission
audits. It does not change shared models, defaults, dataset readers or checkpoint
interfaces. Existing banks are development only. No query or new label release
is implied. Real execution requires a separately pinned registration, measured
cost estimate and explicit update budget; the previous proposal authorizes zero.

Recovery commits the initial selection and then each complete optimizer update,
including any scheduled development selection. Every committed state includes
parameters/buffers, Adam state, Python/NumPy/Torch RNG, module modes, cursor and
first-minimum selection identity. Immutable stage records form a hash chain.
A process interrupted inside an update leaves a pending intent and is rejected;
the executor never silently repeats an uncertain optimizer call. A graceful stop
or interruption after a complete commit resumes only from that exact boundary.
Completed reentry verifies all original pins and stage bytes before construction
or deserialization. No claim of arbitrary mid-instruction continuation is made.

All Python execution, profiling and tests run on the authorized LXC. The first
reconnect timed out; the next succeeded and verified hostname `lbx`, LXC, C run
root and Python 3.10.20. Windows is used only for editing, Git and SSH transport.

## Entry points

* `transactions.py`: immutable optimizer commits and complete-zero-call reentry.
* `tasks.py`: pinned original method adapters; native learned graph64/32 with
  GroupNorm and unchanged three-head/graph losses, DSNet external CE+SAD.
* `cost_profile.py`: actual model shapes, synthetic input only, three updates
  per method/scene. The first successful profile executed18 synthetic updates,
  zero real updates. Complete checkpoint planning for40 sweeps estimates
  129443310132 bytes, exceeding then-current LXC free space.
* `bindings.py`: metadata/hash-only check of original K5/V5 roles and maximum
  tile geometry; plan both3-sweep engineering and40-sweep development budgets.
* `run.py prepare`: require explicit sweep/update/storage budgets and pinned
  cost/role bindings; reject insufficient storage before any real update.
* `run.py run`: consume only the hashed registration. `--resume` is explicit;
  `--stop-after-case-updates 2` exercises a complete boundary in the first
  unfinished case. Completed runs can reenter with no model/data construction.
* `audit.py`: independent plan/exposure/hash chain, confusion/recall, selection,
  Adam cursor and RNG-envelope checks. It does not import the producer.

A3-sweep engineering registration, if admitted, has18 cases and2286 maximum
real optimizer updates, with3600 total label exposures over the same200 released
training points; selection uses the separate200 previously released development
points. It selects at0/1/2/3 sweeps. This is explicitly shorter than the pending
40-sweep proposal and cannot complete its convergence/effectiveness study.
The common Adam1e-4 recipe, different method losses, capacities and contexts
are declared experimental factors; shared project defaults remain untouched.

Tests use `python -m unittest discover -s research/paper_c_recovery -p 'test_*.py' -v`
with `PAPER_C_NATIVE` and `PAPER_C_AUTHOR` pointing to verified pinned checkouts.
Without those variables, the original-model integration test is explicitly
skipped; do not report the skip as a successful actual-model recovery test.
