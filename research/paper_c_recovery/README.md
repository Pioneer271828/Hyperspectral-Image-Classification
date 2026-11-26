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
