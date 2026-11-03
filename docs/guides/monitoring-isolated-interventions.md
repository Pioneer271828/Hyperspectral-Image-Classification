# MON-3B: isolated fixed-model probes and channel interventions

Issue #121 builds on [MON-3A](monitoring-fixed-probes.md). `isolated_probe` accepts a declared reference model, trusted fresh-model factory, trusted forward callback, frozen tensor inputs and a `FrozenProbe` plan. It constructs an independently stored model for every condition, including the no-op control. It executes eval/no-grad and returns detached CPU outputs, conditional observations and an audit. No optimizer, training policy, structural pruning or automatic channel ranking is installed.

```python
import copy
from monitoring.interventions import isolated_probe, ChannelIntervention

result = isolated_probe(
    reference_model, lambda: copy.deepcopy(reference_model),
    forward_on_frozen_inputs, inputs, probe_plan,
    interventions=(ChannelIntervention("cnn-call0", (0, 1)),),
)
```

Freeze candidates before reading intervention scores. An empty intervention tuple is the control. The example zeros the declared module-call output channels0/1 together on the isolated copy, without changing parameters or mutating the original output tensor in place. Compare single channels and groups separately; two individually harmless removals can be harmful together. This operation does not physically shrink a layer or establish speed improvements.

## Execution contract

- Synchronous eager diagnostic execution in a separate process is the supported deployment. Concurrent training threads, arbitrary global hooks, compiled/distributed execution, AMP and gradient attribution are outside scope. Factory and forward callbacks are trusted code, not sandboxed; they must not mutate caller objects, global settings or access undeclared data. Newly initializing another CUDA device inside callbacks is outside the RNG contract; initialize all intended devices before entering the probe.
- Factory must return a distinct model with no shared modules/parameter/buffer storage, the same module types, devices, registered state and checkpoint metadata. Matching state does not authenticate arbitrary nonregistered forward attributes: use the same audited checkpoint loader/configuration. Existing model hooks and nested output selectors are rejected. Tensor module outputs and explicit HWC/NC/NCHW observation calls are supported.
- Inputs are detached-cloned before execution. Dense and sparse COO tensor trees are supported; copy accounting uses stored COO indices/values, not the nominal dense shape. Nonfinite tensors are rejected. Masks and thresholds remain bound to the MON-3A manifest. Do not forward test labels or select candidates with target holdout data.
- Copy budgets default to1GiB input,128MiB output,1GiB retained observation tensors; model factory allocations and temporary operations add memory. These are not total peak-memory guarantees. Return only allowed train/validation predictions rather than entire full-scene outputs when possible.
- Reference registered state, checkpoint metadata, modes and gradients are checked unchanged. Python, NumPy, Torch CPU and already initialized visible CUDA RNG states are restored in `finally`, and installed hooks are removed on success/failure. Isolated model eval state/gradients and cloned inputs must remain unchanged during forward; ordinary BN updates are prevented by eval mode. Ephemeral caches on the fresh copy are discarded with it.
- Trusted callbacks that deliberately modify the reference/input are detected, but not automatically rolled back. Do not blindly retry a failed callback. The API cannot protect against arbitrary captured external objects or authenticate an activation's origin merely from an asserted string.

## Numerical and scientific interpretation

No deterministic CUDA setting or tolerance is changed. Establish a predeclared no-op comparison against the original checkpoint's allowed predictions, retain failures, and report intervention deltas only after that check. A difference under intervention proves the selected output path influences this probe; it does not prove a dependency-aware physical prune is safe or helpful. A zero difference on a small probe does not prove a channel is dead, including across unobserved sensors or small classes.

The returned MON-3A record keeps `model_execution_verified=false` because the stand-alone observation API cannot authenticate caller values. The enclosing audit records the independently checked execution boundary; it is not a claim of untrusted-code isolation or complete experiment provenance.

References are the user RM report sections8/8.6, the MON plan, project MON-0/1/3A and existing message-intervention tests. Independent engineering implementation; no external research code was copied. This is not a new published pruning method.

Remote tests: `MON3B_DEVICE=cuda OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_monitoring_interventions.py -v`; CPU uses `MON3B_DEVICE=cpu`. Coverage includes exact signed zeroing, a known single-safe/group-harmful example, storage alias rejection, modes/gradients/RNG, mutation failures, missing calls, cleanup and sparse/bounded copies. [Scoped real acceptance](../../research/reports/MON3B_隔离探针与单组通道干预验收_20261104.md): CPU7/GPU7/full387 and12 fixed train-only probes passed at8edb9f4, preserving112 original artifacts with zero updates/test scoring; no accuracy gain or controlled overhead follows from the synthetic tests alone.
