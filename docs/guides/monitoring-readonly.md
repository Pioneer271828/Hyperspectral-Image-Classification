# MON-0/1: read-only activation and weight measurements

Implemented as an opt-in Python API in `monitoring/`; existing model/config/CLI/checkpoint/training entry points are unchanged. No hooks exist until entering `ReadOnlyMonitor.sample`. No automatic dead-channel detection or pruning is performed.

Choose `LayerSpec(module_path, layout, epsilon, position, unit, selector=())` explicitly. Supported layouts are HWC, NC, or NCHW with N=1, one spatial block per call. `selector` follows tuple/list indices or dictionary keys. Shared calls get separate call_index values within a sample. Masks are boolean and exactly match all nonchannel dimensions; graph masks refer to regions, not pixel labels. Q-based weighted grouping is not implemented. Caller supplies fixed thresholds and a full run/checkpoint/data/split/step/condition identity dictionary. Only train/val partitions are accepted; this is a selection diagnostic, not a test evaluator.

`ChannelStats` reports element counts, positive/negative/near-zero and absolute-response rates, population variance, mean/RMS/range and block response (`mean(abs(A)) > epsilon`). Empty masks produce count=0 and null statistics. Repeated updates combine moments with valid counts, rather than averaging batch percentages. Nonfinite selected values raise a measurement error rather than silently filtering. Threshold equality belongs to near-zero. Negative LeakyReLU responses count as responses.

`weight_statistics(model, explicit_module_paths, epsilon)` measures direct parameters, with axis 0 explicitly treated as output dimension; weight/bias names and shapes are separate. This does not infer dependencies or rank different layers by raw norms. Norms use float64 reductions. Wrap weight records with the same run/context identity when exporting them. The supplied thresholds are a measurement policy, not a criterion proving a channel is useless.

Typical usage (inside the remote container):

```python
from monitoring import LayerSpec, ReadOnlyMonitor
import json

specs = [LayerSpec('backbone.adapter', 'HWC', 1e-6, 'post-adapter', 'pixel')]
# Open an external output file with mode 'x' to avoid overwriting existing work.
with open('/path/outside/checkout/activation.jsonl', 'x') as stream:
    monitor = ReadOnlyMonitor(model, specs, lambda r: stream.write(json.dumps(r) + '\n'))
    with monitor.sample(context, masks={}):
        outputs = model('paviaU', inputs, wavelengths=wavelengths)
# Backward and optimizer.step, if needed, run after leaving the sampling context.
```

Context requires run_id, commit, checkpoint_hash, data_hash, split_hash, dataset, partition, step, sample_id and condition. String identifiers must be nonempty; caller is responsible for their factual provenance. `step` is the nonnegative actual-update index. Records include schema version, module position/layout/unit, call identity, dtype, thresholds, valid counts and model mode. No dataset or class identity is inferred automatically.

The collector detaches tensors and transfers only O(channels) reduced statistics to CPU. It streams records through a caller sink, retaining no raw tensors or event history. The caller must bound file retention / any in-memory sink. At most max_calls (default 64) are allowed per sample. Hooks are removed after forward or sink exceptions; overlapping collectors on the same model are rejected. Fail-fast measurement errors stop that diagnostic forward and must be distinguished from model numerical failures.

Limits: synchronous eager execution only. Explicitly disable adapter checkpoint_chunks for this diagnostic path; checkpoint replay, DDP, torch.compile, AMP, backward-inside-context and automatic probe/RNG management are not supported in this first module. Shared submodules across separately wrapped model objects are not a supported concurrency pattern. No training callback, gradient collection, actual update statistics, fixed-probe runner, Taylor, PCA, weighted Q grouping, TensorBoard or pruning is implemented. MON-2/3/4 remain planned. GPU reductions synchronize and may allocate temporary float64 tensors; overhead must be measured, not assumed below 5%.

Remote regression commands:

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p 'test_*.py' -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 MON_DEVICE=cuda OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_monitoring_statistics.py -v
```

Selected modules must actually be called in a sample; otherwise the sample raises an explicit measurement error. In this backbone the HGNN convolution is sometimes evaluated through a functional path bypassing its module forward; observe the called normalization wrapper instead of silently accepting an unused convolution hook.

