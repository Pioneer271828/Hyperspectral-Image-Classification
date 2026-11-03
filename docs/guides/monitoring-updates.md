# MON-2: sampled gradients and actual optimizer updates

Issue #117 implements a separate opt-in observer. It does not install hooks until `sample` is entered and does not change the backbone, trainer, configuration, checkpoint format or optimizer defaults. The frozen M4-J2 campaign does not use this observer. Engineering tests and real-data acceptance remain pending until a report is linked here.

```python
from monitoring import LayerSpec
from monitoring.updates import ParameterSpec, UpdateMonitor

monitor = UpdateMonitor(
    model, optimizer,
    [ParameterSpec("backbone.region_residual.projection.weight", 0, "latent_channel")],
    sink=write_record,
)
# context supplies the same source/data/split/step identity fields as MON-0/1.
with monitor.sample(context, contract="fp32_single_backward"):
    existing_training_step()  # includes backward and exactly one optimizer.step()
```

Select existing full parameter names and an explicit channel axis. Bias and normalization parameters remain separately named; a channel is not automatically a neuron or a pruning dependency group. Unregistered parameters, frozen parameters, absent gradients, exact zero gradients and zero actual updates have distinct fields. AdamW can update parameters with zero data gradients through decay; gradient RMS is not an update estimate. Relative update is `||after-before||2 / (||before||2 + epsilon)` per channel, using a detached cloned snapshot immediately before the optimizer step. Changes outside that one step are excluded.

Optional `activations=[LayerSpec(...)]` records gradients with respect to the selected output tensor, with explicit NC/HWC/NCHW layout, selector, call identity and pixel/region/latent unit. It preserves negative gradients, masks only the measurement, uses actual element counts and reports empty masks as null. Detached outputs and outputs disconnected from loss are distinct. A mask is diagnostic; it never changes the loss or network input.

## Supported boundary

- Eager SGD, Adam or AdamW; one ordinary unscaled backward and optimizer step; dense FP32/FP64 measurements. The contract name `fp32_single_backward` also permits FP64 control tests.
- AMP, accumulation, closure-based steps and captured/differentiable optimizers are unsupported. Parameter-only sampling permits checkpoint recomputation because gradients are read after backward, immediately before the optimizer step. Optional activation-gradient hooks require a noncheckpointed model; that combination is rejected for checkpointed models. The explicit contract is a caller declaration: unannounced loss scaling or accumulation cannot reliably be detected from gradient values alone. Exposed autocast/scaler state and repeated backward on a sampled output are rejected.
- Existing optimizer step hooks are rejected to avoid ambiguous before/after ordering. External global hooks and mutation outside the controlled training step are outside the contract. Do not mutate model structure or parameter storage while sampled.
- Exactly one optimizer step must complete. A second attempted step is rejected before its update. Errors do not roll back an already completed first step. Sink failure is reported after training has updated; callers must not blindly replay that step.
- Context must identify a permitted training partition; test labels are excluded. Sampling is selected by the caller using a fixed step schedule, without consuming RNG. `enabled=False` installs no hooks and performs no measurement validation.
- Snapshot and mask storage are each bounded (16 MiB default), and activation forward calls are bounded (32 default). Only selected parameter copies and channel statistics are retained during the sample; no historical tensors are stored by the monitor. Temporary reductions and the normal training graph add memory beyond those snapshot bounds.
- Hooks are removed on body, validation, optimizer and sink errors. Sampling is synchronous; overlapping monitors on one model are rejected. A sink must store/stream small JSON-safe records without retaining model tensors.

The observer supports diagnosis; it does not classify dead units, rank importance, prune, promise an accuracy gain or meet a speed budget by construction. MON-3 fixed probes and conditional class/sensor protection remain separate work. Measure overhead at the intended frequency; do not equate shared-GPU timing with a controlled performance guarantee.

## Reference and reuse record

Design input: user-provided RM report sections8/8.6 and the [monitoring plan](../../research/统一训练监测搭建与训练接入计划_20261031.md). This is independent instrumentation using existing project layout/context conventions, not a new published neural architecture. No third-party source was copied, vendored or executed; no external source SHA/license scope is claimed.

Official API documentation checked on2026-11-04: [optimizer pre hook](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.register_step_pre_hook.html), [optimizer post hook](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.register_step_post_hook.html), [tensor backward hook](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.register_hook.html). Both optimizer and tensor hooks return `None`, preserving arguments and gradients. API documentation is a reference; acceptance uses the installed remote runtime without upgrading it.

Remote focused command: `MON2_DEVICE=cuda PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_monitoring_updates.py -v`. CPU uses `MON2_DEVICE=cpu`. Expected checks include known SGD arithmetic, AdamW decay, missing/frozen/zero states, masked activation gradients, exact optimizer/RNG equality, bounded repeated sampling and failure cleanup. Real joint training and measured overhead require separate acceptance records.
