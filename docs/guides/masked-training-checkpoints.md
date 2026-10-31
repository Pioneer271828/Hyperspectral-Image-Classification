# MASK-2: explicit masked-training checkpoints

`compression.masked_checkpoint` adds a separate checkpoint format without changing legacy project save/load or default training. Save at a completed step boundary **after** `optimizer.zero_grad(set_to_none=True)`. No forward graph may be pending. Use the same model/device/dtype/configuration and construct a fresh optimizer with the same canonical parameter groups and order when restoring.

```python
from compression import save_masked_training, load_masked_training

receipt = save_masked_training(
    external_path, active_controller, optimizer,
    identity={'code': commit, 'data': data_hash, 'split': split_hash},
    progress={'step': completed_updates}, scheduler=scheduler, sampler=sampler)
# Build a fresh identical model and optimizer BEFORE attaching a mask.
restored_controller, progress = load_masked_training(
    external_path, fresh_model, fresh_optimizer,
    expected_sha256=receipt['sha256'],
    expected_identity={'code': commit, 'data': data_hash, 'split': split_hash},
    scheduler=fresh_scheduler, sampler=fresh_sampler)
# The returned controller is already active: do not enter it again.
try:
    # Continue the caller's training loop from progress, then zero gradients before saving again.
    pass
finally:
    restored_controller.detach()
```

Saved content: latent model weights and persistent buffers, fixed mask plan, original masked-value recovery snapshots, per-module training modes, model export/training configuration, optimizer state and canonical parameter names, progress/identity, Python/global NumPy/Torch CPU/relevant CUDA RNG, optional StepLR/ReduceLROnPlateau state, and an explicit NumPy PCG64 sampler state. Initial recovery snapshots are preserved across resume; they are not silently reset to checkpoint-time values.

Only SGD/Adam/AdamW are supported. Parameter groups and order are validated by canonical names because parametrization changes model parameter traversal order; a newly constructed optimizer over an already parametrized model must not be assumed equivalent. Exported model config and module descriptors, state shapes/dtypes, devices, file hash and identity are checked. Caller must truthfully bind code/data/splits, coordinate policy, seed and other experimental semantics in identity/config. Custom module behavior not represented by export_config/extra_repr remains the caller's responsibility.

The file is loaded with weights_only=True and an expected SHA256. Save uses a temporary file and atomic no-clobber hard-link publication on the container filesystem; existing output paths are never overwritten. Store checkpoints and receipts outside the checkout. Concurrent training, save, load or mask mutation is unsupported. Root legacy state_dict still rejects active masks; only this explicit export path bypasses its guard temporarily and restores it in finally.

Load requires fresh disposable objects. Identity/config/group/hash failures are checked before loading model weights. Later unexpected restoration errors clean up parametrizations but are **not** a transaction rolling back every object; discard those fresh objects and retain the original running model/checkpoint. Global RNG is restored at the end of a successful load. Stateful DataLoader workers, augmentation generators, arbitrary torch.Generator instances, AMP scaler, gradient accumulation, other samplers/schedulers, cross-device migration and DDP/compile are not supported; do not claim full campaign resume unless every participating state is covered.

This API enables isolated mask experiments, but is not wired into train_spatial or the existing campaign CLI. It does not alter pruning candidates, masks or accuracy. Structural pruning, quantization and full-budget accuracy experiments remain separate tasks.

Remote checks:

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p 'test_*.py' -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 MASK_DEVICE=cuda OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_masked_checkpoint.py -v
```
