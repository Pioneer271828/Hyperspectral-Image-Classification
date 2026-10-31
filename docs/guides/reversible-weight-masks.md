# MASK-1: explicit reversible weight masks

This opt-in experimental API is independent of default training. It implements fixed elementwise masks for explicitly named `nn.Linear` and `nn.Conv2d` weight parameters. It does not choose candidates, delete neurons/channels, infer dependencies, change losses, or provide acceleration or accuracy guarantees.

```python
from compression import MaskedWeights
import torch

layer = model.backbone.CNN_Branch[0].point_conv
keep = torch.ones_like(layer.weight, dtype=torch.bool)
# Example only, not a recommended pruning decision: mask this output filter's weights.
keep[0] = False
with MaskedWeights(model, {'backbone.CNN_Branch.0.point_conv.weight': keep},
                   recovery='restore_masked_values'):
    # Existing optimizer references remain valid. Use ordinary zero_grad/forward/backward/step.
    # Accessing layer.weight now returns its effective masked value.
    pass
```

A registered PyTorch parametrization returns zero for masked entries and routes zero task-loss gradients to the corresponding underlying weights. The original Parameter object is preserved; optimizers created before attachment still reference it. Masks are copied, binary bool tensors with the exact weight shape; externally changing the supplied mask does not change an active experiment. Fixed masks remain active until detach/context exit. Configure device/dtype before attachment and do not mutate model structure/device, parametrizations or mask buffers while active. Attach between completed optimizer steps, after discarding any previous forward graph; never attach/remove between a forward and its backward.

Bias is not masked. Masking a filter's weights alone does not prove its neuron is inactive: bias, normalization, residual and other paths can still produce outputs. Other module types, biases, tied parameters and existing parametrizations are rejected in this first version. Entire plans are validated before attaching; overlapping masks are rejected and exceptions remove registered parametrizations. No hook-based weight replacement is used.

## Optimizer state and unmasking

Zero task gradients do not guarantee zero optimizer updates. Existing Adam/AdamW moment estimates, optimizer weight decay, or an explicit regularizer on latent parameters can change the underlying masked entries. Their *effective* values remain zero. No optimizer state is reset or edited by the API.

The mandatory recovery policy defines what happens on detach:

- `keep_latent`: expose the latest underlying weights, including any drift while masked.
- `restore_masked_values`: restore only masked positions from attach time, while preserving learned unmasked values and all other parameter updates. Optimizer moments and step counters are retained, so the next unmasked update may immediately move restored values. This is not an optimizer or whole-training rollback.

For pure sensitivity experiments, evaluate without optimizer updates; removing the mask then restores the original model weights. For a true training rollback, restore a separately saved full model/optimizer/scheduler/RNG checkpoint outside this API. Do not infer that masking is training-neutral; that is a hypothesis tested by a dense all-ones control, whereas nontrivial masks intentionally change the computation.

## Mask plans and checkpoints

`mask_plan()` exports only a versioned dictionary of CPU boolean masks. Save it with `torch.save` outside the checkout and load with `torch.load(..., weights_only=True)`. `MaskedWeights.from_plan(model, plan, recovery=...)` recreates the same fixed mask plan; the caller must separately provide the matching model and experiment provenance. A plan is not weights, optimizer state, or a resumable training checkpoint.

While masks are active, the controller rejects root model `state_dict()`/legacy checkpoint serialization rather than silently creating files incompatible with the existing loader. After detach, the original names and Parameter identities are restored and ordinary project checkpoint save/load works. Do not use this API around the existing full training entry point, which saves checkpoints, until a separate explicit masked-checkpoint integration module exists. Saving individual submodules directly is not guarded and is not a supported persistence path. No legacy checkpoint/config interfaces have been modified.

## Validation and limits

Tests cover CPU/GPU all-ones Adam equivalence, warmed AdamW latent drift with persistent effective zeros, zero masked task gradients, both unmasking policies, bias behavior, exact mask-plan roundtrip, repeated/exception cleanup and invalid/shared parameter rejection. The actual small VCA/CNN/HGCN model runs three synthetic Adam steps with a CNN weight mask, then checks ordinary save/load after detach. This is engineering acceptance, not a pruning accuracy experiment. Explicit mask selection uses no real labels or test data.

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p 'test_*.py' -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 MASK_DEVICE=cuda OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p test_weight_masks.py -v
```

Single-process eager execution only. Dynamic mask schedules, sparse kernels, physical structural pruning, AMP/DDP/compile, masked checkpoint resume, optimizer-state rollback and training callbacks remain outside scope. Sparse value count alone is not a speed measurement. Candidate selection, equal-budget dense recovery controls and at least three paired seeds must be planned separately before claiming accuracy improvements or cross-sensor benefits.

MASK-2 now provides a separate [explicit masked-training checkpoint API](masked-training-checkpoints.md). The legacy-save restrictions above remain; the default training CLI is not integrated.
