# MON-3A: frozen conditional probe evidence

Issue #119 adds `monitoring.probes.FrozenProbe` and `class_masks`. This is an observation-contract API, not a general model runner or channel intervention system. The caller must obtain activations from the declared checkpoint in an isolated process. No training defaults, hooks, model structure, optimizer or pruning decisions are installed by this module.

Freeze a plan **before** its first observation: full source/checkpoint/data/split context, allowed domain list, train or validation partition, named input tensors, threshold provenance, `LayerSpec` and call index, and boolean condition masks. Input and mask contents are hashed, masks are cloned, and the JSON manifest has a stable identity. `manifest` returns a defensive copy. Calling `observe` with changed inputs/context, missing calls, changed thresholds/units or corrupted internal masks fails.

```python
from monitoring import LayerSpec
from monitoring.probes import FrozenProbe, class_masks

# All values are explicit; obtain activations in a separate diagnostic process.
masks = class_masks(labels, permitted_training_pixels, class_ids, unit="pixel")
probe = FrozenProbe(
    context, {"cube": cube, "labels": labels},
    {"cnn-0": (LayerSpec("backbone.CNN_Branch.0", "NCHW", 1e-6,
                        "post-block", "pixel"), 0)},
    {"cnn-0": {name: mask.unsqueeze(0) for name, mask in masks.items()}},
    threshold_source="frozen-allowed-training-calibration-identity",
    allowed_datasets=["source_dataset"],
)
save_plan(probe.manifest)
report = probe.observe({"cube": cube, "labels": labels},
                       {"cnn-0": captured_output}, context=context)
```

Each bucket uses its own valid element count. Positive, negative and near-zero rates use the fixed threshold; negative LeakyReLU responses remain responses. Empty buckets retain count zero and null statistics. Buckets may overlap and must not be added as if they partitioned the population. No averaging across plans, datasets, layers or checkpoints is performed. The call index is explicit, so repeated calls to a shared layer cannot silently merge.

`class_masks` accepts positive class IDs and excludes unlabelled zero. It requires explicit permitted pixels and pixel units; it rejects using pixel class labels as region IDs. Graph observations may use separately declared region masks or equal-region groups. Q-weighted class mapping and soft region memberships are not implemented. VCA's latent channels may still be measured over pixel positions; the statistical unit denotes the counted positions, not the number of input spectral bands.

The plan verifies supplied identities and input equality, not the truth of caller declarations or the origin of an arbitrary activation tensor. Accordingly, `model_execution_verified` is false. A separate runner acceptance must prove checkpoint binding, model/mode/gradient/RNG isolation and hook cleanup. Strict held-out-target exclusion is defined by the declared allowed domains; the API cannot infer a user's experimental design. Test partitions are rejected.

Threshold calibration is not automatic. Declare its permitted training source and freeze it before collection; the API does not rescale a threshold per batch or infer a universal dead-channel boundary. Do not prune based on low response. Single/group interventions, stable-window interpretation, small-class protection decisions, graph mapping and MON-4 linkage remain later acceptance stages.

Reference: the project's MON plan, user-provided RM report sections8/8.6 and accepted MON-0/1 statistics. This is independent contract/aggregation code using existing `ChannelStats` and tensor hashes; no external research implementation is copied. External architecture reuse is not claimed.

Remote focused command: `MON3_DEVICE=cuda OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_monitoring_probes.py -v`; use `MON3_DEVICE=cpu` for CPU. Acceptance status and real fixed probes are reported separately. No speed, accuracy or complete MON-3 claim follows from the contract tests.
