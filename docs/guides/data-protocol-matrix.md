# Data protocols for the three research axes

2026-11-03; [Issue #95](https://github.com/Pioneer271828/Hyperspectral-Image-Classification/issues/95). This is a protocol decision guide, not a new executable configuration or a completed benchmark. The [evidence review](../../research/reports/三主线数据划分文献核查与推荐协议_20261103.md) records original counts, literature/code revisions and access limitations. The [current checklist](../../research/当前搭建状态与审查整改清单_20261102.md) owns implementation order.

## Accuracy optimization: preserve the original small training set

The recovered joint PaviaU/Xuzhou/LongKou experiment used `samples_type=ratio`, `train_ratio=0.01`, `val_ratio=0.005`: per annotated foreground class, take `ceil(N_class*0.01)` training labels and `ceil(N_class*0.005)` validation labels; test on the remainder. Unlabelled pixels are outside that denominator. The `train_num=30` and `val_num=10` flags in the old command do not control ratio mode. [split_data.py](../../loadData/split_data.py) and the first-hand logs agree.

| Dataset | Training | Validation | Test | Total annotated |
|---|---:|---:|---:|---:|
| PaviaU | 432 | 219 | 42,125 | 42,776 |
| Xuzhou | 692 | 348 | 67,837 | 68,877 |
| LongKou / WHUL | 2,051 | 1,028 | 201,463 | 204,542 |

**ACC-1P is the primary matched accuracy benchmark:** jointly train on all three datasets with shared VCA/backbone and dataset-specific heads/DomainBN; retain the 1%/0.5% label rule, record actual counts and freeze indices/hashes. Raising the training percentage cannot establish an architectural gain over this benchmark. Optional label-budget curves are separate comparisons with paired controls at every budget. Validation labels also count toward annotation cost.

Match data versions, split identities, graph, normalization, attention, training budget and selection before attributing a gain. Full-scene normalization/graphs with random labelled pixels constitute a transductive scene protocol; the historical train-label-guided LDA-SLIC is part of that declared baseline. A spatially isolated experiment answers an additional question and needs its own paired baseline. Historical logs do not contain sufficient source/data/index provenance to guarantee bitwise reconstruction: BASE-97 must publish the new frozen manifest and disclose this gap. Do not recover old cache bugs to match scores.

## Cross-sensor tasks: record allowed access before percentages

| Protocol family | Target training access | Model selection | Proposed project use |
|---|---|---|---|
| Joint seen-dataset supervision | All three datasets have released training labels | Their validation labels | ACC-1P; does not establish unseen-sensor generalization |
| Few-shot transfer | K labelled target pixels/class plus explicitly allowed unlabelled context | Source validation / source pseudo-target development; no target-test selection | Nested K=1/5/10. Existing frozen G3 primary remains K=10; changes require a new protocol version |
| Unsupervised domain adaptation (UDA) | No target labels; an explicitly declared unlabelled target pool is allowed | Source-side selection; other unsupervised rules must be frozen | Separate transductive full-target and spatially isolated target-pool variants |
| Domain generalization (DG) | No target labels or inputs during fitting, tuning or calibration | Source-only development, ideally leaving source domains out | Leave one sensor/family out; verify metadata and shared class semantics first |

Representative established practices differ. [DCFSL author code](https://github.com/Li-ZK/DCFSL-2021/blob/ec92b1cca5ce3779f0cec4435daa714f902e07e3/DCFSL-UP.py) defaults to five target labels/class. [AGT, IV-G/H](https://arxiv.org/html/2309.12865v4) studies 50/class transfer and 25/50/75 target-label sensitivity. [DANN](https://www.jmlr.org/papers/v17/15-239.html) defines labelled-source/unlabelled-target adaptation. [DomainBed](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/scripts/train.py) defaults to a 20% holdout within each environment; its [IID selection method](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py) uses training-domain validation. These examples support different task families, not a universal split or a measured majority of papers.

Project choices requiring a new frozen task:

- Prefer K=1/5/10 for a small-label curve, with 25/50 only as separately named higher-budget comparisons. Keep existing G3-v1 unchanged. If a target validation set is added, report its labels in addition to K; K training + V validation is a K+V annotation budget.
- Source train/validation 80/20 of the explicitly allowed source pool is a starting proposal inspired by DomainBed, not an automatic change or an HSI-wide rule. Specify source label counts and spatial context. Under severe source-label constraints use a separately budgeted source validation set. Freeze adaptation steps from source-side pseudo-target development before target testing.
- Source checkpoints for held-out sensors must exclude those sensors from supervised pretraining, normalization, graph fitting and calibration. A checkpoint jointly trained on all three datasets cannot be relabelled as an unseen-sensor source model.
- Validate sensor identity/family separately from geography. Same-sensor scene transfer does not prove cross-sensor transfer. Equal numeric class IDs or equal class counts do not establish shared semantics. Few-shot transfer can create a fresh target head; zero-label closed-set UDA/DG needs an explicit semantic mapping or another declared task.
- DCFSL's pinned entry evaluates query/test accuracy and uses it to save the best model. Borrow the label-budget idea, **retain this project's independent model selection**; do not import that checkpoint-selection logic.

## Pruning and quantization: inherit the dense baseline protocol

Start compression on the accepted ACC-1P checkpoint using exactly its train/validation/test identities. For cross-sensor compression, inherit that particular task's source/target access instead. Compression has no independent universal 80/20 or 10% requirement.

| Operation | Permitted fitting data | Required matched control |
|---|---|---|
| Activity/weight/gradient diagnostics and pruning candidate selection | Allowed training inputs/labels; class/domain protection within this allowance | Dense model and no-op mask; separate candidate selection from final evaluation |
| Pruning recovery / QAT | Original allowed training labels; validation for ratio/bitwidth/checkpoint selection | Dense model with the same additional optimization budget; disclose search and recovery costs |
| PTQ | A frozen subset of allowed training inputs; no test labels or target-DG inputs | Same floating-point checkpoint, calibration identities/counts and selection budget |
| Accuracy and speed | Frozen selected models on the declared test and identical runtime/input conditions | Report paired accuracy, per-class/domain protection, storage, memory and measured latency |

[BRECQ](https://arxiv.org/html/2102.05426) uses 1,024 training images for calibration; its [official entry](https://github.com/yhhhli/BRECQ/blob/e455d62e93c70351961f8991c913b59435bd165f/main_imagenet.py) collects inputs from the training loader. This is a model-quantization example, not a prescription to sample 1,024 HSI pixels as independent images.

HSI calibration must define its unit and context. If observers consume all three full-scene inputs, declare **three-scene transductive calibration**, including unlabelled query context; do not describe it as spatially isolated train-only calibration. Masking output labels does not restrict observer inputs. A strict block protocol must limit both observer inputs and graph/context fitting to allowed training blocks. A proposed first pilot is `min(32, number_of_allowed_training_blocks)` preselected blocks, with a fixed seed and hashes; 32 is a project choice, not a literature standard. State counts of scenes, blocks, regions and contributing pixels as applicable. DG permits source calibration only; target calibration changes the task.

Begin with an independently assessed INT8 candidate, then consider lower/mixed precision if justified. Fake-quantized floating-point operations, zero masks and theoretical FLOPs do not establish actual speed gains. Accuracy gain, accuracy preservation, and acceleration each need evidence.

## Implementation gates and repeat policy

1. BASE-97: independently synchronize LongKou; verify data identities and class coverage; freeze ACC-1P split manifests; confirm three datasets enter a short joint training/checkpoint loop. Register the full-budget batch separately.
2. Establish the matched joint accuracy baseline, then assess one mechanism at a time. Keep strict spatial / few-shot / UDA / DG results in separately named tables.
3. Before a new cross-sensor protocol, freeze sensor metadata, semantics, source/target access, annotation costs, support/split seeds and selection. Never choose a split by target-test score.
4. After MON-2/3 and dense-baseline acceptance, independently freeze compression candidates, calibration/context identities, equal-budget recovery controls and speed conditions.
5. Use the existing [seed policy](seed-policy.md): paired seeds; 3/5/10 repetitions according to evidence stage; distinguish training randomness from support/split repetitions. This is our policy, not a universal paper convention. Follow the [post-training review](post-training-review.md) after every training.

This documentation module reads papers/code but copies no external implementation. The review records permission/access gaps; implementation still follows the [reference-code reuse guide](reference-code-reuse.md). No new protocol, default or training run is activated by this document.
