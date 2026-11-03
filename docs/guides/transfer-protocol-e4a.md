# E4a-1: frozen cross-sensor few-shot protocol

This module prepares a **G3 few-shot transfer** task, not zero-shot recognition or G2 unseen-family closed-set generalization. First task: PaviaU (ROSIS) → Xuzhou (airborne HYSPEX according to the bundled Xuzhou_Readme.txt). Target original classes remain unchanged and use a new target head. Equal numeric class IDs in the two datasets are not a semantic mapping. Sensor evidence is task-local and hashed; the global Xuzhou registry remains unknown pending broader metadata review. Sensor, geography, class taxonomy and acquisition conditions all change together, so this pair cannot isolate a causal sensor effect.

Frozen configuration: `configs/protocols/paviau_to_xuzhou_g3.json`.

- Label-blind spatial grid: tile32/gap2/split_seed2026, independently generated per scene. Each block is a separate context. Guard pixels are unused; no adaptive redraw if support is insufficient.
- Support seed2027, nested target budgets1/5/10 per class; primary10/class gives90 training labels for9 target classes. **Target validation labels=0**, total target labels used for optimization/selection=90. Other target train-region labels are unavailable to the trainer.
- The support builder uses an oracle's train-pool class labels for stratified benchmark sampling;90 is the number released for model optimization, not a claim that only90 labels were inspected to acquire a balanced annotation set. Protocol construction/identity inspection is separated from training and can access annotation metadata. Neither support selection nor training uses target val/test label values.
- Source pretrain10000 updates; select source checkpoint using source validation NLL. Source pretraining sees no target inputs/normalization/labels. Target adaptation1000 updates and final-step checkpoint fixed in advance;1000 is a pilot compute budget, not a target-validated optimum or an already source-calibrated value. Do not select a target checkpoint by target validation/test metrics.
- Proposed controls: target-from-scratch full model, source-pretrained backbone plus target-head-only training, source-pretrained full fine-tuning. Same support, target update budget, target head initialization and allowed input conditions. Trainable parameter counts differ and must be reported. Source pretraining costs are additional and reported separately; target-update equality does not mean total compute equality.
- Use separate source/target train-region per-band min/max statistics. Target train-region **unlabelled inputs** may fit normalization and enter learned label-free graphs, including pixels without support labels. No target val/test block enters fitting or adaptation. GroupNorm inference on a block is not global target-statistic refitting.
- Full delivered bands first, explicit M1 metadata and original order. No new spectral augmentation, M4 structure, pruning or quantization is combined into this baseline protocol.

## APIs and access boundaries

`make_support(labels, plan, classes, per_class, seed)` reads only the target train-pool values. It selects per class without replacement, preserves deterministic nested prefixes across budgets, and records indices/labels/seed/counts/plan and support hashes. `validate_support` regenerates the train-only support for audit.

`target_labels_for_tile(..., phase='adapt')` returns zeros except the selected support labels, and rejects val/test tiles. It does not read unselected training labels. Target validation access is rejected. `phase='final_test'` requires a test tile and an explicit64-character frozen checkpoint SHA256; the runner must independently ensure selection is finished, bind that hash to its artifact, and log a single final evaluation. This guard prevents accidental phase mixing, not adversarial access or repeated evaluation by a caller.

`freeze_task` records exact config, source/target dataset identities, plan hashes, support hash/budget and sensor evidence. It enforces this first-task schema. New datasets, shared semantic mappings or different target validation conditions require a separate explicit protocol version. No model code, training entry point, dataset registry or old checkpoint behavior is changed.

## Remote preparation and acceptance

```bash
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python -m unittest discover -s tests -p 'test_*.py' -v
OMP_NUM_THREADS=1 /home/rmvision/venvs/hsi-baseline/bin/python scripts/prepare_transfer_protocol.py \
  --expected-commit FULL_COMMIT_SHA \
  --data-root /home/train/hsi-pioneer271828/datasets/current \
  --out-dir /home/train/hsi-pioneer271828/runs/UNUSED_G3_PROTOCOL_DIRECTORY
```

The script checks actual dataset identity and sensor readme, creates immutable plans/support manifests/task.json externally, validates nested budgets and exact90-label release, and verifies that replacing held-out label values does not change support. No model is trained/scored and no final target test labels are released by the preparer. Existing inspect_dataset reads full label files for integrity and provenance; this is disclosed protocol audit, not source pretraining or target model selection. Insufficient train support causes failure and preserves the failed manifest; no seed search or silent class removal is permitted.

Next module: a separate source-pretrain / target-adaptation runner implementing these contracts, first with synthetic short steps and then a declared real-data pilot. At least three paired model/support seed repetitions are needed for key accuracy conclusions. Strict shared-semantic G2, broader sensor-family evidence, classifier transfer, M1 training augmentation and E1 remain separate work. Compression development stays frozen.
