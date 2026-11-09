# M1-J2 execution and audit scripts

These are exact archived copies of the external scripts used in the authorized LXC run. They are not additional model modules. Training remains frozen at `9942c0f31949bed746f3038e26b53b51d0dddc26` (`experiment-m1j2-full-v1`); later archive/documentation commits are not the training SHA.

The scripts use the recorded dedicated root `/home/lbx/hsi-pioneer271828`, immutable training checkout `checkouts/m1j2-issue127`, external `runs/` and `transfer-records/`. Their source/hash records are included in launch/evaluation/independent receipts. Keep the frozen training checkout separate from this later archive. Do not copy these files into that checkout during execution, because its full source identity is intentionally frozen.

- `m1j2_supervise.py`: creates a new complete six-case launch bank, two GPU queues, bounded resource checks, per-case train/validation review and immutable completed resume. It records the preregistered zero-update finalizer separately if original finalization fails.
- `m1j2_eval_preflight.py`: CPU/GPU12 condition/dataset checks each, including calibrated-clone parameter/source immutability.
- `m1j2_evaluate.py`: only runs after all six cases pass review; evaluates the fixed four-condition bank on frozen test indices, writes compact prediction/true-class-probability packets and checks protected artifacts. Uses observed-condition label-free BN calibration with fixed full-spectrum-derived graphs.
- `m1j2_qa.py`: independent raw-label/packet recomputation of216 head metrics,2880 class recalls,72 complementarity cases and the preregistered paired/AA/small-class gates. Counts are expectations until the final receipt passes.
- `m1j2_watch.py`: waits for this one campaign, executes evaluation and QA, then exits. It sends no messages and starts no subsequent module.

Outputs are exclusive new directories, so these scripts are not idempotent against an existing run. Reproduction in another location requires explicitly adapting paths and recording the new script hashes; do not overwrite the original results. No credentials, labels, weights or per-pixel artifacts are included here. See the experiment preregistration and final report for evidence versus pending work.
