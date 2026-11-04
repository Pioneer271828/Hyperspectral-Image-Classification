# M2-J2 frozen reproduction

Training source: `d945daa15ad8752cceaf3898ebed6fca5d8415d0`, branch `codex/issue-139-m2-full`. Keep the producer checkout frozen. The archived evaluation scripts bind the original LXC paths; review explicit constants before reproducing in a new isolated location. Never reuse an existing output directory or change a frozen training checkout to run an archived evaluator.

Original external execution (all 9 per-training reviews had passed):

```bash
CUDA_VISIBLE_DEVICES=2 /home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
 /home/lbx/hsi-pioneer271828/transfer-records/m4s2_deterministic.py \
 /home/lbx/hsi-pioneer271828/transfer-records/m2j2_evaluate.py
/home/lbx/hsi-pioneer271828/envs/hsi/bin/python \
 /home/lbx/hsi-pioneer271828/transfer-records/m2j2_verify.py
```

`deterministic.py`, `evaluate.py` and `verify.py` archive those exact external scripts. The generic wrapper changes only verification context, not optimizer training. Evaluation was preceded by all-case train/validation numerical verification; no selection by test outcomes. The source snapshot is the original producer, not the later documentation commit. Prediction/label packets, weights, logs and source/input manifests remain outside Git under `runs/m2j2-issue139-*`. The checked-in JSON contains aggregate metrics only. See the full report for failed scientific gates and metadata/effective-capacity limits.
