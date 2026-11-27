# Reproduce the public development intake

Only an authorized Linux host executes these programs. Keep all output roots
outside Git. Each command refuses a dirty checkout or a different full commit.
No existing model, checkpoint, dataset bank or shared training interface changes.

1. At `b3963c0022c786418d0630973ff22495e01cc208`, run
   `ovh_metadata.py --expected-commit <SHA> --out <external>/ovh-official-metadata02`.
   Its original-host ZIP receipts pin the actual metadata/member bytes. They do not
   prove identity with a different distribution snapshot.
2. At `e6837e23bcad1924101c9fe743cc915c2c9e7907`, run
   `catalog.py --expected-commit <SHA> --out <external>/ovh-training-catalog01`.
   Only the three official train ZIP directories are requested. The ZIP names
   contain both official train and validation domains; later payload gates allow
   only the frozen32 training domains. No test archive is requested.
3. At `8bc7611a4bb9afe662bea0d648ef69c5bd575d02`, run
   `intake.py --expected-commit <SHA> --metadata <external>/ovh-official-metadata02
   --catalog <external>/ovh-training-catalog01 --out <external>/public-train-intake01`.
   The four-example label-blind plan is saved before the first payload read.
4. At `c7286ae2c3a36ef3e079aa368046f0a670cc3900`, run
   `paired_intake.py --expected-commit <SHA> --intake <external>/public-train-intake01
   --metadata <external>/ovh-official-metadata02 --catalog <external>/ovh-training-catalog01
   --out <external>/paired-train-intake01`. It chooses the smaller paired Sentinel
   sequence in each already selected camera group, using catalog sizes only.
5. Use the final independently pinned auditor revision recorded with the result
   receipt to run `audit_intake.py` and `audit_paired.py`, passing their documented
   input roots and a new external `--out` JSON path. Run
   `python -m unittest discover -s research/paper_c_device_admission -p 'test_*.py' -v`.
   Never overwrite an earlier failed receipt/log or rerun into an existing intake.

For later repeats, an old ETag may no longer exist. The code then rejects a changed
archive. A reconstructed manifest is required before new bytes can be admitted;
an old catalog must not be silently relabeled. There is no claim to a whole-archive
SHA when only bounded byte ranges were obtained. Archives, selected payloads,
metadata coordinates, upstream source text and raw logs stay on authorized Linux.
Only aggregate receipts and project source go into review packages.

The reader exposes four aerial spectral columns and a separate height channel.
Sentinel output retains all acquired dates and raw cloud/snow masks, with explicit
10-band IDs and a fixed40x40 larger context crop. It does not invent wavelengths,
average dates, filter clouds, align10m labels, normalize radiometry, train a model,
or adopt the upstream benchmark protocol. Public train labels are opt-in and are
never changed on disk. A distinct raw1..12 mapping uses ignore=-1 for13..19.

Future classification requires its own domain/year split, shared-class objective,
spatial label aggregation or scene-level target, budget, model seeds and controls.
Two camera acquisition groups can be admitted without calibrated SRFs, while a
causal internal-passband claim requires additional physical binding and paired
interventions. These are separate gates, and successful intake is not efficacy or
independent confirmation.
