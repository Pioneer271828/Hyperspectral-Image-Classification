# M4-G4A acceptance drivers

`real.py` and `verify.py` archive the external drivers actually used after production source `77e2a3f45ad5c7e8e319221da69d61af0f2ecf80` was frozen. They use the dedicated LXC paths and immutable output names in the acceptance report. Read and adjust paths/output ownership deliberately before any new run; completed artifacts must not be overwritten. No local Windows Python execution.

The production checkout remained frozen and clean throughout the original runs. These drivers were not present in that checkout; their later archival commit is not a substitute tested production SHA. Real receipt:27cases/288updates. Independent receipt:216headmetrics/2880recalls,526trainingfiles/22inputs unchanged,zero query access and analysis updates. Short source checkpoints are engineering-only and cannot serve as full-source scientific pretraining.
