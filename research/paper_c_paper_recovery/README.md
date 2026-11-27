# C11: paper evidence recovery

Issue #533. This package reconciles the unfinished historical archive with later
verified critical capsules. It does not resume training, evaluate query data, or
claim a new cross-host backup. Private inputs, weights and exact point identities
remain inside the authorized LXC project. Only aggregate receipts leave it.

The new inventory is bounded to frozen dependency manifests, the three published
development banks, existing capsule indexes and their recorded source suites.
Historical off-host receipts are evidence of their original verification, not a
live audit of that host today. A second directory on the LXC is same-host coverage.

Run on the authorized LXC from a clean checkout at its exact published SHA:

```
python research/paper_c_paper_recovery/recover.py --expected-commit FULL_SHA --out NEW_EXTERNAL_BACKUP
```

The script checks archive identities, restores every indexed critical object,
reconciles unique historical states, and creates an incremental dependency store
only for hashes absent from the inspected existing archives. It never modifies
an existing backup, source artifact or training directory.
