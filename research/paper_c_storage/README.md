# C14 lossless storage decision package

Issue #539. Base7c27dd9261bc04d406eaf858f45d93e69bd17c4d. Only this directory changes. Old C11/C13 registrations, C3 code, original states, backups and A's integration remain untouched. This is an optional physical format experiment, not training authorization or a new default.

The [contract review](contracts.json) separates research evidence requirements, C3/C13's explicit per-update recovery contract and physical implementation choices. Every admitted original state, Adam/RNG/buffer/mode/cursor/selection-history byte is retained. No retention frequency, dtype, history or optimizer field is reduced. C11's partial/priority transfer records do not authorize removing original intermediates.

## Existing tools reused

C11/C3 already provide whole-file SHA-CAS, full-state records and uncertain-intent recovery semantics. Whole-file deduplication is measured first; it finds no duplicate in this60-state corpus. The new byte codec uses existing NumPy XOR plus Python stdlib zlib/ZIP/SQLite. SQLite WAL with synchronous=FULL supplies durable atomic publication; it keeps content objects in one database rather than creating millions of tiny object files. No external compression trainer or new dependency is installed. The measured LXC SQLite version is recorded in the environment receipt.

`chunk-zlib` splits original files into64KiB chunks and chooses raw/zlib payloads. `zip-xor` additionally aligns chunks to original uncompressed PyTorch ZIP members and may encode a chunk against its prior same-slot content. Every ZIP header, padding byte, pickle, storage, CRC descriptor and central-directory byte is preserved. Shared-storage aliases, negative zero and NaN bit patterns are not decoded/repacked by the codec. Raw payload is the fallback when compression is worse.

Delta depth is limited to2: a chunk requires at most3 object payload reads. Bases are retained indefinitely; no garbage collection or anchor deletion is performed. Object SHA, compressed-payload SHA, reconstructed full-file SHA and frame/intent/contract/seal bindings are verified. Corruption causes refusal, not approximate recovery. This is integrity detection, not error-correcting replication or protection against an attacker rewriting every trusted receipt.

## Atomic write and restoration

The new optional API is:

```python
with store.transaction(stream, step, original_intent_bytes, frozen_contract_bytes) as tx:
    # Future registered caller performs its existing update after durable intent.
    # Serialize the complete state to an in-memory BytesIO, without a retained
    # raw full-file duplicate, then publish all original byte strings together.
    tx.commit({"state.pt": state_bytes,
               "record.json": record_bytes,
               "intent.json": original_intent_bytes})
```

The one-writer lease spans this unit. Intent commits before the caller update. Objects, complete frame and intent consumption commit in one SQLite transaction. Process death before publication preserves the intent and refuses automatic continuation; after publication the new committed cursor is visible and cannot be retried at the old step. Committed readers observe the old or new frame. Tests inject process exit at all four publication boundaries; they do not claim a physical power-cut test. Filesystem durability still depends on the approved volume honoring fsync/WAL semantics.

Restore writes into a new directory guarded by `.restore-pending`, which the original C3 inspector rejects before any task construction. Complete original files and directories are flushed before publishing readiness. Failed partial materialization is preserved and cannot silently restart from an empty prefix. Ordinary continuation restores only the latest committed state and refuses pending intents; arbitrary earlier prefixes are explicitly forensic. A deliberately separate synthetic replay test restores random old prefixes and counts all repeated optimizer calls anew.

The historical writer still writes full `.pt` files. The importer is not a transparent replacement or automatic migration: it leaves all originals in place and increases current occupation. Real use requires a **separate future runner/registration** to call the new transaction API from its checkpoint boundary, with appropriate raw-state/memory/DB/WAL/backup caps. Old contracts and record timings are never relabeled as new compressed costs.

## Measured corpus and costs

`benchmark.py` admits only the12 already-generated C13 shape cases (3 native class/band shapes x4 methods),5 states each,4 synthetic updates in each original case. It performs no neural calls or new optimizer updates. It measures60 states /135355611 original bytes, plus their120 original record/intent files. All180 original hashes remain unchanged.

| Representation | Closed logical bytes | Saving versus raw states | Total pilot seconds including byte verification |
|---|---:|---:|---:|
| Original states |135355611|0%|reference|
| Whole-state zlib payload only |110907384|18.06%|compression-only diagnostic, no durable-store comparison|
| SQLite chunk-zlib |113041408|16.49%|9.22|
| SQLite ZIP-XOR |104660992|22.68%|15.57|

The SQLite totals include frame/contract/index overhead while the raw reference is state bytes only; original record/intent bytes and the exact denominator remain available in result.json. Filesystem-allocated blocks, temporary WAL peaks, payload I/O and verification costs are separately reported. Allocation on this volume is not a guarantee for another filesystem. The delta store read330025438 payload bytes during encoding and verification versus111413072 for chunk-zlib. Full original-byte verification took2.81s versus1.47s. Both retained all60 checkpoints. This is not a recommendation to enable delta solely for its compression ratio.

Component accounting partitions actual ZIP tensor-storage bytes into model, Adam and RNG. `data.pkl` and ZIP/padding overhead are measured independently. Isolated control/history pickle probes are explicitly nonadditive. Generated metadata-only9-event/K5V5 probes estimate extra selection-history bytes without producing model predictions or changing any source checkpoint.

The36-case/96000-update scenario extrapolates observed marginal growth after initial Adam allocation, with raw metadata growth added conservatively. Chunk-zlib scenarios are203.69-205.76GB; ZIP-XOR189.30-212.43GB per logical store. Encode/commit/checkpoint component projections are about3.22h and6.24h, not real-data runtime measurements or stationary-entropy guarantees. Four early synthetic updates cannot establish a long-run capacity bound. Two retained independent copies would double the scenario before journal, workspace and destination-specific margins.

C13's897169554672-byte reserve remains an unchanged original-plan number. It cannot simply be substituted by a measured compression scenario; a new physical format needs its own proven DB/WAL/memory/backup caps. Current free space is recorded at each measurement and remains far below all scenarios. No destination has been newly approved, no old asset is deleted or moved, and no large aggregate archive is created.

## Reproduce on the authorized LXC

Use a clean exact C14 source SHA, the existing environment and fresh checkout-external outputs:

```sh
python -m research.paper_c_storage.test_store -v
python -m research.paper_c_storage.benchmark --expected-commit "$SOURCE" \
  --source /home/lbx/hsi-pioneer271828/runs/paper-c-spatial-baseline-20261127-01/profile03 \
  --out "$NEW_RUN/measurement"
python -m research.paper_c_storage.recovery --expected-commit "$SOURCE" --out "$NEW_RUN/recovery"
python -m research.paper_c_storage.cli pack --expected-commit "$SOURCE" \
  --source "$EXISTING_C13_SYNTHETIC_CASE" --stream fixture \
  --store "$NEW_RUN/optional-store" --layout zip-xor
python -m research.paper_c_storage.cli verify --expected-commit "$SOURCE" \
  --store "$NEW_RUN/optional-store" --stream fixture
python -m research.paper_c_storage.cli restore --expected-commit "$SOURCE" \
  --store "$NEW_RUN/optional-store" --stream fixture --out "$NEW_RUN/restored"
```

The recovery experiment used committed boundaries3/1/1/1 in the four seed64 methods. Ten new synthetic optimizer updates reproduced all10 golden full states;30 original prefix state/record/intent files were restored byte-identically. It makes no real-data or independent-spatial claim. Performance, conversion and replay evidence have distinct producer SHAs; receipts identify them. Repository-wide stale-status failure remains separate from targeted storage acceptance.
