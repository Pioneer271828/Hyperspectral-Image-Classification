# C13: one external method, common spatial footprint

Issue #537. C owns only this directory. This package reuses the completed C1/C3 and A PaviaU DSNet evidence and D's source review. No A WP09, project defaults, shared CLI, backbone, checkpoint format or canonical status files are modified. Real fitting and new query scoring remain paused.

DSNet is the **single** licensed external fallback. The closer CEGCN/DCG-Net fixed trees still have unresolved code grants in the existing review. DSNet is a pixel/subpixel decomposition comparator, not a superpixel residual equivalent. Read [provenance](source-review.json), [future proposal](future-comparison.json), and [paper-ready wording](paper-ready.md). The prior adapter and author constructor/forward are reused, not another copy of their trainer.

## What this fills

Existing native bank models observe a whole released tile; DSNet sees7x7 center patches. That completed development recipe comparison must not be ranked against historical S2's joint, transductive, full-scene1%/.5% experiment. `history.py` reconstructs frozen A confusion matrices and verifies S2's actual head/precision denominators without inference. It records **zero eligible S2/external matched endpoints**, rather than inventing point pairing from confusion marginals.

The new isolated **normal entry** is `python -m research.paper_c_spatial_baseline.run {register,run,audit}`. All arms receive byte-identical normalized7x7 spectral patches, center order and labels. Native arms are residual off, x_residual and the existing capacity_control; the fourth arm loads DSNet. Native uses the pinned C3 GroupNorm64/learned-graph architecture. Patch graphs explicitly use scale10. This is a **new local-context protocol**, not historical S2 reproduction. Capacity control has64 more trainable parameters than x_residual; report actual counts rather than calling them exactly matched. Native additionally uses fixed local coordinates and wavelength metadata. Native three-head/graph losses versus DSNet CE+SAD remain stated method differences.

Only the central native output at flat index24 is scored; its fused head selects the first minimum development NLL. DSNet's third return is classification logits; its first two returns are reconstruction outputs, never extra class heads. Original unclamped abundance/SAD behavior remains unchanged, including numerical failure. Model seeds are initialization repeats, not spatial repeats.

## Inputs and admission

`fixture.py` generates a small synthetic example. `--future-shape` generates the full9/9/22-class,103/436/270-band schema with400 synthetic centers, explicitly tagged `generated_synthetic`. Its future-shaped registration is planning evidence and **cannot execute fitting**.

For future real registration, replace the generated observations and provenance with a separately released, reviewed manifest: `kind=future_development`, `origin=real_observation`, `query_access=false`, `confirmation_access=false`; the same scene schemas and K5/V5 counts are enforced. Each role has its own hash-bound NPZ, containing only float32 `spectra[N,7,7,B]` and int64 `labels[N]` in native zero-based class order. Samples bind array index, physical center,7x7 half-open footprint and original parent. No raw-image, label-raster, query or confirmation reader exists here.

Every original parent has a common spatial grid, acquisition/original-parent identity, source SHA and externally verified footprint. Cross-role parent aliases are rejected. Both full parent footprints are expanded by `ceil(effective_PSF_radius_pixels + geolocation_error_pixels)` and must remain disjoint. Unknown real uncertainty is rejected. An external receipt must hash the complete scene contracts and record its reviewer and exact review scope. The program cannot establish that asserted groups are true fields, that error bounds are correct or that acquisitions are independent. Those are **external dependencies**, not facts created by passing this validator.

Normalization fits per-band minima/maxima on training patches only; both arms apply the same un-clamped transform. Constant training bands fail. K5/V5 here is a new request for400 unique centers, not permission to reuse prior bank labels as untouched confirmation. Confirmation is not part of this executable development stage and remains a separately unfilled scientific requirement.

## LXC-only commands

Run in a clean frozen C13 checkout using the existing LXC Python. Set `SOURCE` to its full actual commit, `NATIVE` to the clean395caef checkout, `AUTHOR` to either the clean e910 author checkout or the authenticated four-file source packet, and `OUT` to a fresh checkout-external directory. All outputs, including generated fixtures, stay outside checkout.

```sh
python -m research.paper_c_spatial_baseline.fixture --out "$OUT/roles"
MANIFEST="$OUT/roles/manifest.json"
MH=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
python -m research.paper_c_spatial_baseline.run register \
  --expected-commit "$SOURCE" --native "$NATIVE" --author "$AUTHOR" \
  --manifest "$MANIFEST" --manifest-sha256 "$MH" --synthetic \
  --max-updates 48 --max-state-bytes 1000000000 --out "$OUT/registration.json"
RH=$(sha256sum "$OUT/registration.json" | cut -d' ' -f1)
python -m research.paper_c_spatial_baseline.run run \
  --expected-commit "$SOURCE" --native "$NATIVE" --author "$AUTHOR" \
  --manifest "$MANIFEST" --manifest-sha256 "$MH" --synthetic \
  --registration "$OUT/registration.json" --registration-sha256 "$RH" \
  --budget-release-sha256 "$RH" --out "$OUT/execution"
python -m research.paper_c_spatial_baseline.run audit \
  --expected-commit "$SOURCE" --registration "$OUT/registration.json" \
  --execution-root "$OUT/execution" --out "$OUT/independent-audit.json"
```

`--resume` requires unchanged registration and existing outputs. An uncertain pending optimizer step is preserved and blocks automatic replay. Complete reentry verifies saved aggregates without model construction, author dummy forwards or new optimizer calls. All initial/selected/current Adam, RNG, module modes, cursor and selection history are preserved by the original C3 state machine. No old40-sweep or three-sweep prefix is rebased into this protocol.

The real future stage uses the same CLI without `--synthetic`, with reviewed real inputs and an independently approved exact registration budget. Planned36 cases use96000 updates,400 unique centers and21600 development-center selection forwards. The registration contains actual parameter counts, exact sample order, source/input hashes, state-retention ceiling, selection and per-class safeguards; it rejects insufficient explicit caps and execution checks remaining storage. A hash option binds a budget; it is not itself human authorization.

For all three seeds and every native class, a proposed development gain must meet the -2pp recall floor, positive mean OA and at least2of3 positive paired seed gains. V5 gives20pp recall increments, so this guard provides no fine-grained independent uncertainty estimate. Real runtime, independent geography and effectiveness are unmeasured.

`profile.py` uses generated spectra with the actual frozen band/class dimensions. It measures patch graph/forward/backward/Adam, validation and checkpoint components. Registration separately records the time spent constructing and registering all case models, including author dummy forwards. Shape-only projections exclude real input checks, larger selection histories, source loading, contention, independent review and backup. The initial profile with213/128 bands was superseded after frozen metadata verified436/270; that earlier estimate is retained as a failed sizing assumption and excluded from the final proposal.

## Offline source recovery

The official author clone is partial; full-history bundling is not assumed available. The portable packet contains the original raw author commit, raw root tree, `model.py` and GPL `LICENSE` as `DSNet.commit`, `DSNet.tree`, `DSNet-model.py`, `DSNet-LICENSE`. The loader checks the original e910 commit hash, its root-tree hash and both fixed Git blobs before importing. This is a file-level authenticated source closure, not a forged replacement Git history or a claim to preserve all author data/demo files. Original names, upstream URL, license and scope remain in provenance. Project/native code is restored from its separate Git bundle.
