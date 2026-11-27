# C5 public real-acquisition intake

Issue521 / Draft522; C is the sole writer of this directory. The candidate is
IGN FLAIR's author-defined UCE/non-UCE acquisition groups and paired Sentinel-2
products. These are real optical multispectral observations. They do not supply
two hyperspectral instruments, calibrated per-camera SRFs, or a causal internal
passband experiment. Training remains separately gated.

The official original-host ZIP directories contain61712 aerial/mask pairs across
40 train/validation domains. Joining only the32 allowed training domains to the
actual hashed camera metadata gives47712 images: UCE24287 in18 domains and
non-UCE23425 in14. No domain contains both groups. In2020 there are13247 UCE
and5775 non-UCE records; other years are segregated in this roster. A classification
transfer question is feasible as a composite camera-group/geography shift, with
explicit year controls. Neither same-year matching nor an acquisition-ID string
identifies a pure camera or passband effect.

Before reading labels, four2020 examples were fixed by training-domain and image-ID
order, two per raw prefix group. All four image/mask objects pass source SHA256,
ZIP CRC, independent pixel decoding, dimensions, affine alignment and role checks.
Aerials are512x512x5 uint8: four optical columns plus one separate height channel.
Masks are512x512 uint8 with TIFF LZW compression. The independent small decoder
matches tifffile aerial pixels and Pillow/libtiff mask pixels exactly.

Three samples satisfy the additional conservative0.21m metadata-to-affine-center
criterion. One has offsets(-2.61,-13.77)m and stays quarantined by the public reader.
This is a project QA exclusion, not an official centroid guarantee or a diagnosis
of an erroneous camera record. The plan and threshold were not changed, and the
example was not silently replaced. It remains retained for investigation. The
other three provide one non-UCE and two UCE examples for runnable development
intake. They do not establish twelve-class coverage: only four raw classes are
observed in both groups in this tiny sample. The independently proposed raw1..12
future objective was not selected from that intersection; raw13..19 are ignored
explicitly and raw masks remain intact. Qualified examples contain2228 ignored
pixels out of786432 (about0.2833%). Class16/17 names conflict across original
sources; no swap was made.

The initial independent audit failed on an incorrect TiffPage metadata attribute;
it was corrected to TiffFile. The next whole-set audit rejected the center mismatch.
The final audit reports byte checks separately from per-sample admission, and the
reader requires that receipt. Failure logs remain on authorized Linux. Nine
focused tests passed on LXC at1fcf26a5094dca99f91de639ed119fc7586fb42a, then on
rmvision atdb2432fc6c1df0419bfcfb1d1abc5cdd09bc062e after the user's host switch.
LXC used Python3.10.20/NumPy2.0.2/Pillow12.2.0; fallback uses
Python3.12.3/NumPy1.26.4/Pillow11.1.0. Both use tifffile2025.5.10.
Only the fallback's isolated intake environment received this small dependency.

The two satellite sequences were chosen by compressed member size among the four
preselected aerials, one per acquisition group, without label-dependent choice.
The official matching dictionary determines a40x40 context crop. The satellite
payload acquisition is complete: six data/mask/product objects,44333333 network
bytes. Together with the initial aerial intake,64693574 bytes were received under
the shared268435456-byte sample budget. Whole-archive SHAs are not asserted for
remote range-read training ZIPs: archive ETag/size, requested-range hashes and
complete decoded-member SHA/CRC are the identities actually verified.

Both sequences pass independent NPY-header/NumPy byte decoding, product-date,
mask-axis and crop checks on rmvision atdb2432fc6c1df0419bfcfb1d1abc5cdd09bc062e.
The non-UCE31-date sequence inherits its aerial center quarantine; the UCE52-date
sequence is admitted. The latter is uint16,52x10x207x156 before cropping and
52x10x40x40 afterward, with52 matching L2A S2A/S2B product records from2020-01-05
through2020-12-30. Its nearest date is3 days from the aerial acquisition, not
simultaneous. Cloud/snow probabilities span0..100 and remain unfiltered. The
isolated reader smoke test successfully reads all three admitted aerials and the
one admitted satellite sequence, and rejects both quarantined views before
opening their payloads. It verifies a runnable real-device development intake.
Pixelwise10m label resampling is
not implemented. The larger satellite context and annual sequence are not a
same-instant, same-pixel counterpart. No model call, optimizer update, test/query
payload, or independent-confirmation result was produced in C5.

For a subsequent classification experiment, first freeze public-development
source/target domains and held-out areas within the permitted roster, the common
taxonomy, source-only versus target adaptation roles, geographic/year controls,
three model seeds, total update/selection cost and stopping criteria. Confirm
adequate class coverage before launching training. Aerial camera-group transfer
can use common four-column observations; satellite transfer additionally needs an
explicit spatial target and temporal rule. Keep this separate from any causal
internal-passband test, which still lacks native response binding and suitable
paired intervention support. No future training is authorized by this document.

SpectralEarth remains the stronger real-HSI alternative, pending actual DLR
access, per-sensor annual-class mapping, and upstream payload rights. Houston's
official contact requirement was not bypassed. Full source/permission facts and
conflicts are in source-review.json. The8.337GB historical archive remains a
separate unfinished transfer with last verified prefix535822336 bytes; C5 neither
retrained those states nor represented that archive as backed up.

The completed C5 inputs were privately migrated with whole-archive verification:
aerial/metadata archive11014936B SHA256
`0b544b8bca6dfe542560e98d06d52eb5e45c9d0c01ea606b628aad3fc1e7fd11`;
paired archive43737175B SHA256
`2e829edf53afd5d63469f51d7dce00ebcba0eb8c0ef38e6871e5fe8f8ed67c98`.
All14 decoded objects were then independently rechecked at the destination. The
Windows relay retained zero payload files and zero credential files. These two
small successful transfers do not close the unrelated historical archive gap.
