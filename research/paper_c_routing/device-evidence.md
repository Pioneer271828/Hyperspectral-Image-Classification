# Device, semantic and version evidence

Checked2026-11-26. These are provenance findings, not permission to load a new
dataset, remap its labels, or create a confirmation role. The complete30-direction
gate is emitted by`catalog.py` from the first package's frozen ordered-pair CSV.
The base CSV is preserved; this package adds executable model eligibility and
new primary-source qualifications without changing the shared dataset registry.

| Dataset | Original evidence inspected | What is established | What remains open |
|---|---|---|---|
| PaviaU | [provider](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes), current fetch403; ROSIS description in original Xuzhou publications | Published ROSIS family and103-band benchmark; existing local payload hash/header retained | Retained103-band product to actual detector/SRF binding; local approximate471.49–856.23nm centers differ from nominal published coverage; exact calibration is unavailable |
| Xuzhou | [author/publisher](https://www.3sobs.cn/share), [Tan et al.2019](https://doi.org/10.1109/JSTARS.2019.2892975), p.634/TableIX; [Wang et al.2019](https://s3.3sobs.cn/paper/2019/Wang%20et%20al_2019_Caps-TripleGAN.pdf), p.7238 | The original product combines VNIR-1600 and SWIR-384 imagery, after resampling and spectral removal; the delivered benchmark is500×260×436 | Delivered-column to original camera/band map, measured centers/SRF and local original class-ID binding |
| HongHu | [WHU provider](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm), §1.3/Table3 | Headwall Nano-Hyperspec17mm acquisition2017-11-20,270bands; current local header matches940×475×270 | Exact retained calibration, serial identity, independent geography including class21 |
| LongKou | Same provider, §1.1/Table1 | Nano-Hyperspec8mm acquisition2018-07-17,270bands; local header550×400×270 | Author product to local role/version binding and new independent roles |
| HanChuan | Same provider, §1.2/Table2 | Nano-Hyperspec17mm acquisition2016-06-17,274bands,1217×303 | No admitted payload in the current authorized snapshot; this does not prove absence at every historical location |
| LaoYuHeSiDi | [AFGNet experiment](https://doi.org/10.3390/rs17050763), §3.1/Table2, reference56 to [DF2Net](https://doi.org/10.1109/JSTARS.2024.3403863) | A concrete candidate is an OHS satellite image,32bands/eight classes/10m,May2024 | Paper reports391×591 while project historical inventory reports391×550. Origin reference, local crop/version, acquisition metadata, label map, payload permission and roles are not bound |

The three WHU acquisitions share a published model family. Different lens,
scene or acquisition date does not establish an unseen sensor family; physical
serial numbers are not verified. Xuzhou's two hardware streams were already
merged before this model receives them. Their existence does not make the present
single-cube router a two-modality fusion implementation.

Tan et al.p.634 reports160 VNIR bands and288 SWIR bands before overlap/absorption
removal, with the VNIR image resampled to the SWIR spatial resolution. The local
loader linearly interpolates436 approximate centers over415–2508nm. Therefore,
this package's122-column approximate-center cut is **not a certified VNIR-only
view**. Stride2/4 mostly change sampling density while retaining almost the full
nominal span. Neither operation reproduces a measured detector response.

## Important class-name conflict

The original TableIX assigns coal toID3 and cement toID4. Historical project G3
reports instead labelID5 as Coal andID6 as Cement. The original frozen G3 target
metadata contains the same nine class counts as the original publication,
includingID3=2783,ID4=5214,ID5=13184 andID6=2436; its target image SHA also matches
the first package's local image inventory. This is strong counterevidence to
the historical names, but count agreement alone is not a custodian-signed
local-file semantic map.

All current outputs therefore retain numeric class IDs1..9, without renaming,
merging, or changing any model head or label array. Claims specifically about
coal/cement protection must remain unresolved until that mapping is bound.
The first package's request to verify class5/6 semantics remains a historical
gap; it must not be repeated as an authoritative semantic assignment. The same
ID in PaviaU, Xuzhou and WHU has no automatic shared meaning.

## Access and reuse record

The publisher PDFs were downloaded to external references and read, with the
original TableIX page visually checked. The full pages/PDFs are excluded from
the review package. AFGNet's publication identifies`Isee-max/AFGNet`; fixed
SHA`f27705a0702441616e6a08d1b99fe9f72d7e23f5` contains only a README, so no executable
implementation or source-code license was established. It is a provenance lead,
not a reproduced baseline. DF2Net's origin link has not been independently read.

The GitHub connector returned a redirect it could not follow through its allowed
endpoint set; a read-only Git metadata clone established the one-file tree. No
external arrays were decoded, no questionnaire was submitted, and no author was
contacted. Existing DSNet author code/license verification remains pinned to the
[first package](../paper1_c/source-review.json); this package does not copy more
external implementation or claim that an available URL implies reproduction.

Independent sample amounts and project authorization are specified in
[confirmation readiness](confirmation-readiness.md). Public dataset availability
does not release the project's held-aside observations or labels.
