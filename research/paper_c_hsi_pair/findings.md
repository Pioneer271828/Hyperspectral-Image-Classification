# Supported pair and its evidence limits

C7 supports a **two-class development input pair** between the existing Xuzhou
HySpex436 and HongHu Nano-Hyperspec270 acquisitions. The same frozen data can
serve either ordered direction. This is a restricted coarse ontology with
disclosed derivation evidence, not canonical-byte reproduction, an accuracy
result, independent confirmation or identification of a causal sensor effect.

| Evidence | Established here | Limit |
| --- | --- | --- |
| [Original Xuzhou DOI metadata](https://api.datacite.org/dois/10.21227/t3c9-h862) and [IEEE provider](https://ieee-dataport.org/documents/xuzhou-hyspex-dataset) | CC-BY4.0;2014;500×260×436;0.73m;nine classes. Recovered DOI bytes exactly match priorWP05 fingerprint | Current original data-file delivery requires login; no account or protected download was used |
| [Author description](https://www.3sobs.cn/share), local README and original TableIX | Same historical product/instructions,68877 labeled pixels; original published ordinal names | Original all_x/data.mat→local derived MAT conversion code/header is absent; publisher byte identity is not certified |
| Original G3 task and coverage metadata, actual curator/release records | Same image SHA3733fd46… and GT SHA783cb5cd…; all9 per-ID counts match TableIX; original numerical labels/point identities reach the current bank unchanged | The earlier descriptive coverage audit examined full labels including test counts. That historical exposure is disclosed; C7 opens no raw GT/query file |
| [WHU provider](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm), section1.3/Table3 and copyright terms | Nano-Hyperspec17mm;2017-11-20;940×475×270; original22-class legend; academic use with citations, commercial use prohibited | No new questionnaire or raw-data download; exact detector/SRF calibration remains unknown |
| Original paper instrument description | Xuzhou VNIR-1600+SWIR-384 streams were resampled/merged before the436-column product | This is one delivered cube, not a new simultaneous multimodal learning experiment |

The identity inference combines the original per-ID table, the exact unique
count vector, matching product/README facts, and the same original file hashes
through the role curator and development bank. It does not infer names from
equal class numbers across datasets, old manual reports, model scores, or an
unavailable calibration table. Historical manual names5=Coal/6=Cement are
counterevidence and are excluded; the original published table has6=Trees.

| Common target | Xuzhou original labels | HongHu original labels | Interpretation |
| --- | --- | --- | --- |
|0 exposed bare ground |1 Bareland-1,7 Bareland-2 |3 Bare soil | Explicit broad coarsening; substrate/contamination and subclass composition may differ |
|1 tree canopy |6 Trees |22 Tree | Broad canopy only; no species, age or individual-tree equivalence |

Coal, cement, lakes, unspecified crops, roads and red roof/tile labels are
excluded. In particular, Xuzhou Red-tiles9 does not establish HongHu Red roof1
equivalence. `class_catalog.py` exports all31 original class decisions from the
authenticated public-level facts. Raw unannotated0 is never treated as common
ground0; the latter is derived only from the explicitly mapped labeled points.

Each scene retains10 fitting and10 exposed-development points,5 per common
class and role. There are40 unique original labeled identities, not80: visiting
the two ordered directions presents the same points twice. Xuzhou's selected
ground fitting points contain3 originalclass1+2 originalclass7; development has
1+4. These fixed hash-selected subclass proportions were determined before any
model call and must be retained in future interpretation. No label or pixel role
was reassigned to repair coverage.

All cross-role shared parents are excluded before selection. The selected
training/development parents and original observation boxes do not overlap;
minimum box gap is4 pixels in each scene. Patches are7×7, clipped to their
original authorized tile with replicated edges. Full-tile NPY file I/O is
separate from returned patch context. Original per-scene devtrain normalization
statistics are reused, including their historically broader unlabeled training
footprint; none are refitted on the common subset or development validation.

Same pixel context does not mean the same ground area: published nominal GSD is
0.73m versus approximately0.043m. Geography, date, platform, preprocessing,
resolution, class granularity and spectral coverage are confounded. Parent
separation is a development safeguard, not evidence of independent class
components or a new acquisition replicate. Original split construction used
train-region class counts and was oracle-assisted, not label-blind.

All processed columns are preserved. Historical approximate wavelengths and
missing SRF remain explicit; they do not block this classification input plan.
They cannot certify camera-to-column mapping, optical-response integration or
a synthetic VNIR-only view. Bank `raw_fp32` means before project normalization,
not unprocessed sensor DN: the source products already have their own published
preprocessing histories.

The pair is restricted to authorized noncommercial academic research and required
attribution: Tan/Wu/Wang, *Xuzhou HYSPEX dataset*,2018,
DOI10.21227/t3c9-h862; Tan et al.,2019,DOI10.1109/JSTARS.2019.2892975;
and the WHU provider's prescribed Zhong et al.2020 *WHU-Hi* citation and companion
references. Code or paper licenses are not substituted for dataset permission.
Raw spectral/label files are not publicly redistributed by this package.

Historical source/checkpoint pins in bank provenance grant no weight reuse.
No learned model is loaded by C7. A future target-unseen source-only study needs
an independently qualified initializer, a frozen same-budget/selection contract,
measured time/cost notice and the applicable user training authority. K5/V5 is a
development setting, not a power or convergence claim; it changes the objective
from full9/22-class classification, so old native/G4 accuracies are not directly
comparable.

The alternate original HyRANK record was also recovered:CC-BY4.0, versionv001,
HyRANK_satellite.zip415652689bytes. Its public training scenes Dioni/Loukia are
distinct from benchmark-hidden validation Erato/Nefeli/Kirki. No payload or new
role was acquired. The MUUFL original record says`other-open`; it does not close
a dataset-specific MIT grant. Existing Xuzhou/HongHu inputs suffice for this
limited package, so the registered new-image/GT download budget stays zero.
