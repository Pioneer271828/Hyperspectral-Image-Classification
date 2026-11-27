Draft not sent

Subject: Original metadata for HyRANK v001 public training scenes

Dear HyRANK dataset maintainers,

We are preparing a noncommercial academic development study comparing your EO-1 Hyperion products with an existing airborne HySpex product. We would like to bind the original metadata for **HyRANK Hyperspectral Satellite Dataset I, v001, DOI 10.5281/zenodo.1222202**, specifically the publicly labeled training scenes **Dioni and Loukia**. Our reference record lists `HyRANK_satellite.zip`, 415,652,689 bytes, MD5 `a940e79ffa65a6b28174b0d2b3ce8836`. We have not downloaded that archive.

At this stage we request documentation and aggregate metadata only. We do not request benchmark-hidden validation labels, image pixels, label coordinates or a new annotation exercise. If an existing README, codebook, header, processing note or manifest already contains the information below, a link or a small text export would suffice.

| Requested field | Purpose | Minimum useful delivery |
|---|---|---|
| Original product/member identity | Bind the cited version to the actual Dioni/Loukia products | Member names, dimensions, data type, interleave/axis order, sizes and available checksums for the public training cubes and annotation files; headers without pixel payloads |
| Delivered class values | Avoid treating figure ordinals as array codes | Original numeric-value/name table, unlabeled/nodata/ignore codes, and the documented value order for this version |
| Water and forest definitions | Review a proposed coarse water/woody-cover task | Definitions and any CLC correspondence for Water, Coastal Waters, Broad-leaved Forest, Coniferous Forest and Mixed Forest; known canopy/mixed-pixel, orchard, watercourse or inland-water distinctions. Please indicate where no quantitative criterion was specified |
| Acquisition and processing | Interpret the processed spectra correctly | Original EO-1 scene/product IDs and dates; delivered radiometric stage and units; scale/offset/nodata; atmospheric/geometric correction and resampling history; the retained 176-column list with its original-band numbering convention and available wavelengths |
| Spatial units and overlap | Determine whether separate images or groups support a spatial comparison | Existing CRS/transform or public footprints and an overlap statement; the number of original annotation polygons or groups per candidate class and scene. Aggregate counts are sufficient initially; no private parcel list or label geometry is requested |
| Candidate-class support | Check feasibility before defining a project split | Aggregate labeled-pixel and original-polygon/group counts for the candidate water/forest categories in each public training scene, with absent or unknown classes explicitly identified |
| Release roles and terms | Preserve publisher and project boundaries | Confirmation that these are the original public training products, their applicable license/attribution and any relevant revision or use conditions. We do not seek Erato, Nefeli or Kirki/Kiriki reference labels |

The original HyRANK report describes surface-reflectance products, whereas standard USGS Hyperion archive descriptions include radiance products. We seek the processing chain of the specific delivered release; we are not assuming that these descriptions refer to the same stage. Likewise, generic current CLC guidance is not being substituted for your dataset's own codebook. Response curves or FWHM would be useful if already available but are not a prerequisite for this classification metadata request.

If the metadata supports the proposed task, any subsequent request for a limited public-training subset would be separately reviewed and authorized by our project. A possible later minimum would be five fitting and five development centers per mapped class in each of Dioni and Loukia, together with their original group identities and only the authorized context needed for clipped 7×7 observations. That would be 40 prospective Hyperion centers in total, not a request for the full archive or for independent confirmation labels. Feasibility and spatial-role assignment would have to be established before choosing centers. We would retain the original labels, release identities and citations, and would not ask you to certify the histories of third-party pretrained models.

Please mark unavailable information as unknown rather than infer a calibration, polygon partition or class correspondence. Our own source-model and data-exposure records would be audited separately.

Thank you for considering this documentation request and for maintaining the benchmark.
