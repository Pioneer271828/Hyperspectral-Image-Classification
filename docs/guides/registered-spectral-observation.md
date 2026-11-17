# Registered spectral observations

The opt-in API does not modify the baseline image loader, network, attention
hooks, spatial forward or legacy checkpoint format. The source review for
Issue #466 is recorded in [the metadata register](../references/wp05-spectral-metadata-sources.json).
No released scene currently has an approved calibration in that register.
The HongHu proposal remains a draft and does not authorize training.

1. Call `transfer.spectral_calibration.load_calibration` with an independently
   reviewed approval, the calibration table and the delivered image file.
   The loader verifies the actual image and table bytes, original band mapping,
   validity mask and physical units. The approval must come from a trusted source.
2. Create a sampling record from the original label-free spatial observation,
   then call `transfer.spectral_registration.register`. The record binds the
   dataset, full and selected metadata, retained model input, pixel IDs, plan
   and partition. Original labels and held-out scores are not part of this API.
3. Pass the registration, graph scale and optional coverage reference to
   `transfer.spectral_checkpoint.save_registered_checkpoint`. Keep the package
   digest with the independent registration. This package cannot be interpreted
   as a legacy state dictionary by the baseline loader.
4. Use `load_registered_checkpoint` with that exact package digest and
   registration. Its spatial forward accepts the original observation and
   authenticated metadata. It replays the fixed subset, rejects an identity
   mismatch before model entry, and uses the scale and coverage reference in
   the pinned checkpoint. For an unregistered analysis, the separate
   `forward_spectral_partition` function accepts explicit settings.

This replaces the optional interfaces proposed in #467/#468. Existing calls to
`read_band_metadata`, `forward_partition`, `_get_attention`, and legacy
`save_model_checkpoint` / `load_model_checkpoint` retain their original
signature and behavior. A bare tensor cannot establish pixel IDs or spatial
plan identity. Registered replay therefore has no public raw-tensor shortcut.
The package fixes the observation and graph/coverage settings; it does not
certify cross-device arithmetic equivalence or calibration quality.
