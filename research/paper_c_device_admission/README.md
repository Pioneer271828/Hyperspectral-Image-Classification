# Public real-device intake

Issue521; sole writer C. Base `a8c4cd2ec88b60167e7dd1d8df03596f6299ae7b`, branch
`codex/paper-c-device-admission-20261127`. Only this directory changes.

Admission is claim-specific: real-device transfer classification does not require
identification of an internal-passband causal effect. The latter separately needs
appropriate measurements and controls. No new real training, fitting, sealed test
labels/query, third-party contact, account creation or questionnaire submission.
Public development intake is not independent confirmation.

Initial candidates are the official SpectralEarth CDL products (real different
instruments, account-gated; D owns literature/semantic review) and official IGN
FLAIR (real camera/Sentinel-2 acquisitions with corresponding training roles).
FLAIR is an optical multispectral alternative, not two hyperspectral instruments.
The elevation channel is not a spectral band. Aerial radiometry is harmonized,
not physical reflectance; acquisition/time/resolution confounds stay explicit.

`fetch_metadata.py` obtains bounded official public text/metadata only. It pins the
Hugging Face repository revision before fetching auxiliary records. No test pixel
or label payload is requested. Data/metadata stay in the external authorized Linux
run root. Code inspection never executes an upstream trainer or installs its stack.

Prior 8.337GB history transfer is separate: original SHA256
`b9cf9adc398d79214c23abd034e4178ff4e82c2fdd0fb2ba81fd5ac092fdb045`, last verified
offset535822336. Originals/partials stay intact; no missing state is retrained.
