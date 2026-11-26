# Independent calibration and confirmation readiness

This is a release request specification, not a release of images, geography,
labels, thresholds or new training. All 30 ordered directions remain blocked for
independent confirmation. Existing V5, its unlabeled context, model seeds and
synthetic band views cannot supply a new confirmation role.

Calibration selects an operating rule using a separately authorized population.
Confirmation evaluates the frozen rule once. This package's half-batch quota
uses no calibration labels and is an offline allocation experiment; it is not an
online threshold or an independent noninferiority demonstration.

## Minimum material needed to open an engineering evaluation

The proposed engineering floor is **five fresh labeled points per class in each
of two roles**, with at least two independently assessed parent geographies per
class per role. This is a proposed robustness floor, not a mathematical minimum
or a claim of sufficient statistical power. Two geographies still cannot support
a stable spatial variance estimate. Do not assign multiple neighboring pixels
the status of independent geographies.

| Target | Classes | Fresh calibration points | Fresh confirmation points | Present blocker |
|---|---:|---:|---:|---|
| PaviaU | 9 | 45 | 45 | Trusted new role release, exact retained-band/device binding |
| Xuzhou | 9 | 45 | 45 | New roles, merged camera band map, original-to-local semantic map |
| HongHu | 22 | 110 | 110 | New geography including class21; existing parent65 is shared |
| LongKou | 9 | 45 | 45 | Local version and observation/label role admission |
| HanChuan | 16 | 80 | 80 | Admitted payload and independent roles |
| LaoYuHeSiDi | 8 | 40 | 40 | Candidate source is not version-bound; size discrepancy |

The table totals365+365 fresh labels for six target populations. It is not30
independent studies. Reusing one confirmation role for several frozen source
models requires an explicit joint comparison family and changes multiplicity;
it does not create additional samples. Source labels, target adaptation supports,
model fitting, selection and geometry selection remain separately accounted.

## Why a statistically sufficient minimum is not yet specified

For a per-class guard, five labels produce20 percentage-point recall increments.
A two-point drop threshold cannot be established reliably by that resolution.
An exact sample size additionally requires a fixed estimand, margin, comparison
family, confidence/power, paired disagreement rate and spatial sampling design.
None may be estimated by opening held-aside confirmation labels first.

For orientation only, under independent identically distributed Bernoulli trials,
zero observed adverse events give the one-sided upper bound
`1 - (alpha/m)**(1/n)`. Thus `ceil(log(alpha/m)/log(1-delta))` trials suffice
to put that particular bound below `delta`. This is the catalog's illustrative
zero-error calculation withalpha=.05, m=target class count anddelta=.02. It is
not a power calculation, not a general paired recall sample size, and not a
license to treat adjacent pixels as independent. Counting multiple arms/views
requires a larger prespecifiedm.

For independent block-balanced paired scores `D_b in [-1,1]`, the elementary
Hoeffding bound gives a conservative one-sided radius
`sqrt(2*log(m/alpha)/B)` overBindependent blocks. At an observed zero difference,
a margin`delta` would require `ceil(2*log(m/alpha)/delta**2)` blocks by this bound.
This illustrates how demanding distribution-free protection can be; it changes
the estimand to equal-weight blocks and is not the proposed acquisition budget.
The inference is our bounded-score derivation, not an empirical uncertainty
estimate from this development run.

Before a formal study, the custodian and reviewer must freeze a feasible
cluster-aware design using external or permitted development estimates of paired
disagreement and between-geography variation. If a class exists in only one
historically exposed geography, acquire another scene or leave that class/direction
unconfirmed. Do not redraw a favorable split or silently drop the class.

## Required authorization and immutable records

An independent trusted custodian must provide: original observation and label
version hashes; original class-ID definitions and any many-to-one mapping; sensor,
acquisition and retained-band provenance; parent/component roles and complete
history of observation/label use; physical separation and full patch/Q/graph
footprint review; released label counts; approved allowed uses for each role; and
a content-bound authority record approved by the project user. Review all class
feasibility before assigning roles. Freeze one accountable writer for that release.

The dataset author's public academic-use permission is distinct from the project's
permission to open held-aside labels. [WHU's provider page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm)
permits academic use with citation and excludes commercial use; its download
questionnaire is not project confirmation authorization. The
[Xuzhou publisher](https://www.3sobs.cn/share) supplies the original acquisition
reference, but supplies no approved project role assignment. No questionnaires,
emails or new source/label downloads were submitted in this package.

For approved confirmation, model and operating-rule selection must already be
finished; normalization, graph building and patch extraction stay role-local;
confirmation context cannot be used to fit the model or calibrate the router.
No authority signature or new geography is fabricated in this package.
