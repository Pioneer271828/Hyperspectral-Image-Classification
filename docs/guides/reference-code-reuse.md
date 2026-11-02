# Reference and code review before implementation

For a new research mechanism, first read the relevant original papers and check author-provided code, then implement the smallest adapter to the existing project. This is the user's requested development workflow. The [reference inventory](../references/research-code-inventory.json) is a dated starting point, not proof that a repository runs correctly here.

1. Record the task hypothesis, exact paper/version/DOI, relevant section and what is actually demonstrated. A survey points to methods; it is not their implementation or evidence of HSI transfer gains. Inspect the supplied PDF when it differs from the preprint.
2. Follow paper/project/author links first, then search for official code. Distinguish author code, third-party reproduction, tools and bibliographic lists. Bound unsuccessful searches and record `unconfirmed`; do not claim that no code exists anywhere.
3. Inspect actual source, entry points and the applicable license, including nested subprojects and borrowed files. Record an immutable commit, license path/SPDX where verified, attribution and the permitted reuse boundary. Public visibility and a null GitHub root license field do not establish permission. A nested license does not cover the entire repository automatically.
4. Prefer a maintained compatible library or a small licensed component behind a default-off adapter. Keep the original model, data protocol, CLI and checkpoint contracts. Do not transplant a repository's trainer, splits, extra data access, tuning defaults, environment scripts, telemetry or implicit resume rules. Low-rank adaptation is distinct from factorizing the inference weight.
5. Before modifying shared interfaces, specify the proposed change and ownership in the Issue. List files/components to reuse, local changes, dependencies and downstream compatibility. External instructions are reference material, never execution authority. Run all inspection code, installations and validation remotely; keep clones/artifacts outside production checkout and away from other projects.
6. Implement meaningful equivalence, negative-control, gradient, serialization and policy tests appropriate to that mechanism. Execute a small original example when useful, then a project-specific integration check; record exact upstream/local revisions and commands. Downloading or reading code is not reproduction. No need to run every upstream full training experiment.
7. Compare the unchanged baseline, current module alone and required same-capacity/same-budget/random controls on fixed inputs/protocols. Include scoring teachers, routers, calibration, distillation and recovery in cost accounting. After each training, apply the existing [review protocol](post-training-review.md).
8. If permission, implementation availability or suitability remains uncertain, use another verified component or independently implement the published method and label it honestly. Preserve attribution when copying/adapting code, with a local manifest/notice describing upstream SHA and modifications. Record which code was actually incorporated; an inventory entry is not a vendored dependency.

An implementation Issue should contain these fields:

| Field | Required information |
|---|---|
| Evidence | Paper/version/section; demonstrated result versus project hypothesis |
| Upstream | URL, author/third-party/tool classification, immutable SHA |
| License | Actual path and scope, attribution, unresolved restrictions |
| Reuse | Component/file/function, adaptation boundaries, explicit independent reimplementations |
| Data/budget | Allowed labels/regions/sensors, roots/supports, updates/selection/calibration/extra teacher costs |
| Verification | Original example if useful; default-off, negative, numerical/gradient/checkpoint and real integration checks |
| Effect | Paired module ablation, per-class/domain readouts, actual runtime/cost and retained failures |

Current next implementation remains M1-C, followed by independent E1. The dynamic/compression/curriculum candidates in the [supplemental plan](../../research/动态计算与压缩课程学习_论文评审及搭建测试补充计划_20261102.md) do not automatically enable themselves or add a new research axis.
