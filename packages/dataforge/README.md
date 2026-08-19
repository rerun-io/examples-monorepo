# dataforge

One package for dataset work: **download → convert → register → view**.
Train and eval are the designed-for second half.

The full design — decisions, evidence from a 9-system study, and the ranked
work plan — is in **[docs/dataforge-design-report.html](docs/dataforge-design-report.html)**
(single self-contained file; open it in a browser).

v1 scope: the simplecv **robocap** and **HOCap** datasets, nothing else.

## Status

Skeleton only: packaging, envs, and tooling wiring exist; the package has no
operational code yet. The remaining work is to implement the dataset verbs, dataset specs,
and Tyro CLIs described in
**[docs/dataforge-design-report.html](docs/dataforge-design-report.html)**.
