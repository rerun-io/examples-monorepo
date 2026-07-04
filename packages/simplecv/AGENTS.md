# Repo Instructions

## Package Manager

SimpleCV is vendored into the examples monorepo under `packages/simplecv`.
Use the root `pixi.toml` for all dependency, environment, and task changes.
Do not add package-local `[tool.pixi.*]` or `[tool.pyrefly]` sections, and do
not use `pip` or `uv`.

With direnv active in this directory, direct commands run in `simplecv-dev`.
From the repository root, use `pixi run -e simplecv-dev --frozen <task>`.

## Performance Benchmarking

Use the default Pixi environment for actual performance measurements. Do not use
`pixi run -e simplecv-dev` for benchmark numbers, because the dev environment
enables runtime checking and can significantly slow down the measured path.
Reserve `simplecv-dev` for tests and validation.

## Implementation Workflow

Use the `tdd` skill for implementation work. Prefer a failing or characterization
test first, then make the smallest change that proves the behavior.

## Existing Pattern First

Before adding new infrastructure, inspect the closest existing implementation and
preserve its default behavior unless there is a demonstrated blocker.

For catalog/server changes:

- The 2-tier flow is canonical: serve = the `rerun server` CLI (`simplecv-catalog-serve`), register =
  `tools/catalog_register.py` / `register_main` (`CatalogClient` → `create_dataset` →
  `register(layer_name="base", on_duplicate=REPLACE)` → `register_blueprint`). The older single-tier
  `mount_catalog` + RRD index-table builders have been removed; consumers (including mv-api's
  prediction-catalog flow) connect to a running catalog with `CatalogClient` and register layers.
- Assembly101-style nested RRD registration must continue to work through
  recursive discovery and explicit RRD file lists.
- Transport-specific workarounds must be opt-in and covered by tests proving the
  normal path is unchanged.

## Python Style

- Use PEP 526-style variable annotations for nontrivial local values.
- Annotate arrays with jaxtyping dtype and shape.
- Follow the repo's existing dataclass field documentation style.

## Pre-release Rerun

Prefer the public `rerun-sdk` release. The catalog runs on the `simplecv-catalog` env, which uses
the shared `rerun-prerelease` lane, currently pinned to `rerun-io/reality#2496` (`deeb4e6` /
`0.34.0a1+dev`, fast OSS-catalog register). See the root `AGENTS.md` "Testing Rerun builds" for how
to pin/repin that lane. Keep prerelease opt-in; move back to a public release once the fix ships.
