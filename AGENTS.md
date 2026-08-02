# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What This Is

A **Pixi workspace monorepo** of computer vision projects. Each package lives in `packages/<name>/` with its own Python module, CLI tools, and tests. All Pixi configuration (deps, tasks, environments) lives in the root `pixi.toml`; per-package `pyproject.toml` files hold standard Python packaging metadata plus per-package tooling config such as `[tool.ruff]` and `[tool.vulture]`.

## Environments

Each package has a prod env (`<name>`) and a dev env (`<name>-dev`, adds ruff, pytest, beartype, pyrefly, hypothesis, vulture). The dev env exposes the tasks `lint`, `typecheck`, `deadcode`, and `tests` (e.g. `pixi run -e <name>-dev tests`). Direnv auto-activates the `*-dev` env when you `cd` into a package directory.

## Commands

```bash
# With direnv active (cd into a package dir first):
ruff check .        # lint
pytest -q           # test
pyrefly check .     # typecheck

# From repo root (needed for tasks with depends-on chains):
pixi run -e monoprior --frozen monoprior-relative-depth   # runs download + demo
pixi run -e robocap-slam-dev --frozen tests
```

Prefer `pixi run --frozen` to skip re-solving deps. Only omit `--frozen` when you've modified dependencies. The dev tasks (`pixi run -e <name>-dev {lint,typecheck,deadcode,tests}`) are the canonical runners — `typecheck` applies the monorepo `pyrefly.toml` plus any per-package baseline, whereas a bare `pyrefly check .` skips the baseline and can surface known false-positives.

## Platforms & lockfile

The workspace `platforms` list defines the full platform vocabulary: the plain `linux-64`, `linux-aarch64`, and `osx-arm64` CPU/macOS subdirs plus the named `linux-64-cuda13` and `linux-aarch64-cuda13` platforms, which carry the CUDA 13.0 and glibc 2.35 virtual packages. A feature opts into a platform by listing its plain subdir or named workspace platform. An environment only solves a platform that every one of its features allows. For a `common`-composing package to run on macOS, `common` and `dev` also list `osx-arm64`, with an osx-scoped `pytorch-cpu` (simplecv imports torch at module load).

**Every linux-only feature MUST declare `platforms = ["linux-64", "linux-aarch64"]` explicitly.** Since pixi 0.71 (PR prefix-dev/pixi#6178), a feature that omits `platforms` defaults to the entire workspace list — including `osx-arm64` and the named CUDA platforms. An env's platforms are the intersection of its features' lists, so one omitting feature can demand unintended macOS or CUDA solves; if that demand is unsolvable (CUDA deps like `libcublas`), **every `pixi install -e <any-env>`/`pixi lock` in the whole workspace aborts** on the next lock write. Solvable missing demands get silently solved and added to the lock instead. When adding a feature, copy the `platforms` line from an existing linux-only feature (e.g. `mv-api`).

**Regenerate the lock on Linux.** Changing a *shared* feature (`common`/`dev`) or a build-from-source dep forces pixi to re-solve the *whole* workspace. Linux-only build-from-source envs — `wilor-nano`'s git `rtmlib` and the `no-build-isolation` deps (`moge`/`gsplat`/`dpvo`/`sam2`) — **cannot be cross-built `osx-arm64` → `linux-64`** ("no compatible Python interpreter"). So on macOS, only `pixi install`/`lock` a package's *own* deps; for anything touching shared features, regenerate `pixi.lock` on a Linux host, copy it back, then `pixi install -e <name> --frozen` on macOS.

## Architecture

**Beartype** is activated conditionally via `PIXI_DEV_MODE` in each package's `__init__.py`:
```python
import os
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```

**`tools/` scripts must be thin shims** — `beartype_this_package()` only instruments
code **inside** the package, so the Tyro `Config` dataclass and `main()` belong in the
package (e.g. `<module>/apis/<name>.py`), and the `tools/` script just wires them up.
Logic placed directly in a `tools/` script is **not** beartype-checked under dev.
```python
# tools/apps/<name>.py  — keep it to a few lines
import tyro
from <module>.apis.<name> import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
```

**Package structure:**
```
packages/<name>/
  pyproject.toml    # [project], [build-system], [tool.ruff]
  <module>/
    __init__.py     # Beartype activation
    apis/           # High-level interfaces + Tyro Config/main (beartype-instrumented)
    gradio_ui/      # Gradio components (if applicable)
  tools/            # THIN CLI shims over <module>/apis/ (demos/ and apps/ subdirs)
  tests/
```

## Code Style

Prefer straightforward inline code over tiny one-off helpers. Only extract a
function when it has meaningful reuse, hides real complexity, names a domain
concept, or improves testability. Thin wrappers that only pass through
arguments or hide a single call should usually be inlined at the call site.

**Ruff** — line length 150, rules: E, F, UP, B, SIM, I. Ignored: E501, F722/F821 (jaxtyping), UP037/UP040, SIM901.

**pyrefly** config is monorepo-wide in root `pyrefly.toml`; do not add `[tool.pyrefly]` to per-package `pyproject.toml`. When you add a package, register it in `pyrefly.toml` in **three** places: `search-path` and `site-package-path` (omit these and imports of the new module resolve to `missing-import`), and `project-includes` (omit it and the package's files aren't typechecked at all). For unavoidable stub false-positives from compiled/untyped deps (e.g. `depthai`), add a per-package `pyrefly-baseline.json` and wire it via `PYREFLY_EXTRA_ARGS = "--baseline pyrefly-baseline.json"` in `[feature.<name>.activation.env]` (see `simplecv`, `live-rerun`).

## Rerun Tools

When adding or updating Tyro-facing Rerun CLIs, prefer the shared `RerunTyroConfig`
from `simplecv.rerun_log_utils` instead of hand-rolling viewer/save/connect
flags or creating a local `rr.RecordingStream`. Add it as a nested dataclass
field such as `rr_config: RerunTyroConfig`, let its `__post_init__` configure
spawn/connect/save/serve/headless behavior, and then use the normal global
`rr.*` logging calls unless a test or library boundary specifically requires an
explicit recording stream. This preserves the flexible viewer and save behavior
expected across SimpleCV tools. For a realtime tool that needs the live viewer
**and** a `.rrd` at once, set `rr_config.live` together with `--rr-config.save`:
`RerunTyroConfig` then fans out to both via `set_sinks` (the `live`/`port` fields).

### OSS catalog file-descriptor limits

Rerun 0.34.1 keeps one descriptor open per registered `.rrd`. One ARKitScenes
dataset uses 5,015 × 7 = 35,105, so two exceed 65,536 during the second `gt`
layer. `arkitscenes-download-serve` uses `ulimit -n 524288`; restart the server
to inherit it. This is a capacity workaround—the upstream fix is open-on-demand
files or an LRU descriptor pool.

## Testing Rerun builds

**One Rerun version repo-wide — Rust follows Python.** The PyPI `rerun-sdk` pin
is the source of truth; the Rust `re_*` crates
(`packages/gsplat-rust-renderer/Cargo.toml`) must match it exactly, or the
viewer silently loses protocol/tooling parity (e.g. no viewer-control MCP
before 0.34). To bump: Python first (rerun-sdk + gradio-rerun together), then
the Rust pins (matching that release's egui family), then re-lock pixi and cargo.

The whole workspace runs **`rerun-sdk == 0.35.0`** (and `gradio-rerun == 0.35.0`) from PyPI:
`common` carries the pin with the `datafusion` extra. The `dataloader` extra stays scoped to
catalog-side features: `rerun-prerelease` for the shared catalog lanes (composed into
`no-default-feature` envs beside `catalog-common`, e.g. `simplecv-catalog`, `mv-api-catalog`)
and `prompt-da-catalog` for ARKitScenes PromptDA inference.
gradio-rerun releases pin an exact `rerun-sdk==<ver>`, so bump both together.

To test an **unreleased** Rerun build, add a `find-links` at
`build.rerun.io/commit/<sha>/wheels/` to `[feature.rerun-prerelease.pypi-options]` (CI builds one
per commit, including PR branches — `curl` the index first to confirm your platform; PR commits
are usually linux-x86_64 only) and match `rerun-sdk == <ver>` to the wheel filename. Re-lock on
linux-64 (pixi 0.70.x) and move back to a public release once the fix ships.

## Gotchas

- **Never use pip** — all dependency management goes through Pixi
- **`hf download` not `huggingface-cli`** — conda's huggingface_hub provides `hf`, not `huggingface-cli`
- **gradio from PyPI, not conda** — conda's gradio package has missing transitive deps
- **`no-build-isolation` deps** (`moge`, `gsplat`, `dpvo`, `sam2`) build from source and need torch at build time (configured in `[pypi-options]`). They can't be cross-built from macOS — see **Platforms & lockfile**.
- **sam3d-body uses `tool/` (singular)** not `tools/` for its CLI scripts
- **Direnv fails after changing `pixi.toml`** — run `pixi install -e <name>-dev` to re-solve, then direnv picks up the updated lockfile. ⚠️ If you touched a *shared* feature (`common`/`dev`) this re-solves the whole workspace and **fails on macOS** — regenerate the lock on Linux (see **Platforms & lockfile**).
- **Never use bare `except Exception` with beartype** — it silently swallows type violations. Always re-raise `BeartypeException`:
  ```python
  from beartype.roar import BeartypeException
  try:
      result = some_typed_function()
  except BeartypeException:
      raise
  except Exception:
      print("failed")
  ```
- **Use `0.0` not `0` for float annotations** — beartype strictly distinguishes `int` from `float`. `last_error: float = 0` will fail; use `last_error: float = 0.0`
- **`vulture` (the `deadcode` task) flags framework-used names** — Tyro/dataclass config fields, `pytestmark`, `__exit__`'s `*exc`, etc. Add them to `[tool.vulture] ignore_names` in the package `pyproject.toml` rather than reworking the code.
- **Pixi collapses multiline `cmd = """..."""` into a single line**, replacing newlines with spaces. If a task has separate commands on different lines (e.g. `export`, `echo`, `python`), they become arguments to the first command and never execute. The task appears to succeed (exit 0) but produces no output. Always use `&&`-chained single-line commands or `\` line continuations instead.
- **Don't poll with `pgrep -f <pat>` when the polling command itself contains `<pat>`** — it matches its own shell and the `until ! pgrep ...` loop never exits (silently hangs forever). Prefer `run_in_background` on the real command (you're notified on its own exit), or wait on a file/sentinel.
- **Always pass `--rr-config.headless` to Rerun CLIs in shells without `DISPLAY`** — the `RerunTyroConfig` default calls `rr.spawn()`; when the viewer fails to start (winit "neither WAYLAND_DISPLAY nor DISPLAY is set"), the recording stream's channel fills and every `rr.log()` blocks forever. The run wedges silently (zombie viewer child, zero CPU) instead of erroring out.
