# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Pixi workspace monorepo** of computer vision projects. Each package lives in `packages/<name>/` with its own Python module, CLI tools, and tests. All Pixi configuration (deps, tasks, environments) lives in the root `pixi.toml` — per-package `pyproject.toml` files only have standard Python packaging metadata.

## Packages & Environments

Each package has a **prod** environment (for running demos/apps) and a **dev** environment (adds ruff, pytest, beartype, pyrefly):

| Package | Prod env | Dev env | Module path | GPU |
|---|---|---|---|---|
| monoprior | `monoprior` | `monoprior-dev` | `packages/monoprior/monopriors/` | CUDA 12.9 |
| wilor-nano | `wilor` | `wilor-dev` | `packages/wilor-nano/src/wilor_nano/` | CUDA 12.9 |
| sam3d-body-rerun | `sam3d` | `sam3d-dev` | `packages/sam3d-body-rerun/src/sam3d_body/` | CUDA 12.9 |
| sam3-rerun | `sam3-rerun` | `sam3-rerun-dev` | `packages/sam3-rerun/src/sam3_rerun/` | CUDA 12.9 |
| robocap-slam | `robocap` | `robocap-dev` | `packages/robocap-slam/robocap_slam/` | None (CPU) |

## Common Commands

Always specify the environment with `-e`:

```bash
# Install an environment
pixi install -e monoprior        # prod only
pixi install -e monoprior-dev    # dev (includes ruff, pytest, beartype, pyrefly)

# List tasks for an environment
pixi task list -e monoprior-dev

# Lint, typecheck, test (unified task names across all *-dev envs)
pixi run -e monoprior-dev lint
pixi run -e monoprior-dev typecheck
pixi run -e monoprior-dev tests

pixi run -e robocap-dev lint
pixi run -e robocap-dev typecheck
pixi run -e robocap-dev tests

# Run a single test file
pixi run -e monoprior-dev -- pytest tests/test_specific.py -q

# Run demos (prod env is sufficient)
pixi run -e monoprior relative-depth
pixi run -e robocap robocap-track
```

Dev tasks (`lint`, `typecheck`, `tests`) are unified — same name in every `*-dev` environment. Demo/app tasks live in the package's own feature and work in both prod and dev envs.

When running tasks (especially demos and apps), prefer `pixi run --frozen` to skip re-solving dependencies. This avoids multi-minute solve times in a large monorepo when `pixi.toml` hasn't changed. Only omit `--frozen` when you've actually modified dependencies.

## Architecture

### pixi.toml Feature Composition

Dependencies are organized as composable **features** in root `pixi.toml`:

- **`common`** — shared deps all envs get (numpy, opencv, rerun-sdk, gradio, jaxtyping, etc.)
- **`cuda`** — CUDA 12.9 toolkit + PyTorch GPU
- **`dev`** — ruff, pytest, beartype, pyrefly, hypothesis + sets `PIXI_DEV_MODE=1`
- **Per-package features** — package-specific deps, editable install, tasks (with `cwd = "packages/<dir>"`) + sets `PACKAGE_DIR`

Each environment has its own **solve-group** for independent dependency resolution. Prod/dev pairs share the same solve-group (no extra resolution cost). Per-package features can **tighten** common's loose version constraints.

### Beartype Activation

Beartype is activated conditionally via `PIXI_DEV_MODE` (set by the `dev` feature's activation env):
```python
import os
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```

This means beartype only runs in `*-dev` environments. No try/except needed since beartype is always available when `PIXI_DEV_MODE` is set.

### Package Internal Structure

Each package follows roughly:
```
packages/<name>/
  pyproject.toml          # [project], [build-system], [tool.ruff]
  <module>/
    __init__.py           # Beartype activation (conditional on PIXI_DEV_MODE)
    apis/                 # High-level inference interfaces
    models/               # Model implementations
    data/                 # Data loading (often a git submodule)
    gradio_ui/            # Gradio components (if applicable)
  tools/                  # CLI scripts (demos/ and apps/ subdirs)
  tests/
```

`pyrefly` config is monorepo-wide and lives only in the root `pyrefly.toml`. Do not add
`[tool.pyrefly]` to per-package `pyproject.toml` files.

### Key External Dependencies

- **simplecv** (git dep) — camera parameters, Rerun logging utilities, shared across packages (pinned to different revs per package)
- **rerun-sdk** — 3D visualization, deeply integrated in all inference pipelines
- **gradio + gradio-rerun** — interactive web UIs with embedded Rerun viewer
- **moge** (Microsoft) — depth and surface normal prediction (requires `no-build-isolation`)
- **tyro** — CLI argument parsing from dataclass configs

### Ruff Config (consistent across packages)

- Line length: 150
- Rules: E, F, UP, B, SIM, I
- Ignored: E501 (line too long), F722/F821 (jaxtyping compatibility), UP037/UP040, SIM901

### Dependency Overrides

Workspace-level `[pypi-options.dependency-overrides]` in root `pixi.toml` overrides transitive dependency pins (gradio, rerun-sdk, iopath, fsspec). Check these when debugging version conflicts.

## Gotchas

- **Never use pip** — all dependency management goes through Pixi
- **`hf download` not `huggingface-cli`** — conda's huggingface_hub provides `hf`, not `huggingface-cli`
- **Download tasks are idempotent** — they check `test -d data/...` before downloading
- **gradio from PyPI, not conda** — conda's gradio package has missing transitive deps
- **`moge` needs no-build-isolation** — it requires torch at build time (configured in `[pypi-options]`)
- **sam3d-body-rerun uses `tool/` (singular)** not `tools/` for its CLI scripts
- **Dev tasks use `$PACKAGE_DIR`** — each package feature sets `PACKAGE_DIR` via activation env, dev tasks `cd $PACKAGE_DIR` before running
