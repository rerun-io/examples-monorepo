# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Pixi workspace monorepo** of computer vision projects. Each package lives in `packages/<name>/` with its own Python module, CLI tools, and tests. All Pixi configuration (deps, tasks, environments) lives in the root `pixi.toml` — per-package `pyproject.toml` files only have standard Python packaging metadata.

## Packages & Environments

Each package has a **prod** environment (for running demos/apps) and a **dev** environment (adds ruff, pytest, beartype, pyrefly):

| Package | Prod env | Dev env | Module path | GPU |
|---|---|---|---|---|
| monoprior | `monoprior` | `monoprior-dev` | `packages/monoprior/monopriors/` | CUDA 12.9 |
| prompt-da | `prompt-da` | `prompt-da-dev` | `packages/prompt-da/src/rerun_prompt_da/` | CUDA 12.9 |
| wilor-nano | `wilor` | `wilor-dev` | `packages/wilor-nano/src/wilor_nano/` | CUDA 12.9 |
| sam3d-body-rerun | `sam3d` | `sam3d-dev` | `packages/sam3d-body-rerun/src/sam3d_body/` | CUDA 12.9 |
| sam3-rerun | `sam3-rerun` | `sam3-rerun-dev` | `packages/sam3-rerun/src/sam3_rerun/` | CUDA 12.9 |
| robocap-slam | `robocap` | `robocap-dev` | `packages/robocap-slam/robocap_slam/` | None (CPU) |
| pysfm | `pysfm` | `pysfm-dev` | `packages/pysfm/pysfm/` | CUDA 12.9 |
| vistadream | `vistadream` | `vistadream-dev` | `packages/vistadream/src/vistadream/` | CUDA 12.9 |
| gsplat-rust-renderer | `gsplat-rust-renderer` | `gsplat-rust-renderer-dev` | `packages/gsplat-rust-renderer/src/` | CUDA 12.9 |
| pyvrs-viewer | `pyvrs-viewer` | `pyvrs-viewer-dev` | `packages/pyvrs-viewer/src/pyvrs_viewer/` | None (CPU) |
| mast3r-slam | `mast3r-slam` | `mast3r-slam-dev` | `packages/mast3r-slam/mast3r_slam/` | CUDA 12.9 |

## Direnv Integration

Each package directory has an `.envrc` that auto-activates the correct `*-dev` pixi environment via [direnv](https://direnv.net/). This eliminates the need for `-e` and `--frozen` flags on every command.

### Setup (one-time)

```bash
pixi global install direnv                    # install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # or: echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.bashrc                               # restart shell or source your rc file
direnv allow                                   # approve root .envrc
for d in packages/*/; do (cd "$d" && direnv allow); done  # approve all packages
```

### Usage

```bash
cd packages/robocap-slam/   # direnv auto-activates robocap-dev
python --version             # → 3.10.x (robocap's pinned Python)
ruff check .                 # just works, no pixi run -e needed
pytest -q                    # just works

cd ../../packages/pysfm/     # direnv switches to pysfm-dev
python --version             # → 3.12.x
```

Each `.envrc` defaults to the `*-dev` environment. To override (e.g., use prod env), create a `.envrc.local` (gitignored):
```bash
# packages/robocap-slam/.envrc.local
PIXI_ENV=robocap
```

### How it works

- Root `.envrc` activates the lightweight `dev` environment (ruff, pytest, pyrefly only)
- Each `packages/<name>/.envrc` activates that package's `*-dev` environment with `--frozen` and `--manifest-path ../..`
- `PIXI_DEV_MODE`, `PACKAGE_DIR`, `CONDA_PREFIX` etc. are all set automatically
- Environment deactivates when you `cd` out

## Common Commands

With direnv active, run dev tools directly from a package directory:

```bash
cd packages/monoprior/
ruff check .          # lint
pytest -q             # test
pyrefly check .       # typecheck
python tools/demos/relative_depth.py --image-path data/examples/single-image/room.jpg
```

The `pixi run -e` workflow still works from the repo root and is needed for tasks with `depends-on` chains (like download tasks):

```bash
# From repo root — pixi run tasks still work
pixi run -e monoprior --frozen relative-depth     # runs download + demo
pixi run -e robocap-dev --frozen lint
pixi run -e robocap-dev --frozen tests
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
- **Direnv fails after changing `pixi.toml`** — `.envrc` uses `--frozen`, so run `pixi install -e <name>-dev` (or `pixi install -a`) to re-solve, then direnv picks up the updated lockfile automatically
- **Never use bare `except Exception` with beartype** — beartype raises `BeartypeDoorHintViolation` which inherits from `Exception`. A catch-all `except Exception` will silently swallow type annotation errors, making bugs invisible in dev mode. Either catch a specific exception type (e.g. `except torch.linalg.LinAlgError`) or re-raise beartype violations:
  ```python
  # BAD — silently swallows beartype violations
  try:
      result = some_typed_function()
  except Exception:
      print("failed")

  # GOOD — let beartype violations propagate
  from beartype.roar import BeartypeException
  try:
      result = some_typed_function()
  except BeartypeException:
      raise  # always re-raise beartype errors
  except Exception:
      print("failed")
  ```
- **Use `0.0` not `0` for float annotations** — beartype strictly distinguishes `int` from `float`. `last_error: float = 0` will fail; use `last_error: float = 0.0`
- **Pixi collapses multiline `cmd = """..."""` into a single line**, replacing newlines with spaces. If a task has separate commands on different lines (e.g. `export`, `echo`, `python`), they become arguments to the first command and never execute. The task appears to succeed (exit 0) but produces no output. Always use `&&`-chained single-line commands or `\` line continuations instead.
