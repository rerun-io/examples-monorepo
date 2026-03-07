# Examples Monorepo

A pixi workspace containing multiple computer vision example projects, each with its own isolated environment and dependencies.

## Packages

| Package | Environment | Python | GPU | Description |
|---|---|---|---|---|
| [monoprior](packages/monoprior/) | `monoprior` | 3.12 | CUDA 12.8 | Monocular geometric priors (depth, normals) |
| [wilor-nano](packages/wilor-nano/) | `wilor` | 3.12 | CUDA 12.8 | Hand pose estimation |
| [sam3d-body-rerun](packages/sam3d-body-rerun/) | `sam3d` | 3.12 | CUDA 12.9 | 3D body segmentation with SAM + Rerun |
| [robocap-slam](packages/robocap-slam/) | `robocap` | 3.10 | None | Multi-camera visual odometry & SLAM |

## Quick start

```bash
pixi install -e robocap        # Install one environment
pixi run -e robocap robocap-track  # Run a task
```

## Listing tasks

Use `-e` to filter tasks by environment (recommended — without it you see all ~30 tasks):

```bash
pixi task list -e robocap
pixi task list -e sam3d
```

## Running from a package subdirectory

If you `cd` into a package directory, use `-m` to point pixi back to the workspace root:

```bash
cd packages/sam3d-body-rerun
pixi run -m ../../ -e sam3d python tool/gradio_sam3d_body.py
```

## Architecture

### How pixi.toml is organized

All pixi configuration lives in the root `pixi.toml`. Per-package `pyproject.toml` files only contain standard Python packaging metadata (`[project]`, `[build-system]`, `[tool.ruff]`, etc.) — no `[tool.pixi.*]` sections.

The root `pixi.toml` is structured around **features** that compose into **environments**:

```
[workspace]                     # Channels, platforms, pixi version
[pypi-options]                  # Workspace-level: no-build-isolation, dependency-overrides

[feature.common]                # Shared base deps (all envs get these)
[feature.cuda128]               # CUDA 12.8 toolkit
[feature.cuda129]               # CUDA 12.9 toolkit
[feature.dev-tools]             # ruff, pytest, beartype, pyrefly, hypothesis

[feature.monoprior]             # Package-specific: conda deps, pypi deps, tasks
[feature.wilor]                 #   "
[feature.sam3d]                 #   "
[feature.robocap]               #   "

[environments]                  # Compose features into named environments
```

### Features

**`common`** — Shared dependencies installed in every environment (conda + pypi):

| Conda | PyPI |
|---|---|
| av, rerun-sdk (>=0.28.1,<0.29), py-opencv, numpy, pyserde, jaxtyping, hf-transfer, scipy, huggingface_hub, tqdm | gradio, gradio-rerun |

**`cuda128`** / **`cuda129`** — Named CUDA features with the full toolkit (compiler, cudnn, cublas, etc.). Attached to environments that need GPU. Adding a future CUDA version is a new root feature, not a per-package change.

**`dev-tools`** — ruff, pytest, beartype, pyrefly, hypothesis, types-tqdm. Included in every environment so there's no need for separate dev environments.

**Per-package features** (`monoprior`, `wilor`, `sam3d`, `robocap`) contain:
- Python version pin
- Deps that **override** common (e.g. `jaxtyping = "<0.3.0"` tightens common's `"*"`)
- Deps **not** in common (package-specific libraries)
- Editable install of the package itself
- Git/PyPI dependencies (simplecv, torch, etc.)
- All tasks with `cwd = "packages/<dir>"` for correct path resolution

### Environments and solve-groups

Each environment has its own **solve-group**, so dependency versions are resolved independently:

```toml
[environments]
monoprior = { features = ["common", "cuda128", "monoprior", "dev-tools"], solve-group = "monoprior", no-default-feature = true }
wilor     = { features = ["common", "cuda128", "wilor",     "dev-tools"], solve-group = "wilor",     no-default-feature = true }
sam3d     = { features = ["common", "cuda129", "sam3d",     "dev-tools"], solve-group = "sam3d",     no-default-feature = true }
robocap   = { features = ["common",            "robocap",   "dev-tools"], solve-group = "robocap",   no-default-feature = true }
```

`no-default-feature = true` means the root `[dependencies]` (empty) is not included — each env is fully defined by its features.

### Per-package pyproject.toml

Each package's `pyproject.toml` lists only deps **not** covered by the common feature. A comment at the top of the deps list reminds you what common provides:

```toml
dependencies = [
    # common: numpy, scipy, jaxtyping, rerun-sdk, gradio, gradio-rerun,
    #         huggingface_hub, hf-transfer, tqdm, pyserde, av, opencv
    "pyyaml",
    "pyarrow",
    "tyro",
]
```

### Dependency override patterns

Per-feature deps can **tighten** common's loose versions for their solve-group. For example:

- Common: `jaxtyping = "*"`, `av = "*"`
- monoprior overrides: `jaxtyping = ">=0.2.36,<0.3"`, `av = ">=15.1.0,<16"`
- robocap overrides: `jaxtyping = ">=0.3.7,<0.4"` (different major version)

This works because each env has its own solve-group — the solver combines the constraints independently.

### pypi-options (workspace-level)

```toml
[pypi-options]
no-build-isolation = ["moge"]   # These need access to torch at build time

[pypi-options.dependency-overrides]
gradio = ">=6.0.0,<7"    # Override monopriors' gradio==5.33.1 pin
rerun-sdk = ">=0.28.1"   # Override transitive pins
iopath = ">=0.1.10"      # Override transitive iopath pins
fsspec = ">=2025.3"       # Override transitive pins
```

### Why gradio is from PyPI (not conda)

The conda `gradio` package is missing transitive deps (e.g. `pytz`) that the pip version correctly resolves. To avoid patching missing deps, gradio is installed from PyPI via `[feature.common.pypi-dependencies]`.

### Why `hf` not `huggingface-cli`

The conda `huggingface_hub` package provides the `hf` binary, not `huggingface-cli` (which comes from pip). All download tasks use `hf download` instead.

## Common workflows

### Run demos

```bash
pixi run -e monoprior monoprior-polycam-inference  # Downloads data first
pixi run -e wilor wilor-image-example
pixi run -e sam3d sam3d-app                        # Gradio UI
pixi run -e robocap robocap-track-slam             # Downloads data first
```

### Lint & typecheck

Every package has `<pkg>-lint` and `<pkg>-typecheck` tasks:

```bash
pixi run -e robocap robocap-lint
pixi run -e robocap robocap-typecheck
```

### Run tests

```bash
pixi run -e monoprior monoprior-tests
pixi run -e wilor wilor-tests
```

## Project structure

```
pixi.toml                         # Workspace manifest (features, envs, tasks, deps)
.gitignore                        # Merged patterns from all packages
packages/
  monoprior/
    pyproject.toml                # [project] + [tool.ruff] + [tool.pyrefly] (no pixi config)
    monopriors/                   # Python package (includes monopriors/data/ submodule)
    tools/                        # CLI scripts
    tests/
  wilor-nano/
    pyproject.toml
    src/wilor_nano/
    tools/
    tests/
  sam3d-body-rerun/
    pyproject.toml
    src/sam3d_body/               # Python package (includes sam3d_body/data/ submodule)
    tool/                         # CLI scripts (note: singular "tool")
  robocap-slam/
    pyproject.toml
    robocap_slam/                 # Python package (includes robocap_slam/data/ submodule)
    tools/
```

## Adding a new package

1. Copy source files into `packages/<name>/` (exclude `.git/`, `.pixi/`, `pixi.lock`)
2. Strip all `[tool.pixi.*]` sections from its `pyproject.toml`
3. Remove deps from `pyproject.toml` that are already in the common feature
4. Add a new feature in root `pixi.toml`:
   - `[feature.<name>.dependencies]` — Python pin + package-specific conda deps
   - `[feature.<name>.pypi-dependencies]` — editable install + git deps
   - `[feature.<name>.tasks.*]` — tasks with `cwd = "packages/<name>"`
5. Add an environment in `[environments]` composing common + cuda (if needed) + package + dev-tools
6. Add `packages/<name>/data/` to `.gitignore`
7. Run `pixi install -e <name>` to verify
