# Examples Monorepo

A pixi workspace containing multiple computer vision example projects, each with its own isolated environment and dependencies.

## Packages

| Package | Prod env | Dev env | Python | GPU | Description |
|---|---|---|---|---|---|
| [monoprior](packages/monoprior/) | `monoprior` | `monoprior-dev` | 3.12 | CUDA 12.9 | Monocular geometric priors (depth, normals) |
| [prompt-da](packages/prompt-da/) | `prompt-da` | `prompt-da-dev` | 3.12 | CUDA 12.9 | Prompt Depth Anything — depth completion on Polycam data |
| [wilor-nano](packages/wilor-nano/) | `wilor` | `wilor-dev` | 3.12 | CUDA 12.9 | Hand pose estimation |
| [sam3d-body-rerun](packages/sam3d-body-rerun/) | `sam3d` | `sam3d-dev` | 3.12 | CUDA 12.9 | 3D body segmentation with SAM + Rerun |
| [sam3-rerun](packages/sam3-rerun/) | `sam3-rerun` | `sam3-rerun-dev` | 3.12 | CUDA 12.9 | SAM3 video segmentation with Rerun |
| [robocap-slam](packages/robocap-slam/) | `robocap` | `robocap-dev` | 3.10 | None | Multi-camera visual odometry & SLAM |
| [pysfm](packages/pysfm/) | `pysfm` | `pysfm-dev` | 3.12 | CUDA 12.9 | COLMAP SfM reconstruction with Rerun + Gradio |
| [vistadream](packages/vistadream/) | `vistadream` | `vistadream-dev` | 3.12 | CUDA 12.9 | Single-image 3D reconstruction via 3D Gaussians |

## Quick start

```bash
pixi install -e robocap            # Install prod environment
pixi run -e robocap robocap-track  # Run a demo task

pixi install -e robocap-dev        # Install dev environment (adds ruff, pytest, beartype, pyrefly)
pixi run -e robocap-dev lint       # Lint
pixi run -e robocap-dev typecheck  # Typecheck
pixi run -e robocap-dev tests      # Run tests
```

### CUDA driver troubleshooting

GPU environments require CUDA 12.9. If `pixi install` fails with `Virtual package '__cuda >=12.9' does not match`, your NVIDIA driver is too old.

**Workaround** (no reboot):
```bash
export CONDA_OVERRIDE_CUDA=12.9
pixi install -a
```

**Proper fix** — upgrade your NVIDIA driver to one that reports CUDA 12.9+ (check with `nvidia-smi`). On Ubuntu:
```bash
sudo apt install nvidia-driver-590-open && sudo reboot
```

## Listing tasks

Use `-e` to filter tasks by environment (recommended — without it you see all tasks):

```bash
pixi task list -e robocap       # Demo/app tasks only
pixi task list -e robocap-dev   # Demo/app tasks + lint, typecheck, tests
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
[feature.cuda]                  # CUDA 12.9 toolkit + PyTorch GPU
[feature.dev]                   # ruff, pytest, beartype, pyrefly, hypothesis + PIXI_DEV_MODE=1
[feature.ide]                   # Standalone IDE/editor tooling (Python 3.12)

[feature.monoprior]             # Package-specific: conda deps, pypi deps, tasks, PACKAGE_DIR
[feature.prompt-da]             #   "
[feature.wilor]                 #   "
[feature.sam3d]                 #   "
[feature.sam3-rerun]            #   "
[feature.robocap]               #   "
[feature.pysfm]                 #   "
[feature.vistadream]            #   "

[environments]                  # Compose features into named environments (prod + dev per package)
```

### Features

**`common`** — Shared dependencies installed in every environment (conda + pypi):

| Conda | PyPI |
|---|---|
| av, rerun-sdk (==0.30.2), py-opencv, numpy, pyserde, jaxtyping, typing_extensions, hf-transfer, scipy, huggingface_hub, tqdm, transformers | gradio (==6.8.0), gradio-rerun (==0.30.2), simplecv (git) |

**`cuda`** — CUDA 12.9 toolkit with the full set of libraries (compiler, cudnn, cublas, etc.) and PyTorch GPU (`>=2.8.0`). Attached to environments that need GPU. Individual packages can tighten the PyTorch floor in their own feature (e.g. vistadream pins `>=2.10.0`).

**`dev`** — ruff, pytest, beartype, pyrefly, hypothesis, types-tqdm. Sets `PIXI_DEV_MODE=1` via activation env. Provides unified `lint`, `typecheck`, and `tests` tasks that use `$PACKAGE_DIR` (set by each package feature).

**Per-package features** (`monoprior`, `prompt-da`, `wilor`, `sam3d`, `sam3-rerun`, `robocap`, `pysfm`, `vistadream`) contain:
- Python version pin
- Deps that **tighten** common's loose versions (e.g. `av = ">=15.1.0,<16"` narrows common's `"*"`)
- Deps **not** in common (package-specific libraries)
- Editable install of the package itself
- Git/PyPI dependencies (simplecv, torch, etc.)
- Demo/app tasks with `cwd = "packages/<dir>"` for correct path resolution
- `PACKAGE_DIR` activation env (used by dev tasks)

### Environments and solve-groups

Each package has a **prod** and **dev** environment sharing the same **solve-group**:

```toml
[environments]
dev            = { features = ["ide", "dev"],                                        solve-group = "dev",        no-default-feature = true }
monoprior      = { features = ["common", "cuda", "monoprior"],                       solve-group = "monoprior",  no-default-feature = true }
monoprior-dev  = { features = ["common", "cuda", "monoprior", "dev"],                solve-group = "monoprior",  no-default-feature = true }
prompt-da      = { features = ["common", "cuda", "prompt-da", "prompt-da-demo"],     solve-group = "prompt-da",  no-default-feature = true }
prompt-da-dev  = { features = ["common", "cuda", "prompt-da", "dev"],                solve-group = "prompt-da",  no-default-feature = true }
wilor          = { features = ["common", "cuda", "wilor"],                           solve-group = "wilor",      no-default-feature = true }
wilor-dev      = { features = ["common", "cuda", "wilor", "dev"],                    solve-group = "wilor",      no-default-feature = true }
sam3d          = { features = ["common", "cuda", "sam3d"],                           solve-group = "sam3d",      no-default-feature = true }
sam3d-dev      = { features = ["common", "cuda", "sam3d", "dev"],                    solve-group = "sam3d",      no-default-feature = true }
sam3-rerun     = { features = ["common", "cuda", "sam3-rerun"],                      solve-group = "sam3-rerun", no-default-feature = true }
sam3-rerun-dev = { features = ["common", "cuda", "sam3-rerun", "dev"],               solve-group = "sam3-rerun", no-default-feature = true }
robocap        = { features = ["common", "robocap"],                                 solve-group = "robocap",    no-default-feature = true }
robocap-dev    = { features = ["common", "robocap", "dev"],                          solve-group = "robocap",    no-default-feature = true }
pysfm          = { features = ["common", "cuda", "pysfm"],                           solve-group = "pysfm",      no-default-feature = true }
pysfm-dev      = { features = ["common", "cuda", "pysfm", "dev"],                   solve-group = "pysfm",      no-default-feature = true }
vistadream     = { features = ["common", "cuda", "vistadream"],                      solve-group = "vistadream", no-default-feature = true }
vistadream-dev = { features = ["common", "cuda", "vistadream", "dev"],               solve-group = "vistadream", no-default-feature = true }
```

`no-default-feature = true` means the root `[dependencies]` (empty) is not included — each env is fully defined by its features.

### Per-package pyproject.toml

Each package's `pyproject.toml` lists only deps **not** covered by the common feature. A comment at the top of the deps list reminds you what common provides:

```toml
dependencies = [
    # common: numpy, scipy, jaxtyping, rerun-sdk, gradio, gradio-rerun, simplecv,
    #         huggingface_hub, hf-transfer, tqdm, pyserde, av, opencv, transformers
    "pyyaml",
    "pyarrow",
    "tyro",
]
```

### Dependency override patterns

Per-feature deps can **tighten** common's loose versions for their solve-group. For example:

- Common: `jaxtyping = "<0.3"`, `av = "*"`
- monoprior overrides: `av = ">=15.1.0,<16"` (tightens common's loose `"*"`)
- robocap overrides: `jaxtyping = ">=0.3.7,<0.4"` (different major version, separate solve-group)
- Shared cuda: `pytorch-gpu = ">=2.8.0"` (loose floor); vistadream tightens to `>=2.10.0`

This works because each env has its own solve-group — the solver combines the constraints independently. Note that features compose by **intersection**, so a per-package feature can only tighten a shared constraint, never relax it. If a package needs an *older* version, the shared feature's floor must be low enough to allow it (e.g. pysfm needs pytorch <2.10 due to pycolmap native lib conflicts, so the shared cuda floor is `>=2.8.0`).

### pypi-options (workspace-level)

```toml
[pypi-options]
no-build-isolation = ["moge", "gsplat"]   # These need access to torch at build time

[pypi-options.dependency-overrides]
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
pixi run -e monoprior polycam-inference       # Downloads data first
pixi run -e prompt-da polycam-prompt_da       # Depth completion (downloads data first)
pixi run -e wilor wilor-image-example
pixi run -e sam3d sam3d-app                   # Gradio UI
pixi run -e robocap robocap-track-slam        # Downloads data first
pixi run -e pysfm sfm-reconstruction-demo     # COLMAP SfM (downloads data first)
pixi run -e pysfm sfm-reconstruction-app      # Gradio UI
pixi run -e vistadream example                # Single-image 3D reconstruction
pixi run -e vistadream gradio-app             # Gradio UI
```

### Lint, typecheck & test

All dev tasks use unified names in `*-dev` environments:

```bash
pixi run -e monoprior-dev lint
pixi run -e monoprior-dev typecheck
pixi run -e monoprior-dev tests

pixi run -e wilor-dev lint
pixi run -e robocap-dev typecheck
```

## Project structure

```
pixi.toml                         # Workspace manifest (features, envs, tasks, deps)
pyrefly.toml                      # Root pyrefly config (single source of truth for IDE + tasks)
.gitignore                        # Merged patterns from all packages
packages/
  monoprior/
    pyproject.toml                # [project] + build backend + [tool.ruff] (no per-package pyrefly)
    monopriors/                   # Python package (includes monopriors/data/ submodule)
    tools/                        # CLI scripts (demos/, apps/)
    tests/
  prompt-da/
    pyproject.toml
    src/rerun_prompt_da/
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
    tests/
  sam3-rerun/
    pyproject.toml
    src/sam3_rerun/
    tools/                        # CLI scripts (demos/, apps/)
    tests/
  robocap-slam/
    pyproject.toml
    robocap_slam/                 # Python package (includes robocap_slam/data/ submodule)
    tools/
    tests/
  pysfm/
    pyproject.toml
    pysfm/                        # Python package (apis/, gradio_ui/)
    tools/                        # CLI scripts (demos/, nodes/, brush/)
    tests/
  vistadream/
    pyproject.toml
    src/vistadream/
    tools/                        # CLI scripts
    tests/
```

`pyrefly` is configured only at the repo root. Package-local `pyproject.toml` files should not
define `[tool.pyrefly]`; per-package typecheck tasks point `pyrefly` at the root `pyrefly.toml`
explicitly.

## Adding a new package

1. Copy source files into `packages/<name>/` (exclude `.git/`, `.pixi/`, `pixi.lock`)
2. Strip all `[tool.pixi.*]` sections from its `pyproject.toml`
3. Remove deps from `pyproject.toml` that are already in the common feature
4. Add a new feature in root `pixi.toml`:
   - `[feature.<name>.dependencies]` — Python pin + package-specific conda deps
   - `[feature.<name>.pypi-dependencies]` — editable install + git deps
   - `[feature.<name>.activation.env]` — set `PACKAGE_DIR = "packages/<name>"`
   - `[feature.<name>.tasks.*]` — demo/app tasks with `cwd = "packages/<name>"`
   - (Optional) `[feature.<name>] platforms = ["linux-64"]` — restrict to specific platforms if needed
5. Add two environments in `[environments]`:
   - `<name> = { features = ["common", "cuda", "<name>"], solve-group = "<name>", no-default-feature = true }`
   - `<name>-dev = { features = ["common", "cuda", "<name>", "dev"], solve-group = "<name>", no-default-feature = true }`
6. Add `packages/<name>/data/` to `.gitignore`
7. Run `pixi install -e <name>-dev` to verify
