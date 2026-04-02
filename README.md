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

### Direnv setup (recommended, one-time)

[direnv](https://direnv.net/) auto-activates the correct pixi environment when you `cd` into a package directory. No more `-e` and `--frozen` flags.

```bash
pixi global install direnv                      # install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc  # or ~/.zshrc for zsh
source ~/.bashrc                                # reload shell
direnv allow                                    # approve root .envrc
for d in packages/*/; do (cd "$d" && direnv allow); done  # approve all packages
```

To reduce output noise, create `~/.config/direnv/direnv.toml`:
```toml
[global]
hide_env_diff = true
```

### Development workflow

With direnv, just `cd` into a package and use tools directly:

```bash
cd packages/robocap-slam/   # direnv activates robocap-dev
python --version             # → 3.10.x
ruff check .                 # just works
pytest -q                    # just works

cd ../../packages/pysfm/     # direnv switches to pysfm-dev
python --version             # → 3.12.x
```

For tasks with `depends-on` chains (like auto-downloading data), use `pixi run` from the repo root:

```bash
pixi run -e monoprior --frozen relative-depth   # downloads data, then runs demo
pixi run -e robocap --frozen robocap-track-slam
pixi run -e pysfm --frozen sfm-reconstruction-demo
```

### Override environment per directory

Each `.envrc` defaults to the `*-dev` environment. To use a different env (e.g., prod), create a `.envrc.local` (gitignored):
```bash
# packages/robocap-slam/.envrc.local
PIXI_ENV=robocap
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

```bash
pixi task list -e robocap       # Demo/app tasks only
pixi task list -e robocap-dev   # Demo/app tasks + lint, typecheck, tests
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

Each package has a **prod** and **dev** environment sharing the same **solve-group** (no extra resolution cost). `no-default-feature = true` means each env is fully defined by its features.

Per-feature deps can **tighten** common's loose versions for their solve-group (e.g., vistadream tightens `pytorch-gpu >= 2.10.0` over common's `>= 2.8.0`). Features compose by intersection, so a per-package feature can only narrow a shared constraint, never relax it.

### pypi-options (workspace-level)

```toml
[pypi-options]
no-build-isolation = ["moge", "gsplat"]   # These need access to torch at build time

[pypi-options.dependency-overrides]
iopath = ">=0.1.10"      # Override transitive iopath pins
fsspec = ">=2025.3"       # Override transitive pins
```

## Project structure

```
pixi.toml                         # Workspace manifest (features, envs, tasks, deps)
pyrefly.toml                      # Root pyrefly config (single source of truth for IDE + tasks)
.envrc                            # Root direnv config (activates dev environment)
packages/
  <name>/
    .envrc                        # Direnv config (activates <name>-dev environment)
    pyproject.toml                # [project] + build backend + [tool.ruff]
    <module>/                     # Python package
    tools/                        # CLI scripts (demos/, apps/)
    tests/
```

Note: `sam3d-body-rerun` uses `tool/` (singular) not `tools/`.

`pyrefly` is configured only at the repo root — do not add `[tool.pyrefly]` to per-package `pyproject.toml` files.

## Adding a new package

1. Create `packages/<name>/` with `pyproject.toml`, source module, `tools/`, `tests/`
2. Add a feature in root `pixi.toml`:
   - `[feature.<name>.dependencies]` — Python pin + package-specific conda deps
   - `[feature.<name>.pypi-dependencies]` — editable install + git deps
   - `[feature.<name>.activation.env]` — set `PACKAGE_DIR = "packages/<name>"`
   - `[feature.<name>.tasks.*]` — demo/app tasks with `cwd = "packages/<name>"`
3. Add two environments in `[environments]`:
   - `<name> = { features = ["common", "cuda", "<name>"], solve-group = "<name>", no-default-feature = true }`
   - `<name>-dev = { features = ["common", "cuda", "<name>", "dev"], solve-group = "<name>", no-default-feature = true }`
4. Create `packages/<name>/.envrc`:
   ```bash
   watch_file ../../pixi.lock
   PIXI_ENV="${PIXI_ENV:-<name>-dev}"
   source_env_if_exists .envrc.local
   eval "$(pixi shell-hook -e $PIXI_ENV --manifest-path ../.. --frozen)"
   ```
5. Add `packages/<name>/data/` to `.gitignore`
6. Run `pixi install -e <name>-dev` to verify

## Gotchas

- **Never use pip** — all dependency management goes through Pixi
- **`hf download` not `huggingface-cli`** — conda's huggingface_hub provides `hf`, not `huggingface-cli`
- **gradio from PyPI, not conda** — conda's gradio package has missing transitive deps
- **`moge`/`gsplat` need no-build-isolation** — they require torch at build time (configured in `[pypi-options]`)
