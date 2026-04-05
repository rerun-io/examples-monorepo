# lietorch (pixi-build workspace package)

Builds [princeton-vl/lietorch](https://github.com/princeton-vl/lietorch) from source as a conda package using `pixi-build-rattler-build`.

LieTorch generalizes PyTorch to 3D transformation groups (SE3, Sim3, SO3, etc.) with CUDA-accelerated backends.

## How it works

The `recipe.yaml` fetches the upstream git repo at a pinned commit and applies two patches before compiling:

| Patch | Purpose |
|-------|---------|
| `setup.py.patch` | Wraps `torch` import in a `build_ext` guard (so pixi can extract metadata without torch) and uses conda-provided eigen headers instead of a vendored eigen submodule |
| `pyproject.toml.patch` | Strips `torch`/`numpy` from build-requires and runtime deps (provided by the pixi environment) |

## GPU arch auto-detection

The build script detects the current GPU's compute capability via `nvidia-smi` and only compiles for that architecture, making first builds much faster (~7 min for one arch vs ~30+ min for all six).

## Produced artifacts

- `lietorch_backends.so` — Core Lie group CUDA kernels
- `lietorch_extras.so` — Additional kernels (altcorr, corr_index, se3_builder, se3_solver)
- `lietorch/` Python package (groups.py, broadcasting.py, etc.)

## Upstream

- **Repo:** https://github.com/princeton-vl/lietorch
- **Pinned commit:** `e7df8655`
- **Paper:** [Tangent Space Backpropagation for 3D Transformation Groups (CVPR 2021)](https://arxiv.org/abs/2103.12032)
