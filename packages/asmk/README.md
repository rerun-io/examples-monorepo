# asmk (pixi-build workspace package)

Builds [jenicek/asmk](https://github.com/jenicek/asmk) from source as a conda package using `pixi-build-rattler-build`.

ASMK (Aggregated Selective Match Kernels) is an image retrieval method used by MASt3R-SLAM for loop closure detection. It includes a Cython C extension for fast Hamming distance computation.

## How it works

The `recipe.yaml` fetches the upstream git repo at a pinned commit and applies two patches before building:

| Patch | Purpose |
|-------|---------|
| `add-pyproject-toml.patch` | Creates `pyproject.toml` (upstream has none) with empty `dependencies` — faiss comes from conda, numpy/pyaml from the pixi environment |
| `simplify-setup-py.patch` | Removes the `InstallWrapper` class (which checked for faiss at install time) and `install_requires`, keeping only the `ext_modules` for the Cython C extension |

### Cython regeneration

The upstream `cython/hamming.c` was generated with an older Cython version that references `longintrepr.h`, which moved to `cpython/longintrepr.h` in CPython 3.11+. The build script regenerates `hamming.c` from `hamming.pyx` using a modern Cython to fix this.

## Produced artifacts

- `asmk/hamming.so` — Cython C extension for Hamming distance
- `asmk/` Python package (asmk_method.py, inverted_file.py, codebook.py, etc.)

## Upstream

- **Repo:** https://github.com/jenicek/asmk
- **Pinned commit:** `2a96d9c0`
- **Paper:** [Learning and Aggregating Deep Local Descriptors for Instance-Level Recognition (ECCV 2020)](https://arxiv.org/abs/2005.10343)
