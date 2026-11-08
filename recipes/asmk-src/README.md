# asmk (pixi-build workspace package)

Builds [jenicek/asmk](https://github.com/jenicek/asmk) from source as a conda package using `pixi-build-rattler-build`.

ASMK (Aggregated Selective Match Kernels) is an image retrieval method used by MASt3R-SLAM for loop closure detection. Its Cython C extension for Hamming distances (`asmk.hamming`) is not built: `packages/asmk-rs` ports it to Rust (`asmk_rs`, bit-exact), and the package is pure Python.

## How it works

The `recipe.yaml` fetches the upstream git repo at a pinned commit and applies two patches before building:

| Patch | Purpose |
|-------|---------|
| `add-pyproject-toml.patch` | Creates `pyproject.toml` (upstream has none) with empty `dependencies` — faiss comes from conda, numpy/pyaml from the pixi environment |
| `hamming-from-asmk-rs.patch` | Deletes `setup.py` (with the extension gone it only duplicated `pyproject.toml`, and it carried the faiss install check) and adds `asmk/hamming.py`, which re-exports `asmk_rs`, so `kernel.py` and upstream's `test/test_hamming.py` work unchanged |

## Produced artifacts

- `asmk/hamming.py` — re-exports `asmk_rs` (packages/asmk-rs, from the consuming feature)
- `asmk/` Python package (asmk_method.py, inverted_file.py, codebook.py, etc.)

## Upstream

- **Repo:** https://github.com/jenicek/asmk
- **Pinned commit:** `2a96d9c0`
- **Paper:** [Learning and Aggregating Deep Local Descriptors for Instance-Level Recognition (ECCV 2020)](https://arxiv.org/abs/2005.10343)
