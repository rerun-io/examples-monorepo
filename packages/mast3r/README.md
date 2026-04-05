# mast3r (pixi-build workspace package)

Builds [naver/mast3r](https://github.com/naver/mast3r) (+ [naver/dust3r](https://github.com/naver/dust3r) and [naver/croco](https://github.com/naver/croco)) from source as a single pure-Python conda package using `pixi-build-rattler-build`.

MASt3R is a 3D reconstruction model that predicts dense 3D pointmaps from image pairs. DUSt3R is its predecessor, and CroCo provides the vision transformer encoder.

## How it works

The `recipe.yaml` fetches mast3r from git and applies patches. The build script then clones dust3r and croco (submodules) at pinned commits, applies additional fixes, and installs everything as one package containing `mast3r`, `dust3r`, and `models` (croco) Python modules.

### Patches applied to mast3r

| Patch | Purpose |
|-------|---------|
| `add-pyproject-toml.patch` | Creates `pyproject.toml` (upstream has none — it's not a pip package) |
| `add-setup-py.patch` | Creates `setup.py` with `find_packages` that discovers mast3r, dust3r, and croco models across the submodule directory layout |
| `fix-path-to-dust3r.patch` | Replaces the `sys.path` hack in `path_to_dust3r.py` with a `try: import dust3r` check — when dust3r is installed as a package, no path manipulation needed |
| `torch-load-weights-only.patch` | Adds `weights_only=False` to `torch.load()` calls in `retrieval/model.py` and `retrieval/processor.py` for PyTorch 2.6+ compatibility |

### Fixes applied to dust3r (via build script)

| File | Purpose |
|------|---------|
| `dust3r-torch-load.patch` | Adds `weights_only=False` to `torch.load()` in `dust3r/model.py` |
| `path_to_croco.py` | Replacement file: same `try: import` pattern as path_to_dust3r, avoids submodule path hack |
| `misc.py` | Replacement file: converts `transpose_to_landscape` closures into picklable classes so the model can be sent to `torch.multiprocessing` subprocesses with `spawn` mode |

### Missing `__init__.py` files

The build script creates `__init__.py` in two directories that upstream treats as namespace packages:
- `dust3r/croco/models/` — needed for `from models.croco import CroCoNet`
- `mast3r/retrieval/` — needed for `from mast3r.retrieval.processor import Retriever`

## curope (not included)

The CUDA RoPE2D kernels from `dust3r/croco/models/curope/` are **not built**. The codebase falls back to a slower pure-PyTorch implementation. See the TODO in `packages/mast3r-slam/README.md` for plans to add curope as a separate workspace package.

## Upstream

- **mast3r:** https://github.com/naver/mast3r @ `f5209afc`
- **dust3r:** https://github.com/naver/dust3r @ `3cc8c88c`
- **croco:** https://github.com/naver/croco @ `d7de0705`
