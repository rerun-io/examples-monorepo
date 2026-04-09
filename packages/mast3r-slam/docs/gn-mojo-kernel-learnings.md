# GN Kernel Learnings

This note is the compact handoff for the remaining GN backend in this branch.
The intended state is:

- keep the original CUDA backend as the oracle
- keep the idiomatic Mojo shared-library GN path as the only alternative implementation
- do not keep the MAX custom-op experiment in-tree
- keep one representative synthetic GN rays test that matches the real `base` graph shape

Important limitation:

- only the rays GN path is fully implemented in Mojo today
- calibrated and points GN still depend on the CUDA backend step kernels

## What the CUDA kernel is doing

The CUDA backend lives behind:

- [`gn.h`](../mast3r_slam/backend/include/gn.h)
- [`gn.cpp`](../mast3r_slam/backend/src/gn.cpp)
- [`gn_kernels.cu`](../mast3r_slam/backend/src/gn_kernels.cu)

For `gauss_newton_rays`, the important structure is:

- Python/C++ passes `Twc`, `Xs`, `Cs`, `ii`, `jj`, `idx_ii2jj`, `valid_match`, and `Q`
- the ray-step kernel accumulates per-edge partial Hessian and gradient blocks
- the public GN loop reduces those partials, solves the dense step, and retracts poses

Conceptually, each edge contributes:

- `Hs[0]`: ii/ii block
- `Hs[1]`: ii/jj block
- `Hs[2]`: jj/ii block
- `Hs[3]`: jj/jj block
- `gs[0]`: ii gradient
- `gs[1]`: jj gradient

The CUDA path remains the correctness oracle.

## What the idiomatic Mojo kernel is doing

The surviving Mojo implementation lives in:

- [`gn.mojo`](../mast3r_slam/backend/mojo/gn.mojo)
- [`gn_kernels.mojo`](../mast3r_slam/backend/mojo/gn_kernels.mojo)
- [`python_interop.mojo`](../mast3r_slam/backend/mojo/python_interop.mojo)

The key decisions that worked:

- keep the Python-to-Mojo unsafe boundary centralized in `python_interop.mojo`
- keep the public GN loop in `gn.mojo`
- keep the hot ray-step math in `gn_kernels.mojo`
- use the CUDA kernel structure as the algorithmic reference
- use one GPU block per edge-partial and reduce inside the block
- keep the representative acceptance surface on synthetic GN inputs shaped like the real `base` workload

The current scope is intentionally narrow:

- `gauss_newton_rays` is the real Mojo implementation
- `gauss_newton_points` and `gauss_newton_calib` still call CUDA step kernels underneath

The idiomatic Mojo path is readable enough because:

- the interop and kernel layers are split cleanly
- the pose update and dense solve logic remain explicit in `gn.mojo`
- the kernel math is structured around the same geometric steps as CUDA

## Why the MAX kernel failed

The failed MAX path tried to build a fully isolated PyTorch custom op around the same GN rays step.

The experiment failed for a few concrete reasons:

1. The package boundary was more fragile than the shared-lib path.
   The custom-op package needed its own package layout, cache handling, and stricter input dtypes.

2. The op boundary was sensitive to tensor dtypes.
   `ii`, `jj`, and `idx_ii2jj` needed to be `int64`, and `valid_match` needed to be `uint8`.

3. Launch-style custom ops were unstable in this repo context.
   Simple zeroing launch ops could be made to work, but the real multi-input GN launch path was brittle.

4. The pure `foreach` custom-op path compiled and ran, but the real GN math still produced `NaN`s.
   Probe ops were finite, which means the boundary itself was not the main issue anymore. The failure moved inside the GN math under the MAX custom-op execution model.

5. The MAX path increased complexity without beating the shared-lib Mojo path.
   The repo already had a working idiomatic Mojo kernel. The MAX path never reached a clean validated state that justified keeping it.

## What to keep testing

Keep one synthetic GN rays test:

- use the real `base`-like shape: `10` poses, `512x512`, `16` undirected edges, sparse valid mask
- compare the idiomatic Mojo result against CUDA
- check both `dx` and final `Twc`

That test lives in:

- [`test_mojo_vs_cuda.py`](../tests/test_mojo_vs_cuda.py)

## Practical recommendation

If more GN work is needed later:

- treat CUDA as the oracle
- work only on the idiomatic Mojo shared-lib path
- do not resurrect the MAX custom-op path unless there is a new reason to change the Python-to-kernel boundary
