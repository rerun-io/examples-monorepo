# hexels-mojo

Three of the hexels-src CUDA extensions in Mojo. `recipes/hexels-src` no longer builds them; its patches
0002–0004 import these instead:

| Upstream | Here | Used by |
|----------|------|---------|
| fused-ssim `fused_ssim` (rahul-goel/fused-ssim@1272e21, 444 lines CUDA) | `hexels_mojo.fused_ssim` | Image-GS loss (L1 + 0.1 SSIM, every `hexels-fit-2d` step), the 2D scorer (`hexels_bench.score2d`), the 3D trainer with `--use_fused_ssim` |
| simple-knn `distCUDA2` (bkerbl/simple-knn@f155ec0, ~250 lines CUDA + cub) | `hexels_mojo.simple_knn` | 2DGS `GaussianModel.create_from_pcd` (imported when hexels loads 2DGS scenes) |
| `scripts/image_compression_prune_scores.cu` (94 lines, JIT-compiled with nvcc at run time) | `hexels_mojo.prune` | `image_compression_allocation.prune_model` (`hexels-fit-2d --variant rich`, 20K → 10K) |

hexel-kernel (the 14k-line gsplat fork with the hexel rasterizers) is still CUDA.

```bash
pixi run -e hexels-dev hexels-mojo-build   # Mojo shared library for the local GPU (cubins, no PTX JIT)
pixi run -e hexels-dev hexels-mojo-tests   # parity against the frozen CUDA outputs
```

The kernels are CUDA-only (as upstream) and run on PyTorch's current stream through lietorch-mojo's
runtime (`lietorch_mojo.backends._runtime`), like vipe-mojo.

## Parity

`tests/fixtures/hexels_references.rrd` holds inputs and outputs of the CUDA extensions (torch 2.12 cu130,
RTX 5090), written by `tests/freeze_hexels_references.py` (deleted with the extensions; see this package's first commit).

| Op | Result |
|----|--------|
| `distCUDA2` | bit-exact (uniform, clustered 30K, duplicate points) |
| `fused_ssim` backward | bit-exact given the same derivative maps |
| `fused_ssim` forward | map ≤ 5e-6, mean SSIM ≤ 1e-7, end-to-end gradient ≤ 4e-5 (relative to max(\|ref\|, 1)); d map / d σ1² up to 3.4e-4 on smooth images, where σ² = E[x²] − μ² cancels. nvcc and Mojo's LLVM fuse different multiply-adds in the moments. |
| prune scores | ≤ 2.2e-5 absolute on scores up to 5.6; upstream uses `--use_fast_math` and float atomics. The kept half agrees away from the boundary score. |

Upstream's SSIM scratch buffer is declared (42, 32) but indexed at column tx + 5, so its last row writes five
floats past the array; the port indexes it at tx (same arithmetic, no overrun).

## Timing (RTX 5090, CUDA events, mean of 50–100 calls)

| Op | CUDA | Mojo |
|----|------|------|
| `fused_ssim` fwd + bwd, 1×3×1080×1920 | 0.577 ms | 0.605 ms |
| `fused_ssim` fwd + bwd, 1×3×512×768 | 0.124 ms | 0.127 ms |
| `removal_scores`, 20K hexels, 1080p | 1.525 ms (+ an nvcc JIT build on first use) | 1.563 ms |
| `distCUDA2`, 100K clustered points | 3.40 ms | 3.16 ms |
