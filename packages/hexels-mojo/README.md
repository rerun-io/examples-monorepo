# hexels-mojo

Three of the hexels-src CUDA extensions in Mojo. `recipes/hexels-src` no longer builds them; its patches
0002–0004 import these instead:

| Upstream | Here | Used by |
|----------|------|---------|
| fused-ssim `fused_ssim` (rahul-goel/fused-ssim@1272e21, 444 lines CUDA) | `hexels_mojo.fused_ssim` | Image-GS loss (L1 + 0.1 SSIM, every `hexels-fit-2d` step), the 2D scorer (`hexels_bench.score2d`), the 3D trainer with `--use_fused_ssim` |
| simple-knn `distCUDA2` (bkerbl/simple-knn@f155ec0, ~250 lines CUDA + cub) | `hexels_mojo.simple_knn` | 2DGS `GaussianModel.create_from_pcd` (imported when hexels loads 2DGS scenes) |
| `scripts/image_compression_prune_scores.cu` (94 lines, JIT-compiled with nvcc at run time) | `hexels_mojo.prune` | `image_compression_allocation.prune_model` (`hexels-fit-2d --variant rich`, 20K → 10K) |

From hexel-kernel (the 14k-line gsplat fork with the hexel rasterizers), the 2D hexel rasterizers, flat and
color-gradient (forward + backward, scalar / per-hexel / per-edge softness, optional peak and emission), and their
tile binning are Mojo too (`hexels_mojo.raster2d`): they are every hexel-kernel call of Image-GS's 2D hexel fits
(`hexels-fit-2d`, rich and flat). Patch 0005 wraps `gsplat_hexels.py`'s `gsplat` handles in `raster2d.overlay`,
which serves those nine functions and passes every other one to hexel-kernel's CUDA (affine 2D variants). Image-GS's Gaussian projection and top-K rasterizer (the `imagegs-2d` baseline) are `hexels_mojo.gaussians2d`;
patches 0003/0004 import them in place of `gsplat`'s (`rasterize_gaussians_no_tiles`, off in Image-GS's configs, stays
CUDA). hexel-kernel's 3D rasterizer forward (preprocess, tile keys, front-to-back render; every color and view model, no
prefiltering) and its cached backward (per-pixel back-to-front gradients, then the transform's gradient back to means,
scales and quaternions) are `hexels_mojo.raster3d`; patch 0006 overlays them on `gsplat.cuda_hexels3d._C`, so a 3D
training step runs no hexel-kernel CUDA. Still CUDA: prefiltering, the contribution / overdraw / surface / owner
variants and their backward, the uncached backward, and the affine 2D kernels. The plan for the rest: `/mnt/nas/ssd-8tb/scratch/overnight-2026-11-08/ports/HEXEL_KERNEL_PLAN.md`.

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
| 2D rasterizer forward, flat and color-gradient (image, top-10 ids, weights sum, entropy) | bit-exact: the kernel issues nvcc's `--use_fast_math` PTX (`.ftz`, `.approx`) in nvcc's order through inline PTX (`mojo/fastmath.mojo`) |
| 2D rasterizer backward | ≤ 1e-5 relative to max(\|ref\|, 1); float atomics sum in scheduling order |
| tile binning | bit-exact (intersection keys, ids, tile ranges) |
| Image-GS projection (forward + backward), top-K forward | bit-exact (projection backward has no atomics: same PTX order) |
| Image-GS top-K backward | ≤ 1e-5 relative (float atomics) |
| 3D forward (radii, transforms, tile ranges, sorted ids, final transmittance, last index, image) | bit-exact, 3 configurations (`tests/test_raster3d.py`, `fixtures/raster3d_references.rrd`) |
| 3D cached backward (all twelve gradients, incl. the (P, 4) means2d statistic) | ≤ 8e-7 of each tensor's largest magnitude (two CUDA runs: ≤ 8e-7); test bound 1e-5 |

Upstream's SSIM scratch buffer is declared (42, 32) but indexed at column tx + 5, so its last row writes five
floats past the array; the port indexes it at tx (same arithmetic, no overrun).

## Timing (RTX 5090, CUDA events, mean of 50–100 calls)

| Op | CUDA | Mojo |
|----|------|------|
| `fused_ssim` fwd + bwd, 1×3×1080×1920 | 0.577 ms | 0.605 ms |
| `fused_ssim` fwd + bwd, 1×3×512×768 | 0.124 ms | 0.127 ms |
| `removal_scores`, 20K hexels, 1080p | 1.525 ms (+ an nvcc JIT build on first use) | 1.563 ms |
| `distCUDA2`, 100K clustered points | 3.40 ms | 3.16 ms |
| 2D rasterizer, color-gradient: binning + forward + backward, 20K hexels, 2048×2048 | 0.148 + 1.076 + 10.946 ms | 0.124 + 1.170 + 10.452 ms |
| 2D rasterizer, flat: binning + forward + backward, same scene | 0.132 + 1.040 + 6.293 ms | 0.129 + 1.004 + 5.675 ms |
| Image-GS step: projection + binning + forward + backward, 10K Gaussians, 2048×2048 | 8.251 ms | 6.220 ms |
| 3D forward, 300K hexels, 1600×1080, linear color + view gradients (render stage) | 2.03 ms (1.68 ms) | 1.54 ms (1.09 ms) |
| 3D forward, same scene, linear color, no view model (the protocol's training config) | 1.99 ms (1.64 ms) | 1.30 ms (0.85 ms) |
| 3D cached backward, same scene, linear color + view gradients | 27.19 ms | 8.47 ms |
| 3D cached backward, same scene, training config | 17.23 ms | 4.89 ms |

Real run, `hexels-fit-2d --variant rich` on anime-2_2k (5,100 steps, 20K → 10K hexels), two runs each: CUDA
36.18 / 36.08 dB, 85.9 / 88.1 s; every ported kernel on Mojo (rasterizer, binning, SSIM, prune) 36.12 / 36.14 dB,
77.8 / 80.7 s. `--variant flat` (constant colors): CUDA 33.88 dB in 52.9 s, Mojo 33.89 dB in 48.5 s.

Real run, 3D: `hexels.train` on DTU scan105 with the protocol's v1 command at 7,000 iterations (hexels_bench
`prepare_command`), scored on the 8 test views. CUDA: 27.06 / 26.97 dB, SSIM 0.880 / 0.867, 78.3 / 76.7 s. The Mojo forward
(backward still CUDA): 26.90 / 26.97 / 26.94 dB, SSIM 0.872 / 0.875 / 0.879, 81.7 / 82.0 / 76.6 s (the last run on the
reconverging render loop below). Runs differ because the CUDA backward's float atomics are not deterministic. The test
views of one Mojo-trained checkpoint, rendered by the CUDA env and by the Mojo env, are byte-identical PNGs.

Render loops branch with nested `if`s, not `continue` / `break`: with `continue`, LLVM placed the warp's reconvergence
point outside the per-hexel loop, the lanes of a warp drifted onto different hexels, and the 3D render ran at 2.3 ms
instead of 1.1 ms for the same instructions.

The 3D backward walks each tile's hexel list together (CUDA gives each pixel its own walk): every lane of a warp
then holds the same hexel, so its gradients are summed across the warp and written by one atomic instead of up to 32.
That makes it 3.5x faster than CUDA; the sums only change order.
