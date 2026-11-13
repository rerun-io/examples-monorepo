# vipe-mojo

ViPE's DROID/SLAM CUDA kernels (`vipe_ext`, ViPE @ ead2e00, `csrc/droid_net_ext` and `csrc/slam_ext`) in
Mojo, for CPU and CUDA: `import vipe_mojo as _C`
exposes the six namespaces `vipe.ext` reads (`droid_net_ext`, `grounding_dino_ext`, `utils_ext`,
`slam_ext`, `scatter_ext`, `corr_ext`).

## What is ported

The four functions MetricTrail calls, with vipe_ext's signatures (calls in one pool.mp4 run):

| function | calls | port |
|---|---|---|
| `droid_net_ext.corr_index_forward` | 9,236 | float16 and float32 volumes; c10::Half rounding of every product and sum |
| `droid_net_ext.altcorr_index_forward` | 500 | both CUDA paths: the per-pixel half kernel (C % 8 == 0, C <= 128, `VIPE_ALTCORR_PIX` honoured) and the staged kernel (float32, other half shapes) |
| `slam_ext.frame_distance` | 1,266 | whole function |
| `slam_ext.ba_extended` | 1,549 | every option: `motion_only`, `optimize_intrinsics`, `compute_energy`, `depth_active`, `alpha` / `disps_sens`, stereo edges (ii == jj) |

`ba_extended` is `ba_cuda_impl` line for line. The CUDA kernels reduce with one 256-thread block per edge
(`blockReduce`); the Mojo kernels fold the 256 running sums in the same tree (in shared memory for
`frame_distance` and the scratch-row `block_reduce`), so the summation order is CUDA's and there are no
float atomics. As in the extension, the reduced sums go to the CPU once per iteration and the pose system
is assembled and solved there in float64 (Eigen's `SimplicialLLT` in the extension, a single-threaded Mojo profile
Cholesky here); a failed factorisation gives a zero step, as in the extension.

Integer BA topology is prepared on the CPU after one edge readback and retained in an eight-entry
LRU. Keys include ordered source/target values, pose bounds, solver modes and device. Only index
and assembly tables are reused: every call recomputes residuals and every solve owns its scratch.
No poses, depths, focal values or calibration results are cached.

Every other function of the six namespaces raises `NotImplementedError` when called: the backward passes,
`altcorr_forward`, `ba_extended_v2`, `utils_ext`, `scatter_ext`, `corr_ext`, `grounding_dino_ext`.

Layout: `vipe_mojo/mojo/droid.mojo` (correlation), `slam.mojo` (frame distance, BA, Schur products),
`se3.mojo` (DPVO's SE3 helpers; ViPE's are the same functions), `_vipe_mojo.mojo` (CPython entry point);
`vipe_mojo/droid_net_ext.py` and `slam_ext.py` are the drop-in functions. The runtimes are
lietorch-mojo's, shared, as in `dpvo.mojo_backend`.

## Parity

References: `tests/fixtures/vipe_references.rrd` (6.7 MB), frozen from one MetricTrail run on pool.mp4 on an
RTX 5090: one late call of each function, cut to a small window and recomputed by `vipe_ext` on exactly the
stored inputs (provenance in `tests/vipe_references.py`; vipe_ext and the freeze script are deleted). Error = max |ours - ref| / max(|ref|, 1).

| case | CPU (x86-64) | CUDA (5090) |
|---|---|---|
| corr_index, float16, 4 levels | bit-exact | bit-exact |
| corr_index, float32 | 1.9e-7 | bit-exact |
| altcorr_index, half pix / half staged / float32 | 1.9e-7 / 2.7e-7 / 2.7e-7 | bit-exact |
| frame_distance | 1.2e-7 | 1.5e-7 |
| ba_extended full (window, 24 edges) poses / disps | 1.6e-6 / 7.3e-6 | 1.6e-6 / 6.2e-6 |
| ba_extended energy (global, 127 poses) poses / disps / energy | 3.0e-7 / 8.3e-7 / 6.1e-9 | 3.0e-7 / 8.9e-7 / 4.7e-9 |
| ba_extended intrinsics + energy, poses / disps / focal | 7.8e-7 / 2.5e-7 / 2.7e-7 | 1.2e-6 / 3.7e-7 / 2.0e-7 |

vipe_ext is built with `--use_fast_math` (approximate division, sqrt, sin, cos); the Mojo kernels use
IEEE division and lietorch-mojo's `sqrt_` / `sincos_`. Kernels that only multiply and add are therefore
bit-exact on GPUs; the rest differ at the float32 rounding level.

The BA bound: vipe_ext's BA has no float atomics, so six CUDA repeats agree bit for bit and the repeat
spread is zero. The noise floor is instead vipe_ext's own movement when the poses and disparities it
updates move by at most one float32 ulp (six seeded draws, `*_ulp_spread`). Over the three cases and
two devices the Mojo error is 0.2-0.8x that floor, except the poses of the `energy` case, 2.5x on every
device (3.0e-7 against a floor of 1.2e-7; both below the 1e-5 floor of the other tests). The tests allow
4x the floor, or 1e-5 if larger; the energy within 1e-4.

Not exercised by real data: `motion_only` (MetricTrail on pool.mp4 never calls it). On a synthetic
problem it matches vipe_ext to 6e-7 (poses) and 4e-6 (focal) on CPU and CUDA.

## Build and test

```bash
pixi run -e metrictrail-dev --frozen vipe-mojo-build   # Linux: CPU + CUDA kernels
pixi run -e metrictrail-dev --frozen vipe-mojo-tests   # CPU, plus CUDA when present
```

