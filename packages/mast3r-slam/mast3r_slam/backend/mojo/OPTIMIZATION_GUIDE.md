# Mojo GPU Kernel Optimization Guide for mast3r-slam

## Goal

Replace CUDA/C++ matching kernels (`iter_proj`, `refine_matches`) with Mojo equivalents that match or beat CUDA performance.

**Status (verified 2026-04-05)**: the current Mojo backend is numerically aligned with the CUDA oracle on the checked test suite, but it does **not** consistently beat CUDA on the latest benchmark run. `iter_proj` remains 1.4-1.8x slower, and `refine_matches` is currently near parity on small cases and somewhat slower on the larger hot case.

## Architecture

```
Python caller (mast3r_slam/ops/matching.py, mast3r_slam/ops/optim.py)
  → try: import mast3r_slam_mojo_backends  (preferred, no C++ build)
    except: from mast3r_slam import _backends  (CUDA fallback, matching only)
  → iter_proj_py() / refine_matches_py() / gauss_newton_rays_step_py() / pose_retr_py()
    → Extract shapes, pointers from PyTorch tensors (torch_ptr[dtype])
    → cached_context()[].enqueue_function[kernel]()
    → torch.cuda.synchronize()  (pose_retr leaves this to the Python GN loop)
  → Return PyTorch tensors

The ray-cost Gauss-Newton loop is Python (ops/optim.gauss_newton_rays: Mojo step
kernel → dense CPU solve → Mojo retraction). The points and calib costs use the
CUDA extension (nodes/factor_graph.py).
```

**Build command (current repo workflow):**
```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam-dev _mast3r-slam-build-mojo-kernels
```

**Equivalent direct Mojo invocation (stable Mojo 1.x from the max channel, plus max-core for `max.gpu` and `layout`):**
```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam-dev mojo build --emit shared-lib \
  -o mast3r_slam_mojo_backends.so \
  mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo
```

**Test command (Mojo vs CUDA parity, integration lane):**
```bash
pixi run -e mast3r-slam-dev --frozen python -m pytest tests/test_mojo_vs_cuda.py -m integration -v -s
```

**Benchmark command (500 runs, median, side-by-side):**
```bash
pixi run -e mast3r-slam-dev python tools/bench_kernels.py
```

**Task wiring note:** the `mast3r-slam` example/app/verify tasks depend on `_mast3r-slam-build-mojo-kernels`, so the Mojo shared library is built automatically before those entrypoints run. The CUDA extension (`_mast3r-slam-build-cuda-kernels`) is needed for the calibrated cost and the parity tests.

## Current Performance (RTX 5090, median of 500 runs, verified 2026-04-05)

```
========================================================================
CUDA vs Mojo Matching Kernel Benchmarks
GPU: NVIDIA GeForce RTX 5090
========================================================================
Kernel               Size                 CUDA (ms)      Mojo (ms)      Ratio
------------------------------------------------------------------------
iter_proj            64x64 (4K pts)          0.013         0.021       1.56x (CUDA)
iter_proj            224x224 (50K pts)       0.027         0.039       1.42x (CUDA)
iter_proj            512x512 (262K pts)      0.131         0.229       1.75x (CUDA)
refine_matches       32x32, d=16             0.036         0.037       1.03x (CUDA)
refine_matches       64x64, d=128            0.226         0.246       1.09x (CUDA)
refine_matches       128x128, d=16           0.039         0.050       1.29x (CUDA)
------------------------------------------------------------------------
```

**End-to-end pipeline (100 frames):**
| Config | CUDA | Mojo | Ratio |
|--------|------|------|-------|
| fast (224px) | 33.0s | 33.4s | 1.01x |
| base (512px) | 30.5s | 30.8s | 1.01x |

The matching kernels are a small fraction of pipeline time (dominated by the MASt3R ViT forward pass), so these differences are much more visible in microbenchmarks than in the full application.

## Why CUDA Is Still Faster on `iter_proj`

The gap is **not** in the GPU kernel compute — it's in the dispatch path:

| Step | CUDA (PyBind11 C++) | Mojo (Python ext) |
|------|--------------------|--------------------|
| Function call | Direct C++ call | Python→Mojo interop |
| Shape extraction | `tensor.size()` in C++ | `Int(py=t.shape[i])` × N Python calls |
| Kernel dispatch | `<<<grid,block>>>` | `DeviceContext` + `enqueue_function` |
| Return | **Async** (immediate) | **`torch.cuda.synchronize()`** (blocks ~2 µs) |

The `torch.cuda.synchronize()` is the critical difference. Mojo's `DeviceContext` uses a **different CUDA stream** than PyTorch. Without a device-wide sync, `torch.empty()` on the next call can return a buffer that Mojo's stream is still writing to. This was proven by Hypothesis fuzz tests that caught stale-buffer corruption.

## Overhead Budget (measured individually)

| Component | Cost (µs) | Status |
|-----------|-----------|--------|
| `cached_context()` | ~1 | **Active** — Python module attr lookup |
| `torch.cuda.synchronize()` | ~2 | **Active** — required for cross-stream correctness |
| `torch.empty()` × 2 output allocs | ~2 | **Active** |
| `Int(py=tensor.shape[i])` × 4-5 | ~0.5 | Negligible |
| `enqueue_function` kernel launch | ~2 | **Active** |
| `.contiguous()` + `.data_ptr()` | ~0.2 | Negligible |
| **Total active overhead** | **~8** | vs **~0** for CUDA C++ |

For a 13 µs kernel (iter_proj 64x64), 8 µs overhead = 1.6x. For a 225 µs kernel (refine_matches 64x64), 8 µs overhead = 1.04x. This explains most of the `iter_proj` gap. `refine_matches` can get closer because compute dominates more often, but on the latest verified run it still does not beat CUDA consistently.

## All Optimizations Applied (chronological)

### Round 1: Overhead elimination
1. **Removed `torch.cuda.synchronize()` before kernel** — CUDA stream ordering handles input deps
2. **Replaced `torch.zeros()` with `torch.empty()`** — kernel writes all positions
3. **Added native Float16 kernel** for `refine_matches` — avoids `.float()` copy (~7 µs saved)
4. **Per-kernel block sizes**: dynamic via `choose_iter_proj_block()` / `choose_refine_block()`
5. **Hoisted loop-invariant loads** — target point and batch base offset above LM loop

### Round 2: Async dispatch + SIMD
6. **Removed `ctx.synchronize()`** — Tested: async dispatch works for single calls
7. **SIMD-4 vectorized dot product** in `refine_matches_kernel` (f32 path)
8. **SIMD-8 vectorized dot product** in `refine_matches_kernel_f16` (f16 path)
9. **Eliminated `.to(torch.bool)` cast** — allocate converged as `torch.bool` directly

### Round 3: Cached DeviceContext + specialization (by another agent)
10. **Cached DeviceContext via Python module attribute** — `alloc[DeviceContext](1)` in `PyInit_`, stored as `_ctx_addr` int, retrieved via `get_cached_context_ptr()`
11. **Dynamic block size selection** — `choose_iter_proj_block()` picks 16/64/128 based on `num_pts`
12. **Specialized f16 cached kernel** — `refine_matches_kernel_f16_cached[FDIM, BLOCK_SIZE]` uses shared memory for query descriptors with `comptime for` unrolled dot product

### Round 4: Correctness fixes from Hypothesis testing
13. **Restored `torch.cuda.synchronize()`** — Mojo uses a different CUDA stream than PyTorch. Without device-wide sync, rapid back-to-back calls corrupt output buffers. Proven by Hypothesis: `max_diff=8.0` pixels from stale `torch.empty()` buffers.
14. **Removed `alignment=16` from SIMD loads** — Crashed with `CUDA_ERROR_MISALIGNED_ADDRESS` on non-aligned fdim values (13, 17, etc.). PyTorch tensors aren't guaranteed 16-byte aligned at arbitrary offsets.
15. **Fixed non-deterministic test data** — `torch.randn_like(p_init)` used unseeded global RNG; switched to seeded generator.

### Round 5: Idiomatic refactoring
16. **Extracted `bilinear_sample()`, `bilinear_corners()`, `bilinear_weights()`, `normalize_ray()` helpers** — `@always_inline`, zero runtime cost. Eliminates 30+ lines of duplicated interpolation code.
17. **Added `── Section ──` comments** matching CUDA kernel structure for 1:1 readability.

### Investigated but not viable
18. **MAX `CustomOpLibrary` / `@register` pattern** — MAX's own internal ops (`max/_interpreter_ops/`) use the **exact same `PythonModuleBuilder` + `UnsafePointer` pattern** we use. The `@register`/`InputTensor`/`OutputTensor` API is for the MAX Graph path only. No speed benefit.
19. **Module-level `var` for DeviceContext** — Mojo compile error: "global vars are not supported". Worked around with Python module attribute + `alloc[DeviceContext]`.
20. **Async dispatch without sync** — Correctness failure under rapid-fire calls (Hypothesis). Mojo and PyTorch use different CUDA streams; `torch.empty()` recycles buffers before Mojo's kernel finishes.

## Key Technical Details

### Cached DeviceContext pattern (python_interop.mojo)
```mojo
# In PyInit_: a process-lifetime allocation, its address stored on the module
var ctx = alloc(Layout[DeviceContext](count=1)).unsafe_leak()
ctx.unsafe_write(DeviceContext())
Python.add_object(module, "_ctx_addr", PythonObject(Int(ctx)))

# In each wrapper
def cached_context() raises -> MutPointer[DeviceContext, MutAnyOrigin]:
    var module = Python.import_module("mast3r_slam_mojo_backends")
    return MutPointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(py=module._ctx_addr))
```

### Typed pointer and TileTensor from a PyTorch tensor
```mojo
var ptr = MutPointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(py=tensor.data_ptr()))
var t = TileTensor(ptr, row_major((batch, num_pts, Idx[3])))   # plain Int = runtime dim, Idx[N] = static
```
- The tensor must be contiguous and hold `dtype` (the wrappers call `.contiguous()`, `.float()`, `.half()`).
- Kernels take `TileTensor[dtype, Layout, ImmutAnyOrigin]` for inputs and `MutAnyOrigin` for outputs.

### SIMD loads on GPU: pass the scalar alignment
```mojo
comptime F16_ALIGN = align_of[Float16]()
var v16 = D11.load[width=8, alignment=F16_ALIGN]((b, v, u, k))   # SIMD[DType.float16, 8]
score += (v16.cast[.float32]() * q.cast[.float32]()).reduce_add()
```
**WARNING**: `TileTensor.load[width=8]` defaults to the vector alignment (16 B for f16x8). A PyTorch view can start at any
element, so descriptor loads pass the scalar alignment explicitly. A 16-B assumption caused `CUDA_ERROR_MISALIGNED_ADDRESS`
crashes discovered by Hypothesis fuzz testing.

### Cross-stream synchronization (REQUIRED)
```mojo
ctx_ptr[].enqueue_function[kernel](args, grid_dim=..., block_dim=...)
# MUST sync — Mojo uses a different CUDA stream than PyTorch
Python.import_module("torch").cuda.synchronize()
```
Removing this sync saves ~2 µs but causes **data corruption** under rapid-fire calls (e.g., Hypothesis fuzz tests, or any code that calls the kernel in a tight loop). The `torch.empty()` caching allocator recycles buffers before Mojo's stream finishes writing.

### Block sizes — dynamic per workload
```mojo
def choose_iter_proj_block(num_pts: Int) -> Int:
    if num_pts <= 4096: return 16      # small: minimize overhead
    if num_pts <= 16384: return 64     # medium: balance
    return 128                          # large: maximize occupancy

def choose_refine_block(num_pts: Int) -> Int:
    if num_pts <= 1024: return 8       # very small
    return 16                           # branch-heavy: keep blocks small
```
Tested: BLOCK=128 made `refine_matches` **2x slower** at fdim=128 due to warp divergence in the dilation loops.

### Specialized kernel for common hot path
```mojo
# refine_matches_kernel_f16_cached[FDIM=16, BLOCK_SIZE=8]
# - Loads query descriptor D21 into shared memory once
# - Uses comptime for unrolled SIMD-8 dot product
# - Hardcodes radius=2, dilation_max=2 (MASt3R-SLAM fast config)
```

## Remaining Attack Vectors

### 1. Eliminate cross-stream sync (~2 µs) — THE MAIN BOTTLENECK
The fundamental issue: Mojo's `DeviceContext` creates its own CUDA stream, separate from PyTorch's default stream. Options:
- **Enqueue on PyTorch's stream**: If Mojo exposed an API to use an external CUDA stream handle, we could launch the kernel on PyTorch's stream directly. No sync needed.
- **CUDA events for lightweight sync**: Insert a CUDA event on Mojo's stream after the kernel, then make PyTorch's stream wait on that event. Lighter than device-wide sync (~0.5 µs vs ~2 µs). Requires access to CUDA event APIs from Mojo.
- **Wait for Mojo stream API**: If Mojo adds `DeviceContext(stream=pytorch_stream_handle)`, this is solved cleanly.

### 2. Profile with nsys/ncu
We haven't done GPU profiling yet. `nsys` would show:
- Exact kernel duration vs dispatch overhead
- Whether the SIMD loads generate vector instructions
- Memory bandwidth utilization
- Stream-level timeline showing the Mojo vs PyTorch stream interaction
```bash
nsys profile --trace=cuda PYTHONPATH=. python tools/bench_matching_kernels.py
ncu --target-processes all PYTHONPATH=. python -c "..."
```

### 3. torch.compile integration
Register the Mojo kernel as a `torch.library.custom_op` so `@torch.compile` can fuse the Python dispatch overhead. This could potentially eliminate the Python→Mojo boundary crossing entirely by compiling the call path.

### 4. Fuse Python wrapper calls
Each Mojo wrapper makes ~10 Python API calls (shape, data_ptr, torch.empty, etc.). A pre-packed approach could reduce this to 2-3 calls:
```python
info = torch.tensor([batch, h, w, num_pts, max_iter, ...], dtype=torch.int64)
mojo_be.iter_proj_packed(info, rays_img, pts_3d_norm, p_init, p_new, converged)
```

## Correctness: What The Tests Currently Establish

The Hypothesis property-based fuzz tests (50 random examples per test, randomizing shapes 8-128, batch 1-3, seeds, LM parameters, feature dims including non-SIMD-aligned values) currently support the following:

1. **Single-iteration iter_proj**: p99 diff < 0.1 pixels across all tested configurations. The kernel is numerically equivalent to CUDA for one LM step.

2. **Multi-iteration iter_proj**: With near-zero damping (lambda=1e-8), a 0.007 pixel FP diff at iteration 1 cascades to 221 pixel difference by iteration 10. This is NOT a bug — it's the chaotic nature of the LM accept/reject branch. The test uses statistical comparison (convergence rate gap < 10%, median position gap < 2 pixels).

3. **refine_matches**: differences stay small across the tested shapes, feature dims, radii, and dilation levels, but the strongest honest statement is not `< 2%` for every case. The current fuzz test uses `< 3%` because tiny half-precision searches with many near-ties can flip a few winners even in the CUDA oracle when the same values are reallocated at different addresses.

4. **Cross-stream correctness**: Without `torch.cuda.synchronize()`, rapid-fire calls produce `max_diff=8-20 pixels` from stale `torch.empty()` buffers. With sync restored, the current verification run passes the full comparison suite (`24 passed`).

## File Inventory

| File | Purpose |
|------|---------|
| `mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo` | Extension module entry point (`PyInit_`) |
| `mast3r_slam/backend/mojo/matching.mojo` | `iter_proj` and `refine_matches` kernels + Python wrappers |
| `mast3r_slam/backend/mojo/gn_kernels.mojo` / `gn.mojo` | Gauss-Newton ray-cost step and Sim(3) retraction kernels / wrappers |
| `mast3r_slam/backend/mojo/python_interop.mojo` | Cached DeviceContext and torch tensor pointers |
| `mast3r_slam/backend/mojo/OPTIMIZATION_GUIDE.md` | This document |
| `tests/test_mojo_vs_cuda.py` | Mojo vs CUDA parity (integration lane) |
| `tools/bench_kernels.py` | Side-by-side CUDA vs Mojo benchmark (500 runs, median) |
| `mast3r_slam/ops/matching.py`, `mast3r_slam/ops/optim.py` | Production callers — matching tries Mojo first, falls back to CUDA |
| `mast3r_slam/backend/src/matching_kernels.cu` | CUDA oracle (DO NOT MODIFY) |
| `mast3r_slam/backend/src/gn_kernels.cu` | GN CUDA kernels (oracle for the rays cost; points and calib costs) |
| `pixi.toml` (root) | `mojo` + `max-core` from the stable max channel + cached `_mast3r-slam-build-mojo-kernels` task |

## Mojo API Gotchas (stable Mojo 1.1)

1. **No module-level `var`** — Workaround: `alloc(Layout[T](count=1)).unsafe_leak()` + store the address as a Python module attribute
2. **`gpu` and `layout` are not in the stdlib** — `from max.gpu import ...`, `from max.gpu.host import DeviceContext`, `from layout import ...`; they ship with `max-core`, which the pixi feature pins next to `mojo`
3. **Kernel scalars must be fixed-width** — `Int` and `UInt` are not `DevicePassable`; pass `Int32` and convert inside the kernel
4. **Pass the scalar alignment on SIMD loads from PyTorch views** — the default vector alignment causes `CUDA_ERROR_MISALIGNED_ADDRESS`
5. **`Float32` type casts**: Use `Float32(int_value)` explicitly; no implicit int→float
6. **Kernel functions can't raise** — No `raises` keyword on GPU kernels
7. **`comptime` not `alias`** — For compile-time constants
8. **`def` not `fn`** — `fn` is removed
9. **`min(max(x, lo), hi)` on purpose in iter_proj** — `clamp(NaN, lo, hi)` returns `hi`; CUDA's `fmin(fmax(..))` returns `lo`
10. **`SIMD.reduce_add()`** — Not `.sum()` or `.reduce[add]()`
11. **`floor()` returns Float** — Need `Int(floor(x))` for integer index
12. **`Array[T, N](fill=default)`** for fixed-size per-thread arrays (`InlineArray` was renamed)
13. **Cross-stream sync is mandatory** — matching and ray-step wrappers synchronise before returning; the Python GN loop owns the final sync after `pose_retr`
14. **Pointers**: `Pointer`/`MutPointer`/`ImmPointer` (`UnsafePointer` is deprecated), `p[unsafe_offset=i]`, `p.unsafe_load[width=w]`, `p.unsafe_write(v)`

## How the CUDA Kernel Works (for reference)

### iter_proj_kernel (matching_kernels.cu:119-275)
- 1 thread per point, BLOCK=16
- Bilinear interpolation of [b,h,w,9] ray image (ray + gx + gy)
- LM iterations: normalize ray → compute error → build 2×2 JᵀJ → solve → accept/reject
- Uses `PackedTensorAccessor32` for 4D/3D indexing (compiles to pointer + stride arithmetic)
- Returns: `p_new[b,hw,2]` (float32), `converged[b,hw]` (bool)

### refine_matches_kernel (matching_kernels.cu:26-81)
- 1 thread per match point, BLOCK=16
- Dilation loop: for d in [dilation_max..1]: search (2*radius*d+1)² neighborhood
- Inner loop: dot product of D21[b,n,:] with D11[b,v,u,:] over fdim dimensions
- Updates best position, re-centres search for next dilation level
- Uses `AT_DISPATCH_FLOATING_TYPES_AND_HALF` for fp16/fp32 dispatch
- Returns: `p1_new[b,hw,2]` (long)
