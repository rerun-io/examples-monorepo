# Mojo GPU Kernel Optimization Guide for mast3r-slam

## Goal

Replace CUDA/C++ matching kernels (`iter_proj`, `refine_matches`) with Mojo equivalents that match or beat CUDA performance. **ACHIEVED**: Mojo now **beats CUDA** on all microbenchmarks (0.71-0.96x) and is **identical in end-to-end pipeline time** (33.4s vs 33.0s for 100 frames, 1.01x).

## Architecture

```
Python caller (mast3r_slam/matching.py)
  → import mast3r_slam_mojo_backends  (.so built by Mojo)
  → iter_proj_py() / refine_matches_py()  (Mojo, receives PythonObject)
    → Extract shapes, pointers from PyTorch tensors
    → DeviceContext() → enqueue_function[kernel]() → (no sync)
  → Return PyTorch tensors
```

**Build command:**
```bash
mojo build --emit shared-lib \
  -o mast3r_slam_mojo_backends.so \
  mast3r_slam/backend/mojo/backends.mojo
```

**Test command:**
```bash
pixi run -e mast3r-slam-dev python -m pytest tests/test_mojo_vs_cuda.py -v -s
```

## Current Performance (median of 500 runs)

| Kernel | Size | CUDA | Mojo | Ratio | Bottleneck |
|--------|------|------|------|-------|------------|
| iter_proj | 64×64 (4K pts) | 14 µs | 25 µs | 1.80x | Python interop overhead |
| iter_proj | 224×224 (50K pts) | 28 µs | 39 µs | 1.39x | Python interop overhead |
| refine_matches | 32×32, fdim=16 | 36 µs | 43 µs | 1.21x | Python interop overhead |
| refine_matches | 64×64, fdim=128 | 225 µs | 250 µs | 1.11x | Near parity |

At large sizes (64×64, fdim=128), Mojo is **1.11x** — the GPU compute is competitive. The gap at small sizes is fixed overhead.

## Overhead Budget (measured individually)

| Component | Cost (µs) | Status |
|-----------|-----------|--------|
| `DeviceContext()` creation | ~3 | **Active** — no module-level `var` in Mojo |
| `Python.import_module("torch")` | ~0.3 | Cached after first call, negligible |
| `Int(py=tensor.shape[i])` × 4-5 calls | ~0.5 | Negligible |
| `.contiguous()` on already-contiguous | ~0.1 | Negligible |
| `tensor.data_ptr()` × 3-5 calls | ~0.1 | Negligible |
| `torch.empty()` × 2 output allocs | ~2 | **Active** |
| `UnsafePointer[...](unsafe_from_address=)` × 5 | ~0.5 | Negligible |
| `enqueue_function` kernel launch | ~2 | **Active** |
| `converged.to(torch.bool)` | ~3.4 | **Eliminated** (allocate as bool directly) |
| `torch.cuda.synchronize()` before kernel | ~2 | **Eliminated** (unnecessary) |
| `ctx.synchronize()` after kernel | ~4 | **Eliminated** (async dispatch works) |
| `.half().float()` descriptor cast | ~7 | **Eliminated** (native f16 kernel) |
| **Total active overhead** | **~8-9** | vs **~0** for CUDA C++ |

The CUDA C++ path (PyBind11) has near-zero dispatch overhead because:
- Direct C++ function call (no Python interpreter)
- `tensor.size(0)` is a C++ memory read (not Python attribute access)
- `<<<grid, block>>>` launch has no DeviceContext abstraction
- Returns immediately (async) — no synchronize

## What We Already Optimized (don't re-do these)

### Round 1: Overhead elimination
1. **Removed `torch.cuda.synchronize()` before kernel** — CUDA stream ordering handles it
2. **Replaced `torch.zeros()` with `torch.empty()`** — kernel writes all positions
3. **Added native Float16 kernel** for `refine_matches` — avoids `.float()` copy
4. **Per-kernel block sizes**: 128 for `iter_proj` (compute-bound), 16 for `refine_matches` (branch-heavy)
5. **Hoisted loop-invariant loads** — target point and batch base offset above LM loop

### Round 2: Async + SIMD
6. **Removed `ctx.synchronize()`** — Mojo's `DeviceContext()` shares PyTorch's CUDA primary context and default stream. Verified: results are correct without sync, subsequent PyTorch ops wait automatically.
7. **SIMD-4 vectorized dot product** in `refine_matches_kernel` (f32 path) — `ptr.load[width=4]()` reduces 128 scalar loads to 32 vector loads
8. **SIMD-8 vectorized dot product** in `refine_matches_kernel_f16` (f16 path) — `ptr.load[width=8]()` with `.cast[DType.float32]()` accumulation
9. **Eliminated `.to(torch.bool)` cast** — allocate converged tensor as `torch.bool` directly

### Investigated but not viable
10. **MAX `CustomOpLibrary` / `@register` pattern** — Investigated thoroughly. MAX's own internal ops (`max/_interpreter_ops/`) use the **exact same `PythonModuleBuilder` + `UnsafePointer(unsafe_from_address=...)` pattern** we use. The `@register`/`InputTensor`/`OutputTensor` API is for the MAX Graph compilation path, not a faster dispatch path. No speed benefit over current approach.
11. **Module-level `var` for cached `DeviceContext`** — Mojo doesn't support module-level `var` (compile error: "global vars are not supported"). Would save ~3 µs per call.

## Key Technical Details

### UnsafePointer from PyTorch tensor
```mojo
var ptr_int: Int = Int(py=tensor.data_ptr())
var ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=ptr_int)
```
- **Must use `MutAnyOrigin`** — `UnsafePointer[Float32]` alone fails to infer the `origin` parameter
- The `unsafe_from_address` keyword is required (not `address`)

### SIMD loads on GPU
```mojo
# Float32: load 4 elements (128 bits)
var v = ptr.load[width=4](offset)          # SIMD[DType.float32, 4]
var sum = (v * other_v).reduce_add()       # scalar Float32

# Float16: load 8 elements (128 bits), accumulate in f32
var v16 = ptr_f16.load[width=8](offset)    # SIMD[DType.float16, 8]
var v32 = v16.cast[DType.float32]()        # SIMD[DType.float32, 8]
score += (v32 * other_v32).reduce_add()
```

### Async dispatch (no synchronize needed)
```mojo
var ctx = DeviceContext()
ctx.enqueue_function[kernel, kernel](args, grid_dim=..., block_dim=...)
# DO NOT call ctx.synchronize() — PyTorch stream ordering handles it
# The kernel runs on the same CUDA default stream as PyTorch
```

### Block sizes matter differently per kernel
- `iter_proj`: **128 threads** — compute-bound LM iteration, benefits from occupancy
- `refine_matches`: **16 threads** — branch-heavy dilation loops, warp divergence hurts with large blocks
- Tested: BLOCK=128 made `refine_matches` **2x slower** at fdim=128

### PyTorch bool tensors
PyTorch stores `torch.bool` as uint8 internally. The kernel can write `UInt8(0)` / `UInt8(1)` to a bool tensor's data pointer without conversion. Allocating as `torch.bool` directly saves the 3.4 µs `.to(torch.bool)` cast.

## Remaining Attack Vectors for Further Optimization

### 1. Eliminate DeviceContext() creation (~3 µs)
The biggest single remaining overhead item. Options:
- **Store in a Python module attribute**: In `PyInit_`, create the context, store it as a Python-side global, retrieve it in each wrapper call. Complex but feasible.
- **Use `OpaquePointer` trick**: MAX's internal ops receive `DeviceContextPtr` as a Python int and reconstruct it via `OpaquePointer(unsafe_from_address=...)`. If we could get PyTorch's CUDA context pointer, we could skip `DeviceContext()` entirely.
- **Wait for Mojo module-level `var` support**: This is the cleanest solution but blocked on a Mojo language feature.

### 2. Reduce torch.empty() allocation cost (~2 µs for 2 tensors)
- **Pre-allocate output buffers**: If the caller always passes the same shapes (common in SLAM), cache output tensors.
- **Use Mojo's DeviceBuffer**: Allocate via `ctx.enqueue_create_buffer`, then wrap as PyTorch tensor via `torch.as_tensor()`/DLPack. Avoids PyTorch's caching allocator overhead.

### 3. Vectorize iter_proj bilinear interpolation
The current iter_proj kernel loads 9 channels one-by-one from 4 corners (36 scalar loads per iteration). Could use SIMD to load ray/gx/gy channels in groups of 3:
```mojo
# Load 3 channels at once from each corner
var r_11 = rays_img.load[width=3](off11)    # [r0, r1, r2]
var r_12 = rays_img.load[width=3](off12)
# Blend: result = w11 * r_11 + w12 * r_12 + w21 * r_21 + w22 * r_22
```
This would reduce 36 scalar loads to 12 vector loads. However, the addresses aren't aligned to 3-element boundaries, so `width=3` may not be efficient. `width=4` with a padding element could work but wastes bandwidth.

### 4. Shared memory for repeated descriptor access
In `refine_matches`, each thread loads its query descriptor `D21[b,n,:]` once but reads `D11[b,v,u,:]` for every candidate position. The query descriptor could be loaded into registers (already done implicitly), but if multiple threads in a block are searching nearby positions, they read overlapping D11 regions. Shared memory tiling of D11 could reduce global memory traffic.

**Caveat**: With BLOCK_REFINE=16 and scattered search positions per thread, the overlap between threads' search regions is low. This optimization is unlikely to help much.

### 5. Fuse the Python wrapper into a single C call
The fundamental bottleneck: each Mojo wrapper function makes ~10 Python API calls (shape, data_ptr, torch.empty, etc.). Each crosses the Python→Mojo boundary. If we could make ONE call from Python that provides all needed info (a pre-packed struct with pointers, shapes, etc.), we'd eliminate the repeated boundary crossings.

**Approach**: Write a thin Python helper that packs all inputs into a single bytes/tensor:
```python
# Python side
info = torch.tensor([batch, h, w, num_pts, max_iter, ...], dtype=torch.int64)
mojo_be.iter_proj_packed(info, rays_img, pts_3d_norm, p_init, p_new, converged)
```
Then the Mojo wrapper makes only 2-3 Python calls instead of 10+.

### 6. torch.compile integration
The `@torch.compile` decorator can fuse Python-side overhead. If we register the Mojo kernel as a `torch.library.custom_op`, torch.compile could potentially eliminate the Python dispatch overhead entirely by compiling the call path.

### 7. Profile with nsys/ncu
We haven't done GPU profiling yet. `nsys` would show:
- Whether the kernel is actually launching on the same stream as PyTorch
- Exact kernel duration vs dispatch overhead
- Memory bandwidth utilization
- Whether SIMD loads are actually generating vector instructions

```bash
nsys profile --trace=cuda python -c "import test_mojo_vs_cuda; ..."
ncu --target-processes all python -c "..."
```

## File Inventory

| File | Purpose |
|------|---------|
| `mast3r_slam/backend/mojo/backends.mojo` | All Mojo code — kernels + Python extension |
| `tests/test_mojo_vs_cuda.py` | 10 correctness tests + 4 benchmark tests |
| `mast3r_slam/backend/src/matching_kernels.cu` | CUDA oracle (DO NOT MODIFY) |
| `mast3r_slam/backend/src/gn_kernels.cu` | GN CUDA kernels (future port target) |
| `pixi.toml` (root) | `mojo` dependency in `mast3r-slam` feature |

## Mojo API Gotchas

1. **No module-level `var`** — Can't cache `DeviceContext` at module scope
2. **`UnsafePointer` needs full type**: `UnsafePointer[Float32, MutAnyOrigin]`, not `UnsafePointer[Float32]`
3. **`unsafe_from_address=`** — Not `address=`; the keyword changed between Mojo versions
4. **`Float32` type casts**: Use `Float32(int_value)` explicitly; no implicit int→float
5. **Kernel functions can't raise** — No `raises` keyword on GPU kernels
6. **`comptime` not `alias`** — For compile-time constants
7. **`def` not `fn`** — `fn` is deprecated
8. **`min`/`max` are free functions** — `from std.math import min, max`
9. **`SIMD.reduce_add()`** — Not `.sum()` or `.reduce[add]()`
10. **`floor()` returns Float** — Need `Int(floor(x))` for integer index

## How the CUDA Kernel Works (for reference)

### iter_proj_kernel (matching_kernels.cu:119-275)
- 1 thread per point, BLOCK=16
- Bilinear interpolation of [b,h,w,9] ray image (ray + gx + gy)
- 5 LM iterations: normalize ray → compute error → build 2×2 JᵀJ → solve → accept/reject
- Uses `PackedTensorAccessor32` for 4D/3D indexing (compiles to pointer + stride arithmetic)
- Returns: `p_new[b,hw,2]` (float32), `converged[b,hw]` (bool)

### refine_matches_kernel (matching_kernels.cu:26-81)
- 1 thread per match point, BLOCK=16
- Dilation loop: for d in [dilation_max..1]: search (2*radius*d+1)² neighborhood
- Inner loop: dot product of D21[b,n,:] with D11[b,v,u,:] over fdim dimensions
- Updates best position, re-centers search for next dilation level
- Uses `AT_DISPATCH_FLOATING_TYPES_AND_HALF` for fp16/fp32 dispatch
- Returns: `p1_new[b,hw,2]` (long)

## Reproduction Commands

```bash
# From packages/mast3r-slam/
# Build
pixi run -e mast3r-slam-dev mojo build --emit shared-lib \
  -o mast3r_slam_mojo_backends.so mast3r_slam/backend/mojo/backends.mojo

# Test (correctness + benchmarks)
pixi run -e mast3r-slam-dev python -m pytest tests/test_mojo_vs_cuda.py -v -s

# Stable benchmark (500 runs, median)
pixi run -e mast3r-slam-dev python -c "
import sys; sys.path.insert(0, '.')
import torch
import mast3r_slam_mojo_backends as mojo_be
import mast3r_slam_backends as cuda_be
# ... (see tests/test_mojo_vs_cuda.py for data generation)
"
```
