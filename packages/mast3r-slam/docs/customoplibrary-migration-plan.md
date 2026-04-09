# CustomOpLibrary Migration Plan (Deferred)

This document captures all learnings from the CustomOpLibrary proof-of-concept
so the migration can be completed at a later date. The POC was successful —
both `iter_proj` and `refine_matches` compiled, ran on GPU, and matched CUDA
output — but the scope was too large for a single session.

## What CustomOpLibrary replaces

Currently: `mojo build --emit shared-lib` → `PythonModuleBuilder` → manual
`UnsafePointer` casting from `tensor.data_ptr()` → manual `torch.cuda.synchronize()`
→ ~350 lines of bridge/wrapper code across `python_interop.mojo`,
`mast3r_slam_mojo_backends.mojo`, and Python wrappers in `matching.mojo`/`gn.mojo`.

CustomOpLibrary: auto-compiles `.mojo` files from a directory, auto-marshals tensors
via DLPack, auto-syncs CUDA streams, enables `torch.compile`. No manual pointer
casting, no DeviceContext management, no build step.

## Dependency setup (VERIFIED WORKING)

The `modular` conda metapackage pulls in `starlette==1.0.0` which conflicts with
`gradio>=6.8.0`. The fix: install `max` directly (not `modular`):

```toml
# pixi.toml — [feature.mast3r-slam.dependencies]
# max Python SDK (not modular metapackage!) for CustomOpLibrary.
# Installing max directly avoids max-pipelines → uvicorn → starlette==1.0
# which conflicts with gradio's starlette<1.0 requirement.
#
# Package hierarchy:
#   modular (metapackage) → max-pipelines → uvicorn → starlette==1.0  ← conflicts
#   max (Python SDK)       → max-core                                  ← no conflict
max = { version = ">=26.3.0.dev2026040905,<27", channel = "https://conda.modular.com/max-nightly/" }
```

This was verified: `from max.experimental.torch import CustomOpLibrary` imports
correctly, no starlette conflict, all existing tests pass.

## Directory structure

```
mast3r_slam/backend/mojo/operations/
  __init__.mojo          # Vec3 struct defined here (visible to all siblings)
  matching_ops.mojo      # @compiler.register("iter_proj"), @compiler.register("refine_matches")
  gn_ops.mojo            # (future) gauss_newton_rays_step, pose_retr
  test_add_one.mojo      # Minimal test op for verifying the pipeline
```

`CustomOpLibrary(Path("mast3r_slam/backend/mojo/operations"))` auto-compiles
this directory into a `.mojopkg` at runtime (cached in `/tmp/.modular_1000/mojo_pkg/`).

## Kernel registration pattern

### The `@compiler.register` struct

```mojo
from compiler import register
from tensor import InputTensor, OutputTensor
from layout import LayoutTensor

@register("iter_proj")
struct IterProj[max_iter: Int, lambda_init_x1e8: Int, cost_thresh_x1e6: Int]:
    @staticmethod
    def execute[target: StaticString](
        # Outputs first (destination-passing style)
        p_new: OutputTensor[dtype=DType.float32, rank=3, ...],
        converged: OutputTensor[dtype=DType.uint8, rank=2, ...],
        # Inputs
        rays_img: InputTensor[dtype=DType.float32, rank=4, ...],
        pts_3d_norm: InputTensor[dtype=DType.float32, rank=3, ...],
        p_init: InputTensor[dtype=DType.float32, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var rays_lt = rays_img.to_layout_tensor()  # InputTensor → LayoutTensor
        var dev_ctx = ctx.get_device_context()
        iter_proj_launch(dev_ctx, rays_lt, ...)
```

### The GPU kernel function (parameterized on LayoutTensor types)

```mojo
def iter_proj_gpu_kernel[
    rays_dtype: DType, rays_layout: Layout,
    pts_dtype: DType, pts_layout: Layout,
    pinit_dtype: DType, pinit_layout: Layout,
    pnew_dtype: DType, pnew_layout: Layout,
    conv_dtype: DType, conv_layout: Layout,
](
    rays_img: LayoutTensor[rays_dtype, rays_layout, MutAnyOrigin],
    pts_3d_norm: LayoutTensor[pts_dtype, pts_layout, MutAnyOrigin],
    ...
):
    # Kernel body — same LM iteration as before
```

### The launch function (comptime specialization)

```mojo
def iter_proj_launch(ctx: DeviceContext, rays_img: LayoutTensor, ...) raises:
    comptime kernel = iter_proj_gpu_kernel[
        rays_img.dtype, rays_img.layout,
        pts_3d_norm.dtype, pts_3d_norm.layout,
        ...
    ]
    ctx.enqueue_function_experimental[kernel](
        rays_img, ...,
        grid_dim=(...), block_dim=...,
    )
```

### Python call site

```python
from max.experimental.torch import CustomOpLibrary

ops = CustomOpLibrary(Path("mast3r_slam/backend/mojo/operations"))
iter_proj_op = ops.iter_proj[{
    "max_iter": cfg["max_iter"],
    "lambda_init_x1e8": int(cfg["lambda_init"] * 1e8),
    "cost_thresh_x1e6": int(cfg["convergence_thresh"] * 1e6),
}]

p_new = torch.empty(batch, num_pts, 2, device=device)
converged = torch.empty(batch, num_pts, device=device, dtype=torch.uint8)
iter_proj_op(p_new, converged, rays_img, pts_3d_norm, p_init)
```

## Key learnings / gotchas

### 1. Scalar params must be compile-time integers

CustomOpLibrary only passes tensors at runtime. Scalar params (max_iter,
lambda_init, etc.) must be struct-level compile-time parameters. Float scalars
are encoded as scaled integers since Mojo doesn't support Float compile-time
params:
```mojo
struct IterProj[max_iter: Int, lambda_init_x1e8: Int, cost_thresh_x1e6: Int]:
    # lambda_init = lambda_init_x1e8 * 1e-8
```

### 2. LayoutTensor indexing returns SIMD, needs `.cast[DType.float32]()[0]`

Inside parameterized kernels, LayoutTensor's dtype is generic. Reading a scalar:
```mojo
var u: Float32 = p_init[b, ni, 0].cast[DType.float32]()[0]  # NOT just [0]
```
Writing a scalar:
```mojo
p_new[b, ni, 0] = Scalar[pnew_dtype](u)  # NOT just = u
```

### 3. Kernel functions need full type parameterization

`enqueue_function_experimental` can't infer LayoutTensor types automatically.
Each tensor param needs its own `dtype` and `layout` compile-time parameter:
```mojo
def kernel[rays_dtype: DType, rays_layout: Layout, ...](
    rays_img: LayoutTensor[rays_dtype, rays_layout, MutAnyOrigin],
    ...
)
```
The launch function extracts these via `comptime` from the unparameterized
LayoutTensor returned by `.to_layout_tensor()`.

### 4. Sibling imports don't work in mojo package

`from sim3_types import Vec3` fails in `mojo package` compilation. The fix:
define shared types in `__init__.mojo` where they're automatically available
to all files in the package.

### 5. UnsafePointer still needed for SIMD loads

The refine_matches kernel needs `ptr.load[width=8]()` for vectorized dot products.
Extract the raw pointer from LayoutTensor: `D11.ptr.bitcast[Float16]()`.

### 6. `enqueue_function` vs `enqueue_function_experimental`

Use `enqueue_function_experimental` for parameterized kernels. The regular
`enqueue_function` can't handle the complex LayoutTensor type params.

### 7. Cache invalidation

CustomOpLibrary caches compiled `.mojopkg` in `/tmp/.modular_1000/mojo_pkg/`.
After changing kernel code, delete this cache or it may use stale code:
```bash
rm -f /tmp/.modular_1000/mojo_pkg/*.mojopkg
```

### 8. The `_bsample` capturing closure

The bilinear sampling helper inside the kernel captures `rays_img` from the
outer scope. Use `capturing` keyword:
```mojo
def _bsample(...) capturing -> Float32:
```

## Verified results

- `iter_proj`: matches old Mojo backend exactly (0.000000 diff), matches CUDA
  within tolerance (0.000164 max diff)
- `refine_matches`: 0.6% pixel diff vs CUDA (within 3% f16 tolerance)
- `example-fast`: completes successfully with CustomOpLibrary matching ops

## What remains to do

1. **Port GN kernels** (gauss_newton_rays_step, pose_retr) to operations/gn_ops.mojo
   - Add Quat, Sim3Pose, TangentVec7 to __init__.mojo
   - Port the complex GN kernel with shared memory + block reduction
   - Need to handle the 10-tensor signature with full type parameterization
2. **Move GN Python orchestration** (solve_step_system, gauss_newton_impl) from
   gn.mojo to global_opt.py (these are pure Python/PyTorch via PythonObject)
3. **Update global_opt.py** to use CustomOpLibrary for GN rays
4. **Delete old bridge code**: python_interop.mojo, mast3r_slam_mojo_backends.mojo,
   matching.mojo, gn.mojo, gn_kernels.mojo
5. **Remove `_build-mojo-kernels` pixi task** and all depends-on references
6. **Update tests + benchmarks** to use CustomOpLibrary path

## Relevant Modular docs

### Core references
- [Custom kernels for PyTorch](https://docs.modular.com/max/develop/custom-kernels-pytorch/) — the main tutorial; shows `CustomOpLibrary`, `@register`, `foreach`, destination-passing style
- [Build custom ops for GPUs](https://docs.modular.com/max/develop/build-custom-ops/) — device-specific kernel dispatch (`comptime if target == "gpu"`), `enqueue_function_experimental`
- [Custom ops overview](https://docs.modular.com/max/develop/custom-ops/) — high-level architecture
- [Custom ops matmul example](https://docs.modular.com/max/develop/custom-ops-matmul/) — advanced example with shared memory, tiling, TensorCore

### Mojo language features used
- [LayoutTensor](https://docs.modular.com/mojo/manual/layout/tensors/) — multi-dim tensor view, `RuntimeLayout`, `UNKNOWN_VALUE`
- [Layouts](https://docs.modular.com/mojo/manual/layout/layouts/) — `Layout.row_major()`, layout algebra
- [Metaprogramming / parameters](https://docs.modular.com/mojo/manual/metaprogramming/) — `comptime`, compile-time specialization
- [Pointers](https://docs.modular.com/mojo/manual/pointers/) — when UnsafePointer is unavoidable (SIMD loads)
- [GPU programming](https://docs.modular.com/mojo/manual/gpu/basics/) — `block_idx`, `thread_idx`, `barrier`, shared memory

### GitHub examples (working code)
- [pytorch_custom_ops/](https://github.com/modular/modular/tree/main/max/examples/pytorch_custom_ops) — grayscale, fused_attention, whisper integration
- [operations/fused_attention.mojo](https://github.com/modular/modular/blob/main/max/examples/pytorch_custom_ops/operations/fused_attention.mojo) — the best reference for complex GPU kernels (shared memory, `enqueue_function_experimental`, compile-time kernel specialization)
- [operations/add_custom.mojo](https://github.com/modular/modular/blob/main/max/examples/pytorch_custom_ops/operations/add_custom.mojo) — shows compile-time struct params (`struct AddConstantCustom[value: Int]`)
- [whisper.py](https://github.com/modular/modular/blob/main/max/examples/pytorch_custom_ops/whisper.py) — shows `op_library.fused_attention_custom[{"BN": 16, "BD": 8}]` parameter specialization from Python

### GPU puzzles (learning resource)
- [Mojo GPU puzzles](https://puzzles.modular.com/) — interactive exercises for learning Mojo GPU programming patterns

## Files from the POC (to reference when resuming)

The POC code was on branch `vk/mojo-only-kernels` in these commits:
- `cd3a9d9` — pixi.toml: add max conda dep
- `2153859` — operations/__init__.mojo + test_add_one.mojo scaffold
- `3fe6bd0` — Vec3 in __init__.mojo, test_vec3_norm verified
- `072db91` — iter_proj CustomOpLibrary kernel (compiles, runs, matches CUDA)
- `87fabb0` — refine_matches added (both matching ops verified)
- `6bfb09f` — matching.py updated to use CustomOpLibrary in production pipeline
