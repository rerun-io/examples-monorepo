# MASt3R-SLAM Bottleneck Analysis

## Pipeline Overview

MASt3R-SLAM runs two concurrent processes:

```
Main Process (Tracker)                  Backend Process (Global Optimizer)
┌───────────────────────┐               ┌────────────────────────────┐
│ for each frame:       │               │ for each new keyframe:     │
│  1. Load & resize img │               │  1. Retrieval DB query     │
│  2. Create frame      │  ──queue──▶   │  2. Symmetric matching     │
│  3. Encode (ViT)      │               │     (4 decoder passes +    │
│  4. Decode (1 pass)   │               │      dense matching)       │
│  5. Dense matching    │               │  3. Factor graph update    │
│  6. GN tracking (50i) │               │  4. Global GN solve (10i)  │
│  7. Rerun logging     │               │  5. Update shared poses    │
└───────────────────────┘               └────────────────────────────┘
```

---

## Bottleneck #1: MASt3R Decoder Forward Passes (DOMINANT)

**Where:** `mast3r_utils.py:40-45` (decoder fn), called from tracking + backend

**What happens:** The ViT-Large decoder is a heavy neural network forward pass.
Every tracked frame requires **1 encoder + 1 decoder** call (asymmetric).
Every new keyframe triggers **4 decoder passes** in the backend (symmetric matching: `ii→jj` + `jj→ii`, both directions).

**Cost breakdown per frame:**

| Operation | Decoder Passes | Context |
|---|---|---|
| Tracking (asymmetric) | 1 encode + 1 decode | `mast3r_asymmetric_inference`, line 189 |
| INIT / RELOC (mono) | 1 encode + 1 decode | `mast3r_inference_mono`, line 124 |
| Backend symmetric match | 4 decodes per pair | `mast3r_decode_symmetric_batch`, line 89 |

At 512px resolution, each decoder pass processes `(1, N_patches, 1024)` tokens through cross-attention layers. This is the single most expensive operation in the entire pipeline.

**Evidence:** The `fast.yaml` config achieves speedup primarily by reducing resolution from 512→224 (4x fewer pixels = fewer patches = much faster decoder).

**Why it's the bottleneck:**
- ViT-Large has ~300M parameters
- Each forward pass involves full attention over all patch tokens
- The backend does 4 decoder passes **per keyframe pair** (sequential loop at line 94-100 in `mast3r_decode_symmetric_batch`)
- With k=3 retrieval + 1 consecutive = ~4 pairs per keyframe = **16 decoder calls per new keyframe** in the backend

### Potential Solutions

1. **Half-precision inference (FP16/BF16):** The decoder already uses `torch.amp.autocast(enabled=False)` at line 42 — this explicitly *disables* AMP for the downstream head. Enabling AMP for the full decoder+head could yield ~1.5-2x speedup. The `autocast(enabled=False)` was likely added for numerical stability; test whether BF16 maintains acceptable accuracy.

2. **torch.compile the decoder:** Wrapping `model._decoder` and `model._downstream_head` with `torch.compile(mode="reduce-overhead")` can fuse kernels and reduce Python overhead. The model is already in `torch.inference_mode`, making it a good candidate.

3. **Batch decoder calls in the backend:** `mast3r_decode_symmetric_batch` (line 89-120) runs decoder pairs **sequentially in a Python loop**. For batch>1 (which happens with retrieval candidates), this loop could be parallelized or batched at the decoder level if memory allows.

4. **Feature caching across frames:** When a frame becomes a keyframe, its encoder features (`frame.feat`, `frame.pos`) are already cached. But the decoder is still called per-pair. Consider caching intermediate decoder states for the keyframe side.

5. **Reduce retrieval k:** `retrieval.k=3` means 3 extra pairs per keyframe. Reducing to k=1 or k=2 cuts backend decoder calls proportionally (from 16 to 8 or 12 per keyframe).

---

## Bottleneck #2: Dense Matching Kernels (iter_proj + refine_matches)

**Where:** `matching.py:134-199` → CUDA/Mojo kernels

**What happens:** After decoding, every frame-to-keyframe pair needs dense correspondence matching:
1. **iter_proj:** 10 iterations of gradient-descent projection of 3D points onto a ray image (`matching.max_iter=10`)
2. **refine_matches:** Descriptor-based refinement in a local neighborhood (radius=3, dilation_max=5)

At 512px, this operates on ~262K points per frame.

**Evidence:** The existing `tools/bench_matching_kernels.py` already benchmarks these kernels in isolation. Both iter_proj and refine_matches have dedicated CUDA and Mojo implementations, indicating they were already identified as performance-sensitive.

### Potential Solutions

1. **Reduce matching iterations:** `max_iter=10` could potentially be reduced to 5-7 if convergence is fast (the kernel has a `convergence_thresh=1e-6` early exit). Profile average actual iterations to convergence.

2. **Warm-start from previous frame:** Already partially implemented — `idx_i2j_init` is passed from the previous tracking result (line 63 in `tracker.py`). This helps consecutive frames but not the backend's symmetric matching.

3. **Descriptor refinement radius:** `radius=3` with `dilation_max=5` creates a search window up to 15x15. Reducing to radius=2 or dilation_max=3 would shrink the search space significantly.

4. **Use Mojo kernels over CUDA:** The codebase already has Mojo alternatives (`mast3r_slam_mojo_backends`). The matching module prefers Mojo when available (line 13-16 in `matching.py`). Benchmark results from `bench_matching_kernels.py` should guide which to use per resolution.

---

## Bottleneck #3: Rerun Visualization Logging

**Where:** `rerun_log_utils.py:89-246` → `log_frame()`, called every frame

**What happens:** On **every single frame**, the logger:
1. Estimates focal length via Weiszfeld iteration (10 iterations, CPU) — line 113
2. Converts all keyframe poses to SE3 → 4x4 matrix → GL convention — lines 162-217
3. Re-logs **all keyframes' transforms** (not just new/dirty ones) — line 162 loop
4. Recomputes gravity alignment when keyframe count changes — line 107
5. Logs edge graph by iterating all edges — lines 230-246

The keyframe loop at line 162 is O(N_keyframes) **per frame**, so as the map grows, visualization cost grows linearly.

**Evidence:** The focal estimation at line 113 (`estimate_focal_knowing_depth` with `weiszfeld` mode) runs 10 iterations of reweighted least squares on CPU. The keyframe iteration loop has no dirty-checking (the `get_dirty_idx` method exists at line 481 in `frame.py` but is commented out at line 160 in `rerun_log_utils.py`).

### Potential Solutions

1. **Enable dirty-flag optimization:** `SharedKeyframes.get_dirty_idx()` already exists (frame.py:481). The commented-out `dirty_idx = keyframes.get_dirty_idx()` at line 160 in `rerun_log_utils.py` shows this was planned. Only re-log keyframes whose poses have changed.

2. **Skip focal estimation per frame:** The Weiszfeld focal estimation at line 113 runs for every frame. Since focal length doesn't change rapidly, cache it and recompute only every N frames or on new keyframes.

3. **Subsample visualization:** Log every Nth frame to Rerun instead of every frame. The `no_viz` flag exists but is all-or-nothing.

4. **Async visualization:** Move logging to a separate thread so it doesn't block the tracking loop. Currently `rr_logger.log_frame()` is synchronous in the main loop.

---

## Bottleneck #4: Tracking Gauss-Newton Optimizer (Per-Frame CPU)

**Where:** `tracker.py:261-331` → `opt_pose_ray_dist_sim3` / `opt_pose_calib_sim3`

**What happens:** For every tracked frame, a Gauss-Newton solver runs up to **50 iterations** (`tracking.max_iters=50`):
- Each iteration: construct Jacobian (hw×4×7 tensor), form H=J'J and g=-J'b, Cholesky factorize, solve
- Matrix sizes: for 512px, hw ≈ 262K, so A is ~262K×7, H is 7×7
- The Cholesky solve is 7×7 so it's cheap, but the Jacobian construction is not

The solver operates on torch tensors on CUDA but the logic is pure Python with torch operations — no custom kernel.

**Evidence:** The convergence check (`check_convergence` at line 314) provides early exit, but worst case is 50 iterations of dense tensor operations.

### Potential Solutions

1. **Reduce max_iters with good initialization:** The pose is initialized from the previous frame, so convergence should be fast. Profile how many iterations are actually used on average. Reducing from 50→20 may have no accuracy impact.

2. **Move to CUDA kernel:** The backend's global optimizer already uses `mast3r_slam_backends.gauss_newton_rays()` (a compiled CUDA kernel). A similar kernel for the per-frame tracker would eliminate Python loop overhead.

3. **Subsample residuals:** Instead of using all ~262K point correspondences, randomly sample or use a confidence-weighted subset (e.g. top-K by quality score). The 7-DOF Sim3 only needs ~7 good correspondences, so using 10K high-quality ones may suffice.

4. **Levenberg-Marquardt damping:** The current solver is undamped Gauss-Newton. Adding LM damping could improve convergence speed and reduce iterations needed.

---

## Bottleneck #5: Backend Global Gauss-Newton Solve

**Where:** `global_opt.py:185-236` → `solve_GN_rays()` / `solve_GN_calib()`

**What happens:** After adding factors, the backend solves a global pose graph over ALL unique keyframes:
- `mast3r_slam_backends.gauss_newton_rays()` — compiled CUDA kernel
- Runs up to 10 iterations (`local_opt.max_iters=10`)
- Data sizes scale with `n_unique_kf × hw` for point maps and `n_edges × hw` for correspondences

As the map grows, this becomes increasingly expensive because it optimizes over all keyframes, not a sliding window.

**Evidence:** `window_size` is set to `1e+6` (effectively infinite) in `base.yaml`, meaning no windowing is applied.

### Potential Solutions

1. **Sliding window optimization:** Set `window_size` to a reasonable value (e.g., 20-50 keyframes) to limit the optimization to recent keyframes. Only optimize the local subgraph.

2. **Reduce global solve frequency:** Currently solves after every new keyframe. Could solve every N keyframes or only when loop closures are detected.

3. **Covisibility-based subgraph:** Only optimize keyframes connected to the new keyframe within K hops in the factor graph, rather than all unique keyframes.

---

## Bottleneck #6: Image I/O and Preprocessing (CPU-bound)

**Where:** `dataloader.py:86-101` (get_image/read_img) + `mast3r_utils.py:239-283` (resize_img) + `frame.py:164-194` (create_frame)

**What happens per frame:**
1. `cv2.imread()` — disk read + JPEG/PNG decode (CPU)
2. `cv2.cvtColor()` — BGR→RGB conversion (CPU)
3. Optional `cv2.remap()` for undistortion (CPU)
4. `float32` conversion + normalization (CPU)
5. `PIL.Image.fromarray()` + resize via LANCZOS/BICUBIC (CPU, Python GIL)
6. `ImgNorm()` — ImageNet normalization (CPU)
7. Transfer to GPU

For MP4 inputs, `torchcodec.VideoDecoder` is used when available (line 311-313), which is more efficient than OpenCV's `cap.read()`. But the PIL resize step still happens on CPU.

### Potential Solutions

1. **GPU-accelerated resize:** Replace PIL resize with `torchvision.transforms.functional.resize()` on GPU tensors, or use NVIDIA DALI for the full preprocessing pipeline.

2. **Prefetch with DataLoader:** Use `torch.utils.data.DataLoader` with `num_workers>0` and `prefetch_factor` to overlap I/O with GPU computation. Currently frames are loaded synchronously in the main loop.

3. **Eliminate PIL intermediary:** The `resize_img` function converts numpy→PIL→numpy→torch. This could be done entirely in torch/torchvision on GPU.

---

## Bottleneck #7: Retrieval Database Scaling

**Where:** `retrieval_database.py:73-121` → `update()` method

**What happens:** For each new keyframe, the retrieval database:
1. Extracts whitened local features via MLP layers (GPU, fast)
2. Transfers to CPU numpy (line 94)
3. Quantizes against codebook centroids via L2 distance (GPU, line 230)
4. Transfers back to CPU numpy (line 231)
5. Runs ASMK inverted file search (CPU)
6. Adds to inverted file (CPU)

**Scaling concern:** The ASMK inverted file search is O(n_images × n_local_features). As the database grows to hundreds of keyframes, query time increases linearly.

### Potential Solutions

1. **Keep quantization on GPU:** The `quantize_custom` method (line 174-196) already runs on GPU, but results are immediately transferred to CPU numpy. If the ASMK library supported GPU tensors, this transfer could be eliminated.

2. **Limit database size:** Prune old/redundant keyframes from the database based on spatial coverage.

3. **Batch quantization and search:** Group multiple queries if keyframes arrive in bursts.

---

## Bottleneck #8: Shared Memory Synchronization

**Where:** `frame.py:197-510` (SharedStates, SharedKeyframes) — RLock on every read/write

**What happens:** Both SharedStates and SharedKeyframes use `manager.RLock()` for thread safety. Every operation (get_mode, set_frame, append keyframe, read keyframe) acquires the lock. The backend process polls in a tight loop (10ms sleep at `inference.py:339,354`) and the tracker writes on every frame.

**Evidence:** The comment at `inference.py:263` explicitly notes: "The lock slows viz down but safer this way..."

### Potential Solutions

1. **Lock-free ring buffer:** Replace the locked buffer with a lock-free SPSC (single-producer, single-consumer) ring buffer for the keyframe queue.

2. **Finer-grained locking:** Instead of one global RLock for all keyframes, use per-keyframe locks so the backend can read keyframe K while the tracker writes keyframe K+1.

3. **Reduce lock contention in visualization:** The visualization loop (Bottleneck #3) holds the keyframes lock while iterating all keyframes. Moving visualization to a separate process with snapshot-based reads would eliminate this contention.

---

## Summary: Bottleneck Priority Ranking

| Priority | Bottleneck | Est. % of Frame Time | Difficulty |
|---|---|---|---|
| **1** | MASt3R decoder passes | ~50-60% | Medium (AMP, compile) |
| **2** | Rerun visualization logging | ~15-20% | Low (dirty flags, caching) |
| **3** | Dense matching kernels | ~10-15% | Medium (already optimized) |
| **4** | Per-frame GN tracking | ~5-10% | Low (reduce iters, subsample) |
| **5** | Image I/O + preprocessing | ~5% | Low (prefetch, GPU resize) |
| **6** | Backend global GN solve | Async (blocks backend) | Low (windowing) |
| **7** | Retrieval DB | Async (blocks backend) | Medium |
| **8** | Shared memory sync | Indirect overhead | High (architectural) |

## Recommended First Steps

1. **Enable `torch.compile` on the decoder** — likely the single highest-impact change
2. **Enable AMP (BF16) for decoder** — test accuracy impact, potentially 1.5-2x speedup
3. **Uncomment dirty-flag visualization** — trivial change, big win at high keyframe counts
4. **Profile actual GN iteration counts** — reduce `max_iters` if convergence is typically fast
5. **Add async image prefetch** — overlap I/O with GPU compute

## Profiling Methodology

To validate these estimates, instrument the pipeline with `torch.cuda.Event` timing:

```python
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... operation ...
end.record()
torch.cuda.synchronize()
print(f"Operation: {start.elapsed_time(end):.2f}ms")
```

Add timing around:
- `model._encode_image()` — encoder cost
- `decoder()` — per-call decoder cost
- `mast3r_slam_backends.iter_proj()` — matching kernel cost
- `mast3r_slam_backends.refine_matches()` — refinement cost
- `self.solve()` in tracker — per-GN-iteration cost
- `rr_logger.log_frame()` — visualization cost
- `create_frame()` — preprocessing cost

The existing `tools/bench_matching_kernels.py` provides a good template for this approach.
