# Mini-DPVO Pipeline Architecture

This document explains the mini-dpvo inference architecture: the patch-based visual odometry pipeline, its state machine, and the data flow from video input to 3D trajectory output.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Caller (CLI or Gradio)                              │
│                                                                             │
│  CLI: inference_dpvo()                   Gradio: run_dpvo()                 │
│  ┌───────────────────────────┐           ┌──────────────────────────────┐   │
│  │ 1. Load config (yacs)     │           │ 1. Load model (module-level) │   │
│  │ 2. Estimate calib (dust3r)│           │ 2. @rr.thread_local_stream   │   │
│  │ 3. Process all frames     │           │ 3. Process frames, yield     │   │
│  │ 4. Return DPVOPrediction  │           │    stream.read() per frame   │   │
│  └───────────┬───────────────┘           └──────────────┬───────────────┘   │
│              └──────────────┬───────────────────────────┘                   │
│                             ▼                                               │
│              ┌──────────────────────────────┐                               │
│              │    DPVO.__call__(t, img, K)  │  ◄── Per-frame entry point    │
│              │    (dpvo.py)                 │                                │
│              └──────────────┬───────────────┘                               │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  Patchifier   │  │  Edge Manager    │  │  Bundle Adjustment  │
│  (net.py)     │  │  (dpvo.py)       │  │  (ba.py + fastba/)  │
│               │  │                  │  │                     │
│  BasicEncoder │  │  __edges_forw()  │  │  Jacobians          │
│  → fmap, imap │  │  __edges_back()  │  │  Schur complement   │
│  → patches    │  │  → ii, jj, kk   │  │  SE3 retraction     │
└───────┬───────┘  └────────┬─────────┘  └──────────┬──────────┘
        │                   │                        │
        └───────────────────┼────────────────────────┘
                            ▼
              ┌──────────────────────────────┐
              │  Correlation + GRU Update    │
              │  (altcorr/ + net.py)         │
              │                              │
              │  reproject() → corr()        │
              │  → GRU → delta, weight       │
              └──────────────────────────────┘
```

## Data Flow

```
Input Video/Images
    │
    ▼
[Stream] (stream.py)
    multiprocessing.Process reads frames into Queue
    Output: (timestamp: int, bgr_hw3: uint8[h,w,3], intrinsics: float64[4] | None)
    │
    ▼
[Camera Calibration] (api/inference.py)
    If no calibration file: estimate K via mini-dust3r on first frame
    Output: intrinsics [fx, fy, cx, cy] as float64[4]
    │
    ▼
[Patchify] (net.py: Patchifier)
    Input:  image float32[1, 1, 3, H, W] (normalized to [-0.5, 0.5])
    ├─ fnet (BasicEncoder4) → fmap float32[1, 1, 128, H/4, W/4]
    ├─ inet (BasicEncoder4) → imap float32[1, 1, DIM=384, H/4, W/4]
    └─ gradient-biased random sampling → M=PATCHES_PER_FRAME patch locations
    Output:
      fmap:    float16[1, 1, 128, H/4, W/4]   feature map (2 pyramid levels)
      imap:    float16[1, M, DIM]              per-patch descriptors
      gmap:    float16[1, M, 128, 3, 3]        correlation template
      patches: float32[1, M, 3, 3, 3]          patch coords (x, y, disp) at 3×3
      clr:     uint8[M, 3]                     RGB at patch centers
    │
    ▼
[Buffer Storage] (dpvo.py)
    Rolling buffers indexed by keyframe counter n:
      tstamps_[n]              float64            timestamp
      poses_[n, 7]             float32            SE3: [tx,ty,tz, qx,qy,qz,qw]
      patches_[n, M, 3, 3, 3]  float32            patch coords
      intrinsics_[n, 4]        float32            [fx,fy,cx,cy] / RES
      colors_[n, M, 3]         uint8              RGB
      imap_[n%mem, M, DIM]     float16            descriptors
      gmap_[n%mem, M, 128,3,3] float16            correlation templates
      fmap1_[1, n%mem, 128, H/4, W/4]   float16   pyramid level 1
      fmap2_[1, n%mem, 128, H/16,W/16]  float16   pyramid level 2
    │
    ▼
[Edge Creation] (dpvo.py)
    __edges_forw(): recent patches → current frame
    __edges_back(): current frame's patches → recent frames
    Output: (ii, jj, kk) int64 tensors
      ii[E]: source frame index (patch origin)
      jj[E]: measurement frame index (observed in)
      kk[E]: global patch index (ii * M + local_idx)
    │
    ▼
[Correlation + Update] (projective_ops.py, altcorr/, net.py)
    1. reproject(poses, patches, intrinsics, ii, jj, kk)
       - iproj(): backproject patches to 3D using inverse depth
       - SE3 transform: Gij = poses[jj] * poses[ii].inv()
       - proj(): project to frame jj image coords
       → coords float32[1, E, 3, 3, 2]

    2. corr(gmap, fmap1, fmap2, coords, ...)
       - CUDA kernel: bilinear sample at reprojected coords
       - Multi-scale: radius-3 neighborhood at 2 pyramid levels
       → corr float32[1, E, 98]  (7×7 × 2 scales)

    3. Update network (GRU):
       - Input: hidden_state[1,E,DIM], imap[1,E,DIM], corr[1,E,98]
       - Neighbor aggregation across spatial/temporal edges
       - 2× GRU blocks with gated residuals
       → delta  float32[1, E, 2]   pixel displacement residual
       → weight float32[1, E, 2]   confidence (sigmoid, 0-1)
    │
    ▼
[Bundle Adjustment] (ba.py → fastba/ CUDA)
    Input: target (coords+delta), weight, poses, patches, intrinsics, ii, jj, kk
    1. Compute Jacobians:
       Ji float32[1,E,2,6]: ∂projection / ∂pose_ii
       Jj float32[1,E,2,6]: ∂projection / ∂pose_jj
       Jz float32[1,E,2,1]: ∂projection / ∂inverse_depth
    2. Build normal equations: H·dx = b
    3. Schur complement: eliminate depths → pose-only system
    4. Solve and retract:
       - pose_retr(): SE3 exponential map update
       - disp_retr(): inverse-depth additive update
    │
    ▼
[Keyframe Management] (dpvo.py: keyframe())
    Check motion between frames (i, i+4, i+6) via motionmag()
    If motion < KEYFRAME_THRESH:
      - Store relative pose in delta dict for later interpolation
      - Remove all edges connecting to this frame
      - Shift buffers to keep contiguous indexing
      - Decrement n, m counters
    │
    ▼
[Output]
    points_ float32[N*M, 3]: 3D point cloud (world coords)
    poses_  float32[N, 7]:   SE3 camera poses
    colors_ uint8[N, M, 3]:  point colors
    → Logged to Rerun for visualization
```

## State Machine

The DPVO class operates in two phases controlled by `is_initialized: bool`:

```
                    ┌────────────────────┐
                    │                    │
                    ▼                    │
┌──────────────────────────────┐        │
│         COLLECTING           │        │
│    (is_initialized=False)    │        │
│                              │        │
│  Frames 0–7:                 │        │
│  • Patchify each frame       │        │
│  • Create edges              │        │
│  • motion_probe(): check     │        │
│    median flow > 2.0 px      │        │
│  • If insufficient motion:   │────────┘
│    store delta, skip frame    (retry next frame)
│                              │
│  At frame 8:                 │
│  • Run 12 full BA iterations │
│  • Set is_initialized=True   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          TRACKING            │
│    (is_initialized=True)     │
│                              │
│  Per frame:                  │
│  • Patchify                  │
│  • Create edges              │
│  • 1× update() iteration:   │
│    reproject → corr → GRU    │
│    → BA                      │
│  • keyframe(): remove        │
│    low-motion frames         │
│  • Log to Rerun              │
└──────────────┬───────────────┘
               │ (all frames processed)
               ▼
┌──────────────────────────────┐
│         TERMINATED           │
│      (terminate())           │
│                              │
│  • 12 final BA iterations    │
│  • Build pose trajectory     │
│  • Interpolate deleted       │
│    keyframes via delta chain │
│  • Return (poses, tstamps)   │
└──────────────────────────────┘
```

## Tensor Shape Reference

### Constants
| Name | Value | Description |
|------|-------|-------------|
| N | 2048 | BUFFER_SIZE — max keyframes |
| M | 80 | PATCHES_PER_FRAME |
| DIM | 384 | Feature dimension |
| P | 3 | Patch size (3×3) |
| RES | 4 | Network stride |
| mem | 32 | Rolling feature map buffer |

### Buffers (DPVO class attributes)
| Buffer | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `tstamps_` | `[N]` | float64 | Frame timestamps |
| `poses_` | `[N, 7]` | float32 | SE3: [tx,ty,tz, qx,qy,qz,qw] |
| `patches_` | `[N, M, 3, 3, 3]` | float32 | Patch coords (x, y, disparity) |
| `intrinsics_` | `[N, 4]` | float32 | [fx,fy,cx,cy] / RES |
| `points_` | `[N*M, 3]` | float32 | 3D point cloud |
| `colors_` | `[N, M, 3]` | uint8 | RGB per patch |
| `imap_` | `[mem, M, DIM]` | float16 | Descriptor features |
| `gmap_` | `[mem, M, 128, 3, 3]` | float16 | Correlation templates |
| `fmap1_` | `[1, mem, 128, H/4, W/4]` | float16 | Feature pyramid L1 |
| `fmap2_` | `[1, mem, 128, H/16, W/16]` | float16 | Feature pyramid L2 |

### Edge Tracking
| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `ii` | `[E]` | int64 | Patch source frame |
| `jj` | `[E]` | int64 | Measurement frame |
| `kk` | `[E]` | int64 | Global patch index |
| `net` | `[1, E, DIM]` | float16 | GRU hidden state |

### Update Outputs
| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `coords` | `[1, E, 3, 3, 2]` | float32 | Reprojected patch centers |
| `corr` | `[1, E, 98]` | float32 | Correlation (7×7 × 2 scales) |
| `delta` | `[1, E, 2]` | float32 | Pixel displacement residual |
| `weight` | `[1, E, 2]` | float32 | Measurement confidence |

### Bundle Adjustment
| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `Ji` | `[1, E, 2, 6]` | float32 | Jacobian w.r.t. pose ii |
| `Jj` | `[1, E, 2, 6]` | float32 | Jacobian w.r.t. pose jj |
| `Jz` | `[1, E, 2, 1]` | float32 | Jacobian w.r.t. inv-depth |

## Network Architecture

```
VONet (net.py)
│
├─ Patchifier
│  ├─ fnet: BasicEncoder4(output_dim=128)
│  │  Conv2d(3→32, stride=2) → 2×ResBlock → Conv2d(32→64, stride=2)
│  │  → Conv2d(64→128) → fmap [batch, 128, H/4, W/4]
│  │
│  ├─ inet: BasicEncoder4(output_dim=DIM=384)
│  │  Same arch → imap [batch, 384, H/4, W/4]
│  │
│  └─ Patch extraction:
│     - Gradient-biased sampling if GRADIENT_BIAS=True
│     - Random otherwise
│     → M patches at 3×3 grid points
│
└─ Update (per BA iteration)
   Input: net[1,E,DIM], imap[1,E,DIM], corr[1,E,98]
   ├─ corr embedding: Linear(98→DIM)
   ├─ Neighbor aggregation (SoftAgg):
   │  scatter_softmax attention over spatial/temporal neighbors
   ├─ 2× GRU blocks: LayerNorm → GatedResidual
   └─ Output heads:
      ├─ delta: Linear(DIM→2)   pixel displacement
      └─ weight: Linear(DIM→2) → Sigmoid   confidence
```

## Module Map

| Module | Purpose |
|--------|---------|
| `dpvo.py` | Core SLAM class — buffers, state machine, keyframe management |
| `net.py` | VONet neural network (Patchifier + Update) |
| `ba.py` | Bundle adjustment orchestration (calls fastba CUDA) |
| `fastba/ba.py` | Schur complement solver (Python + CUDA via `_cuda_ba`) |
| `projective_ops.py` | Camera projection: iproj, proj, transform, point_cloud |
| `altcorr/correlation.py` | Local correlation via CUDA (`_cuda_corr`) |
| `scatter_utils.py` | Pure-PyTorch scatter_sum / scatter_softmax |
| `blocks.py` | GatedResidual, SoftAgg, GradientClip modules |
| `extractor.py` | BasicEncoder4 CNN backbone |
| `stream.py` | Video/image frame reader (multiprocessing) |
| `config.py` | YACS config defaults |
| `api/inference.py` | High-level inference pipeline + Rerun logging |
| `gradio_ui/dpvo_ui.py` | Gradio web interface |
| `data_readers/` | Dataset loaders (TartanAir, RGBD, generic) |
