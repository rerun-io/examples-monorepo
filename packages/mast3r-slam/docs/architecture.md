# MASt3R-SLAM architecture

The pipeline is one dagkit graph (`mast3r_slam/graphs/slam.py`); the CLI and the generated gradio app both run
`mast3r_slam.apis.slam.build`.

```
video ─frames─▶ frontend (main thread) ─task─▶ backend (own process) ─map, keyframes, cloud─▶ sinks
  └─rig (AsOf, calibrated only)─▶ ▲  └─tracked──────────────────────────────────────────▶ sinks
```

| node | layer module | does |
|---|---|---|
| `video` | `simplecv.sources` (`VideoSource`, `VideoFileSource`, `ImageDirSource`, `WebcamSource`) | one-view `MultiViewFrames`, every `subsample`-th frame |
| `frontend` | `nodes/frontend.py` (+ `nodes/tracking.py`) | INIT / TRACKING / RELOC per frame; emits `tracked` poses and `KeyframeAdded` / `RelocRequest` tasks |
| `backend` | `nodes/backend.py` (+ `nodes/factor_graph.py`, `models/retrieval.py`) | matching factors, retrieval, global Gauss-Newton; emits `MapUpdate` per task, a new keyframe's `KeyframeCloud`, and the final keyframe poses at flush |
| sinks | `dagkit.sinks.RerunSink` / `LayerSink`, `nodes/nerfstudio.py` | viewer, catalog layer `mast3r_slam`, Nerfstudio export |

Resources: `mast3r` (the model, shared with the backend by CUDA IPC) and `shared` (`nodes/shared.py`: keyframe buffer and
current-frame state in shared memory over an `mp.Manager`). The shared state is the feedback path of the SLAM loop
(poses, mode, relocalisation), so the graph has no cycle. CUDA IPC does not order GPU work across processes: each side
calls `SharedSlam.publish` (a device sync) before it hands over (the frontend before each task, the backend before each
acknowledgement), and the backend refuses a relocalisation request whose frame is not the shared current frame.

Modes: **lossless** (default, every eval run) — before each frame and at flush the frontend waits until the backend has
acknowledged every task it handed over (`SharedSlam.wait`: tasks are numbered in hand-over order, the backend counts
the ones it applied and wakes the waiter), so every frame sees one fixed map; **realtime** — no wait, but lost frames that
arrive before the backend acknowledged the relocalisation request are skipped (the shared current frame stays the requested one);
at the end the backend drops what is still queued and a stop is never undone by a late relocalisation. A failed backend
raises in the runner, which aborts the frontend's wait. Only the threaded runner fits the frontend's wait.
The backend process uses expandable CUDA segments and releases cached blocks after each task: the global solve's
edge-sized buffers grow with the map.

Types (`types.py`): `SlamConfig` (pyserde; `profiles/x86_64-cuda.toml` presets override it), `Frame` (working state),
`TrackedFrame`, `KeyframeAdded`, `RelocRequest`, `MapUpdate`, `KeyframeCloud` (the edge values carry `to_rerun` / `from_layer`). Outputs live under
`/mast3r_slam` on the input's `video_time`: final keyframe camera poses at `/mast3r_slam/<camera>` (one per keyframe at
its own frame time), tracked poses at `/mast3r_slam/tracked/<camera>`, the map at `/mast3r_slam/map` (each keyframe's cloud under its corrected
pose at `/mast3r_slam/map/keyframes/<index>/points`; viewer only, never in the catalog layer).

Kernels: matching (`ops/matching.py`) and the ray-cost Gauss-Newton step run as Mojo GPU kernels
(`mast3r_slam_mojo_backends`); `ops/optim.gauss_newton_rays` drives the iterations in Python (the dense solve on the CPU
is freed every iteration). The calibrated cost uses the CUDA extension (`_mast3r-slam-build-cuda-kernels`).
