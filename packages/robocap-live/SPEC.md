# robocap-live: file formats and conventions

robocap-live runs on a RoboCap (RK3588): it takes the six cameras and the IMU, runs slam-rs VIO and the hand tracker (DetNet and
KeyNet on the NPU, the handfit fit), and streams the result to a Rerun viewer. The same core replays a dump of a recorded session
and writes a catalog segment's `hands` layer.

This file holds what only it says: the dump format, the hands catalog layer, the `--record` JSONL, and the code conventions. The types live in the code:

- `crates/robocap-types`: camera indices and names, `FrameMeta`, `CameraFrame`, `Frameset`, `ImuSample`, `SourceEvent`, and
  `SlamPose` / `SlamStatus` / `SlamStages` (no runtime, device or estimator; `frame.rs` and `slam.rs` re-export them).
- `crates/robocap-live/src/frame.rs`: `Rig`, `DumpMeta`, `FrameHeader`.
- `crates/robocap-live/src/nets/mod.rs`: the `HandNets` trait and the raw net outputs; `models/MODELS.md` for the model files.
- `crates/robocap-live/src/hands/mod.rs`: the tracker's inputs, config and per-frameset results.
- `crates/robocap-live/src/sched/record.rs`: `RecordLine`, the record JSONL below.
- `crates/robocap-live/src/main.rs`: the command line (`robocap-live --help`); `scripts/` for build, deploy, start and stop.

The small image: the runtime makes one 640x360 image per camera from the 1920x1080 frame, an exact area /3 (each output pixel is
the rounded mean of a 3x3 block; `src/downsample.rs`). SLAM, DetNet's letterbox (`src/hands/letterbox.rs`) and the viewer use it.

## The dump format (`robocap-live-dump/1`)

A directory that the replay source (`--source replay <dir>`, `src/source/replay.rs`) reads. The s66 dumps were exported from the
catalog by a Python tool that is not in this tree; `frame.rs`'s `FrameWriter` writes `frames.bin` for tests and examples.

- `meta.json`: `DumpMeta`: `{format: "robocap-live-dump/1", source, segment, device ("cap_a" or "cap_b"), frames, width, height,
  cameras: [6 names], first_t_ns, last_t_ns}`. The replay source accepts only 1920x1080.
- `rig.json`: `Rig`: six `RigCamera { name, width, height, cam_from_rig (row-major 4x4, metres), focal: [fx, fy],
  principal: [cx, cy], fisheye62: [k1..k6, p1, p2] or null }` in index order, plus `source` and `device`. Unknown fields are
  refused. The live source reads the same file (`--rig`, default `/root/robocap-live/rig.json`).
- `frames.bin`: `frames` records. Each starts with the 80-byte little-endian `FrameHeader`: `magic b"RLF1"`, `u32 version = 1`,
  `u64 index`, `i64 t_ns`, `u8 present_mask` (bit c = camera c present), `7 x u8 0`, `i64 cam_t_ns[6]` (0 when absent). After it,
  for each present camera in index order, `width * height` luma bytes (row-major, stride = width). `FrameHeader` is normative.
- `imu.bin`: combined IMU0 records, 56 bytes little-endian: `i64 t_ns`, `f64 gyro[3]` (rad/s), `f64 accel[3]` (m/s^2), in time
  order, with accel linearly interpolated onto the gyro stamps.
- `reference_world_from_rig.bin` (optional): per frameset, `i64 t_ns` + 16 `f64` (row-major): the catalog's SLAM pose, which
  `--slam reference` replays instead of running VIO.

Times are nanoseconds: CLOCK_MONOTONIC on the cap, the catalog's video time in a dump.

## The hands catalog layer

One core, two front ends. The cap binary runs `sched::run` on the live source. `robocap_live._core` (`crates/robocap-live-py`,
PyO3; built in place by the `robocap-live-build` pixi task and not a default workspace member, so the cap build never compiles
it) runs the same `sched::run` through `layer::HandsLayerWriter`: Python pushes framesets into a `source::channel` source, and the
pipeline is that of `--source replay <dump> --slam reference --hands on`: lossless queues, `downsample::small_images`, the
`ReferencePoses` and the pose store, the hands stage's tracker step, and the output stage's `log::LoggerSink` with the logger in
`scene::Content::HandsLayer` mode. On the CPU ONNX Runtime the layer of a catalog segment equals that replay's of the segment's
dump.

The Python side (`robocap_live.catalog_segment`) reads the rig from the `base` layer's statics, the six H.264 streams in one bulk
query, decoded to the raw Y plane (the cap's NV12 Y), groups framesets as the live adapter does (3 ms), and gives each frameset
the nearest `slam_rs` pose of `/world/rig_00` within 5 ms. `robocap_live.apis.hands_layer` (`pixi run -e robocap-live
robocap-hands-layer`) writes the layer and registers it as the segment's `hands` layer, replacing an earlier one.

The layer holds only the hand entities (`/world/hands/{side}/{keypoints,mesh}`, the camera-pane overlays in full-resolution
pixels, the skeleton's `AnnotationContext` on `/world`) and no recording properties; its recording id is the segment id and its
times are the catalog's `video_time`.

## SLAM lane, lag and pose time

`--slam-lane gpu|cpu` defaults to GPU on aarch64 and CPU on other hosts. Cap builds enable `robocap-live/gpu-wgpu`.
GPU startup failure emits a warning and falls back to CPU. The startup line and summary fields `slam_lane`, `slam_frontend_lag`
and `slam_threads` report the actual lane, lag and worker count; those fields are null when SLAM is off or uses reference poses.
GPU defaults to one worker and `port.frontend_lag=true`; CPU defaults to two workers and lag off. `--slam-threads` accepts
one or two workers; the existing cap affinity keeps SLAM and the threads it creates on cores 6–7.
`--slam-set port.frontend_lag=true|false` overrides the lane default, including after fallback. The panel offers GPU/CPU and
auto/on/off lag. `start-live.sh --slam-lag auto|true|false` offers the same selection; auto sends no lag override.

SLAM consumes the front-left, front-right, left and right cameras, in that order (rig indices `[0, 1, 4, 5]`). The embedded
Cap A and Cap B Basalt calibrations use the same order. The front stereo pair gives camera 0 overlapping views for initial
triangulation; the six-camera capture and hand-tracking order is unchanged.

The [slam-rs estimator](../slam-rs/README.md) defines when visual tracking starts; until then no usable pose is published.

With lag, the first accepted frameset buffers without a pose. Each later call returns the previous accepted frameset's pose.
SLAM matches the result timestamp to a pending input record, preserving that input's index even after rate skips or drops.
An already queued next selected frameset supplies an optional GPU lookahead hint; the SLAM stage never waits for a hint.
EOF and normal stop drain the queues and flush the last result before the pose store closes. A world is reset only on a
backwards input timestamp, a gap **greater than 3 seconds** between accepted inputs, or an estimator error. Shorter gaps,
including exactly 3 seconds, keep the world. Backwards time is checked before rate selection can reject the frame.
Time boundaries first flush the valid pending old-world pose. Failed track/flush calls discard pending estimates. Every reset
emits one stderr line with its reset count and cause: backwards timestamps, gap in milliseconds, or estimator error text.
Pose progress and world boundaries use monotonic source frameset indices, so rewound timestamps cannot reuse old-world poses.
Queued readers of earlier frames retain their published poses, with lag either on or off.

IMU is read only through the current frame's coverage (a sample strictly after its timestamp). Without coverage, live mode
waits up to the IMU timeout; lossless mode waits for samples or EOF. An uncovered frame is skipped and counted in
`slam_imu_timeouts`; it does not by itself reset SLAM. On backwards time, the previous world's queued IMU tail is discarded
until sample timestamps return to the earlier range. A first sample just after the new frame supplies coverage as usual.
Internal IMU sample spacing has no separate reset threshold.
Replay loops keep monotonically increasing output timestamps and insert a 3.5-second gap to start a new world.

- **Live/realtime:** progress advances when SLAM consumes or skips the input. Hands use the newest usable pose at or before
  their timestamp, subject to `--hands-wait-ms`. They do not wait an extra frameset for a lagged pose.
- **Lossless/offline:** progress cannot pass an accepted input until its pose is published or flushed. Hands and output use
  that frame's own estimate, including an estimate that has not yet reached visual tracking. A rate/input skip holds the last
  pose. Before a lossless skip, SLAM flushes any pending estimate so bounded queues cannot stall waiting for the next selected
  input. A missing catalog reference is also a skip; it holds the last real reference, never an identity placeholder.

Pose indices, timestamps, visual-support counts and keyframe flags describe the returned estimate. `compute_ms` describes
the track/flush call that produced it; with lag the frontend timers therefore describe the *next* submitted input. Estimator
and frontend work overlap, so their times must not be added. Flush has zero frontend time. Summary `slam` rate and timings
count track calls, including the buffered first call; `slam_flush` records drain work separately. Counters `slam_buffered`
and `slam_lookahead` expose buffering and supplied hints. A supplied hint may be discarded if its queued frame later changes.

## Shipping the Vulkan loader

The cap does not have a system Vulkan loader. Deploy `lib/libvulkan.so.1` beside `bin/`; keep the cap's own Mali ICD.
The loader is the conda-forge `libvulkan-loader` package that `pixi.lock` pins for the robocap-live environment on linux-aarch64
(declared in `pixi.toml`, `feature.robocap-live.target.linux-aarch64`). `scripts/stage-vulkan.sh` reads that package's URL and
sha256 from the lock, downloads it once into `~/.cache/robocap-live/vulkan`, verifies the sha256, checks glibc <= 2.34
and direct shared-library dependencies, then copies the loader and a provenance note (package URL and package checksum).
`deploy.sh` includes these files in its checksum-verified staging tree.
The aarch64 binary's `$ORIGIN/../lib` RUNPATH finds the loader for every launcher; `build-arm.sh` checks that path.
Vulkan is opened at runtime; the cap build still refuses a Vulkan, GStreamer, RKNN or RGA link dependency.

## Runtime: the `--record` JSONL

`--record <file>` writes one JSON object per frameset (`RecordLine` in `src/sched/record.rs` is normative):

- `index`, `t_ns`;
- `slam_index`, `slam_t_ns`: identity of the pose actually used, or null without a pose;
- `world_from_rig`: 16 `f64`, row-major (identity without a pose);
- `slam_ok`, `slam_status` (`no_visual_features`, `tracking`, `failed`, `off`, `reference`, or `none` before
  any pose), `slam_landmarks` (landmarks in the window), `slam_tracked` (observations of them in this frameset), `slam_optimised`;
- `hands`: empty when the hands stage did not run, else left then right, each `{tracked, reported, landmarks (21 x 3, world
  metres, or null), views: [{camera, keypoints_px (21 x 2, full-resolution pixels), presence, pinch (or null)}], detnet_camera,
  detnet_circle ([cx, cy, r] in the DetNet net frame, or null)}`;
- `detnet_camera`: the camera DetNet ran on in this frameset, or null; `scale`: the hand scale in use, or null without hands;
- `timings_ms`: `downsample_ms`, `slam_ms`, `slam_frontend_ms`, `slam_optimize_ms`, `slam_marginalize_ms`, `pose_wait_ms`,
  `hands_ms`, `detnet_ms`, `crops_ms`, `keynet_ms`, `fit_ms`, `tracker_ms`, `pipeline_ms`; null for a stage that did not run.

## Code conventions (kornia-rs style)

- kornia-rs comes from the git revision pinned in `Cargo.toml`, the one slam-rs uses, so both share one `Image` type. Do not add
  crates.io kornia crates beside it. Images are `kornia_image::Image<u8, 1>` (tight rows, no stride), shared as `Arc` (`Luma`).
- Image ops write into a destination: `fn op(src: &Image<..>, dst: &mut Image<..>, params) -> Result<(), ImageError>`, named
  output first with a dtype suffix (`gray_from_rgb`, `_u8`, `_f32`); parallel work is rayon over row chunks; SIMD kernels have
  NEON and scalar paths.
- Errors are `thiserror` enums per module. `anyhow` only in `main.rs`; no `unwrap`/`expect`/`panic!` in library code. Every
  `unsafe` block has a `// SAFETY:` comment.
- Modules meant for upstream (UPSTREAM.md, `src/kornia_ext/*`) carry `#![deny(missing_docs)]`-level docs: each pub fn has
  `# Arguments`, `# Returns`, `# Errors` and, where cheap, a doctest.
- The cap binary links neither GStreamer nor librknnrt (nor librga): librknnrt is dlopened at run time, and H.264 goes through
  `gst-launch-1.0` child processes fed by pipes. `scripts/build-arm.sh` checks the link-time dependencies and the glibc 2.34 floor.
