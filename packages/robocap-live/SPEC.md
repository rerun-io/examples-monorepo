# robocap-live: file formats and conventions

robocap-live runs on a RoboCap (RK3588): it takes the six cameras and the IMU, runs slam-rs VIO and the hand tracker (DetNet and
KeyNet on the NPU, the handfit fit), and streams the result to a Rerun viewer. The same core replays a dump of a recorded session.

This file holds what only it says: the dump format, the `--record` JSONL, and the code conventions. The types live in the code:

- `crates/robocap-live/src/frame.rs`: camera indices and names, `Frameset`, `ImuSample`, `Rig`, `DumpMeta`, `FrameHeader`.
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

## Runtime: the `--record` JSONL

`--record <file>` writes one JSON object per frameset (`RecordLine` in `src/sched/record.rs` is normative):

- `index`, `t_ns`;
- `world_from_rig`: 16 `f64`, row-major (identity without a pose);
- `slam_ok`, `slam_status` (`waiting_for_imu`, `no_visual_features`, `tracking`, `failed`, `off`, `reference`, or `none` before
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
