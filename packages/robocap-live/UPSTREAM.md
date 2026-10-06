# Pieces written in kornia-rs style, for upstreaming

Each line: module in this crate -> target kornia crate/repo: what it is.

- `crates/robocap-types` (FrameMeta / Frameset / ImuSample / SourceEvent) and `src/frame.rs`'s dump format -> kornia-sensors (kornia-slam repo): multi-camera frameset + combined IMU sample
  types with integer-ns time; a raw multi-camera replay format.
- `src/source/mod.rs` FrameSource (rig, next_event) -> kornia-slam: an N-camera + IMU source trait (its roadmap item
  "multi-camera rigs, ported from slam-rs").
- Integer area resize is staged in `kornia-staging-imgproc::resize::area`; see [the staging tracker](../kornia-staging/README.md).
- `src/capture/v4l2_mplane.c` + `capture/camera.rs` -> kornia-io `v4l`: multi-planar (`VIDEO_CAPTURE_MPLANE`) NV12 capture with
  `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` checked and the luma plane copied out; kornia-io's `V4lVideoCapture` is single-planar only
  and drops the buffer flags. Adapted from PR #270.
- `src/capture/{iio.rs, device.rs, imu.rs}` -> kornia-sensors (kornia-slam repo): Linux IIO buffered IMU capture (scan layout
  validation, attribute save/restore), accel interpolated onto gyro stamps into combined `ImuMeasurement`-shaped samples, and a
  clock guard. Adapted from PR #270.
- `src/capture/matcher.rs` -> kornia-sensors / sensor-rt: timestamp-tolerance multi-camera frameset assembly.
- `src/kornia_ext/remap.rs` remap_f32_from_u8 -> kornia-imgproc `interpolation::remap`: u8 source to scaled f32 output (e.g. a [0, 1]
  network input) in one pass, zero padding per tap, matching torch `grid_sample(bilinear, zeros, align_corners=False)`.
- `src/kornia_ext/heatmap.rs` argmax_first / refine_peak_log_quadratic / decode_peak_2d -> kornia-imgproc `features` (or
  kornia-tensor-ops): separable log-quadratic sub-pixel heatmap peak decoding (exact for sampled Gaussians).
- `src/hands/letterbox.rs` BarLetterbox -> kornia-imgproc `preprocess`: the mono letterbox with its public pixel-centre maps
  (`to_net`/`from_net`), built on `spatial_padding`.
- `src/hands/circles.rs` min_enclosing_circle -> kornia-imgproc (contours/features): OpenCV's `minEnclosingCircle` (Welzl, f64,
  deterministic), with docs, a doctest and unit tests.
- `src/nets/rknn/api.rs` RknnRuntime / RknnModel -> kornia-rs `examples/rknn` (beside `examples/onnx`), or a small kornia runtime
  crate: a dlopened RKNN 2.x C-API binding (one context per NPU core, u8/f16/f32 inputs, float outputs into caller buffers,
  typed errors, rknn_destroy on drop). kornia has no inference crate, so this stays ours until one exists.
- `src/kornia_ext/pool.rs` pool4_u8 / pool4_mean_f32 -> kornia-imgproc resize: integer-factor box downscale (u8 rounded half to
  even like torch, and an unrounded f32 mean), the 4x4 case of the `resize_area_u8` gap.
- `src/hands/heatmaps.rs` decode_heatmaps (one decoder for the tracker and the golden comparison) -> kornia-tensor-ops / kornia-imgproc features: per-channel heatmap arg-max with a
  separable log-quadratic sub-pixel refinement (handtrack's decode_heatmaps).
- `src/log/video.rs` `AccessUnitSplitter` + `H264Encoder` -> kornia-io (gstreamer / video): a hardware H.264 video sink that does not
  link GStreamer. kornia-io's GStreamer `VideoWriter` links libgstreamer, which the cap binary must not (SPEC: no GStreamer at build
  time); this runs `gst-launch-1.0 filesrc location=/dev/stdin ! queue ! mpph264enc ! h264parse ! fdsink` (any encoder element) as a
  child, writes NV12 frames to its stdin and splits the Annex-B stdout into access units with their input timestamps. Cap A's
  GStreamer has no `rawvideoparse`; `filesrc` on `/dev/stdin` with `blocksize` = one frame does the framing.
