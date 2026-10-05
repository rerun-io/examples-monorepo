# Pieces written in kornia-rs style, for upstreaming

Each line: module in this crate -> target kornia crate/repo: what it is.

- `crates/robocap-types` (FrameMeta / Frameset / ImuSample / SourceEvent) and `src/frame.rs`'s dump format -> kornia-sensors (kornia-slam repo): multi-camera frameset + combined IMU sample
  types with integer-ns time; a raw multi-camera replay format.
- `src/source/mod.rs` FrameSource (rig, next_event) -> kornia-slam: an N-camera + IMU source trait (its roadmap item
  "multi-camera rigs, ported from slam-rs").
- `src/downsample.rs` `resize_area_u8` -> kornia-imgproc `resize`: exact integer-factor box (area) downscale for any `C`, rayon over
  row chunks, NEON `vld3q_u8` kernel for the 3x3 mono case with software prefetch (the A55 prefetcher does not follow three
  interleaved row streams). kornia has no Area mode; at 1080p -> 360p on an RK3588 A55 it runs 1.27 ms vs `resize_fast_mono_aa`
  bicubic+aa 57 ms / bilinear 5.5 ms (bilinear at an exact /3 is point sampling). Needs an OpenCV `INTER_AREA` parity test.
- `src/capture/v4l2_mplane.c` + `capture/camera.rs` -> kornia-io `v4l`: multi-planar (`VIDEO_CAPTURE_MPLANE`) NV12 capture with
  `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` checked and the luma plane copied out; kornia-io's `V4lVideoCapture` is single-planar only
  and drops the buffer flags. Adapted from PR #270.
- `src/capture/{iio.rs, device.rs, imu.rs}` -> kornia-sensors (kornia-slam repo): Linux IIO buffered IMU capture (scan layout
  validation, attribute save/restore), accel interpolated onto gyro stamps into combined `ImuMeasurement`-shaped samples, and a
  clock guard. Adapted from PR #270.
- `src/capture/matcher.rs` -> kornia-sensors / sensor-rt: timestamp-tolerance multi-camera frameset assembly.
- `src/kornia_ext/fisheye.rs` MonotonicFisheye -> kornia-3d `camera::fisheye` (bug report + fix): `FisheyeCamera::unproject` starts Newton
  at `theta = theta_d`, so when the KB4 polynomial peaks before 90 deg (s66 right_front: ~84 deg) edge pixels converge on the far,
  decreasing branch (7 deg off at pixel (100, 80), yet reprojecting onto it). The fix finds the peak once and solves on [0, theta_max].
- `src/kornia_ext/virtual_camera/` maps_from_virtual_pinhole_f32 (+ `_kb4` f32/NEON fast path, `RayProjection` trait) -> kornia-3d `camera` /
  kornia-imgproc `calibration`: remap maps for a rotated virtual pinhole sharing a source camera's centre (perspective crops of a
  fisheye image), beside the undistort maps.
- `src/kornia_ext/remap.rs` remap_f32_from_u8 -> kornia-imgproc `interpolation::remap`: u8 source to scaled f32 output (e.g. a [0, 1]
  network input) in one pass, zero padding per tap, matching torch `grid_sample(bilinear, zeros, align_corners=False)`.
- `src/kornia_ext/heatmap.rs` argmax_first / refine_peak_log_quadratic / decode_peak_2d -> kornia-imgproc `features` (or
  kornia-tensor-ops): separable log-quadratic sub-pixel heatmap peak decoding (exact for sampled Gaussians).
- `src/hands/letterbox.rs` BarLetterbox -> kornia-imgproc `preprocess`: the mono letterbox with its public pixel-centre maps
  (`to_net`/`from_net`), built on `spatial_padding`.
- `src/hands/circles.rs` min_enclosing_circle -> kornia-imgproc (contours/features): OpenCV's `minEnclosingCircle` (Welzl, f64,
  deterministic), with docs, a doctest and unit tests.
- `src/hands/scale.rs` calibrate_scale -> handfit (or kornia-3d's BA tooling): one shared model scale + per-frame poses by LM with a
  Schur complement on the scale, analytic scale column.
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
