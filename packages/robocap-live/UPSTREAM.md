# Pieces written in kornia-rs style, for upstreaming

Each line: module in this crate -> target kornia crate/repo: what it is.

- `crates/robocap-types` (FrameMeta / Frameset / ImuSample / SourceEvent) and `src/frame.rs`'s dump format -> kornia-sensors (kornia-slam repo): multi-camera frameset + combined IMU sample
  types with integer-ns time; a raw multi-camera replay format.
- `src/source/mod.rs` FrameSource (rig, next_event) -> kornia-slam: an N-camera + IMU source trait (its roadmap item
  "multi-camera rigs, ported from slam-rs").
- `src/capture/v4l2_mplane.c` + `capture/camera.rs` -> kornia-io `v4l`: multi-planar (`VIDEO_CAPTURE_MPLANE`) NV12 capture with
  `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC` checked and the luma plane copied out; kornia-io's `V4lVideoCapture` is single-planar only
  and drops the buffer flags. Adapted from PR #270.
- `src/capture/{iio.rs, device.rs, imu.rs}` -> kornia-sensors (kornia-slam repo): Linux IIO buffered IMU capture (scan layout
  validation, attribute save/restore), accel interpolated onto gyro stamps into combined `ImuMeasurement`-shaped samples, and a
  clock guard. Adapted from PR #270.
- `src/capture/matcher.rs` -> kornia-sensors / sensor-rt: timestamp-tolerance multi-camera frameset assembly.
- `src/hands/letterbox.rs` BarLetterbox -> kornia-imgproc `preprocess`: the mono letterbox with its public pixel-centre maps
  (`to_net`/`from_net`), built on `spatial_padding`.
- `src/nets/rknn/api.rs` RknnRuntime / RknnModel -> kornia-rs `examples/rknn` (beside `examples/onnx`), or a small kornia runtime
  crate: a dlopened RKNN 2.x C-API binding (one context per NPU core, u8/f16/f32 inputs, float outputs into caller buffers,
  typed errors, rknn_destroy on drop). kornia has no inference crate, so this stays ours until one exists.
- `src/log/video.rs` `AccessUnitSplitter` + `H264Encoder` -> kornia-io (gstreamer / video): a hardware H.264 video sink that does not
  link GStreamer. kornia-io's GStreamer `VideoWriter` links libgstreamer, which the cap binary must not (SPEC: no GStreamer at build
  time); this runs `gst-launch-1.0 filesrc location=/dev/stdin ! queue ! mpph264enc ! h264parse ! fdsink` (any encoder element) as a
  child, writes NV12 frames to its stdin and splits the Annex-B stdout into access units with their input timestamps. Cap A's
  GStreamer has no `rawvideoparse`; `filesrc` on `/dev/stdin` with `blocksize` = one frame does the framing.
