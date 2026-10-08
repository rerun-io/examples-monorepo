# Pieces written in kornia-rs style, for upstreaming

Each line: module in this crate -> target kornia crate/repo: what it is.

Sensor payloads, timestamp matching, IMU interpolation, IIO ownership,
V4L2 capture, and subprocess video encoding now live in `../kornia-staging/`.
See its README table for their upstream destinations. RoboCap keeps its six-camera
constants, FrameSource, mounting orientation, encoder presets, dump schema, trigger
configuration, device profiles and clock guard.

- `src/hands/letterbox.rs` BarLetterbox -> kornia-imgproc `preprocess`: the mono letterbox with its public pixel-centre maps
  (`to_net`/`from_net`), built on `spatial_padding`.
- `src/nets/rknn/api.rs` RknnRuntime / RknnModel -> kornia-rs `examples/rknn` (beside `examples/onnx`), or a small kornia runtime
  crate: a dlopened RKNN 2.x C-API binding (one context per NPU core, u8/f16/f32 inputs, float outputs into caller buffers,
  typed errors, rknn_destroy on drop). kornia has no inference crate, so this stays ours until one exists.
