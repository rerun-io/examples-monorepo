# Worked example — LiteAnyStereo V2 (2026-08-29)

Fork: `pablovela5620/LiteAnyStereo` (`main` = upstream 8c97bd4, `pixi` default). Monorepo PR stack
#151 `liteanystereo/1-vendor` → #152 `2-predictor` → #154 `3-typed` → #155 `4-app` → #157 `5-catalog-layer`.

## What landed

- `monopriors/third_party/liteanystereo/` (typed owned fork: `liteanystereov2.py`, `_H.py`, `fnet.py`,
  `aggregation_fasternet.py`, `submodule.py`, `padding.py`), fixtures under `tests/reference_data/liteanystereo/`.
- `monopriors/models/stereo_depth/`: `StereoDepthPrediction`, `BaseStereoPredictor`, `disparity_to_metric_depth`,
  `LiteAnyStereoPredictor(device, model_size="m", checkpoint=None, max_disp=192)`, `rectify.py`
  (`fisheye_stereo_rectify` on `Fisheye62Parameters` → `StereoRectification` with `PinholeParameters`).
- `apis/stereo_depth.py` + `tools/demos/stereo_depth.py` (ETH3D demo), `gradio_ui/stereo_depth_ui.py` +
  `tools/apps/stereo_depth_app.py`, `apis/stereo_catalog.py` + `tools/apps/stereo_catalog.py`
  (robocap front pair → rectify → stereo → `EncodedDepthImage` + incremental TSDF mesh, streamed to the viewer).
- simplecv additions: `rig.stereo_rig_calibration`, `rerun_dataloader(codec=)`, `log_open3d_mesh`.

## Numbers

ETH3D `playground_1l` (non-occluded, gt < 192): LAS2-M EPE 0.350 / bad1 2.24 % (paper 2.59), LAS2-H 0.250 / 1.12 %
(paper 1.83). 5090 fp32 warm 384×1248: S 5.4 / M 6.7 / L 10.4 / H 14.8 ms (launch-bound); 1080p M 28 ms, H 62 ms.

## Gotchas hit (all now encoded in the phase files)

- `simplecv` git dep lacks runtime deps (`av`, `pyarrow`, `einops`) and needs `typing-extensions<4.16`.
- Upstream S/M/L ignore `max_disp` at build time — tests use 192.
- Rectified cameras logged as children of `cam_00` double-applied `rig_T_cam` (cloud pointed down); a level TSDF
  mesh did not catch it. Fix: rectified views are rig sensors (`cam_10`/`cam_11`) with `R_rect @ cam_T_rig`.
- `cv2.fisheye.stereoRectify`'s estimated `P` is unusable for wide fisheyes (`cx` 2721 px) → own `K_rect`
  (centre principal point, `focal_scale` 0.8).
- `$origin/**` in a 3D view pulls 2D-only entities in (error badge); sky disparities streak the cloud (mask > 20 m).
- Gradio: `np.float32` defaults not JSON-serialisable; int sliders need `float | int`; localhost self-check fails
  inside the sandbox; plain http on the tailnet breaks the embedded viewer (`crypto.randomUUID`) → tailscale HTTPS.
- `VideoStream:codec` fourcc endianness → `rr.VideoCodec(value)`. Conda `py-opencv` needs ffmpeg 9 (no ffmpeg pin).
- Depth colour range 0–20 m hid indoor scenes → default 0–6 m; blueprint overrides via
  `rr.EncodedDepthImage.from_fields(depth_range=...)`; sending a blueprint after a fresh `rr.init` makes empty
  recordings → bind the `recording_id`.
- A process-kill pattern that also appeared in the same command line killed the tool shell four times → a
  PreToolUse hook now blocks it; use exact names, saved PIDs, `fuser -k <port>/tcp`, or tmux sessions.
- Live `rr.spawn()` mode of the catalog tool was ~10× slower than headless save + open (unresolved); use save.
