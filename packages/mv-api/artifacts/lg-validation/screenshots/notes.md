# LG Synced Videos Validation

- Input root: `/mnt/8tb/data/exoego-self-collected/lg/ServerAssembly_4Views_11-3-25`
- Pipeline output: `packages/mv-api/artifacts/lg-validation/lg-full-synced-videos-1f.rrd`
- Data-only RRD for blueprint experiments: `packages/mv-api/artifacts/lg-validation/lg-full-synced-videos-1f-data-only.rrd`
- Full native Viewer screenshot: `packages/mv-api/artifacts/lg-validation/screenshots/lg-full-synced-videos-native.png`
- Per-view crops from the full native screenshot:
  - `packages/mv-api/artifacts/lg-validation/screenshots/views/rerun_crops/exo_FRONT.png`
  - `packages/mv-api/artifacts/lg-validation/screenshots/views/rerun_crops/exo_LEFT.png`
  - `packages/mv-api/artifacts/lg-validation/screenshots/views/rerun_crops/exo_RIGHT.png`
  - `packages/mv-api/artifacts/lg-validation/screenshots/views/rerun_crops/ego_TOP_visible.png`

## Command

```bash
pixi run -e mv-api mv-api-full-app \
  --rr-config.save artifacts/lg-validation/lg-full-synced-videos-1f.rrd \
  --rr-config.headless \
  --calib-config.no-refine-depth-maps \
  --calib-config.no-segment-people \
  --max-frames 1 \
  --camera-source estimated \
  synced-videos \
  --root-directory /mnt/8tb/data/exoego-self-collected/lg/ServerAssembly_4Views_11-3-25
```

## Result

- The LG run completed successfully in 11.37 seconds for one frame.
- The RRD contains 3 exo videos, 1 ego video, 4 pinholes/transforms, one triangulated 3D keypoint entity, one environment pointcloud, and one environment mesh.
- Rerun native screenshots had to run under `xvfb-run` because this shell has no `DISPLAY` or Wayland socket.
- Rerun still reports the known `40000x40000` WGPU validation toast under Xvfb, but it does save valid screenshots.
- The `view_id`/one-view `.rbl` screenshot route still did not produce useful individual 2D screenshots in this environment, so the per-view PNGs are crops from the full native Viewer screenshot.

## 2D Keypoint Ranges

Finite keypoints in the generated RRD:

- `exo_FRONT`: 43 finite, x `206.63..450.88`, y `261.23..319.67`
- `exo_LEFT`: 33 finite, x `0.12..131.35`, y `193.64..444.53`
- `exo_RIGHT`: 28 finite, x `542.27..571.68`, y `297.55..477.70`
- `ego_TOP`: 0 finite projected keypoints

The exo screenshots show projected hand keypoints in-frame. The ego screenshot only shows the video because the first-frame projected ego layer contains no finite keypoints after masking.
