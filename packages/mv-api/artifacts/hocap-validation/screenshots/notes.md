# HoCap Viewer Screenshot Notes

- RRD: `packages/mv-api/artifacts/hocap-validation/hocap-full-ego.rrd`
- Screenshot: `packages/mv-api/artifacts/hocap-validation/screenshots/hocap-full-ego-native.png`
- Capture command:

```bash
xvfb-run -s '-screen 0 1920x1080x24' pixi run -e mv-api rerun \
  --renderer vulkan \
  --window-size 1920x1080 \
  --screenshot-to packages/mv-api/artifacts/hocap-validation/screenshots/hocap-full-ego-native.png \
  packages/mv-api/artifacts/hocap-validation/hocap-full-ego.rrd
```

- Result: Rerun saved the screenshot, then exited with a shutdown panic (`Failed to take store hub from the Viewer`). The PNG is present and visually valid.
- Visual check: the screenshot shows the HoCap 3D scene reconstruction, the ego camera view with projected hand keypoints and left/right hand boxes, the exo camera video strips with overlaid keypoints, and the `video_time` timeline.
- Pixel check: `(3840, 2160)` image, RGB mean `(44.94, 51.82, 37.89)`, RGB stddev `(41.85, 41.35, 39.97)`, per-channel extrema `0..255`.

## Screenshot-Driven Alignment Fix

- Corrected RRD: `packages/mv-api/artifacts/hocap-validation/hocap-full-filtered.rrd`
- Native Viewer screenshot: `packages/mv-api/artifacts/hocap-validation/screenshots/hocap-full-filtered-native.png`
- Individual 2D view crops:
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/ego_hololens_kv5h72_video.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_108222250342.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_117222250549.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_115422250549.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_043422252387.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_105322251564.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_037522251142.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_046122250168.png`
  - `packages/mv-api/artifacts/hocap-validation/screenshots/views/rerun_crops/exo_105322251225.png`
- Root cause: finite 2D points were logged outside each camera image, especially for ego (`x=-568..2549`, `y=610..2579`), which made Rerun auto-fit a much larger 2D extent than the video plane.
- Fix: keypoints and confidences are masked to `NaN` when outside the camera intrinsics bounds before logging 2D overlays.
- Post-fix RRD coordinate ranges:
  - ego `hololens_kv5h72`: `x=256.84..1259.76`, `y=610.62..715.85`, `n=13`
  - exo views are bounded within `640x480`; the largest finite ranges are `x=24.71..600.43`, `y=126.50..476.42`.
- Rerun 0.31.3 notes:
  - `ViewerClient.save_screenshot(..., view_id=...)` rejected both RRD blueprint IDs and live-sent view IDs with `View ... not found for screenshot`.
  - One-view `.rbl` screenshots and full screenshots reported a `40000x40000` WGPU validation error, but after resetting Rerun state the full native screenshot still saved. The per-view PNGs above are crops from that native Viewer screenshot.
