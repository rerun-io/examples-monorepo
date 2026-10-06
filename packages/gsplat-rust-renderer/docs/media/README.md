# Media provenance

Captured on 2026-10-06 from the integrated stack, with Rerun 0.38.1 and pinned
Brush `1388f74c`. These are viewer pixels, not illustrations or bundled predictions.

- `pretrained-lego.png`: custom viewer, native PLY import, no visualizer override.
  The portable-recording integration check captures the full Lego and base.
- `training-dashboard.gif`: a fresh `gsplat-rust-renderer-train lego 7000 video`
  recording. The Rerun viewer-validation skill's video helper captured 30 frames
  across `iterations=50..7000`, at 1280×720, with one second and five render steps
  of settling per frame. The GIF is 960×540, 5 fps, six seconds, with a 96-color
  palette. The report-local adapter maps the helper's older MCP names to 0.38.1.
- `training-progression.png`: the same fixed `Render 3` pane at the 1k, 3k, and 7k
  evaluation snapshots, from capture frames 4, 13, and 29. Only crop, resize, and
  labels were applied.
- `eval-pairs.png`: the four GT/render pairs at step 7,000, cropped from frame 29.
  These are the recording's fixed thumbnail views, not a full test-split score.

The training dashboard uses stock Rerun and the native archetype. The pretrained
capture uses the custom compute viewer. No old-pipeline media remains.

Evidence is retained under `~/gsplat-modern-work/reports/w4/`: the fresh RRD and
MP4 in `media/`, binary/recording hashes in `media/SHA256SUMS`, the helper adapter,
`media-video-helper.log`, `media-train.log`, and the full integration captures.
The training task is a correctness demonstration; its runtime is not a benchmark.
