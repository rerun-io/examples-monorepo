# Checked-in test fixtures

Small binaries the tests need on **any** machine, GPU or not. Anything that has
to prove the encoder itself works stays an NVENC-gated encode in
`test_encoding.py`; these files exist so everything downstream of an mp4 — the
`log_video_stream` remux and its retiming — runs without one.

| file | how it was made |
| --- | --- |
| `av1_48f_192x160.mp4` | `ffmpeg -f lavfi -i "color=c=gray:s=192x160:r=24:d=2" -frames:v 48 -c:v av1_nvenc -bf 0 -g 12 -cq 40 -pix_fmt yuv420p -movflags +faststart av1_48f_192x160.mp4` — 48 samples, keyframes every 12, no B-frames (`rr.VideoStream` rejects reordered samples), 2.3 KB. |
| `msd/{index,g2,odyssey}-calibration.json` | Verbatim copies of `M_monado_datasets/<device>/extras/calibration.json` from the upstream HuggingFace dataset `collabora/monado-slam-datasets` (CC-BY 4.0). Nothing is stripped, so `load_calibration` is tested against what upstream really ships. |
