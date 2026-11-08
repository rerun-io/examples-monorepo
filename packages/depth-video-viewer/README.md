# depth-video-viewer

A Rerun 0.42 viewer that shows depth stored as AV1 10-bit monochrome lossless video (`DepthVideoStream`) in the
built-in 2D and 3D spatial views. In 2D the depth is a colormapped image; under a `Pinhole` in 3D it is a depth cloud,
exactly as for `DepthImage`/`EncodedDepthImage`.

It is an out-of-tree extension, like `gsplat-rust-renderer`'s viewer: published `=0.42.0` crates from crates.io,
no Rerun fork, no `[patch]`. It adds one visualizer to `Spatial2DView` and `Spatial3DView` through
`App::extend_view_class`.

The Rust archetype, decoder and visualizer and the Python `log.py` are out-of-tree scaffolding for checking depth as
video end to end. They are deleted once `DepthVideoStream` ships in Rerun; the encoder settings, the comparison tool
and the evidence stay here.

## Why a custom viewer

Stock Rerun keeps only 8 bits of 10-bit AV1 luma: rav1d decodes 10-bit streams in every build, and `re_video` then
right-shifts the samples to 8 bits (`decode/av1.rs:295-297`, TODO #7594). This viewer decodes with the same crate
(`re_rav1d`) and copies the Y plane as is, so every 10-bit code comes back exact. The codes then go through the same public upload
as `DepthImage` (`gpu_bridge::image_to_gpu`, an `R16Uint` texture), so the pixels equal those of a PNG
`EncodedDepthImage` of the same codes.

## Use

```bash
# From the repo root.
pixi run -e depth-video-viewer --frozen depth-video-viewer-build
pixi run -e depth-video-viewer --frozen depth-video-viewer -- recording.rrd     # or --headless, --port 9876
```

Log from Python with the stock rerun-sdk (`depth_video_viewer/log.py`, `encode.py`):

```python
from depth_video_viewer.encode import encode_av1
from depth_video_viewer.log import DepthVideoProperties, log_depth_video

samples = encode_av1(codes, gop=10)  # (n, h, w) uint16 codes <= 1023 -> one AV1 temporal unit per frame
log_depth_video("world/cam/pinhole/depth_video", samples, times_ns, DepthVideoProperties(meter=1000.0 / unit_mm))
```

`tools/make_comparison_rrd.py` builds a test recording from a published millimetre depth layer: raw uint16 mm
(`DepthImage`), the PNG layer as published and the `DepthVideoStream`, with a per-pixel error map |video - raw| in mm
and per-frame max/p99 error plots (the PNG error is zero by construction). Its default blueprint shows raw / PNG /
video in 2D and as 3D clouds from one eye, then the error map and the plot; view names carry bytes per frame and the largest error. It also logs the 10-bit codes
as 16-bit PNG (`depth_codes_png`), which the viewer must draw pixel-identical to the video.

## The archetype

`DepthVideoStream` uses built-in component types: `codec` (`VideoCodec`, required, AV1 only), `sample`
(`VideoSample`, one temporal unit per frame), `meter` (`DepthMeter`, codes per meter; default 1000), `colormap`
(default: Rerun's depth colormap), `depth_range` (`ValueRange` in codes; default `[0, 2^bit_depth - 1]`) and `point_fill_ratio`
(`FillRatio`). The descriptors are those of the in-tree prototype, `rerun.archetypes.DepthVideoStream` /
`DepthVideoStream:<field>`, so one recording works in both viewers.

Encoding (`encode.py`): libaom with `lossless=1`, so every code comes back exact; a fast `realtime` preset
(`cpu-used 8`), since a lossless stream decodes to the same codes at any preset, only its size changes; a fixed GOP (`-g` and
`-keyint_min`, default 10), so a seek decodes at most that many samples; `gray10le` input in full range, so no
limited-range scaling touches the codes. Codes are linear in depth with a per-recording unit, `meter = 1000 / unit_mm`;
code 0 means no depth, so depths below half a unit are rejected.

## Decoding and seeking

Decoding is synchronous in the visualizer (`src/decoder.rs`). Each entity keeps an index of its samples in decode
order (one range query, sorted by time then row id, rebuilt when its chunks change), one rav1d decoder, and the last
decoded sample: playing forward decodes one sample per frame; a seek walks back in the index to the keyframe at or
before the target and decodes from there (at most one GOP). Samples may share a time point.

## What is lost compared with the in-tree route

The in-tree prototype (branch `pablo/depth-video-stream` in a Rerun checkout) adds a `RawLuma` decode output to
`re_video` and a built-in visualizer. Out of tree, these are lost:

- **2D hover depth readout**: picking uses the crate-private `PickableTexturedRect`. Hovering the image shows no depth.
- **Auto-framing and the empty-view overlay**: the scene bounds (`SpatialViewVisualizerData`) are not exported, so the
  views ignore the depth when they frame the scene. Set `VisualBounds2D` / `EyeControls3D`. A 2D view whose only
  content is the depth video also draws "No data available." over the image (`ui_2d.rs:411` checks the bounds). The
  `Pinhole` at the view origin does not help; a transparent `Boxes2D` over the image does (the comparison tool logs one). (gsplat works around the 3D framing with fallback
  providers for the eye; not done here.)
- **Typed SDK archetype**: no generated Python/C++ `DepthVideoStream`; the helper logs built-in components under the
  archetype's descriptors.
- **Stock viewer**: recordings render only in this binary; the stock viewer ignores the entity.
- **Shared video machinery**: no reuse of the viewer's video cache, decode-ahead thread, or video error UI.
- **Web**: native only. The web viewer decodes through WebCodecs, which gives 8-bit RGBA; a wasm build would need
  rav1d compiled to wasm. The in-tree branch has the same web gap.

## Checks

`pixi run -e depth-video-viewer-dev --frozen gate` runs Ruff, pyrefly, vulture, pytest, and
`depth-video-viewer-rust-check` (rustfmt, Clippy `-D warnings`, `cargo test`). The Rust test decodes a 24-frame
32x16 AV1 10-bit stream (`src/testdata/`, made with `encode_av1`) and checks every code, then seeks forward, across
GOPs and backward.
