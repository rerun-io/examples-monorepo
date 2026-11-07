"""Comparison recording: one depth layer as raw `DepthImage`, PNG `EncodedDepthImage` and `DepthVideoStream`, with error maps.

Reads a published layer that holds millimetre PNG depth under a pinhole (e.g. Gen-HumanEgo `depth_pinhole`), copies
it into the output, and adds beside the depth entity:

- `depth_raw_mm`: the decoded millimetres, uncompressed uint16, a stock `DepthImage` (the reference);
- `depth_video`: the depth as 10-bit codes in AV1 Main 10-bit monochrome lossless, logged with `log_depth_video`;
- `depth_codes_png`: the same 10-bit codes as 16-bit PNGs (control: the viewer must draw it pixel-identical to the video);
- `error_png_mm`, `error_video_mm`: per-pixel |decoded - raw| in mm, and `/errors/{png,video}/{max,p99}` per frame.

Codes are linear: code = round(mm / unit), unit = layer max depth / 1023 mm, 0 = no depth; `meter` = 1000 / unit, so the
video error is at most unit / 2. The default blueprint shows raw / PNG / video in 2D and as 3D clouds from one eye, then
the two error maps and the error plot; view names carry the bytes per frame and the largest error.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from jaxtyping import Int64, UInt16
from numpy import ndarray
from PIL import Image

from depth_video_viewer.encode import MAX_CODE, decode_av1, encode_av1
from depth_video_viewer.log import DepthVideoProperties, log_depth_video


@dataclass
class Config:
    source: Path
    """Layer .rrd with an `EncodedDepthImage` of millimetre PNGs."""
    output: Path
    """Where to write the comparison recording."""
    depth_entity: str = "/world/rig_00/cam_07/pinhole/depth"
    frames: int = 0
    """Keep only the first N frames (0 = all)."""
    gop: int = 10
    range_mm: float = 2000.0
    """Visible depth range, the same for every depth entity."""
    error_range_mm: float = 5.0
    """Colormap range of the error maps; 0 is black."""
    point_fill_ratio: float = 0.35
    timeline: str = "video_time"


def read_depth_pngs(source: Path, entity: str, timeline: str) -> tuple[Int64[ndarray, "n"], list[bytes]]:
    """Times (ns) and PNG blobs of an `EncodedDepthImage` entity, in time order."""
    times: list[int] = []
    blobs: list[bytes] = []
    for chunk in rrc.RrdReader(source).store().stream():
        if str(chunk.entity_path) != entity:
            continue
        batch = chunk.to_record_batch()
        if "EncodedDepthImage:blob" not in batch.schema.names:
            continue
        times += batch.column(timeline).cast("int64").to_pylist()
        blobs += [bytes(row[0]) for row in batch.column("EncodedDepthImage:blob").to_pylist()]
    order = np.argsort(np.asarray(times, np.int64), kind="stable")
    return np.asarray(times, np.int64)[order], [blobs[i] for i in order]


def png_u16(codes: UInt16[ndarray, "h w"]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(codes).save(buf, format="PNG")
    return buf.getvalue()


def main(config: Config) -> None:
    times, pngs = read_depth_pngs(config.source, config.depth_entity, config.timeline)
    if config.frames:
        times, pngs = times[: config.frames], pngs[: config.frames]
    if not pngs:
        raise ValueError(f"no EncodedDepthImage blobs at {config.depth_entity} in {config.source}")
    depth_mm = np.stack([np.asarray(Image.open(io.BytesIO(p)), np.uint16) for p in pngs])
    unit = max(1.0, int(depth_mm.max()) / MAX_CODE)
    codes = np.clip(np.rint(depth_mm / unit), 0, MAX_CODE).astype(np.uint16)
    samples = encode_av1(codes, gop=config.gop)
    if not np.array_equal(decode_av1(samples, *codes.shape[1:]), codes):
        raise RuntimeError("AV1 lossless round trip changed a code")

    decoded = decode_av1(samples, *codes.shape[1:])
    if not np.array_equal(decoded, codes):
        raise RuntimeError("AV1 lossless round trip changed a code")

    meter = 1000.0 / unit
    parent = config.depth_entity.rsplit("/", 1)[0]
    n, h, w = codes.shape
    # Errors against the raw millimetres, from decoded values: the PNG layer as stored, the video as ffmpeg decodes it
    # (lossless AV1 decoding is bit-exact across conforming decoders; the viewer's rav1d path is tested to match).
    errors = {"png": np.abs(depth_mm.astype(np.float32) - depth_mm), "video": np.abs(decoded.astype(np.float32) * np.float32(unit) - depth_mm)}
    valid = depth_mm > 0
    stats = {key: (err.reshape(n, -1).max(axis=1), np.array([np.percentile(e[v], 99) if v.any() else 0.0 for e, v in zip(err, valid, strict=True)]))
             for key, err in errors.items()}  # fmt: skip
    pngs_codes = [png_u16(c) for c in codes]
    kb = {"raw": depth_mm[0].nbytes / 1e3, "png": sum(map(len, pngs)) / len(pngs) / 1e3, "video": sum(map(len, samples)) / len(samples) / 1e3}
    max_mm = {"raw": 0.0, **{key: float(m.max()) for key, (m, _) in stats.items()}}

    rec = rr.RecordingStream("depth_video_viewer_comparison")
    rec.save(config.output)
    rec.send_chunks(rrc.RrdReader(config.source).store().stream())
    index = [rr.TimeColumn(config.timeline, duration=times * 1e-9)]
    log_depth_video(f"{parent}/depth_video", samples, times,
                    DepthVideoProperties(meter, rr.components.Colormap.Turbo, (0.0, config.range_mm / unit), config.point_fill_ratio),
                    timeline=config.timeline, recording=rec)  # fmt: skip
    rr.log(f"{parent}/depth_codes_png", rr.EncodedDepthImage.from_fields(media_type="image/png", meter=meter, depth_range=(0.0, config.range_mm / unit),
                                                                         colormap="turbo", point_fill_ratio=config.point_fill_ratio), static=True, recording=rec)  # fmt: skip
    rr.send_columns(f"{parent}/depth_codes_png", indexes=index, columns=rr.EncodedDepthImage.columns(blob=pngs_codes), recording=rec)
    u16 = rr.components.ImageFormat(width=w, height=h, channel_datatype="U16")
    rr.log(f"{parent}/depth_raw_mm", rr.DepthImage.from_fields(format=u16, meter=1000.0, depth_range=(0.0, config.range_mm), colormap="turbo", point_fill_ratio=config.point_fill_ratio),
           static=True, recording=rec)  # fmt: skip
    rr.send_columns(f"{parent}/depth_raw_mm", indexes=index, columns=rr.DepthImage.columns(buffer=[d.tobytes() for d in depth_mm]), recording=rec)
    # The published millimetre layer gets the same style, so every depth entity compares 1:1.
    rr.log(config.depth_entity, rr.EncodedDepthImage.from_fields(depth_range=(0.0, config.range_mm), colormap="turbo", point_fill_ratio=config.point_fill_ratio), static=True, recording=rec)

    f32 = rr.components.ImageFormat(width=w, height=h, channel_datatype="F32")
    for key, err in errors.items():
        entity = f"{parent}/error_{key}_mm"
        rr.log(entity, rr.DepthImage.from_fields(format=f32, meter=1000.0, colormap="inferno", depth_range=(0.0, config.error_range_mm)),
               static=True, recording=rec)  # fmt: skip
        rr.send_columns(entity, indexes=index, columns=rr.DepthImage.columns(buffer=[e.tobytes() for e in err]), recording=rec)
        for stat, values in zip(("max", "p99"), stats[key], strict=True):
            rr.send_columns(f"/errors/{key}/{stat}", indexes=index, columns=rr.Scalars.columns(scalars=values), recording=rec)

    # A transparent box over the image gives the 2D views bounds: the out-of-tree visualizer cannot add them, and a view
    # with no bounds draws "No data available." over the video.
    rr.log(f"{parent}/{BOUNDS}", rr.Boxes2D(mins=[0, 0], sizes=[w, h], colors=[[0, 0, 0, 0]]), static=True, recording=rec)
    rec.send_blueprint(three_way_blueprint(config.depth_entity, (w, h), kb, max_mm, config.error_range_mm))
    rec.flush()
    print(f"{n} frames {w}x{h}, unit {unit:.4f} mm (meter {meter:.2f}), kB/frame raw {kb['raw']:.1f} png {kb['png']:.1f} video {kb['video']:.1f}; "
          f"max error png {max_mm['png']:.3f} mm video {max_mm['video']:.3f} mm (p99 max {stats['video'][1].max():.3f}) -> {config.output}")  # fmt: skip


BOUNDS = "image_bounds"
VARIANTS = {"raw": ("depth_raw_mm", "Raw uint16 mm (DepthImage)"), "png": ("depth", "PNG mm (EncodedDepthImage)"),
            "video": ("depth_video", "AV1 10-bit (DepthVideoStream)")}  # fmt: skip
HIDDEN_IN_3D = ["depth_codes_png", "error_png_mm", "error_video_mm", BOUNDS]


def three_way_blueprint(depth_entity: str, size: tuple[int, int], kb: dict[str, float], max_mm: dict[str, float], error_range_mm: float) -> rrb.Blueprint:
    """Raw / PNG / video in 2D, the same three as 3D clouds from one eye, then the two error maps and the error plot."""
    pinhole = depth_entity.rsplit("/", 1)[0]
    camera = pinhole.rsplit("/", 1)[0]
    bounds = rrb.VisualBounds2D(x_range=[0.0, size[0]], y_range=[0.0, size[1]])
    eye = rrb.EyeControls3D(position=[0.0, -0.25, -0.35], look_target=[0.0, 0.0, 0.45], eye_up=[0.0, -1.0, 0.0])
    views_2d, views_3d = [], []
    for key, (entity, label) in VARIANTS.items():
        name = f"{label}: {kb[key]:.1f} kB/frame" + ("" if key == "raw" else f", max err {max_mm[key]:.2f} mm")
        views_2d.append(rrb.Spatial2DView(name=name, origin=pinhole, contents=[f"$origin/{BOUNDS}", f"$origin/{entity}"], visual_bounds=bounds))
        hidden = [f"- {pinhole}/{other}" for other in [e for e, _ in VARIANTS.values()] + HIDDEN_IN_3D if other != entity]
        views_3d.append(rrb.Spatial3DView(name=name, origin=camera, contents=["$origin/**", *hidden], eye_controls=eye, background=[20, 20, 24]))
    errors = [rrb.Spatial2DView(name=f"|{key} - raw| mm, 0-{error_range_mm:g} mm inferno: max {max_mm[key]:.2f} mm", origin=pinhole,
                                contents=[f"$origin/{BOUNDS}", f"$origin/error_{key}_mm"], visual_bounds=bounds) for key in ("png", "video")]  # fmt: skip
    plot = rrb.TimeSeriesView(name="Error per frame (mm): max and p99", origin="/errors")
    return rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*views_2d), rrb.Horizontal(*views_3d), rrb.Horizontal(*errors, plot)), collapse_panels=True)
