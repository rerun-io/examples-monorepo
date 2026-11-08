"""Comparison recording: one depth layer as raw `DepthImage`, PNG `EncodedDepthImage` and `DepthVideoStream`, with an error map.

Reads a published layer that holds millimetre PNG depth under a pinhole (e.g. Gen-HumanEgo `depth_pinhole`), copies
it into the output, and adds beside the depth entity:

- `depth_raw_mm`: the decoded millimetres, uncompressed uint16, a stock `DepthImage` (the reference);
- `depth_video`: the depth as 10-bit codes in AV1 Main 10-bit monochrome lossless, logged with `log_depth_video`;
- `depth_codes_png`: the same 10-bit codes as 16-bit PNGs (control: the viewer must draw it pixel-identical to the video);
- `error_video_mm`: per-pixel |video - raw| in mm, and `/errors/video/{max,p99}` per frame.

The PNG layer is lossless, so its error is zero by construction and only its view name says so. The default blueprint
shows raw / PNG / video in 2D and as 3D clouds from one eye, then the video error map and plot; view names carry the
bytes per frame and the largest error.
"""

import io
from collections.abc import Iterator
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

BOUNDS = "image_bounds"
CODES_PNG = "depth_codes_png"
ERROR = "error_video_mm"
EYE = rrb.EyeControls3D(position=[0.0, -0.25, -0.35], look_target=[0.0, 0.0, 0.45], eye_up=[0.0, -1.0, 0.0])


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
    """Colormap range of the error map; 0 is black."""
    point_fill_ratio: float = 0.35
    timeline: str = "video_time"


@dataclass(frozen=True, slots=True)
class Variant:
    """One way of storing the depth, as one 2D and one 3D view."""

    entity: str
    """Child of the pinhole entity."""
    label: str
    kb_per_frame: float
    error: str
    """What the view name says about the error against raw."""


def read_depth_pngs(chunks: list[rrc.Chunk], entity: str, timeline: str) -> tuple[Int64[ndarray, "n"], list[bytes]]:
    """Times (ns) and PNG blobs of an `EncodedDepthImage` entity, in time order."""
    blob_column = rr.EncodedDepthImage.descriptor_blob().component
    times: list[int] = []
    blobs: list[bytes] = []
    for chunk in chunks:
        if str(chunk.entity_path) != entity:
            continue
        batch = chunk.to_record_batch()
        if blob_column not in batch.schema.names:
            continue
        times += batch.column(timeline).cast("int64").to_pylist()
        blobs += [bytes(row[0]) for row in batch.column(blob_column).to_pylist()]
    order = np.argsort(np.asarray(times, np.int64), kind="stable")
    return np.asarray(times, np.int64)[order], [blobs[i] for i in order]


def to_codes(depth_mm: UInt16[ndarray, "n h w"]) -> tuple[UInt16[ndarray, "n h w"], float]:
    """Linear 10-bit codes and their unit in mm: code = round(mm / unit), unit = max depth / 1023 (at least 1 mm).

    Code 0 means no depth, so a positive depth below unit / 2 would vanish: such inputs are rejected.
    """
    unit = max(1.0, int(depth_mm.max()) / MAX_CODE)
    codes = np.clip(np.rint(depth_mm / unit), 0, MAX_CODE).astype(np.uint16)
    lost = int(np.count_nonzero((depth_mm > 0) & (codes == 0)))
    if lost:
        raise ValueError(f"{lost} depths below {unit / 2:.3f} mm would quantise to code 0 (no depth)")
    return codes, unit


def source_chunks(chunks: list[rrc.Chunk], timeline: str, end_ns: int | None) -> Iterator[rrc.Chunk]:
    """The source chunks; with `end_ns`, temporal rows after it on `timeline` are dropped (static data stays)."""
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if end_ns is None or chunk.is_static or timeline not in batch.schema.names:
            yield chunk
            continue
        kept = batch.filter(batch.column(timeline).cast("int64").to_numpy() <= end_ns)
        if kept.num_rows:
            yield from rrc.Chunk.from_record_batch(kept)


def png_u16(codes: UInt16[ndarray, "h w"]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(codes).save(buf, format="PNG")
    return buf.getvalue()


def main(config: Config) -> None:
    chunks = list(rrc.RrdReader(config.source).store().stream())
    times, pngs = read_depth_pngs(chunks, config.depth_entity, config.timeline)
    if config.frames:
        times, pngs = times[: config.frames], pngs[: config.frames]
    if not pngs:
        raise ValueError(f"no EncodedDepthImage blobs at {config.depth_entity} in {config.source}")
    depth_mm = np.stack([np.asarray(Image.open(io.BytesIO(p)), np.uint16) for p in pngs])
    codes, unit = to_codes(depth_mm)
    samples = encode_av1(codes, gop=config.gop)
    decoded = decode_av1(samples, *codes.shape[1:])
    if not np.array_equal(decoded, codes):
        raise RuntimeError("AV1 lossless round trip changed a code")
    if not np.array_equal(decoded > 0, depth_mm > 0):
        raise RuntimeError("the decoded video has depth where the source has none, or the other way round")

    meter = 1000.0 / unit
    pinhole = config.depth_entity.rsplit("/", 1)[0]
    n, h, w = codes.shape
    # The error from decoded values, as ffmpeg decodes the video: lossless AV1 decoding is bit-exact across conforming
    # decoders, and the viewer's rav1d path is tested to return the same codes.
    error = np.abs(decoded.astype(np.float32) * np.float32(unit) - depth_mm)
    error_max = error.reshape(n, -1).max(axis=1)
    error_p99 = np.array([np.percentile(e[d > 0], 99) if (d > 0).any() else 0.0 for e, d in zip(error, depth_mm, strict=True)])
    variants = [
        Variant("depth_raw_mm", "Raw uint16 mm (DepthImage)", depth_mm[0].nbytes / 1e3, "reference"),
        Variant(config.depth_entity.rsplit("/", 1)[1], "PNG mm (EncodedDepthImage)", sum(map(len, pngs)) / n / 1e3, "max err 0 (lossless)"),
        Variant("depth_video", "AV1 10-bit (DepthVideoStream)", sum(map(len, samples)) / n / 1e3, f"max err {error_max.max():.2f} mm"),
    ]

    rec = rr.RecordingStream("depth_video_viewer_comparison")
    rec.save(config.output)
    rec.send_chunks(source_chunks(chunks, config.timeline, int(times[-1]) if config.frames else None))
    index = [rr.TimeColumn(config.timeline, duration=times.astype("timedelta64[ns]"))]
    code_range = (0.0, config.range_mm / unit)
    log_depth_video(f"{pinhole}/depth_video", samples, times, DepthVideoProperties(meter, rr.components.Colormap.Turbo, code_range, config.point_fill_ratio),
                    timeline=config.timeline, recording=rec)  # fmt: skip
    rr.log(f"{pinhole}/{CODES_PNG}", rr.EncodedDepthImage.from_fields(media_type="image/png", meter=meter, depth_range=code_range, colormap="turbo",
                                                                      point_fill_ratio=config.point_fill_ratio), static=True, recording=rec)  # fmt: skip
    rr.send_columns(f"{pinhole}/{CODES_PNG}", indexes=index, columns=rr.EncodedDepthImage.columns(blob=[png_u16(c) for c in codes]), recording=rec)
    u16 = rr.components.ImageFormat(width=w, height=h, channel_datatype="U16")
    rr.log(f"{pinhole}/depth_raw_mm", rr.DepthImage.from_fields(format=u16, meter=1000.0, depth_range=(0.0, config.range_mm), colormap="turbo",
                                                                point_fill_ratio=config.point_fill_ratio), static=True, recording=rec)  # fmt: skip
    rr.send_columns(f"{pinhole}/depth_raw_mm", indexes=index, columns=rr.DepthImage.columns(buffer=[d.tobytes() for d in depth_mm]), recording=rec)
    # The published millimetre layer gets the same style, so every depth entity compares 1:1.
    rr.log(config.depth_entity, rr.EncodedDepthImage.from_fields(depth_range=(0.0, config.range_mm), colormap="turbo", point_fill_ratio=config.point_fill_ratio),
           static=True, recording=rec)  # fmt: skip

    f32 = rr.components.ImageFormat(width=w, height=h, channel_datatype="F32")
    rr.log(f"{pinhole}/{ERROR}", rr.DepthImage.from_fields(format=f32, meter=1000.0, colormap="inferno", depth_range=(0.0, config.error_range_mm)),
           static=True, recording=rec)  # fmt: skip
    rr.send_columns(f"{pinhole}/{ERROR}", indexes=index, columns=rr.DepthImage.columns(buffer=[e.tobytes() for e in error]), recording=rec)
    for stat, values in (("max", error_max), ("p99", error_p99)):
        rr.send_columns(f"/errors/video/{stat}", indexes=index, columns=rr.Scalars.columns(scalars=values), recording=rec)

    # A transparent box over the image gives the 2D views bounds: the out-of-tree visualizer cannot add them, and a view
    # with no bounds draws "No data available." over the video.
    rr.log(f"{pinhole}/{BOUNDS}", rr.Boxes2D(mins=[0, 0], sizes=[w, h], colors=[[0, 0, 0, 0]]), static=True, recording=rec)
    rec.send_blueprint(three_way_blueprint(pinhole, (w, h), variants, float(error_max.max()), config.error_range_mm))
    rec.flush()
    sizes = ", ".join(f"{v.entity} {v.kb_per_frame:.1f}" for v in variants)
    print(f"{n} frames {w}x{h}, unit {unit:.4f} mm (meter {meter:.2f}), kB/frame: {sizes}; video error max {error_max.max():.3f} mm, "
          f"p99 max {error_p99.max():.3f} mm -> {config.output}")  # fmt: skip


def three_way_blueprint(pinhole: str, size: tuple[int, int], variants: list[Variant], error_max: float, error_range_mm: float) -> rrb.Blueprint:
    """The variants in 2D, the same as 3D clouds from one eye, then the video error map and the error plot."""
    camera = pinhole.rsplit("/", 1)[0]
    bounds = rrb.VisualBounds2D(x_range=[0.0, size[0]], y_range=[0.0, size[1]])
    not_in_3d = [v.entity for v in variants] + [CODES_PNG, ERROR, BOUNDS]
    views_2d, views_3d = [], []
    for v in variants:
        name = f"{v.label}: {v.kb_per_frame:.1f} kB/frame, {v.error}"
        views_2d.append(rrb.Spatial2DView(name=name, origin=pinhole, contents=[f"$origin/{BOUNDS}", f"$origin/{v.entity}"], visual_bounds=bounds))
        hidden = [f"- {pinhole}/{other}" for other in not_in_3d if other != v.entity]
        views_3d.append(rrb.Spatial3DView(name=name, origin=camera, contents=["$origin/**", *hidden], eye_controls=EYE, background=[20, 20, 24]))
    error_map = rrb.Spatial2DView(name=f"|video - raw| mm, 0-{error_range_mm:g} mm inferno: max {error_max:.2f} mm", origin=pinhole,
                                  contents=[f"$origin/{BOUNDS}", f"$origin/{ERROR}"], visual_bounds=bounds)  # fmt: skip
    plot = rrb.TimeSeriesView(name="Video error per frame (mm): max and p99", origin="/errors")
    return rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*views_2d), rrb.Horizontal(*views_3d), rrb.Horizontal(error_map, plot)), collapse_panels=True)
