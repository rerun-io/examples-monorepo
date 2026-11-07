"""Comparison recording: one depth layer as a `DepthVideoStream` next to a PNG `EncodedDepthImage` of the same codes.

Reads a published layer that holds millimetre PNG depth under a pinhole (e.g. Gen-HumanEgo `depth_pinhole`), copies
it into the output, and adds two entities beside the depth entity:

- `depth_video`: the depth as AV1 Main 10-bit monochrome lossless, logged with `log_depth_video`;
- `depth_codes_png`: the same 10-bit codes as 16-bit PNGs, a stock `EncodedDepthImage` (the control).

Codes are linear: code = round(mm / unit), unit = layer max depth / 1023 mm, 0 = no depth; `meter` = 1000 / unit.
The viewer must draw both entities with identical pixels.
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
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
    """Visible depth range, the same for both entities."""
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

    meter = 1000.0 / unit
    depth_range = (0.0, config.range_mm / unit)
    parent = config.depth_entity.rsplit("/", 1)[0]
    rec = rr.RecordingStream("depth_video_viewer_comparison")
    rec.save(config.output)
    rec.send_chunks(rrc.RrdReader(config.source).store().stream())
    log_depth_video(f"{parent}/depth_video", samples, times, DepthVideoProperties(meter, rr.components.Colormap.Turbo, depth_range, config.point_fill_ratio),
                    timeline=config.timeline, recording=rec)  # fmt: skip
    control = f"{parent}/depth_codes_png"
    rr.log(control, rr.EncodedDepthImage.from_fields(media_type="image/png", meter=meter, colormap="turbo", depth_range=depth_range,
                                                     point_fill_ratio=config.point_fill_ratio), static=True, recording=rec)  # fmt: skip
    rr.send_columns(control, indexes=[rr.TimeColumn(config.timeline, duration=times * 1e-9)],
                    columns=rr.EncodedDepthImage.columns(blob=[png_u16(c) for c in codes]), recording=rec)  # fmt: skip
    # The published millimetre layer gets the same visible range, so all three entities compare 1:1.
    rr.log(config.depth_entity, rr.EncodedDepthImage.from_fields(depth_range=(0.0, config.range_mm)), static=True, recording=rec)
    rec.flush()
    kb = sum(map(len, samples)) / len(samples) / 1e3
    print(f"{len(samples)} frames {codes.shape[2]}x{codes.shape[1]}, unit {unit:.4f} mm (meter {meter:.2f}), AV1 {kb:.1f} kB/frame -> {config.output}")
