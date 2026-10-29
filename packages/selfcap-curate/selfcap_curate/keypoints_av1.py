"""Reference-preserving AV1 TU skip, using FFmpeg's maintained header parser."""

import re
import subprocess
from pathlib import Path

import av
import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray


def reference_packets(path: Path) -> list[bool]:
    """Keep refresh/control TUs without decoding pixels; abort on unparsed headers.

    Only explicit nonrefreshing INTER/INTRA_ONLY headers without state changes
    can be dropped. FFmpeg handles implicit KEY/SWITCH refresh and frame IDs.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-copyts",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-bsf:v",
        "trace_headers",
        "-f",
        "null",
        "-",
    ]
    result: list[bool] = []
    fields: dict[str, int] = {}
    active: bool = False
    keep: bool = False
    frames: int = 0

    def finish_frame() -> None:
        nonlocal keep, frames
        if not fields:
            return
        frames += 1
        kind = fields.get("frame_type")
        keep |= (
            fields.get("show_existing_frame") == 1 or kind in (0, 3) or bool(fields.get("error_resilient_mode", 0)) or "current_frame_id" in fields
        )
        if kind not in (0, 3) and fields.get("show_existing_frame") != 1:
            if "refresh_frame_flags" not in fields:
                raise ValueError("AV1 header missing refresh flags")
            keep |= bool(fields["refresh_frame_flags"])
        fields.clear()

    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as proc:
        assert proc.stderr is not None
        for line in proc.stderr:
            if "Packet:" in line:
                finish_frame()
                if active:
                    result.append(keep or not frames)
                active, keep, frames = True, False, 0
            elif active and "Frame Header" in line:
                finish_frame()
            elif active:
                if any(word in line for word in ("Sequence Header", "Metadata", "Tile List", "Redundant Frame Header")):
                    keep = True
                match = re.search(r"\s(show_existing_frame|frame_type|refresh_frame_flags|error_resilient_mode|current_frame_id)\s+\S+ = (\d+)", line)
                if match:
                    fields[match[1]] = int(match[2])
        finish_frame()
        if active:
            result.append(keep or not frames)
        if proc.wait() != 0 or not result:
            raise ValueError(f"FFmpeg AV1 header parse failed: {path}")
    return result


def filter_video(source: Path, dest: Path, wanted: Float64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Copy reference/control/selected TUs; verify PTS/counts before decoding."""
    keep: list[bool] = reference_packets(source)
    retained: list[float] = []
    samples: set[float] = set(wanted.tolist())
    with av.open(str(source)) as reader, av.open(str(dest), "w", options={"avoid_negative_ts": "disabled"}) as writer:
        stream = reader.streams.video[0]
        assert stream.time_base is not None
        target = writer.add_stream_from_template(stream, opaque=True)
        target.time_base = stream.time_base
        packets = (packet for packet in reader.demux(stream) if packet.pts is not None)
        for packet, required in zip(packets, keep, strict=True):
            assert packet.pts is not None
            stamp: float = float(packet.pts * stream.time_base)
            if required or stamp in samples:
                retained.append(stamp)
                packet.stream = target
                writer.mux(packet)
    with av.open(str(dest)) as reader:
        stream = reader.streams.video[0]
        assert stream.time_base is not None
        pts: Float64[ndarray, "m"] = np.sort(np.asarray([float(p.pts * stream.time_base) for p in reader.demux(stream) if p.pts is not None]))
    indices: Int64[ndarray, "n"] = np.searchsorted(pts, wanted)
    if (
        not np.array_equal(pts, np.sort(retained))
        or len(np.unique(pts)) != len(pts)
        or np.any(indices >= len(pts))
        or not np.array_equal(pts[indices], wanted)
    ):
        raise ValueError("AV1 remux changed/lost sample timestamps")
    return indices
