"""Encode PNG frames into an H.264 ``rr.VideoStream`` on a given entity path.

H.264 with ``yuv420p`` requires even dimensions; sequences with odd width or
height (e.g. ETH's 739x458) are centre-cropped by 1px on the affected axis.
B-frames are disabled so per-packet PTS equals the presentation time without
reordering — ``rr.VideoStream`` explicitly rejects B-frames.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import rerun as rr
from jaxtyping import Int64, UInt8
from numpy import ndarray


def _iter_rgb_frames(image_paths: list[Path]) -> Iterator[UInt8[ndarray, "h w 3"]]:
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"cv2 failed to read image: {p}")
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield rgb


def _probe_dimensions(first_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(first_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cv2 failed to read image: {first_path}")
    h, w = img.shape[:2]
    return w, h


def _pad_to_even(frame: UInt8[ndarray, "h w 3"]) -> UInt8[ndarray, "h w 3"]:
    h, w = frame.shape[:2]
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    if h_even == h and w_even == w:
        return frame
    return frame[:h_even, :w_even]


def encode_and_log_video(
    *,
    entity_path: str,
    image_paths: list[Path],
    timestamps_ns: Int64[ndarray, "n"],
    recording: rr.RecordingStream,
    timeline: str = "ts",
    fps_hint: float | None = None,
) -> tuple[int, int, int]:
    """Encode ``image_paths`` with H.264 and log as a ``rr.VideoStream``.

    Returns ``(width, height, num_frames_emitted)``. ``num_frames_emitted`` may
    be smaller than ``len(image_paths)`` if the encoder drops duplicate frames;
    that shouldn't happen with B-frames off and one input frame per step, but
    the caller can log a warning if the count is unexpected.
    """
    if len(image_paths) == 0:
        raise ValueError(f"no images to encode for {entity_path}")
    if len(image_paths) != len(timestamps_ns):
        raise ValueError(f"image_paths vs timestamps length mismatch: {len(image_paths)} vs {len(timestamps_ns)}")

    width, height = _probe_dimensions(image_paths[0])
    enc_w = width - (width % 2)
    enc_h = height - (height % 2)

    # H.264 wants an integer framerate; derive from source deltas when possible.
    if fps_hint is None or fps_hint <= 0:
        if len(timestamps_ns) > 1:
            dt_ns = np.median(np.diff(timestamps_ns))
            fps_hint = float(1e9 / max(float(dt_ns), 1.0))
        else:
            fps_hint = 30.0
    fps_int = max(1, int(round(fps_hint)))

    codec_ctx = av.codec.CodecContext.create("libx264", "w")
    codec_ctx.width = enc_w
    codec_ctx.height = enc_h
    codec_ctx.pix_fmt = "yuv420p"
    codec_ctx.framerate = Fraction(fps_int, 1)
    codec_ctx.time_base = Fraction(1, 1_000_000_000)  # PTS in nanoseconds
    codec_ctx.options = {
        "preset": "veryfast",
        "tune": "zerolatency",
        "bf": "0",         # disable B-frames — VideoStream forbids them
        "g": "30",         # keyframe interval
    }

    rr.log(entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True, recording=recording)

    t0 = int(timestamps_ns[0])
    emitted = 0
    for i, frame_rgb in enumerate(_iter_rgb_frames(image_paths)):
        frame_rgb = _pad_to_even(frame_rgb)
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = int(timestamps_ns[i]) - t0

        for packet in codec_ctx.encode(frame):
            # B-frames are disabled (bf=0), so PTS is always populated; guard just
            # in case and fall through to the input frame's PTS if the encoder
            # returns a packet without one.
            pts_ns = int(packet.pts) if packet.pts is not None else frame.pts
            rr.set_time(timeline, duration=pts_ns * 1e-9, recording=recording)
            rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)), recording=recording)
            emitted += 1

    for packet in codec_ctx.encode(None):
        pts_ns = int(packet.pts) if packet.pts is not None else 0
        rr.set_time(timeline, duration=pts_ns * 1e-9, recording=recording)
        rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)), recording=recording)
        emitted += 1

    return enc_w, enc_h, emitted
