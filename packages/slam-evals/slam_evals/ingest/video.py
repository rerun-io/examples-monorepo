"""Encode PNG frames into an H.265 ``rr.VideoStream`` via NVENC.

NVENC saturates the GB10 GPU at SLAM-benchmark resolutions (~1100 fps),
so the practical bottleneck is PNG decode + ndarray reformat, not the
encoder. AV1 produces smaller files at matched PSNR but rerun's web /
local viewers don't decode AV1 reliably yet — HEVC is the safe choice
and at ``cq=32`` gives PSNR equal-or-better than AV1 ``cq=40`` on every
test sequence (color and grayscale).

``yuv420p`` still requires even dimensions; sequences with odd width or
height (e.g. ETH's 739x458) are centre-cropped 1px on the affected axis.
B-frames are disabled so per-packet PTS equals the presentation time
without reordering — ``rr.VideoStream`` explicitly rejects B-frames.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import cv2
import numpy as np
import rerun as rr
from jaxtyping import Int64, UInt8
from numpy import ndarray


def _decode_one(path: Path) -> UInt8[ndarray, "h w 3"]:
    """Decode a PNG to BGR (uint8, 3 channels). Releases GIL inside libpng.

    cv2 returns BGR natively; we feed it straight to PyAV as ``bgr24`` and
    let the encoder reformat to ``yuv420p``, skipping a per-frame
    ``cv2.cvtColor`` call we would otherwise need to produce ``rgb24``.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cv2 failed to read image: {path}")
    if img.ndim == 2:
        # Grayscale → 3-channel BGR (all channels equal).
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # Drop alpha; cv2 returns BGRA so swap to BGR.
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _iter_rgb_frames_threaded(
    image_paths: list[Path],
    *,
    n_workers: int = 2,
    queue_size: int = 8,
) -> Iterator[UInt8[ndarray, "h w 3"]]:
    """Yield decoded frames in input order, with N background workers reading ahead.

    A bounded queue (default 8 slots) caps memory while keeping the encoder
    fed. Workers split the index range round-robin; the consumer holds a
    small reorder buffer so frames flow out in PTS order regardless of which
    worker finishes first. ``cv2.imread`` releases the GIL during libpng
    decode, so 2–4 worker threads recover most of the I/O gap on top of a
    GPU encoder.

    Workers honour a shared ``cancel`` event so an exception or generator
    teardown stops them promptly; the ``finally`` block always joins
    threads (drained or not) so file handles and decoded buffers don't
    outlive the call.
    """
    n = len(image_paths)
    if n == 0:
        return
    n_workers = max(1, min(n_workers, n))
    q: queue.Queue[tuple[int, np.ndarray] | BaseException] = queue.Queue(maxsize=queue_size)
    cancel = threading.Event()

    def worker(start: int) -> None:
        try:
            for i in range(start, n, n_workers):
                if cancel.is_set():
                    return
                q.put((i, _decode_one(image_paths[i])))
        except BaseException as exc:  # noqa: BLE001 -- propagate to consumer
            q.put(exc)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(n_workers)]
    for t in threads:
        t.start()

    pending: dict[int, np.ndarray] = {}
    next_idx = 0
    seen = 0
    try:
        while seen < n:
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            idx, rgb = item
            seen += 1
            if idx == next_idx:
                yield rgb
                next_idx += 1
                while next_idx in pending:
                    yield pending.pop(next_idx)
                    next_idx += 1
            else:
                pending[idx] = rgb
    finally:
        cancel.set()
        # Drain anything the workers already enqueued so they can return
        # past their bounded ``q.put`` calls and exit; we don't need the
        # values on the abort path.
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        for t in threads:
            t.join()


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
    timeline: str = "video_time",
    fps_hint: float | None = None,
) -> tuple[int, int, int]:
    """Encode ``image_paths`` with HEVC (NVENC) and log as a ``rr.VideoStream``.

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

    # HEVC (and the underlying av container) wants an integer framerate;
    # derive it from source deltas when ``rgb.csv`` doesn't supply one.
    if fps_hint is None or fps_hint <= 0:
        if len(timestamps_ns) > 1:
            dt_ns = np.median(np.diff(timestamps_ns))
            fps_hint = float(1e9 / max(float(dt_ns), 1.0))
        else:
            fps_hint = 30.0
    fps_int = max(1, int(round(fps_hint)))

    # CodecContext.create returns VideoCodecContext for video codecs but the
    # static signature is the base CodecContext; cast so pyrefly resolves the
    # video-specific width/height/pix_fmt/encode attributes.
    codec_ctx = cast(av.VideoCodecContext, av.codec.CodecContext.create("hevc_nvenc", "w"))
    codec_ctx.width = enc_w
    codec_ctx.height = enc_h
    codec_ctx.pix_fmt = "yuv420p"
    codec_ctx.framerate = Fraction(fps_int, 1)
    codec_ctx.time_base = Fraction(1, 1_000_000_000)  # PTS in nanoseconds
    # NVENC HEVC with constant-quality VBR. p4 is the balanced preset; cq=32
    # lands at ~31-39 dB PSNR depending on content (verified via PSNR sweep
    # on ETH/einstein_1, YOUTUBE/fpv-drone, KITTI/04) — equal-or-better than
    # av1_nvenc cq=40 on every test sequence. bf=0 disables B-frames per
    # rr.VideoStream's no-reordering requirement.
    codec_ctx.options = {
        "preset": "p4",
        "rc": "vbr",
        "cq": "32",
        "bf": "0",
        "g": "30",
    }

    rr.log(entity_path, rr.VideoStream(codec=rr.VideoCodec.H265), static=True, recording=recording)

    t0 = int(timestamps_ns[0])
    emitted = 0
    for i, frame_bgr in enumerate(_iter_rgb_frames_threaded(image_paths, n_workers=2)):
        frame_bgr = _pad_to_even(frame_bgr)
        frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
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
