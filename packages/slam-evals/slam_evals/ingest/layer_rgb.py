"""Write a ``rgb_<i>`` layer — one per camera with image data.

What this layer contributes to the composed segment:

- static ``VideoStream(codec=H265)`` at ``/world/rig_0/cam_<i>/pinhole/video``
- per-packet HEVC chunks over the ``video_time`` timeline (PTS in seconds
  relative to ``t0_ns``)

Recording properties on this layer:

- ``rgb_<i>`` — codec, fps, num_frames, encoded width/height. Source
  resolution is centre-cropped 1px on odd-dimensioned axes (HEVC requires
  even dims for ``yuv420p``).

Encoding details (NVENC HEVC at PSNR-balanced cq=32) are inherited from
``slam_evals.ingest.video.encode_and_log_video`` — the layer writer is just
a thin shell that opens an isolated ``RecordingStream`` and adds the
property bag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

from slam_evals.data.parse import RgbCsv
from slam_evals.data.types import Sequence
from slam_evals.ingest.video import encode_and_log_video


def _stream_for_cam(rgb_csv: RgbCsv, cam_idx: int) -> tuple[tuple[str, ...], Int64[ndarray, "n"]] | None:
    """Pick (paths, timestamps) for ``cam_idx``. Returns ``None`` if absent."""
    if cam_idx == 0:
        return rgb_csv.path_rgb_0, rgb_csv.ts_rgb_0_ns
    if cam_idx == 1 and rgb_csv.path_rgb_1 is not None and rgb_csv.ts_rgb_1_ns is not None:
        return rgb_csv.path_rgb_1, rgb_csv.ts_rgb_1_ns
    return None


def _fps_from_timestamps(ts_ns: Int64[ndarray, "n"]) -> float | None:
    if ts_ns.shape[0] < 2:
        return None
    dt_ns = float(np.median(np.diff(ts_ns)))
    if dt_ns <= 0:
        return None
    return 1e9 / dt_ns


def write_rgb_layer(
    sequence: Sequence,
    *,
    cam_idx: int,
    rgb_csv: RgbCsv,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``rgb_<cam_idx>.rrd`` for ``sequence``. Returns the output path.

    Raises ``ValueError`` if ``cam_idx`` doesn't have a corresponding stream
    in ``rgb.csv`` — caller should gate via ``sequence.has_camera(cam_idx)``.
    """
    stream = _stream_for_cam(rgb_csv, cam_idx)
    if stream is None:
        raise ValueError(f"sequence {sequence.slug!r} has no rgb_{cam_idx} stream in rgb.csv")
    rel_paths, ts_ns = stream

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = [sequence.root / p for p in rel_paths]
    fps = _fps_from_timestamps(ts_ns)

    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )
    with rec:
        enc_w, enc_h, num_emitted = encode_and_log_video(
            entity_path=f"/world/rig_0/cam_{cam_idx}/pinhole/video",
            image_paths=image_paths,
            timestamps_ns=ts_ns,
            recording=rec,
            fps_hint=fps,
        )

        rec.send_property(
            f"rgb_{cam_idx}",
            rr.AnyValues(
                codec="hevc",
                fps=float(fps) if fps is not None else -1.0,
                num_frames=int(num_emitted),
                width=int(enc_w),
                height=int(enc_h),
            ),
        )

    rec.save(str(out_path))
    return out_path
