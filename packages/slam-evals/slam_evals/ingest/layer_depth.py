"""Write a ``depth_<i>`` layer — one per depth camera.

What this layer contributes to the composed segment:

- per-frame ``EncodedDepthImage`` (PNG passthrough) at
  ``/world/rig_0/cam_<i>/pinhole/depth`` over the ``video_time`` timeline

Recording properties on this layer:

- ``depth_<i>`` — depth_factor, num_frames, plus width/height of the first
  frame (read directly from the source PNG header).

Depth is logged under the *RGB camera*'s pinhole because VSLAM-LAB depth is
pre-registered to its paired RGB sensor (``calibration.yaml`` encodes this
via ``cam_type: rgb+depth, depth_name: depth_<i>``). Sharing the pinhole
makes the viewer hover-sync RGB samples and depth metres at the pixel
level.

PNG passthrough — VSLAM-LAB depth files are uniformly 16-bit single-channel
PNG. We ship the on-disk bytes verbatim into ``EncodedDepthImage`` instead
of decoding to ndarray and re-encoding via ``DepthImage.compress``: ~12
ms/frame becomes ~0.07 ms/frame at the cost of ~28% larger encoded bytes
on average (source PNGs use looser libpng filters than rerun's encoder).
"""

from __future__ import annotations

import struct
from pathlib import Path

import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

from slam_evals.data.parse import RgbCsv
from slam_evals.data.types import Calibration, Sequence


def _stream_for_depth(rgb_csv: RgbCsv, cam_idx: int) -> tuple[tuple[str, ...], Int64[ndarray, "n"]] | None:
    """Pick (paths, timestamps) for ``depth_<cam_idx>``. Returns ``None`` if absent."""
    if cam_idx == 0 and rgb_csv.path_depth_0 is not None and rgb_csv.ts_depth_0_ns is not None:
        return rgb_csv.path_depth_0, rgb_csv.ts_depth_0_ns
    if cam_idx == 1 and rgb_csv.path_depth_1 is not None and rgb_csv.ts_depth_1_ns is not None:
        return rgb_csv.path_depth_1, rgb_csv.ts_depth_1_ns
    return None


def _depth_factor_for_cam(calibration: Calibration | None, cam_idx: int) -> float | None:
    """Locate ``depth_factor`` for the depth stream paired with ``cam_<idx>``.

    VSLAM-LAB ships ``depth_factor`` either on the ``rgb_<i>`` calibration
    entry (when depth is pre-registered to it) or on a separate ``depth_<i>``
    entry. Try both names.
    """
    if calibration is None:
        return None
    candidates = (f"rgb_{cam_idx}", f"depth_{cam_idx}")
    for cam in calibration.cameras:
        if cam.cam_name in candidates and cam.depth_factor is not None and cam.depth_factor > 0:
            return float(cam.depth_factor)
    return None


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read width/height from a PNG file's IHDR chunk without decoding pixels."""
    with path.open("rb") as fh:
        header = fh.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def write_depth_layer(
    sequence: Sequence,
    *,
    cam_idx: int,
    rgb_csv: RgbCsv,
    calibration: Calibration | None,
    t0_ns: int,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``depth_<cam_idx>.rrd`` for ``sequence``. Returns the output path.

    Raises ``ValueError`` if ``cam_idx`` doesn't have a corresponding depth
    stream in ``rgb.csv`` — caller should gate via ``sequence.has_depth(cam_idx)``.
    """
    stream = _stream_for_depth(rgb_csv, cam_idx)
    if stream is None:
        raise ValueError(f"sequence {sequence.slug!r} has no depth_{cam_idx} stream in rgb.csv")
    rel_paths, ts_ns = stream

    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_factor = _depth_factor_for_cam(calibration, cam_idx)
    # VSLAM-LAB's ``depth_factor`` is the divisor that maps native uint16
    # depth values to metres (``metres = pixel / depth_factor``). Rerun's
    # ``EncodedDepthImage(meter=...)`` is the *multiplier* in the opposite
    # direction (``metres = pixel * meter``), so the right value to pass
    # is ``1 / depth_factor`` — passing ``depth_factor`` directly scales
    # the rendered depth by ``depth_factor**2``.
    meter: float | None = (1.0 / depth_factor) if depth_factor is not None and depth_factor > 0 else None

    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )

    width = 0
    height = 0
    num_emitted = 0

    with rec:
        for i, rel in enumerate(rel_paths):
            path = sequence.root / rel
            try:
                blob = path.read_bytes()
            except OSError:
                continue
            if num_emitted == 0:
                # Cheap one-time read of the first frame's header for properties.
                try:
                    width, height = _png_dimensions(path)
                except (ValueError, OSError):
                    width = 0
                    height = 0
            t_rel_s = (int(ts_ns[i]) - t0_ns) * 1e-9
            rr.set_time("video_time", duration=t_rel_s, recording=rec)
            depth = rr.archetypes.EncodedDepthImage.from_fields(
                blob=blob,
                media_type="image/png",
                meter=meter,
            )
            rr.log(f"/world/rig_0/cam_{cam_idx}/pinhole/depth", depth, recording=rec)
            num_emitted += 1

        rec.send_property(
            f"depth_{cam_idx}",
            rr.AnyValues(
                depth_factor=float(depth_factor) if depth_factor is not None else -1.0,
                num_frames=int(num_emitted),
                width=int(width),
                height=int(height),
            ),
        )

    rec.save(str(out_path))
    return out_path
