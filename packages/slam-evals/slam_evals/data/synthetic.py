"""Synthetic VSLAM-LAB-shaped sequences for tests + visual smoke checks.

Materialises a tiny benchmark layout on disk (rgb_0/, optional rgb_1/,
optional depth_0/, rgb.csv, groundtruth.csv, optional imu_0.csv,
calibration.yaml) so the ingester's per-modality code paths can be
exercised without touching the multi-GB real benchmark.

Default dimensions are 192x128: NVENC's AV1 encoder rejects anything
smaller than ~160x120, so we sit just above that lower bound to keep
fixture build time minimal while still being a valid encoder input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from slam_evals.data.types import Modality, Sequence


@dataclass(frozen=True)
class Fixture:
    """A disk-materialised VSLAM-LAB sequence with known modality + frame count."""

    sequence: Sequence
    n_frames: int


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")


def _make_rgb(i: int, *, w: int = 192, h: int = 128) -> np.ndarray:
    # Per-frame gradient so the H.264 encoder doesn't collapse identical frames.
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    r = np.clip(x[None, :] + i * 7, 0, 255).astype(np.uint8)
    g = np.clip(y + i * 11, 0, 255).astype(np.uint8)
    b = np.full((h, w), (i * 23) % 256, dtype=np.uint8)
    return np.stack([b, np.broadcast_to(g, (h, w)), np.broadcast_to(r, (h, w))], axis=-1)  # BGR for cv2


def _make_depth_uint16(i: int, *, w: int = 192, h: int = 128) -> np.ndarray:
    base = 1000 + (i * 10)  # ~0.2 m at depth_factor=5000
    return np.full((h, w), base, dtype=np.uint16)


_IDENTITY_T_BS = (
    "[1.0, 0.0, 0.0, 0.0, "
    "0.0, 1.0, 0.0, 0.0, "
    "0.0, 0.0, 1.0, 0.0, "
    "0.0, 0.0, 0.0, 1.0]"
)


def _write_calibration(seq_dir: Path, *, modality: Modality, w: int = 192, h: int = 128) -> None:
    cams: list[str] = []

    def _cam(cam_name: str, *, include_depth: bool = False, cam_type: str = "gray") -> str:
        depth_bits = ""
        if include_depth:
            depth_bits = (
                "\n     depth_name: depth_0,"
                "\n     depth_factor: 5000.0,"
            )
            cam_type = "rgb+depth"
        return (
            f"  - {{cam_name: {cam_name},\n"
            f"     cam_type: {cam_type},\n"
            f"     cam_model: pinhole,\n"
            f"     distortion_type: radtan4,\n"
            f"     focal_length: [320.0, 320.0],\n"
            f"     principal_point: [{w / 2:.1f}, {h / 2:.1f}],\n"
            f"     distortion_coefficients: [0.0, 0.0, 0.0, 0.0],\n"
            f"     image_dimension: [{w}, {h}],\n"
            f"     fps: 10.0,{depth_bits}\n"
            f"     T_BS: {_IDENTITY_T_BS}\n"
            f"    }}"
        )

    if modality.has_depth:
        cams.append(_cam("rgb_0", include_depth=True))
    else:
        cams.append(_cam("rgb_0"))
    if modality.has_stereo:
        cams.append(_cam("rgb_1"))

    imu_block = ""
    if modality.has_imu:
        imu_block = (
            "\nimus:\n"
            "  - {imu_name: imu_0,\n"
            "     fps: 100.0,\n"
            f"     T_BS: {_IDENTITY_T_BS}\n"
            "    }\n"
        )

    yaml = "%YAML 1.2\n---\ncameras:\n" + ",\n".join(cams) + "\n" + imu_block
    (seq_dir / "calibration.yaml").write_text(yaml)


def _write_rgb_csv(
    seq_dir: Path,
    *,
    modality: Modality,
    n_frames: int,
    period_ns: int = 100_000_000,
) -> None:
    cols = ["ts_rgb_0 (ns)", "path_rgb_0"]
    if modality.has_depth:
        cols += ["ts_depth_0 (ns)", "path_depth_0"]
    if modality.has_stereo:
        cols += ["ts_rgb_1 (ns)", "path_rgb_1"]

    lines = [",".join(cols)]
    for i in range(n_frames):
        ts = i * period_ns
        row = [str(ts), f"rgb_0/{i:04d}.png"]
        if modality.has_depth:
            row += [str(ts), f"depth_0/{i:04d}.png"]
        if modality.has_stereo:
            row += [str(ts), f"rgb_1/{i:04d}.png"]
        lines.append(",".join(row))
    (seq_dir / "rgb.csv").write_text("\n".join(lines) + "\n")


def _write_groundtruth_csv(
    seq_dir: Path,
    *,
    n_frames: int,
    period_ns: int = 100_000_000,
) -> None:
    """Write a slow forward translation along +x so the trajectory is visible."""
    lines = ["ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw"]
    for i in range(n_frames):
        ts = i * period_ns
        tx = i * 0.1
        lines.append(f"{ts},{tx},0.0,0.0,0.0,0.0,0.0,1.0")
    (seq_dir / "groundtruth.csv").write_text("\n".join(lines) + "\n")


def _write_imu_csv(
    seq_dir: Path,
    *,
    n_frames: int,
    period_ns: int = 10_000_000,
) -> None:
    """IMU runs at 10× the RGB rate (≈100 Hz vs 10 Hz)."""
    n_samples = n_frames * 10
    lines = [
        "ts (ns),wx (rad s^-1),wy (rad s^-1),wz (rad s^-1),"
        "ax (m s^-2),ay (m s^-2),az (m s^-2)"
    ]
    for i in range(n_samples):
        ts = i * period_ns
        lines.append(f"{ts},0.01,0.0,0.0,0.0,0.0,9.81")
    (seq_dir / "imu_0.csv").write_text("\n".join(lines) + "\n")


def build_fixture(
    root: Path,
    *,
    modality: Modality,
    n_frames: int = 5,
    dataset: str = "MOCK",
    name: str | None = None,
) -> Fixture:
    """Build a tiny VSLAM-LAB-shaped sequence on disk and return its ``Sequence``.

    Parameters
    ----------
    root:
        Base directory under which ``<dataset>/<name>/`` will be created.
    modality:
        Which streams to emit (mono / mono-vi / stereo / stereo-vi / rgbd / rgbd-vi).
    n_frames:
        Number of RGB / depth frames + GT poses. IMU samples are 10×.
    """
    name = name or f"mock_{modality.value.replace('-', '_')}"
    seq_dir = root / dataset / name
    seq_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        _write_png(seq_dir / "rgb_0" / f"{i:04d}.png", _make_rgb(i))
        if modality.has_stereo:
            _write_png(seq_dir / "rgb_1" / f"{i:04d}.png", _make_rgb(i + 1))
        if modality.has_depth:
            _write_png(seq_dir / "depth_0" / f"{i:04d}.png", _make_depth_uint16(i))

    _write_rgb_csv(seq_dir, modality=modality, n_frames=n_frames)
    _write_groundtruth_csv(seq_dir, n_frames=n_frames)
    if modality.has_imu:
        _write_imu_csv(seq_dir, n_frames=n_frames)
    _write_calibration(seq_dir, modality=modality)

    sequence = Sequence(
        dataset=dataset,
        name=name,
        root=seq_dir,
        modality=modality,
        has_calibration=True,
    )
    return Fixture(sequence=sequence, n_frames=n_frames)
