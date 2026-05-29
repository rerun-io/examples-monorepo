"""Parsing utilities for HOT3D dataset files (Aria and Quest 3).

Handles camera calibration (``camera_models.json``, ``online_calibration.jsonl``),
device trajectories (MPS SLAM, ``headset_trajectory.csv``), UmeTrack hand
annotations, and the preprocessed calibration JSON produced by
``tools/preprocess_hot3d.py``.
"""

from __future__ import annotations

import json
from dataclasses import field
from pathlib import Path

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import from_json, to_json

# ──────────────────────── Serde calibration classes ──────────────────────── #


@serde
class AriaStreamCalibration:
    """Calibration for a single Meta camera stream (Aria or Quest 3).

    Both headsets use Meta's FISHEYE624 (FisheyeRadTanThinPrism) projection
    model, but with different parameter layouts:

    - **Aria** (15 params): ``[f, cx, cy, k1-k6, p1-p2, s1-s4]`` — single
      focal length ``f`` shared for both axes.
    - **Quest 3** (16 params): ``[fx, fy, cx, cy, k1-k6, p1-p2, s1-s4]`` —
      separate focal lengths per axis.

    We map FISHEYE624 → simplecv's ``Fisheye62Parameters`` (Kannala-Brandt
    model with k1-k6 radial + p1-p2 tangential distortion).  The four
    thin-prism terms (s1-s4) are dropped — validated to produce <1px error
    against the full OVR624 model on HOT3D data.
    """

    stream_label: str
    """Stream label, e.g. 'camera-rgb', 'camera-slam-left'."""
    width: int
    height: int
    fl_x: float
    """Focal length (x). Aria: fl_x == fl_y == f. Quest: may differ."""
    fl_y: float
    """Focal length (y). Aria: fl_x == fl_y == f. Quest: may differ."""
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    device_T_camera: list[list[float]] = field(default_factory=lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    """4x4 transform from camera frame to device (CPF) frame, row-major."""


# Backward-compatible aliases
Hot3dStreamCalibration = AriaStreamCalibration


@serde
class AriaSequenceCalibration:
    """All camera calibrations for a sequence (Aria or Quest 3)."""

    streams: list[AriaStreamCalibration]


# Backward-compatible alias
Hot3dSequenceCalibration = AriaSequenceCalibration


def save_calibration(cal: AriaSequenceCalibration, path: Path) -> None:
    """Serialize calibration to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(cal))


def load_calibration(path: Path) -> AriaSequenceCalibration:
    """Deserialize calibration from JSON."""
    return from_json(AriaSequenceCalibration, path.read_text())


# ──────────────────── Headset detection ────────────────────────────────── #


def detect_headset(seq_dir: Path) -> str:
    """Read ``metadata.json`` to determine headset type (``'Aria'`` or ``'Quest3'``).

    Raises ``AssertionError`` if ``metadata.json`` is missing — every valid
    HOT3D sequence must have this file.
    """
    metadata_path: Path = seq_dir / "metadata.json"
    assert metadata_path.exists(), (
        f"metadata.json not found in {seq_dir}. "
        f"Is this a valid HOT3D sequence directory?"
    )
    metadata: dict = json.loads(metadata_path.read_text())
    return metadata.get("headset", "Aria")


# ─────────────────── Quaternion / transform helpers ──────────────────────── #
# TODO: These are general rotation/transform utilities and could be moved to
# simplecv/ops/ for reuse across other datasets. Kept here for now to avoid
# touching import sites across the codebase.


def aria_unit_quaternion_to_matrix(uq: list) -> Float32[ndarray, "3 3"]:
    """Convert Aria ``UnitQuaternion`` ``[w, [x, y, z]]`` to a 3x3 rotation matrix.

    scipy expects ``[x, y, z, w]`` ordering.
    """
    w: float = uq[0]
    x: float = uq[1][0]
    y: float = uq[1][1]
    z: float = uq[1][2]
    rot: Rotation = Rotation.from_quat([x, y, z, w])
    return rot.as_matrix().astype(np.float32)


def quat_wxyz_to_matrix(q_wxyz: list[float]) -> Float32[ndarray, "3 3"]:
    """Convert ``[w, x, y, z]`` quaternion to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    rot: Rotation = Rotation.from_quat([x, y, z, w])
    return rot.as_matrix().astype(np.float32)


def build_4x4(R: Float32[ndarray, "3 3"], t: list[float] | Float32[ndarray, "3"]) -> Float32[ndarray, "4 4"]:
    """Compose a 4x4 homogeneous transform from rotation and translation."""
    T: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float32)
    return T


# ──────────────── MPS SLAM closed-loop trajectory parser ─────────────────── #


def parse_mps_closed_loop_trajectory(
    csv_path: Path,
) -> tuple[Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"], Float32[ndarray, "n_poses"]]:
    """Parse MPS SLAM closed_loop_trajectory.csv.

    Returns
    -------
    timestamps_ns
        Timestamps in nanoseconds (converted from microseconds in the CSV).
    world_T_device
        4x4 world-to-device transforms.
    quality_scores
        Per-pose quality scores (0.0 = bad, 1.0 = good).
    """
    import pandas as pd

    df: pd.DataFrame = pd.read_csv(csv_path)

    timestamps_us: Int64[ndarray, "n_poses"] = df["tracking_timestamp_us"].values.astype(np.int64)
    timestamps_ns: Int64[ndarray, "n_poses"] = timestamps_us * np.int64(1000)

    n_poses: int = len(df)
    world_T_device: Float32[ndarray, "n_poses 4 4"] = np.zeros((n_poses, 4, 4), dtype=np.float32)

    tx: Float32[ndarray, "n_poses"] = df["tx_world_device"].values.astype(np.float32)
    ty: Float32[ndarray, "n_poses"] = df["ty_world_device"].values.astype(np.float32)
    tz: Float32[ndarray, "n_poses"] = df["tz_world_device"].values.astype(np.float32)
    qx: Float32[ndarray, "n_poses"] = df["qx_world_device"].values.astype(np.float32)
    qy: Float32[ndarray, "n_poses"] = df["qy_world_device"].values.astype(np.float32)
    qz: Float32[ndarray, "n_poses"] = df["qz_world_device"].values.astype(np.float32)
    qw: Float32[ndarray, "n_poses"] = df["qw_world_device"].values.astype(np.float32)

    # Vectorized quaternion to rotation matrix
    quats_xyzw: Float32[ndarray, "n_poses 4"] = np.stack([qx, qy, qz, qw], axis=-1)
    rotations: Rotation = Rotation.from_quat(quats_xyzw)
    rot_matrices: Float32[ndarray, "n_poses 3 3"] = rotations.as_matrix().astype(np.float32)

    world_T_device[:, :3, :3] = rot_matrices
    world_T_device[:, 0, 3] = tx
    world_T_device[:, 1, 3] = ty
    world_T_device[:, 2, 3] = tz
    world_T_device[:, 3, 3] = 1.0

    quality_scores: Float32[ndarray, "n_poses"] = df["quality_score"].values.astype(np.float32)

    return timestamps_ns, world_T_device, quality_scores


# ───────────────── Online calibration parser (first entry) ────────────────── #

# Aria image dimensions per stream label (standard Aria hardware)
ARIA_IMAGE_DIMENSIONS: dict[str, tuple[int, int]] = {
    "camera-rgb": (1408, 1408),
    "camera-slam-left": (640, 480),
    "camera-slam-right": (640, 480),
}

# Stream ID to label mapping for Aria VRS
ARIA_STREAM_ID_TO_LABEL: dict[str, str] = {
    "214-1": "camera-rgb",
    "1201-1": "camera-slam-left",
    "1201-2": "camera-slam-right",
}

ARIA_LABEL_TO_STREAM_ID: dict[str, str] = {v: k for k, v in ARIA_STREAM_ID_TO_LABEL.items()}

# Quest 3 image dimensions per stream label
QUEST_IMAGE_DIMENSIONS: dict[str, tuple[int, int]] = {
    "camera-slam-left": (1280, 1024),
    "camera-slam-right": (1280, 1024),
}

# Stream ID to label mapping for Quest 3 VRS (no RGB stream)
QUEST_STREAM_ID_TO_LABEL: dict[str, str] = {
    "1201-1": "camera-slam-left",
    "1201-2": "camera-slam-right",
}


def parse_online_calibration_first(jsonl_path: Path) -> AriaSequenceCalibration:
    """Parse the first entry of online_calibration.jsonl to get factory calibration.

    The online calibration updates per-frame, but the intrinsics change very
    little.  We use the first entry for factory-like intrinsics and the
    ``device_T_camera`` transform.

    FisheyeRadTanThinPrism params order (15 values):
        [f, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4]
    We drop s1-s4 (thin-prism) for the Fisheye62 model.
    """
    with open(jsonl_path) as f:
        first_entry: dict = json.loads(f.readline())

    streams: list[AriaStreamCalibration] = []
    for cam in first_entry["CameraCalibrations"]:
        label: str = cam["Label"]
        params: list[float] = cam["Projection"]["Params"]

        # FISHEYE624 (Aria online_calibration): [f, cx, cy, k1-k6, p1-p2, s1-s4]
        # We extract k1-k6 + p1-p2 for Fisheye62; s1-s4 thin-prism dropped (<1px error).
        f_val: float = params[0]
        cx: float = params[1]
        cy: float = params[2]

        # Determine image dimensions from label
        width: int
        height: int
        if label in ARIA_IMAGE_DIMENSIONS:
            width, height = ARIA_IMAGE_DIMENSIONS[label]
        else:
            # Fallback: infer from principal point (rough)
            width = int(cx * 2)
            height = int(cy * 2)

        # Build device_T_camera from Aria's UnitQuaternion format [w, [x,y,z]]
        t_dev_cam: dict = cam["T_Device_Camera"]
        R_dev_cam: Float32[ndarray, "3 3"] = aria_unit_quaternion_to_matrix(t_dev_cam["UnitQuaternion"])
        t_translation: list[float] = t_dev_cam["Translation"]
        device_T_camera: Float32[ndarray, "4 4"] = build_4x4(R_dev_cam, t_translation)

        stream_cal: AriaStreamCalibration = AriaStreamCalibration(
            stream_label=label,
            width=width,
            height=height,
            fl_x=f_val,
            fl_y=f_val,  # Aria fisheye uses single focal length
            cx=cx,
            cy=cy,
            k1=params[3],
            k2=params[4],
            k3=params[5],
            k4=params[6],
            k5=params[7],
            k6=params[8],
            p1=params[9],
            p2=params[10],
            device_T_camera=device_T_camera.tolist(),
        )
        streams.append(stream_cal)

    return AriaSequenceCalibration(streams=streams)


# ─────────── camera_models.json parser (Aria + Quest) ─────────────────────── #


def parse_camera_models_json(json_path: Path) -> AriaSequenceCalibration:
    """Parse ``camera_models.json`` — works for both Aria and Quest 3.

    Aria has 15 projection params: ``[f, cx, cy, k1-k6, p1-p2, s1-s4]``
    (single focal length).  Quest 3 has 16: ``[fx, fy, cx, cy, k1-k6, p1-p2, s1-s4]``
    (separate fx/fy).  Both use FISHEYE624 model.

    The ``T_Device_Camera`` transform uses ``quaternion_wxyz`` + ``translation_xyz``
    format (same for both headsets).
    """
    with open(json_path) as f:
        cameras: list[dict] = json.load(f)

    streams: list[AriaStreamCalibration] = []
    for cam in cameras:
        label: str = cam["label"]
        params: list[float] = cam["projectionParams"]
        n_params: int = len(params)

        # Handle Aria (15 params: [f, cx, cy, ...]) vs Quest (16 params: [fx, fy, cx, cy, ...])
        # See hand_tracking_toolkit camera.from_json() for reference.
        if n_params == 15:
            # Aria: single focal length
            fl_x: float = params[0]
            fl_y: float = params[0]
            cx: float = params[1]
            cy: float = params[2]
            distortion_offset: int = 3
        else:
            # Quest 3 (and standard): separate fx/fy
            fl_x = params[0]
            fl_y = params[1]
            cx = params[2]
            cy = params[3]
            distortion_offset = 4

        width: int = cam["imageWidth"]
        height: int = cam["imageHeight"]

        # Build device_T_camera from quaternion_wxyz + translation_xyz
        t_dev_cam: dict = cam["T_Device_Camera"]
        R_dev_cam: Float32[ndarray, "3 3"] = quat_wxyz_to_matrix(t_dev_cam["quaternion_wxyz"])
        device_T_camera: Float32[ndarray, "4 4"] = build_4x4(R_dev_cam, t_dev_cam["translation_xyz"])

        # FISHEYE624 distortion layout after focal/principal: k1-k6, p1-p2, s1-s4.
        # We extract k1-k6 (radial) and p1-p2 (tangential) for the Fisheye62
        # model. Thin-prism terms s1-s4 are dropped — validated to produce <1px
        # error against the full OVR624Distortion model on HOT3D data.
        d: int = distortion_offset
        stream_cal: AriaStreamCalibration = AriaStreamCalibration(
            stream_label=label,
            width=width,
            height=height,
            fl_x=fl_x,
            fl_y=fl_y,
            cx=cx,
            cy=cy,
            k1=params[d] if d < n_params else 0.0,
            k2=params[d + 1] if d + 1 < n_params else 0.0,
            k3=params[d + 2] if d + 2 < n_params else 0.0,
            k4=params[d + 3] if d + 3 < n_params else 0.0,
            k5=params[d + 4] if d + 4 < n_params else 0.0,
            k6=params[d + 5] if d + 5 < n_params else 0.0,
            p1=params[d + 6] if d + 6 < n_params else 0.0,
            p2=params[d + 7] if d + 7 < n_params else 0.0,
            device_T_camera=device_T_camera.tolist(),
        )
        streams.append(stream_cal)

    return AriaSequenceCalibration(streams=streams)


# ─────────── headset_trajectory.csv parser (Aria + Quest) ─────────────────── #


def parse_headset_trajectory(
    csv_path: Path,
) -> tuple[Int64[ndarray, "n_poses"], Float32[ndarray, "n_poses 4 4"]]:
    """Parse ``headset_trajectory.csv`` — available for both Aria and Quest 3.

    Columns: ``object_uid, timestamp[ns], t_wo_x/y/z[m], q_wo_w/x/y/z``

    Timestamps are in **timecode domain** for Aria (need
    ``timecode_devicetime_mapping.csv`` to convert to device time) and in
    the **native VRS domain** for Quest 3.

    Returns
    -------
    timestamps_ns
        Timestamps in nanoseconds (as-is from the CSV).
    world_T_device
        4x4 world-to-device transforms.
    """
    import pandas as pd

    df: pd.DataFrame = pd.read_csv(csv_path)

    timestamps_ns: Int64[ndarray, "n_poses"] = df["timestamp[ns]"].values.astype(np.int64)

    n_poses: int = len(df)
    world_T_device: Float32[ndarray, "n_poses 4 4"] = np.zeros((n_poses, 4, 4), dtype=np.float32)

    tx: Float32[ndarray, "n_poses"] = df["t_wo_x[m]"].values.astype(np.float32)
    ty: Float32[ndarray, "n_poses"] = df["t_wo_y[m]"].values.astype(np.float32)
    tz: Float32[ndarray, "n_poses"] = df["t_wo_z[m]"].values.astype(np.float32)
    qx: Float32[ndarray, "n_poses"] = df["q_wo_x"].values.astype(np.float32)
    qy: Float32[ndarray, "n_poses"] = df["q_wo_y"].values.astype(np.float32)
    qz: Float32[ndarray, "n_poses"] = df["q_wo_z"].values.astype(np.float32)
    qw: Float32[ndarray, "n_poses"] = df["q_wo_w"].values.astype(np.float32)

    quats_xyzw: Float32[ndarray, "n_poses 4"] = np.stack([qx, qy, qz, qw], axis=-1)
    rotations: Rotation = Rotation.from_quat(quats_xyzw)
    rot_matrices: Float32[ndarray, "n_poses 3 3"] = rotations.as_matrix().astype(np.float32)

    world_T_device[:, :3, :3] = rot_matrices
    world_T_device[:, 0, 3] = tx
    world_T_device[:, 1, 3] = ty
    world_T_device[:, 2, 3] = tz
    world_T_device[:, 3, 3] = 1.0

    return timestamps_ns, world_T_device


# ──────────── Timecode ↔ device-time mapping ──────────────────────────────── #


def load_timecode_to_devicetime_mapping(csv_path: Path) -> Int64[ndarray, "n_entries"]:
    """Load ``timecode_devicetime_mapping.csv`` and return device-time timestamps.

    HOT3D uses two time domains:
    - **Timecode**: Used by hand annotation JSONL, headset_trajectory.csv
    - **Device time**: Used by VRS frame timestamps, MPS SLAM trajectory

    The mapping CSV has columns ``timecode_ns`` and ``devicetime_ns`` with a
    1:1 correspondence to the JSONL annotation entries.  This function returns
    the ``devicetime_ns`` column so label timestamps can be expressed in the
    same domain as VRS / MPS data.
    """
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        devicetime_ns: list[int] = [int(row["devicetime_ns"]) for row in reader]
    return np.array(devicetime_ns, dtype=np.int64)


# ──────────── Nearest-neighbor pose interpolation ─────────────────────────── #


def lookup_nearest_poses(
    query_ts_ns: Int64[ndarray, "n_frames"],
    trajectory_ts_ns: Int64[ndarray, "n_poses"],
    world_T_device: Float32[ndarray, "n_poses 4 4"],
) -> Float32[ndarray, "n_frames 4 4"]:
    """Look up the nearest device pose for each query timestamp.

    Uses nearest-neighbor (no interpolation) via ``np.searchsorted``.
    """
    # Find insertion points
    indices: Int64[ndarray, "n_frames"] = np.searchsorted(trajectory_ts_ns, query_ts_ns, side="right") - 1
    indices = np.clip(indices, 0, len(trajectory_ts_ns) - 1).astype(np.int64)

    # Check if the next index is closer
    next_indices: Int64[ndarray, "n_frames"] = np.minimum(indices + 1, len(trajectory_ts_ns) - 1)
    diff_left: Int64[ndarray, "n_frames"] = np.abs(query_ts_ns - trajectory_ts_ns[indices])
    diff_right: Int64[ndarray, "n_frames"] = np.abs(query_ts_ns - trajectory_ts_ns[next_indices])
    use_next: ndarray = diff_right < diff_left
    indices = np.where(use_next, next_indices, indices)

    return world_T_device[indices]
