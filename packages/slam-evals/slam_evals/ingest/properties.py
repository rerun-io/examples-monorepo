"""Recording-level properties: queryable metadata on the Rerun catalog."""

from __future__ import annotations

import rerun as rr

from slam_evals.data.types import Calibration, Modality, Sequence


def send_sequence_properties(
    *,
    sequence: Sequence,
    num_rgb_frames: int,
    num_gt_poses: int,
    trajectory_len_m: float,
    duration_s: float,
    fps_rgb: float | None,
    num_imu_samples: int,
    calibration: Calibration | None,
    recording: rr.RecordingStream,
) -> None:
    """Populate ``/__properties`` with info + calibration summary.

    Column names after mount are ``property:info:<field>`` and
    ``property:calibration:<field>`` — the names flow from the first argument
    of each ``send_property`` call, so keep them stable across sequences.
    """
    rr.send_recording_name(sequence.slug, recording=recording)

    recording.send_property(
        "info",
        rr.AnyValues(
            dataset=sequence.dataset,
            sequence=sequence.name,
            slug=sequence.slug,
            modality=str(sequence.modality),
            has_imu=sequence.modality.has_imu,
            has_depth=sequence.modality.has_depth,
            has_stereo=sequence.modality.has_stereo,
            num_rgb_frames=num_rgb_frames,
            num_gt_poses=num_gt_poses,
            num_imu_samples=num_imu_samples,
            trajectory_len_m=float(trajectory_len_m),
            duration_s=float(duration_s),
            fps_rgb=float(fps_rgb) if fps_rgb is not None else -1.0,
            has_calibration=sequence.has_calibration,
        ),
    )

    if calibration is not None and calibration.cameras:
        cam0 = calibration.cameras[0]
        recording.send_property(
            "calibration",
            rr.AnyValues(
                num_cameras=len(calibration.cameras),
                cam0_name=cam0.cam_name,
                cam0_width=cam0.width,
                cam0_height=cam0.height,
                cam0_fx=cam0.fx,
                cam0_fy=cam0.fy,
                cam0_cx=cam0.cx,
                cam0_cy=cam0.cy,
                cam0_distortion_type=cam0.distortion_type or "",
                has_imu_params=calibration.imu_params is not None,
                depth_factor=cam0.depth_factor if cam0.depth_factor is not None else -1.0,
            ),
        )


def modality_of(sequence: Sequence) -> Modality:
    """Trivial re-export for downstream code that only has a ``Sequence``."""
    return sequence.modality
