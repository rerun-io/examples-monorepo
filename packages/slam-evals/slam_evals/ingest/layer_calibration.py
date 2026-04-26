"""Write the ``calibration`` layer — static sensor tree + cross-cutting properties.

What this layer contributes to the composed segment:

- static ``Transform3D`` (``rig_0_T_cam_<i>``) at ``/world/rig_0/cam_<i>``
- static ``Pinhole``/``PinholeWithDistortion`` at ``/world/rig_0/cam_<i>/pinhole``
- static ``Transform3D`` (``rig_0_T_imu_<i>``) at ``/world/rig_0/imu_<i>`` when
  ``calibration.yaml`` declares an IMU with a ``T_BS``
- static world view convention (``RDF``) at ``/``

Recording properties on this layer:

- ``info`` — cross-cutting metadata about the segment as a whole (modality,
  dataset name, sequence name, slug, presence flags). Hosted here because
  ``calibration.rrd`` is always present and naturally static — see
  ``docs/schema.md`` for why this isn't a separate ``info.rrd``.
- ``calibration`` — quick-look summary (num_cameras, cam_0 intrinsics, depth
  scale factor). The full per-camera intrinsics live as static logs in the
  entity tree above.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from slam_evals.data.types import Calibration, Sequence
from slam_evals.ingest.calibration import log_camera_static


def _log_imu_extrinsic_static(
    *,
    imu_idx: int,
    t_bs_flat: list[float] | None,
    recording: rr.RecordingStream,
) -> None:
    """Log ``rig_0_T_imu_<idx>`` as a static ``Transform3D``.

    VSLAM-LAB calibration.yaml stores ``T_BS`` as a flat list of 16 floats
    (row-major 4x4). ``T_BS`` is body-from-sensor — exactly the
    parent-to-child transform Rerun's ``Transform3D`` defaults to when
    logged at the child entity path.
    """
    if t_bs_flat is None or len(t_bs_flat) != 16:
        return
    m = np.asarray(t_bs_flat, dtype=np.float64).reshape(4, 4)
    rr.log(
        f"/world/rig_0/imu_{imu_idx}",
        rr.Transform3D(translation=m[:3, 3].tolist(), mat3x3=m[:3, :3].tolist()),
        static=True,
        recording=recording,
    )


def write_calibration_layer(
    sequence: Sequence,
    calibration: Calibration | None,
    *,
    out_path: Path,
    application_id: str = "slam-evals",
) -> Path:
    """Write ``calibration.rrd`` for ``sequence``. Returns the output path.

    Idempotent over the same inputs: re-running with the same ``sequence`` +
    ``calibration`` produces a byte-equivalent file (modulo recording-id
    timestamps inside the chunk format itself).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=application_id,
        recording_id=sequence.recording_id,
        send_properties=True,
    )
    with rec:
        # World coordinate convention — matches simplecv / robocap-slam usage.
        rr.log("/", rr.ViewCoordinates.RDF, static=True, recording=rec)

        if calibration is not None:
            # Camera intrinsics + extrinsics. Only ``rgb_<i>`` entries get
            # entity paths — depth_<i> calibration is intentionally skipped
            # because depth is pre-registered to its paired RGB camera and
            # shares that camera's pinhole tree.
            for cam in calibration.cameras:
                if not cam.cam_name.startswith("rgb_"):
                    continue
                idx = cam.cam_name.split("_", 1)[1]
                log_camera_static(cam, entity_path=f"/world/rig_0/cam_{idx}", recording=rec)

            # IMU extrinsic. ``imu_params`` is the first IMU's full field bag
            # from calibration.yaml; we only need ``T_BS`` to place the IMU
            # in the rig frame. Sequences whose IMU is the rig reference
            # (EUROC-style) have ``T_BS = identity`` already.
            if calibration.imu_params is not None and sequence.has_imu(0):
                t_bs = calibration.imu_params.get("T_BS")
                if isinstance(t_bs, list):
                    _log_imu_extrinsic_static(imu_idx=0, t_bs_flat=t_bs, recording=rec)

        # Cross-cutting `info` properties — visible at the segment level
        # via segment_table() as ``property:info:<key>``.
        rr.send_recording_name(sequence.slug, recording=rec)
        rec.send_property(
            "info",
            rr.AnyValues(
                dataset=sequence.dataset,
                sequence=sequence.name,
                slug=sequence.slug,
                modality=str(sequence.modality),
                has_imu=sequence.modality.has_imu,
                has_depth=sequence.modality.has_depth,
                has_stereo=sequence.modality.has_stereo,
                has_calibration=sequence.has_calibration,
            ),
        )

        if calibration is not None and calibration.cameras:
            cam0 = calibration.cameras[0]
            rec.send_property(
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

    rec.save(str(out_path))
    return out_path
