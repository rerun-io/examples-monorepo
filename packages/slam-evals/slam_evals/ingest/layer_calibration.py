"""Write the ``calibration`` layer — static sensor tree + cross-cutting properties.

What this layer contributes to the composed segment:

- static ``Transform3D`` (``rig_0_T_cam_<i>``) at ``/world/rig_0/cam_<i>``
- static ``Pinhole``/``PinholeWithDistortion`` at ``/world/rig_0/cam_<i>/pinhole``
- static ``Transform3D`` (``rig_0_T_imu_<i>``) at ``/world/rig_0/imu_<i>`` when
  ``calibration.yaml`` declares an IMU with a ``T_BS``

(World-frame ``ViewCoordinates`` is logged separately by
``layer_view_coordinates.py`` — see that module for the per-dataset
convention registry.)

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

# Wireframe colour for cam_0 — the rig's reference sensor — so it's
# trivially distinguishable from cam_1 in stereo views. Same green as the
# GT path's start marker for visual consistency.
_REFERENCE_CAM_COLOR: tuple[int, int, int, int] = (40, 200, 80, 255)


def _log_imu_extrinsic_static(
    *,
    imu_idx: int,
    rig_T_imu: list[float] | None,
    recording: rr.RecordingStream,
) -> None:
    """Log ``rig_0_T_imu_<idx>`` as a static ``Transform3D``.

    The parameter is the rig→IMU extrinsic — the parent-to-child
    transform Rerun's ``Transform3D`` defaults to when logged at the
    child entity path. Its on-disk source is VSLAM-LAB's calibration
    YAML field ``T_BS`` (body-from-sensor), which is the same quantity
    by a different name; we read it once at parse time and use the
    project's own ``rig_T_sensor`` naming everywhere else.

    Pass it as a flat 16-float list in row-major 4×4 order, the
    natural shape coming out of YAML.
    """
    if rig_T_imu is None or len(rig_T_imu) != 16:
        return
    m = np.asarray(rig_T_imu, dtype=np.float64).reshape(4, 4)
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
    # World-frame ViewCoordinates is *not* logged here — that's the
    # ``view_coordinates`` layer's job (see ``layer_view_coordinates.py``).
    # Splitting it out keeps spec changes additive: adding a new dataset
    # entry only writes a fresh ``view_coordinates.rrd``, leaving every
    # other layer file byte-identical.
    with rec:

        if calibration is not None:
            # Camera intrinsics + extrinsics. Only ``rgb_<i>`` entries get
            # entity paths — depth_<i> calibration is intentionally skipped
            # because depth is pre-registered to its paired RGB camera and
            # shares that camera's pinhole tree.
            for cam in calibration.cameras:
                if not cam.cam_name.startswith("rgb_"):
                    continue
                idx = cam.cam_name.split("_", 1)[1]
                color = _REFERENCE_CAM_COLOR if idx == "0" else None
                log_camera_static(
                    cam,
                    entity_path=f"/world/rig_0/cam_{idx}",
                    color=color,
                    recording=rec,
                )

            # IMU extrinsic. ``imu_params`` is the first IMU's full field bag
            # from calibration.yaml. We pull ``T_BS`` (the YAML's own field
            # name, body-from-sensor) and log it as ``rig_0_T_imu_0`` —
            # the project-internal name for the same transform. Sequences
            # whose IMU is the rig reference (EUROC-style) ship identity.
            if calibration.imu_params is not None and sequence.has_imu(0):
                rig_T_imu = calibration.imu_params.get("T_BS")
                if isinstance(rig_T_imu, list):
                    _log_imu_extrinsic_static(imu_idx=0, rig_T_imu=rig_T_imu, recording=rec)

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
            cam0_intr = cam0.parameters.intrinsics
            cam0_distortion = (
                type(cam0.parameters.distortion).__name__ if cam0.parameters.distortion is not None else ""
            )
            rec.send_property(
                "calibration",
                rr.AnyValues(
                    num_cameras=len(calibration.cameras),
                    cam0_name=cam0.cam_name,
                    cam0_width=int(cam0_intr.width),
                    cam0_height=int(cam0_intr.height),
                    cam0_fx=float(cam0_intr.fl_x or 0.0),
                    cam0_fy=float(cam0_intr.fl_y or 0.0),
                    cam0_cx=float(cam0_intr.cx or 0.0),
                    cam0_cy=float(cam0_intr.cy or 0.0),
                    cam0_distortion_type=cam0_distortion,
                    has_imu_params=calibration.imu_params is not None,
                    depth_factor=cam0.depth_factor if cam0.depth_factor is not None else -1.0,
                ),
            )

    rec.save(str(out_path))
    return out_path
