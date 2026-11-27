"""A calibration as a Rerun catalog layer: `<root>/<camera name>` the way a segment's own camera nodes carry it.

Each camera is a static `Transform3D` (`cam_T_rig`, child-from-parent) and a static `Pinhole` with simplecv's distortion
components on `pinhole` (`simplecv.rerun_log_utils.log_pinhole`), so the read side is the camera nodes' own reader:
`colsfm.catalog.read_catalog_cameras(..., calibration_entity=root)`. `colsfm calibrate` writes one under
`CALIBRATION_ENTITY`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import rerun as rr
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from colsfm.catalog import CatalogCameraModel
from colsfm.rerun_layer import IMAGE_PLANE_DISTANCE, open_layer

CALIBRATION_ENTITY: Final[str] = "/world/runs/colsfm_calibrate/rig"
"""Where `colsfm calibrate` writes its cameras, one child per camera name."""


def simplecv_camera(camera: CatalogCameraModel) -> PinholeParameters | Fisheye62Parameters:
    """The simplecv camera `log_pinhole` logs, with the catalog camera's lens and `rig_T_cam`.

    Raises:
        ValueError: When the distortion has the wrong number of coefficients for the projection.
    """
    extrinsics: Extrinsics = Extrinsics(world_R_cam=camera.rig_T_cam[:3, :3], world_t_cam=camera.rig_T_cam[:3, 3])
    intrinsics: Intrinsics = Intrinsics(camera_conventions="RDF", k_matrix=camera.k_matrix, width=camera.width, height=camera.height)
    wanted: int = {"OPENCV_FISHEYE": 4, "DISTORTED_PINHOLE": 8, "PINHOLE": 0}[camera.projection]
    if len(camera.distortion) != wanted:
        raise ValueError(f"{camera.name}: {camera.projection} takes {wanted} distortion coefficients, got {len(camera.distortion)}")
    if camera.projection == "OPENCV_FISHEYE":
        k1, k2, k3, k4 = camera.distortion
        return Fisheye62Parameters(camera.name, extrinsics, intrinsics, KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4))
    if camera.projection == "DISTORTED_PINHOLE":
        k1, k2, p1, p2, k3, k4, k5, k6 = camera.distortion
        return PinholeParameters(camera.name, extrinsics, intrinsics, BrownConradyDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, k4=k4, k5=k5, k6=k6))
    return PinholeParameters(camera.name, extrinsics, intrinsics)


def write_calibration_layer(
    rrd_path: Path, segment: str, cameras: Sequence[CatalogCameraModel], properties: rr.AnyValues, root: str = CALIBRATION_ENTITY
) -> None:
    """Write cameras as a calibration layer of `segment`; the file appears whole or not at all.

    Args:
        rrd_path: Where the layer goes; parent directories are created.
        segment: The catalog segment id (the layer's recording id).
        cameras: The calibration to write.
        properties: The `colsfm_calibrate` recording property (how the calibration was made).
        root: The entity the cameras go under.

    Raises:
        ValueError: When a camera's distortion does not fit its projection (nothing is written).
    """
    models: list[PinholeParameters | Fisheye62Parameters] = [simplecv_camera(camera) for camera in cameras]
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    # A staging file of this call's own, beside the destination, so `os.replace` stays on one file system and two
    # writers of the same layer never share one.
    descriptor, name = tempfile.mkstemp(prefix=f".{rrd_path.name}.", suffix=".partial", dir=rrd_path.parent)
    os.close(descriptor)
    staging: Path = Path(name)
    try:
        recording: rr.RecordingStream = open_layer(staging, segment)
        try:
            for model in models:
                log_pinhole(model, Path(f"{root}/{model.name}"), image_plane_distance=IMAGE_PLANE_DISTANCE, static=True, recording=recording)
            recording.send_property("colsfm_calibrate", properties)
            recording.flush()
        finally:
            recording.disconnect()
        staging.chmod(0o644)  # mkstemp's 0600 would follow the file; a layer is read by other users' catalog servers
        os.replace(staging, rrd_path)
    finally:
        staging.unlink(missing_ok=True)
