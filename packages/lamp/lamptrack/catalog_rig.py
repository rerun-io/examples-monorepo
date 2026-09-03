"""Small exoego:v2 catalog readers duplicated pending a shared SimpleCV reader.

These functions mirror ``monopriors.apis.stereo_catalog`` so ``lamptrack`` does
not depend on the monoprior package. Move both callers to SimpleCV when its
catalog rig reader becomes public.
"""

import numpy as np
import pyarrow as pa
from jaxtyping import Float64
from rerun.catalog import DatasetView
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.rrd_query_utils import first_valid_value

TIMELINE = "video_time"
RIG = "world/rig_00"


def _read_static(view: DatasetView, entity: str, component: str) -> object:
    """Read the first value of one static component."""
    table: pa.Table = view.filter_contents(entity).reader(index=None).select(f"/{entity}:{component}").to_arrow_table()
    return first_valid_value(table.column(0), component_name=component)


def read_fisheye_camera(view: DatasetView, cam: str) -> Fisheye62Parameters:
    """Read one dataforge exoego:v2 camera whose extrinsics are ``cam_T_rig``."""
    pinhole = f"{RIG}/{cam}/pinhole"
    model = str(np.asarray(_read_static(view, pinhole, "simplecv.components.DistortionModel")).ravel()[0])
    if model != "kannala_brandt":
        raise ValueError(f"{pinhole}: expected kannala_brandt distortion, got {model!r}")
    K_33: Float64[np.ndarray, "3 3"] = np.asarray(
        _read_static(view, pinhole, "Pinhole:image_from_camera"), dtype=np.float64
    ).reshape(3, 3, order="F")
    resolution = np.asarray(_read_static(view, pinhole, "Pinhole:resolution"), dtype=np.float64)
    coefficients = np.asarray(
        _read_static(view, pinhole, "simplecv.components.DistortionCoefficients"), dtype=np.float64
    )
    return Fisheye62Parameters(
        name=str(np.asarray(_read_static(view, f"{RIG}/{cam}", "name")).ravel()[0]),
        extrinsics=Extrinsics(
            cam_R_world=np.asarray(_read_static(view, f"{RIG}/{cam}", "Transform3D:mat3x3"), dtype=np.float64).reshape(
                3, 3, order="F"
            ),
            cam_t_world=np.asarray(_read_static(view, f"{RIG}/{cam}", "Transform3D:translation"), dtype=np.float64),
        ),
        intrinsics=Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=K_33,
            height=int(resolution[1]),
            width=int(resolution[0]),
        ),
        distortion=KannalaBrandtDistortion(
            k1=float(coefficients[0]),
            k2=float(coefficients[1]),
            k3=float(coefficients[2]),
            k4=float(coefficients[3]),
            k5=float(coefficients[4]),
            k6=float(coefficients[5]),
            p1=float(coefficients[6]),
            p2=float(coefficients[7]),
        ),
    )


def read_rig_poses(view: DatasetView) -> tuple[np.ndarray, Float64[np.ndarray, "n 4 4"]]:
    """Read temporal ``world_T_rig`` timestamps and poses from the SLAM layer."""
    table: pa.Table = (
        view.filter_contents(RIG)
        .reader(index=TIMELINE)
        .select(TIMELINE, f"/{RIG}:Transform3D:translation", f"/{RIG}:Transform3D:quaternion")
        .sort(TIMELINE)
        .to_arrow_table()
    )
    if table.num_rows == 0:
        raise ValueError(f"{RIG} carries no temporal Transform3D — is the slam layer registered?")
    times = np.array([timestamp.value for timestamp in table.column(0)], dtype="timedelta64[ns]")
    translations: Float64[np.ndarray, "n 3"] = np.array([value[0] for value in table.column(1).to_pylist()], dtype=np.float64)
    quaternions: Float64[np.ndarray, "n 4"] = np.array([value[0] for value in table.column(2).to_pylist()], dtype=np.float64)
    poses: Float64[np.ndarray, "n 4 4"] = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    poses[:, :3, 3] = translations
    return times, poses


__all__ = ("RIG", "TIMELINE", "read_fisheye_camera", "read_rig_poses")
