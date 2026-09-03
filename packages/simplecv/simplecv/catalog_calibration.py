"""Read a camera's static calibration back out of a Rerun catalog layer.

The inverse of ``simplecv.rerun_log_utils.log_pinhole``: the camera node carries a
static ``Transform3D`` (``cam_R_world`` / ``cam_t_world``, logged ``from_parent``)
and its ``pinhole`` child a static ``Pinhole`` (``image_from_camera`` plus the
``resolution``). Rerun stores both 3x3 matrices column-major, so they are
transposed here — the one place that convention lives on the read side.
"""

from collections.abc import Sequence

import numpy as np
import pyarrow as pa
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rrd_query_utils import first_valid_value


def read_camera_calibration(dataset: DatasetEntry, segment_id: str, camera_entities: Sequence[str]) -> dict[str, PinholeParameters]:
    """Read the static pinhole calibration of the given camera nodes.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        camera_entities: Camera node entity paths, e.g. ``/world/rig_00/cam_00``; each carries the
            ``Transform3D`` and has a ``pinhole`` child carrying the ``Pinhole``.

    Returns:
        Float64 pinhole parameters (RDF, ``cam_T_world``) keyed by camera entity path.

    Raises:
        ValueError: If a camera lacks one of the four calibration components in this segment.
    """
    view = dataset.filter_segments(segment_id).filter_contents([f"{entity}/**" for entity in camera_entities])
    table: pa.Table = view.reader(index=None).to_arrow_table()

    def static_value(column_name: str) -> Float64[ndarray, " n"]:
        if column_name not in table.column_names:
            raise ValueError(f"column {column_name} has no data in segment {segment_id}")
        return np.asarray(first_valid_value(table.column(column_name), component_name=column_name), dtype=np.float64).reshape(-1)

    calibration: dict[str, PinholeParameters] = {}
    for entity in camera_entities:
        cam_R_world: Float64[ndarray, "3 3"] = static_value(f"{entity}:Transform3D:mat3x3").reshape(3, 3).T
        cam_t_world: Float64[ndarray, "3"] = static_value(f"{entity}:Transform3D:translation").reshape(3)
        image_from_camera: Float64[ndarray, "3 3"] = static_value(f"{entity}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        resolution_wh: Float64[ndarray, "2"] = static_value(f"{entity}/pinhole:Pinhole:resolution").reshape(2)
        calibration[entity] = PinholeParameters(
            name=entity.removeprefix("/world/").replace("/", "_"),
            intrinsics=Intrinsics(
                camera_conventions="RDF", k_matrix=image_from_camera, width=int(resolution_wh[0]), height=int(resolution_wh[1])
            ),
            extrinsics=Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world),
        )
    return calibration
