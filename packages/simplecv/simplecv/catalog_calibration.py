"""Read a camera's static calibration back out of a Rerun catalog layer.

The inverse of ``simplecv.rerun_log_utils.log_pinhole``: the camera node carries a
static ``Transform3D`` (``cam_R_world`` / ``cam_t_world``, logged ``from_parent``)
and its ``pinhole`` child a static ``Pinhole`` (``image_from_camera`` plus the
``resolution``). Rerun stores both 3x3 matrices column-major, so they are
transposed here — the one place that convention lives on the read side. A camera
logged with distortion (``PinholeWithDistortion``) also carries the simplecv
``DistortionModel`` / ``DistortionCoefficients`` components on its ``pinhole`` child;
``read_camera_models`` returns them alongside the pinhole. ``camera_models_from_static_table`` decodes the static
columns whatever they come from: the catalog (``read_camera_models``) or a recording file
(``read_camera_models_from_store``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun.chunk as rrc
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_custom_types import _DISTORTION_COEFF_COMPONENT as DISTORTION_COEFFICIENTS_COMPONENT
from simplecv.rerun_custom_types import _DISTORTION_MODEL_COMPONENT as DISTORTION_MODEL_COMPONENT
from simplecv.rrd_query_utils import first_valid_value


@dataclass(frozen=True, slots=True)
class CatalogDistortion:
    """A camera's lens distortion as ``PinholeWithDistortion`` logged it."""

    model: str
    """The ``DistortionModel`` name, lower case, e.g. ``brown_conrady`` or ``kannala_brandt``."""
    coefficients: Float64[ndarray, " n"]
    """The ``DistortionCoefficients`` in that model's logged order (``rerun_custom_types.PinholeWithDistortion``)."""


@dataclass(frozen=True, slots=True)
class CatalogCameraCalibration:
    """One camera node's static calibration: the pinhole and, when logged, its distortion."""

    pinhole: PinholeParameters
    """Intrinsics (``Pinhole``) and ``cam_T_world`` (the node's ``Transform3D``)."""
    distortion: CatalogDistortion | None
    """None when the camera carries no distortion components."""


def read_camera_calibration(dataset: DatasetEntry, segment_id: str, camera_entities: Sequence[str]) -> dict[str, PinholeParameters]:
    """Read the static pinhole calibration of the given camera nodes.

    Needs only the four calibration components: a camera's distortion components, complete or not, are not read.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        camera_entities: Camera node entity paths, e.g. ``/world/rig_00/cam_00``.

    Returns:
        Float64 pinhole parameters (RDF, ``cam_T_world``) keyed by camera entity path.

    Raises:
        ValueError: If a camera lacks one of the four calibration components in this segment.
    """
    return {entity: model.pinhole for entity, model in _read_cameras(dataset, segment_id, camera_entities, with_distortion=False).items()}


def read_camera_models(dataset: DatasetEntry, segment_id: str, camera_entities: Sequence[str]) -> dict[str, CatalogCameraCalibration]:
    """Read the static pinhole calibration and the distortion of the given camera nodes, in one query.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        camera_entities: Camera node entity paths, e.g. ``/world/rig_00/cam_00``; each carries the
            ``Transform3D`` and has a ``pinhole`` child carrying the ``Pinhole``.

    Returns:
        The calibration keyed by camera entity path.

    Raises:
        ValueError: If a camera lacks one of the four calibration components in this segment, or carries a
            distortion model without coefficients.
    """
    return _read_cameras(dataset, segment_id, camera_entities, with_distortion=True)


def read_camera_models_from_store(store: rrc.LazyStore, camera_entities: Sequence[str], where: str) -> dict[str, CatalogCameraCalibration]:
    """``read_camera_models`` for a recording file: the same static columns, read from its store.

    Raises:
        ValueError: As ``read_camera_models``, naming ``where`` (e.g. the file).
    """
    chunks: list[rrc.Chunk] = store.stream().filter(content=[f"{entity}/**" for entity in camera_entities], is_static=True).to_chunks()
    table: pa.Table = rrc.ChunkStore.from_chunks(chunks).reader(index=None).to_arrow_table()
    return camera_models_from_static_table(table, camera_entities, where)


def _read_cameras(
    dataset: DatasetEntry, segment_id: str, camera_entities: Sequence[str], *, with_distortion: bool
) -> dict[str, CatalogCameraCalibration]:
    """One query for the camera nodes; the pinhole always, the distortion components only when asked for."""
    view = dataset.filter_segments(segment_id).filter_contents([f"{entity}/**" for entity in camera_entities])
    return camera_models_from_static_table(
        view.reader(index=None).to_arrow_table(), camera_entities, f"segment {segment_id}", with_distortion=with_distortion
    )


def camera_models_from_static_table(
    table: pa.Table, camera_entities: Sequence[str], where: str, *, with_distortion: bool = True
) -> dict[str, CatalogCameraCalibration]:
    """Decode the camera nodes' calibration from a static table (``reader(index=None)`` of a catalog view or a store).

    Raises:
        ValueError: If a camera lacks one of the four calibration components, naming ``where``.
    """

    def static_value(column_name: str) -> Float64[ndarray, " n"]:
        if column_name not in table.column_names:
            raise ValueError(f"column {column_name} has no data in {where}")
        return np.asarray(first_valid_value(table.column(column_name), component_name=column_name), dtype=np.float64).reshape(-1)

    calibration: dict[str, CatalogCameraCalibration] = {}
    for entity in camera_entities:
        cam_R_world: Float64[ndarray, "3 3"] = static_value(f"{entity}:Transform3D:mat3x3").reshape(3, 3).T
        cam_t_world: Float64[ndarray, "3"] = static_value(f"{entity}:Transform3D:translation").reshape(3)
        image_from_camera: Float64[ndarray, "3 3"] = static_value(f"{entity}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        resolution_wh: Float64[ndarray, "2"] = static_value(f"{entity}/pinhole:Pinhole:resolution").reshape(2)
        pinhole: PinholeParameters = PinholeParameters(
            name=entity.removeprefix("/world/").replace("/", "_"),
            intrinsics=Intrinsics(camera_conventions="RDF", k_matrix=image_from_camera, width=int(resolution_wh[0]), height=int(resolution_wh[1])),
            extrinsics=Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world),
        )
        model_column: str = f"{entity}/pinhole:{DISTORTION_MODEL_COMPONENT}"
        distortion: CatalogDistortion | None = None
        if with_distortion and model_column in table.column_names:
            model: object = first_valid_value(table.column(model_column), allow_none=True, component_name=model_column)
            # The model is a one-element string batch; unwrap it the way the coefficients are flattened.
            if isinstance(model, list):
                model = model[0] if model else None
            if model is not None:
                distortion = CatalogDistortion(
                    model=str(model).lower(), coefficients=static_value(f"{entity}/pinhole:{DISTORTION_COEFFICIENTS_COMPONENT}")
                )
        calibration[entity] = CatalogCameraCalibration(pinhole=pinhole, distortion=distortion)
    return calibration
