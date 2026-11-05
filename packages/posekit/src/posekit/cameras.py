"""Calibrated source cameras for models that undo lens distortion (EgoHand's perspective crops).

A pure record: the lens projection itself lives with the model that needs it, so
importing this module pulls in no GPU or warp code.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from serde import SerdeError, from_dict, serde

LensModel: TypeAlias = Literal["PINHOLE", "RATIONAL_POLYNOMIAL", "FISHEYE624", "KB4"]
"""Lens models EgoHand's camera code projects: no distortion, OpenCV 14-term, Meta fisheye 6-2-4, Kannala-Brandt 4."""
LENS_COEFFICIENTS: dict[LensModel, int] = {"PINHOLE": 0, "RATIONAL_POLYNOMIAL": 14, "FISHEYE624": 12, "KB4": 4}
"""Distortion coefficient count of each lens model."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    """Intrinsics and lens distortion of the camera that took the frames, in its source pixels."""

    intrinsics: Float64[ndarray, "3 3"]
    """Pixel intrinsics ``K`` of the original decoded image (zero skew)."""
    distortion: Float64[ndarray, "d"]
    """Distortion coefficients in the order of ``lens``: OpenCV ``k1,k2,p1,p2,k3,k4,k5,k6,s1..s4,tx,ty``;
    FISHEYE624 ``k0..k5, p0, p1, s0..s3``; KB4 ``k1..k4``; PINHOLE none."""
    width: int
    """Original image width in pixels."""
    height: int
    """Original image height in pixels."""
    lens: LensModel = "RATIONAL_POLYNOMIAL"
    """Lens model the coefficients belong to."""

    def __post_init__(self) -> None:
        expected: int = LENS_COEFFICIENTS[self.lens]
        if self.distortion.shape != (expected,):
            raise ValueError(f"{self.lens} needs {expected} distortion coefficients, got {self.distortion.shape[0]}.")
        if min(self.width, self.height) <= 0:
            raise ValueError(f"Camera size must be positive, got {self.width}x{self.height}.")


def save_calibrated_camera(camera: CalibratedCamera, path: Path) -> None:
    """Write a camera as a static ``posekit-calibration`` recording: ``/camera`` holds a ``Pinhole`` and the
    ``distortion`` (float64) and ``lens`` AnyValues.

    Raises:
        ValueError: If the intrinsics do not survive the float32 the Pinhole stores.
    """
    if not np.array_equal(camera.intrinsics.astype(np.float32), camera.intrinsics):
        raise ValueError(f"{path}: intrinsics are not float32-exact, a Pinhole would round them.")
    pinhole: Chunk = Chunk.from_columns(
        "/camera", indexes=[], columns=rr.Pinhole.columns(image_from_camera=[camera.intrinsics], resolution=[[camera.width, camera.height]])
    )
    lens: Chunk = Chunk.from_columns(
        "/camera",
        indexes=[],
        columns=[*rr.AnyValues.columns(distortion=camera.distortion).partition([len(camera.distortion)]), *rr.AnyValues.columns(lens=[camera.lens])],
    )
    LazyChunkStream.from_iter([pinhole, lens]).write_rrd(path, application_id="posekit-calibration", recording_id=path.stem)


def load_calibrated_camera(path: Path) -> CalibratedCamera:
    """Read a ``CalibratedCamera`` .rrd written by ``save_calibrated_camera``.

    Args:
        path: .rrd holding one camera at ``/camera``.

    Returns:
        The validated camera.

    Raises:
        ValueError: If the recording lacks a component or does not match the schema.
    """
    row: dict[str, Any] = {}
    for chunk in RrdReader(path).stream():
        if chunk.entity_path == "/camera":
            batch = chunk.to_record_batch()
            row |= {name: batch.column(name)[0].as_py() for name in batch.schema.names if not name.startswith("rerun.")}
    try:
        # Rerun matrices are column-major.
        intrinsics: list[list[float]] = np.reshape(row["Pinhole:image_from_camera"][0], (3, 3)).T.tolist()
        width, height = row["Pinhole:resolution"][0]
        fields: dict[str, Any] = {"intrinsics": intrinsics, "distortion": row["distortion"], "width": int(width), "height": int(height), "lens": row["lens"][0]}
        return from_dict(CalibratedCamera, fields)
    except (SerdeError, KeyError) as error:
        raise ValueError(f"{path}: not a CalibratedCamera: {error!r}") from error
