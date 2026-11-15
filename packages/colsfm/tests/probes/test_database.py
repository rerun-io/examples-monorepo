"""Probe (a): what `pycolmap.Database` accepts, and which tables COLMAP 4.2 has."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32, Float64, UInt8, UInt32
from numpy import ndarray

from probes.synthetic import (
    REF_CAMERA_ID,
    RIG_ID,
    SECOND_CAMERA_ID,
    SyntheticScene,
    make_pinhole_camera,
    matches_between,
    write_database,
)


def test_write_methods_are_named_write_not_add() -> None:
    """COLMAP 4.2 renamed the pycolmap 0.x `add_*` database writers to `write_*`.

    Any code carried over from pycolmap 0.6 (`db.add_camera`, `db.add_image`,
    `db.add_keypoints`, `db.add_descriptors`, `db.add_matches`,
    `db.add_two_view_geometry`) will raise `AttributeError` against this build.
    """
    for legacy in (
        "add_camera",
        "add_image",
        "add_keypoints",
        "add_descriptors",
        "add_matches",
        "add_two_view_geometry",
    ):
        assert not hasattr(pycolmap.Database, legacy), f"unexpected legacy API {legacy}"
    for current in (
        "write_camera",
        "write_image",
        "write_keypoints",
        "write_descriptors",
        "write_matches",
        "write_two_view_geometry",
        "write_rig",
        "write_frame",
        "write_pose_prior",
    ):
        assert hasattr(pycolmap.Database, current), f"missing writer {current}"


def test_full_round_trip(scene: SyntheticScene, tmp_path: Path) -> None:
    """Cameras, rig, frames, images, keypoints, matches and geometries survive SQLite."""
    database_path: Path = tmp_path / "database.db"
    num_pairs: int = write_database(scene, database_path)
    assert num_pairs > 0

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    assert database.num_cameras() == 2
    assert database.num_rigs() == 1
    assert database.num_frames() == scene.reconstruction.num_frames()
    assert database.num_images() == scene.reconstruction.num_images()
    assert database.num_matched_image_pairs() == num_pairs
    assert database.num_verified_image_pairs() == num_pairs

    first_image_id: int = min(scene.keypoints_px)
    keypoints: Float32[ndarray, "n_obs 2"] = database.read_keypoints(first_image_id)
    assert keypoints.shape == scene.keypoints_px[first_image_id].shape
    assert np.allclose(keypoints, scene.keypoints_px[first_image_id], atol=1e-3)

    second_image_id: int = sorted(scene.keypoints_px)[1]
    stored: UInt32[ndarray, "n_matches 2"] = database.read_matches(first_image_id, second_image_id)
    assert np.array_equal(stored, matches_between(scene, first_image_id, second_image_id))

    geometry: pycolmap.TwoViewGeometry = database.read_two_view_geometry(
        first_image_id, second_image_id
    )
    assert geometry.config == int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
    database.close()


def test_rig_and_frame_tables_exist(scene: SyntheticScene, tmp_path: Path) -> None:
    """COLMAP 4.2 stores rigs and frames in the database, with a first-class API.

    `Database.write_rig` / `read_rig` / `read_all_rigs` and the corresponding
    frame calls are what carry cuSFM's `sensor_to_vehicle_transform` and the
    per-timestamp multi-camera grouping.
    """
    database_path: Path = tmp_path / "database.db"
    write_database(scene, database_path)
    database: pycolmap.Database = pycolmap.Database.open(database_path)

    rigs: list[pycolmap.Rig] = database.read_all_rigs()
    assert len(rigs) == 1
    rig: pycolmap.Rig = rigs[0]
    assert rig.rig_id == RIG_ID
    assert rig.num_sensors() == 2
    assert rig.is_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, REF_CAMERA_ID))
    cam_from_rig: pycolmap.Rigid3d | None = rig.sensor_from_rig(
        pycolmap.sensor_t(pycolmap.SensorType.CAMERA, SECOND_CAMERA_ID)
    )
    assert cam_from_rig is not None
    assert np.allclose(cam_from_rig.translation, [-0.25, 0.0, 0.0])

    frames: list[pycolmap.Frame] = database.read_all_frames()
    assert len(frames) == scene.reconstruction.num_frames()
    assert all(frame.num_data_ids() == 2 for frame in frames)
    # The database frame carries the grouping only; the pose lives in the
    # Reconstruction, not in the database's frames table.  Note the asymmetry:
    # `Frame.has_pose` is a *method*, while `Image.has_pose` is a *property*.
    assert not frames[0].has_pose()
    database.close()


def test_pose_prior_table(tmp_path: Path) -> None:
    """The database has a pose_prior table, keyed by `data_t`, with position covariance."""
    database: pycolmap.Database = pycolmap.Database.open(tmp_path / "database.db")
    camera_id: int = database.write_camera(make_pinhole_camera(1))
    image: pycolmap.Image = pycolmap.Image(name="a.png", camera_id=camera_id)
    image_id: int = database.write_image(image)

    prior: pycolmap.PosePrior = pycolmap.PosePrior()
    prior.corr_data_id = pycolmap.data_t(
        pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id), image_id
    )
    prior.position = np.array([1.0, 2.0, 3.0])
    prior.position_covariance = np.diag([0.01, 0.01, 0.04])
    prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
    prior_id: int = database.write_pose_prior(prior)

    assert database.num_pose_priors() == 1
    read_back: pycolmap.PosePrior = database.read_pose_prior(prior_id, False)
    assert read_back.has_position()
    assert read_back.has_position_cov()
    assert np.allclose(read_back.position, [1.0, 2.0, 3.0])
    assert read_back.coordinate_system == pycolmap.PosePriorCoordinateSystem.CARTESIAN
    # Priors are position-and-gravity only: there is no rotation prior field.
    assert not read_back.has_gravity()
    database.close()


def test_descriptors_require_the_feature_descriptors_wrapper(tmp_path: Path) -> None:
    """A bare numpy array is rejected; descriptors must be a `FeatureDescriptors`.

    The SQLite column is an opaque uint8 blob plus a row/column count, so the
    binding refuses the ambiguity of a plain array.
    """
    database: pycolmap.Database = pycolmap.Database.open(tmp_path / "database.db")
    camera_id: int = database.write_camera(make_pinhole_camera(1))
    image_id: int = database.write_image(pycolmap.Image(name="a.png", camera_id=camera_id))

    raw: UInt8[ndarray, "5 128"] = np.zeros((5, 128), dtype=np.uint8)
    with pytest.raises(TypeError):
        database.write_descriptors(image_id, raw)
    database.close()


@pytest.mark.parametrize("descriptor_dim", [64, 128, 256])
def test_float_descriptors_round_trip_at_any_dimension(
    tmp_path: Path, descriptor_dim: int
) -> None:
    """ALIKED-style float32 descriptors of arbitrary width survive the database.

    `FeatureDescriptorsFloat.to_bytes()` reinterprets the float32 buffer as
    uint8 (so an Nx128 float block is stored as an Nx512 uint8 blob) and
    `FeatureDescriptors.to_float()` reverses it exactly.  The dimension is not
    constrained to SIFT's 128, and the dtype is not constrained to uint8 —
    but the *storage* dtype always is.
    """
    database: pycolmap.Database = pycolmap.Database.open(tmp_path / "database.db")
    camera_id: int = database.write_camera(make_pinhole_camera(1))
    image_id: int = database.write_image(pycolmap.Image(name="a.png", camera_id=camera_id))

    values: Float32[ndarray, "5 dim"] = (
        np.arange(5 * descriptor_dim, dtype=np.float32).reshape(5, descriptor_dim) / 7.0
    )
    descriptors: pycolmap.FeatureDescriptorsFloat = pycolmap.FeatureDescriptorsFloat(
        data=values, type=pycolmap.FeatureExtractorType.ALIKED_N32
    )
    database.write_descriptors(image_id, descriptors.to_bytes())

    stored: pycolmap.FeatureDescriptors = database.read_descriptors(image_id)
    assert stored.data.dtype == np.uint8
    assert stored.data.shape == (5, descriptor_dim * 4)
    assert stored.type == pycolmap.FeatureExtractorType.ALIKED_N32

    recovered: pycolmap.FeatureDescriptorsFloat = stored.to_float()
    assert recovered.data.dtype == np.float32
    assert np.array_equal(recovered.data, values)
    database.close()


def test_feature_extractor_types_include_aliked() -> None:
    """COLMAP 4.2 tags descriptors with a feature type, and ALIKED is one of them."""
    names: set[str] = {
        name for name in dir(pycolmap.FeatureExtractorType) if not name.startswith("_")
    }
    assert {"SIFT", "ALIKED_N16ROT", "ALIKED_N32"} <= names


def test_keypoints_store_any_column_width(tmp_path: Path) -> None:
    """Keypoints are float32 Nx`k` and COLMAP stores `k` verbatim, without validating it.

    COLMAP itself writes Nx2 (xy), Nx4 (xy + scale + orientation) or Nx6
    (xy + 2x2 affine), but the database enforces none of that: an Nx3 or Nx5
    block round-trips unchanged.  A malformed width is therefore a silent
    corruption, not an error, and the caller must get it right.

    Writing twice for the same image hits a UNIQUE constraint, so each width is
    probed on a fresh image row.
    """
    database: pycolmap.Database = pycolmap.Database.open(tmp_path / "database.db")
    camera_id: int = database.write_camera(make_pinhole_camera(1))
    accepted: dict[int, bool] = {}
    for num_columns in (1, 2, 3, 4, 5, 6, 8):
        image_id: int = database.write_image(
            pycolmap.Image(name=f"cols{num_columns}.png", camera_id=camera_id)
        )
        keypoints: Float32[ndarray, "5 cols"] = np.zeros((5, num_columns), dtype=np.float32)
        try:
            database.write_keypoints(image_id, keypoints)
            accepted[num_columns] = database.read_keypoints(image_id).shape == (5, num_columns)
        except (RuntimeError, TypeError, ValueError):
            accepted[num_columns] = False
    print(f"[probe] keypoint column widths accepted: {accepted}")
    assert accepted[2] and accepted[4] and accepted[6]
    assert accepted[3] and accepted[5], "widths are not validated on write"
    database.close()


def test_keypoints_are_cast_to_float32(tmp_path: Path) -> None:
    """Keypoints are stored as float32 whatever the input dtype, so sub-pixel
    coordinates lose float64 precision on the way into the database."""
    database: pycolmap.Database = pycolmap.Database.open(tmp_path / "database.db")
    camera_id: int = database.write_camera(make_pinhole_camera(1))
    image_id: int = database.write_image(pycolmap.Image(name="a.png", camera_id=camera_id))
    exact: Float64[ndarray, "1 2"] = np.array([[123.456789012345, 67.891234567890]])
    database.write_keypoints(image_id, exact)
    stored: Float32[ndarray, "1 2"] = database.read_keypoints(image_id)
    assert stored.dtype == np.float32
    assert not np.array_equal(stored.astype(np.float64), exact)
    assert np.allclose(stored, exact, atol=1e-4)
    database.close()
