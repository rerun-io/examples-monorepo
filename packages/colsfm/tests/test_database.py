"""The COLMAP database `colsfm` hands to feature extraction and matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32
from numpy import ndarray

from colsfm.database import (
    ImagePair,
    create_database,
    image_ids_by_name,
    keypoint_counts,
    pair_inlier_counts,
    pairs_with_matches,
    raw_match_counts,
    read_keypoints,
    read_two_view_geometry,
)
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.reconstruction import RIG_ID, camera_sensor_id, vehicle_sensor_id


@pytest.fixture(scope="module")
def galileo(galileo_input_meta: Path) -> FramesMeta:
    """The 226-keyframe Galileo input metadata."""
    return read_frames_meta(galileo_input_meta)


@pytest.fixture(scope="module")
def galileo_database(galileo: FramesMeta, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A database written from the whole Galileo collection, built once."""
    database_path: Path = tmp_path_factory.mktemp("database") / "galileo.db"
    create_database(database_path, galileo)
    return database_path


def test_cameras_carry_the_cusfm_ids_and_a_focal_prior(galileo: FramesMeta, galileo_database: Path) -> None:
    """Every `camera_params_id` becomes a COLMAP camera id verbatim.

    `has_prior_focal_length` must be True or `estimate_two_view_geometry` takes
    the uncalibrated path, which returns DEGENERATE for fisheye cameras
    (pycolmap-capabilities.md §5).
    """
    database: pycolmap.Database = pycolmap.Database.open(galileo_database)
    assert database.num_cameras() == len(galileo.cameras)
    for camera_params_id, camera in sorted(galileo.cameras.items()):
        stored: pycolmap.Camera = database.read_camera(camera_params_id)
        assert stored.has_prior_focal_length is True
        assert stored.width == camera.image_width
        assert stored.height == camera.image_height
    database.close()


def test_images_are_keyframe_ids_named_relative_to_the_frames_meta(galileo: FramesMeta, galileo_database: Path) -> None:
    """`image_id` is the cuSFM keyframe id and `name` is `image_name` verbatim."""
    database: pycolmap.Database = pycolmap.Database.open(galileo_database)
    assert database.num_images() == len(galileo.keyframes)
    for keyframe in galileo.keyframes:
        image: pycolmap.Image = database.read_image(keyframe.keyframe_id)
        assert image.name == keyframe.image_name
        assert image.camera_id == keyframe.camera_params_id
        assert image.frame_id == keyframe.synced_sample_id
    database.close()


def test_rig_and_frames_describe_the_vehicle(galileo: FramesMeta, galileo_database: Path) -> None:
    """The database carries cuSFM's rig: a vehicle reference sensor plus every camera.

    COLMAP 4.2 stores rigs and frames in the database, so a `pycolmap.Database`
    consumer sees the multi-camera structure without re-reading `frames_meta.json`.
    """
    database: pycolmap.Database = pycolmap.Database.open(galileo_database)
    rigs: list[pycolmap.Rig] = database.read_all_rigs()
    assert len(rigs) == 1
    rig: pycolmap.Rig = rigs[0]
    assert rig.rig_id == RIG_ID
    assert rig.is_ref_sensor(vehicle_sensor_id())
    assert rig.num_sensors() == len(galileo.cameras) + 1

    cam_from_rig: pycolmap.Rigid3d | None = rig.sensor_from_rig(camera_sensor_id(0))
    assert cam_from_rig is not None
    assert np.allclose(cam_from_rig.matrix(), galileo.cameras[0].vehicle_T_cam.inverse().matrix())

    frames: list[pycolmap.Frame] = database.read_all_frames()
    assert len(frames) == len(galileo.rig_frames())
    frame_ids: set[int] = {frame.frame_id for frame in frames}
    assert frame_ids == {rig_frame.synced_sample_id for rig_frame in galileo.rig_frames()}
    assert sum(frame.num_data_ids() for frame in frames) == len(galileo.keyframes)
    database.close()


def test_image_ids_by_name_round_trips(galileo: FramesMeta, galileo_database: Path) -> None:
    """The name index is what turns a pair of keyframe ids into a COLMAP pair list."""
    index: dict[str, int] = image_ids_by_name(galileo_database)
    assert len(index) == len(galileo.keyframes)
    for keyframe in galileo.keyframes:
        assert index[keyframe.image_name] == keyframe.keyframe_id


def test_create_database_refuses_to_clobber(galileo: FramesMeta, galileo_database: Path) -> None:
    """Writing over an existing database needs `overwrite=True`."""
    with pytest.raises(FileExistsError):
        create_database(galileo_database, galileo)
    create_database(galileo_database, galileo, overwrite=True)
    assert image_ids_by_name(galileo_database)


def test_keypoint_and_geometry_readers(galileo: FramesMeta, tmp_path: Path) -> None:
    """`read_keypoints` returns xy only, and the geometry readers see what was written.

    COLMAP writes Nx6 keypoint rows (xy plus a 2x2 affine block); the pipeline
    only ever wants the first two columns.
    """
    database_path: Path = tmp_path / "readers.db"
    create_database(database_path, galileo)
    first: int = galileo.keyframes[0].keyframe_id
    second: int = galileo.keyframes[1].keyframe_id

    written: Float32[ndarray, "4 6"] = np.zeros((4, 6), dtype=np.float32)
    written[:, 0] = [1.0, 2.0, 3.0, 4.0]
    written[:, 1] = [5.0, 6.0, 7.0, 8.0]
    geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
    geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
    geometry.inlier_matches = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    database.write_keypoints(first, written)
    database.write_keypoints(second, written)
    database.write_two_view_geometry(first, second, geometry)
    database.close()

    keypoints_xy: Float32[ndarray, "4 2"] = read_keypoints(database_path, first)
    assert keypoints_xy.shape == (4, 2)
    assert np.allclose(keypoints_xy[:, 0], [1.0, 2.0, 3.0, 4.0])

    assert keypoint_counts(database_path)[first] == 4
    assert read_two_view_geometry(database_path, first, second).config == int(
        pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    )

    pairs: list[ImagePair] = [(first, second), (second, galileo.keyframes[2].keyframe_id)]
    counts: dict[ImagePair, int] = pair_inlier_counts(database_path, pairs)
    assert counts[(first, second)] == 3
    assert counts[(second, galileo.keyframes[2].keyframe_id)] == 0


def test_a_matched_pair_with_no_matches_is_present_but_uncounted(galileo: FramesMeta, tmp_path: Path) -> None:
    """`pairs_with_matches` reports finished work; `raw_match_counts` reports how much of it.

    The loop-closure stage used to ask for counts and treat 0 as "not matched", so a
    pair the matcher had already run and found nothing in was re-matched to produce
    the same nothing. Presence and count are different questions and now have
    different readers.
    """
    database_path: Path = tmp_path / "presence.db"
    create_database(database_path, galileo)
    first, second, third = (keyframe.keyframe_id for keyframe in galileo.keyframes[:3])

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    database.write_matches(first, second, np.array([[0, 1], [1, 2]], dtype=np.uint32))
    database.write_matches(first, third, np.zeros((0, 2), dtype=np.uint32))
    database.close()

    matched: ImagePair = (min(first, second), max(first, second))
    empty: ImagePair = (min(first, third), max(first, third))
    never: ImagePair = (min(second, third), max(second, third))

    present: set[ImagePair] = pairs_with_matches(database_path, [matched, empty, never])
    assert present == {matched, empty}, "an empty result is a result; only `never` is missing work"

    counts: dict[ImagePair, int] = raw_match_counts(database_path, [matched, empty, never])
    assert counts[matched] == 2
    assert counts[empty] == 0
    assert counts[never] == 0


def test_selected_subset_only_writes_kept_frames(galileo: FramesMeta, tmp_path: Path) -> None:
    """A filtered collection produces a database holding exactly those images.

    This is how the extractor stage feeds keyframe selection into COLMAP: select
    first, filter the metadata, then create the database.
    """
    kept_ids: list[int] = [keyframe.keyframe_id for keyframe in galileo.keyframes[:12]]
    subset: FramesMeta = galileo.filtered(kept_ids)
    database_path: Path = tmp_path / "subset.db"
    create_database(database_path, subset)

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    assert database.num_images() == len(kept_ids)
    # Cameras are written for every calibration, used or not, exactly as
    # `kpmap_to_colmap` does (export.md §3.3).
    assert database.num_cameras() == len(galileo.cameras)
    database.close()
