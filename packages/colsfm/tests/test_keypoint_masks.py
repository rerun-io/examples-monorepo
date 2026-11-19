"""Keypoint masks: a keypoint on a masked pixel is never stored; the others keep their order and descriptors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Bool, Float32
from numpy import ndarray

from colsfm.catalog import CatalogCameraModel, frames_meta_document
from colsfm.database import create_database, read_descriptors_from_database, read_keypoints
from colsfm.frame_manifest import FrameSourceManifest, StreamKeyframes
from colsfm.frames_meta import FRAMES_META_NAME, read_frames_meta
from colsfm.keypoint_masks import CatalogMasks, DictMasks, keypoints_outside_mask, open_mask_source

WIDTH: int = 64
HEIGHT: int = 48


def test_keypoints_on_masked_pixels_are_dropped_and_the_rest_keep_their_order() -> None:
    mask: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:20, 30:40] = True
    keypoints: Float32[ndarray, "5 2"] = np.asarray([[5.0, 5.0], [35.0, 15.0], [29.4, 15.0], [39.6, 19.4], [63.0, 47.0]], dtype=np.float32)

    kept: Bool[ndarray, " 5"] = keypoints_outside_mask(keypoints, mask)

    # 29.4 rounds to column 29 (outside); 39.6 rounds to 40 (outside); 35,15 is inside.
    assert kept.tolist() == [True, False, True, True, True]


def test_keypoints_past_the_last_pixel_centre_are_clipped_into_the_image() -> None:
    mask: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[-1, -1] = True

    kept: Bool[ndarray, " 1"] = keypoints_outside_mask(np.asarray([[63.6, 47.6]], dtype=np.float32), mask)

    assert kept.tolist() == [False]


def test_a_held_mask_of_the_wrong_size_is_refused() -> None:
    masks: DictMasks = DictMasks({"cam/1": np.zeros((HEIGHT, WIDTH), dtype=bool)})

    with pytest.raises(ValueError, match="mask is 64x48"):
        masks.mask("cam/1", HEIGHT, WIDTH)
    assert masks.mask("other/1", WIDTH, HEIGHT) is None


def test_an_input_without_mask_entities_opens_no_mask_source() -> None:
    manifest: FrameSourceManifest = FrameSourceManifest(
        catalog_url="rerun+http://127.0.0.1:1",
        dataset="d",
        segment="s",
        timeline="video_time",
        streams=[StreamKeyframes(entity="world/rig_00/cam_02/pinhole/video", names=["a/1"], t_ns=[1])],
    )

    assert open_mask_source(manifest) is None
    assert open_mask_source(None) is None


def _two_camera_database(tmp_path: Path) -> Path:
    """A database with one image per camera: a 64x48 and a 48x64 camera (two image sizes)."""
    k_matrix = np.asarray([[32.0, 0.0, 31.5], [0.0, 32.0, 23.5], [0.0, 0.0, 1.0]])
    cameras: list[CatalogCameraModel] = [
        CatalogCameraModel(
            entity=f"/world/rig_00/cam_0{index}",
            name=name,
            rig_T_cam=np.eye(4),
            k_matrix=k_matrix,
            width=width,
            height=height,
            projection="PINHOLE",
            distortion=(),
        )
        for index, (name, width, height) in enumerate([("wide", WIDTH, HEIGHT), ("tall", HEIGHT, WIDTH)])
    ]
    document = frames_meta_document(cameras, [{"wide": (1_000, np.eye(4)), "tall": (1_000, np.eye(4))}], ())
    (tmp_path / FRAMES_META_NAME).write_text(json.dumps(document))
    database_path: Path = tmp_path / "masked.db"
    create_database(database_path, read_frames_meta(tmp_path / FRAMES_META_NAME))
    return database_path


def test_storing_a_masked_batch_drops_masked_keypoints_and_keeps_the_other_descriptors(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt", reason="colsfm.features_native imports tensorrt at module scope")
    pytest.importorskip("cuda.bindings.runtime", reason="colsfm.features_native allocates through cuda-python")
    import pycolmap

    from colsfm.features_native import FeatureBatch, ImageTask, image_tasks, store_feature_batch

    database_path: Path = _two_camera_database(tmp_path)
    tasks: list[ImageTask] = image_tasks(database_path, ["wide/1000", "tall/1000"])
    # Normalised [-1, 1] coordinates; on the 64x48 image x = (k + 1) * 31.5, y = (k + 1) * 23.5.
    keypoints_bn2: Float32[ndarray, "2 4 2"] = np.asarray(
        [[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]], dtype=np.float32
    )
    descriptors_bnd: Float32[ndarray, "2 4 128"] = np.tile(np.arange(4, dtype=np.float32).reshape(1, 4, 1), (2, 1, 128))
    wide_mask: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    wide_mask[20:28, 28:36] = True  # covers the centre keypoint (31.5, 23.5) -> pixel (32, 24)
    wide_mask[47, 0] = True  # covers the bottom-left keypoint (0, 47)

    with pycolmap.Database.open(database_path) as database:
        store_feature_batch(
            database,
            tasks,
            FeatureBatch(keypoints_bn2=keypoints_bn2, descriptors_bnd=descriptors_bnd, scores_bn=np.ones((2, 4), dtype=np.float32)),
            min_score=0.0,
            max_num_features=0,
            masks=DictMasks({"wide/1000": wide_mask}),  # "tall" has no mask: all four stay
        )

    wide = read_keypoints(database_path, tasks[0].image_id)
    np.testing.assert_array_equal(wide, np.asarray([[0.0, 0.0], [63.0, 47.0]], dtype=np.float32))
    stored = read_descriptors_from_database(database_path, [task.image_id for task in tasks])
    np.testing.assert_array_equal(stored[tasks[0].image_id][:, 0], np.asarray([0.0, 2.0], dtype=np.float32))
    assert read_keypoints(database_path, tasks[1].image_id).shape == (4, 2)
    np.testing.assert_array_equal(stored[tasks[1].image_id][:, 0], np.arange(4, dtype=np.float32))


@pytest.mark.integration
def test_catalog_masks_read_a_segmentation_image_per_stamp_from_a_catalog(tmp_path: Path) -> None:
    """Needs a catalog server (an in-process one): the mask layer as the people-mask stage writes it."""
    import rerun as rr

    rrd: Path = tmp_path / "masks.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="seg")
    recording.save(rrd)
    first: np.ndarray = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    first[0, 0] = 1
    second: np.ndarray = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    second[HEIGHT - 1, WIDTH - 1] = 1
    for t_ns, mask in ((1_000, first), (2_000, second)):
        recording.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        recording.log("/world/rig_00/cam_02/pinhole/people_mask", rr.SegmentationImage(mask))
    recording.flush()
    recording.disconnect()

    with rr.server.Server(host="127.0.0.1", port=0, datasets={"d": [str(rrd)]}) as server:
        masks: CatalogMasks = CatalogMasks(
            FrameSourceManifest(
                catalog_url=server.url(),
                dataset="d",
                segment="seg",
                timeline="video_time",
                streams=[
                    StreamKeyframes(
                        entity="world/rig_00/cam_02/pinhole/video",
                        names=["f/1000", "f/2000", "f/3000"],
                        t_ns=[1_000, 2_000, 3_000],
                        mask_entity="world/rig_00/cam_02/pinhole/people_mask",
                    )
                ],
            )
        )
        got_first = masks.mask("f/1000", WIDTH, HEIGHT)
        got_second = masks.mask("f/2000", WIDTH, HEIGHT)
        with pytest.raises(ValueError, match="no mask at t_ns 3000"):
            masks.mask("f/3000", WIDTH, HEIGHT)
        with pytest.raises(ValueError, match="pixels"):
            masks.mask("f/1000", WIDTH + 1, HEIGHT)

    assert got_first is not None and got_second is not None
    assert np.argwhere(got_first).tolist() == [[0, 0]]
    assert np.argwhere(got_second).tolist() == [[HEIGHT - 1, WIDTH - 1]]
