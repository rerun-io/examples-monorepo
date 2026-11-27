"""Keypoint masks: a keypoint on a masked pixel is never stored; the others keep their order and descriptors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pycolmap
import pytest
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray

from colsfm.catalog import CatalogCameraModel, frames_meta_document
from colsfm.database import create_database, read_descriptors_from_database, read_keypoints
from colsfm.frame_manifest import FrameSourceManifest, StreamKeyframes
from colsfm.frames_meta import FRAMES_META_NAME, read_frames_meta
from colsfm.keypoint_masks import (
    VALID_ANGLE_MARGIN_DEG,
    CatalogMasks,
    DictMasks,
    KeypointMask,
    MaskSource,
    drop_masked_keypoints,
    keypoints_outside_mask,
    open_mask_source,
    pack_mask_batches,
    valid_angle,
    valid_angle_mask,
    with_valid_angle_masks,
)

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
    # Normalised [-1, 1] coordinates; on the 64x48 image x = (k + 1) * 32 - 0.5, y = (k + 1) * 24 - 0.5.
    keypoints_bn2: Float32[ndarray, "2 4 2"] = np.asarray(
        [[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]], dtype=np.float32
    )
    descriptors_bnd: Float32[ndarray, "2 4 128"] = np.tile(np.arange(4, dtype=np.float32).reshape(1, 4, 1), (2, 1, 128))
    wide_mask: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    wide_mask[20:28, 28:36] = True  # covers the centre keypoint (31.5, 23.5) -> pixel (32, 24)
    wide_mask[47, 0] = True  # covers the bottom-left keypoint (-0.5, 47.5), clipped into pixel (0, 47)

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
    np.testing.assert_array_equal(wide, np.asarray([[-0.5, -0.5], [63.5, 47.5]], dtype=np.float32))
    stored = read_descriptors_from_database(database_path, [task.image_id for task in tasks])
    np.testing.assert_array_equal(stored[tasks[0].image_id][:, 0], np.asarray([0.0, 2.0], dtype=np.float32))
    assert read_keypoints(database_path, tasks[1].image_id).shape == (4, 2)
    np.testing.assert_array_equal(stored[tasks[1].image_id][:, 0], np.arange(4, dtype=np.float32))


@pytest.mark.integration
def test_a_mask_stream_is_read_window_by_window_and_each_mask_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small windows, a stamp on every window's first nanosecond: no mask is lost between windows or read twice."""
    import rerun as rr
    from rerun.catalog import CatalogClient

    from colsfm import keypoint_masks

    entity: str = "world/rig_00/cam_02/pinhole/people_mask"
    stamps: list[int] = list(range(1_000, 5_000, 100))
    rrd: Path = tmp_path / "masks.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="seg")
    recording.save(rrd)
    for index, t_ns in enumerate(stamps):
        mask: np.ndarray = np.zeros((8, 6), dtype=np.uint8)
        mask.flat[index] = 1
        recording.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        recording.log(f"/{entity}", rr.SegmentationImage(mask))
    recording.flush()
    recording.disconnect()
    monkeypatch.setattr(keypoint_masks, "MASK_READ_WINDOW_NS", 300)

    with rr.server.Server(host="127.0.0.1", port=0, datasets={"d": [str(rrd)]}) as server:
        packed = keypoint_masks.read_packed_masks(CatalogClient(server.url()).get_dataset("d"), "seg", entity, "video_time")

    assert sorted(packed) == stamps
    for index, t_ns in enumerate(stamps):
        width, height, bits = packed[t_ns]
        assert (width, height) == (6, 8)
        assert np.flatnonzero(np.unpackbits(bits, count=48)).tolist() == [index]


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
        with pytest.raises(ValueError, match="does not cover"):
            masks.mask("f/1000", WIDTH + 1, HEIGHT)

    assert got_first is not None and got_second is not None
    assert np.argwhere(got_first).tolist() == [[0, 0]]
    assert np.argwhere(got_second).tolist() == [[HEIGHT - 1, WIDTH - 1]]


@pytest.mark.integration
def test_catalog_masks_read_a_native_resolution_mask_through_its_scale(tmp_path: Path) -> None:
    """The mask layer contract: a mask at the resolution it was computed at, with a static `Transform3D` scale onto its
    image. Each image pixel takes the mask pixel that covers its centre."""
    import rerun as rr

    entity: str = "/world/rig_00/cam_02/pinhole/people_mask"
    rrd: Path = tmp_path / "masks.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="seg")
    recording.save(rrd)
    # A 3x2 (w x h) mask over a 12x10 image: scale 4 x 5; the top-right mask pixel covers columns 8..11, rows 0..4.
    native: np.ndarray = np.zeros((2, 3), dtype=np.uint8)
    native[0, 2] = 1
    recording.log(entity, rr.Transform3D(scale=[4.0, 5.0, 1.0]), static=True)
    recording.set_time("video_time", duration=np.timedelta64(1_000, "ns"))
    recording.log(entity, rr.SegmentationImage(native))
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
                        names=["f/1000"],
                        t_ns=[1_000],
                        mask_entity=entity.lstrip("/"),
                    )
                ],
            )
        )
        got = masks.mask("f/1000", 12, 10)
        with pytest.raises(ValueError, match="does not cover"):
            masks.mask("f/1000", 13, 10)

    expected: np.ndarray = np.zeros((10, 12), dtype=bool)
    expected[0:5, 8:12] = True
    assert got is not None
    np.testing.assert_array_equal(got, expected)


@pytest.mark.integration
def test_catalog_masks_refuse_a_mask_with_more_than_one_byte_per_pixel(tmp_path: Path) -> None:
    """The layer is uint8: a uint16 mask's bytes are not its pixels, so it is refused, not misread."""
    import rerun as rr

    entity: str = "/world/rig_00/cam_02/pinhole/people_mask"
    rrd: Path = tmp_path / "masks.rrd"
    recording = rr.RecordingStream("colsfm-test", recording_id="seg")
    recording.save(rrd)
    recording.set_time("video_time", duration=np.timedelta64(1_000, "ns"))
    recording.log(entity, rr.SegmentationImage(np.array([[0, 1]], dtype=np.uint16)))
    recording.flush()
    recording.disconnect()

    with rr.server.Server(host="127.0.0.1", port=0, datasets={"d": [str(rrd)]}) as server:
        masks: CatalogMasks = CatalogMasks(
            FrameSourceManifest(
                catalog_url=server.url(),
                dataset="d",
                segment="seg",
                timeline="video_time",
                streams=[StreamKeyframes(entity="world/rig_00/cam_02/pinhole/video", names=["f/1000"], t_ns=[1_000], mask_entity=entity.lstrip("/"))],
            )
        )
        with pytest.raises(ValueError, match="not one per pixel"):
            masks.mask("f/1000", 2, 1)


def test_packing_a_mask_table_skips_rows_without_a_mask_and_reads_every_batch() -> None:
    """The loader walks record batches through their offsets; rows where the entity logged no mask are skipped."""
    entity: str = "world/cam/people_mask"
    masks: list[ndarray | None] = [np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8), None, np.array([[2, 0, 0], [0, 0, 3]], dtype=np.uint8)]
    whole: pa.Table = _mask_table(
        entity, [None if m is None else [m.ravel().tolist()] for m in masks], [None if m is None else [{"width": 3, "height": 2}] for m in masks]
    )
    # Two batches, the second a slice, so offsets that do not start at zero are exercised.
    table = pa.Table.from_batches([whole.slice(0, 2).to_batches()[0], whole.slice(2, 1).to_batches()[0]])
    packed = pack_mask_batches(table.to_batches(), "video_time", entity)
    assert sorted(packed) == [10, 30]
    for stamp, mask in ((10, masks[0]), (30, masks[2])):
        width, height, bits = packed[stamp]
        assert (width, height) == (3, 2)
        assert np.array_equal(np.unpackbits(bits, count=6).astype(bool).reshape(2, 3), mask != 0)


def _mask_table(entity: str, buffers: list[list[list[int]] | None], formats: list[list[dict[str, int]] | None]) -> pa.Table:
    format_type = pa.struct([("width", pa.uint32()), ("height", pa.uint32())])
    return pa.table(
        {
            "video_time": pa.array([10 * (index + 1) for index in range(len(buffers))], type=pa.int64()),
            f"/{entity}:SegmentationImage:buffer": pa.array(buffers, type=pa.list_(pa.list_(pa.uint8()))),
            f"/{entity}:SegmentationImage:format": pa.array(formats, type=pa.list_(format_type)),
        }
    )


def test_an_empty_mask_row_is_absent_and_borrows_nothing() -> None:
    """A cleared row (`[]`, valid but empty) has no mask; it must not read the next row's pixels or run past the end."""
    entity: str = "world/cam/people_mask"
    one = [1, 0, 0, 0, 0, 1]
    size = [{"width": 3, "height": 2}]
    packed = pack_mask_batches(_mask_table(entity, [[], [one], []], [[], size, []]).to_batches(), "video_time", entity)
    assert sorted(packed) == [20]
    assert np.array_equal(np.unpackbits(packed[20][2], count=6), np.array(one, dtype=np.uint8))


@pytest.mark.parametrize(
    ("buffers", "formats", "message"),
    [
        ([[[1] * 6, [0] * 6]], [[{"width": 3, "height": 2}] * 2], "2 mask instances"),
        ([[[1] * 6]], [None], "no format"),
        ([[[1] * 6]], [[{"width": 3, "height": 3}]], "holds 6 bytes"),
    ],
    ids=["two-instances", "no-format", "wrong-size"],
)
def test_a_malformed_mask_row_names_its_entity_and_stamp(buffers: list, formats: list, message: str) -> None:
    """One mask per present row, with its format, one byte per pixel; anything else stops the load and says where."""
    entity: str = "world/cam/people_mask"
    with pytest.raises(ValueError, match=rf"/world/cam/people_mask at t_ns 10: .*{message}"):
        pack_mask_batches(_mask_table(entity, buffers, formats).to_batches(), "video_time", entity)


# The valid-angle guard: no keypoint is stored where the camera model's unprojection folds its ray.

TARTANAIR_FOCAL: float = 1000.0 / np.radians(195.0)
"""TartanAir's 195 degree equidistant fisheye, 1000x1000, as colsfm maps it (COLMAP OPENCV_FISHEYE, k = 0)."""


def _fisheye(focal: float, width: int, height: int, k: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> pycolmap.Camera:
    centre_x, centre_y = (width - 1) / 2, (height - 1) / 2
    return pycolmap.Camera(model="OPENCV_FISHEYE", width=width, height=height, params=[focal, focal, centre_x, centre_y, *k])


def _off_axis_degrees(camera: pycolmap.Camera, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, " n"]:
    """The angle of each pixel's ray off the optical axis, from the model's forward projection (exact for k = 0)."""
    normalised: Float64[ndarray, "n 2"] = (pixels - camera.params[2:4]) / camera.params[0:2]
    return np.degrees(np.hypot(normalised[:, 0], normalised[:, 1]))


def test_perspective_cameras_have_no_valid_angle_limit() -> None:
    pinhole = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[100.0, 100.0, 319.5, 239.5])
    # MSD MGO07 cam 2's rational pole sits inside its image; `point_filters.before_rational_pole` owns that guard.
    full_opencv = pycolmap.Camera(
        model="FULL_OPENCV", width=640, height=480, params=[200.0, 200.0, 319.5, 239.5, 0.1, 0.01, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0]
    )

    for camera in (pinhole, full_opencv):
        assert valid_angle(camera) is None
        assert valid_angle_mask(camera) is None


def test_an_equidistant_fisheye_wider_than_180_degrees_is_cut_just_below_its_horizon() -> None:
    camera = _fisheye(TARTANAIR_FOCAL, 1000, 1000)

    limit: float | None = valid_angle(camera)
    mask: KeypointMask | None = valid_angle_mask(camera)

    assert limit is not None and np.degrees(limit) == pytest.approx(90.0 - VALID_ANGLE_MARGIN_DEG)
    assert mask is not None and mask.shape == (1000, 1000)
    rows, columns = np.mgrid[0:1000, 0:1000]
    pixels: Float64[ndarray, "n 2"] = np.stack([columns.ravel(), rows.ravel()], axis=1).astype(np.float64)
    angles: Float64[ndarray, " n"] = _off_axis_degrees(camera, pixels)
    dropped: Bool[ndarray, " n"] = mask.ravel()
    assert not dropped[angles < 88.9].any()
    assert dropped[angles > 89.1].all()
    # COLMAP's unprojection folds every ray past 90 degrees into the front hemisphere; no kept pixel does.
    rays: Float64[ndarray, "n 3"] = np.concatenate([np.asarray(camera.cam_from_img(pixels), dtype=np.float64), np.ones((len(pixels), 1))], axis=1)
    unprojected: Float64[ndarray, " n"] = np.degrees(np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2]))
    folded: Bool[ndarray, " n"] = np.abs(unprojected - angles) > 0.01
    assert folded.sum() > 0.1 * len(pixels)
    assert dropped[folded].all()


def test_a_folding_kannala_brandt_polynomial_is_cut_at_its_fold() -> None:
    # theta_d = theta (1 - 0.3 theta^2) stops growing at theta = 1 / sqrt(0.9) = 60.4 degrees: past it, two rays share a pixel.
    camera = _fisheye(200.0, 640, 480, k=(-0.3, 0.0, 0.0, 0.0))

    limit: float | None = valid_angle(camera)
    mask: KeypointMask | None = valid_angle_mask(camera)

    assert limit is not None and limit == pytest.approx(1.0 / np.sqrt(0.9))
    fold_radius_px: float = 200.0 * limit * (1.0 - 0.3 * limit**2)
    assert mask is not None
    assert not mask[240, int(319.5 + fold_radius_px - 1)]
    assert mask[240, int(319.5 + fold_radius_px + 2)]


def test_a_fisheye_whose_image_stays_inside_its_valid_angle_gets_no_mask() -> None:
    # 640x480 at f = 400: the corner is 1.0 rad (57 degrees) off the axis.
    assert valid_angle_mask(_fisheye(400.0, 640, 480)) is None


def _fisheye_and_pinhole_database(tmp_path: Path) -> Path:
    """A database with a TartanAir-like 195 degree fisheye ("fish", 100x100 at a tenth of the scale) and a pinhole ("cube")."""
    focal: float = TARTANAIR_FOCAL / 10
    cameras: list[CatalogCameraModel] = [
        CatalogCameraModel(
            entity="/world/rig_00/cam_00",
            name="fish",
            rig_T_cam=np.eye(4),
            k_matrix=np.asarray([[focal, 0.0, 49.5], [0.0, focal, 49.5], [0.0, 0.0, 1.0]]),
            width=100,
            height=100,
            projection="OPENCV_FISHEYE",
            distortion=(0.0, 0.0, 0.0, 0.0),
        ),
        CatalogCameraModel(
            entity="/world/rig_00/cam_01",
            name="cube",
            rig_T_cam=np.eye(4),
            k_matrix=np.asarray([[32.0, 0.0, 31.5], [0.0, 32.0, 23.5], [0.0, 0.0, 1.0]]),
            width=WIDTH,
            height=HEIGHT,
            projection="PINHOLE",
            distortion=(),
        ),
    ]
    document = frames_meta_document(cameras, [{"fish": (1_000, np.eye(4)), "cube": (1_000, np.eye(4))}], ())
    (tmp_path / FRAMES_META_NAME).write_text(json.dumps(document))
    database_path: Path = tmp_path / "fisheye.db"
    create_database(database_path, read_frames_meta(tmp_path / FRAMES_META_NAME))
    return database_path


THREE_KEYPOINTS: Float32[ndarray, "3 6"] = np.zeros((3, 6), dtype=np.float32)
THREE_KEYPOINTS[:, :2] = [[49.5, 49.5], [0.0, 0.0], [59.5, 49.5]]
"""Centre, corner and a point 10 px right of the centre (Nx6, as the `pycolmap` backend writes them)."""
THREE_DESCRIPTORS: UInt8[ndarray, "3 8"] = np.repeat(np.arange(3, dtype=np.uint8)[:, None], 8, axis=1)
"""One distinct uint8 descriptor row per keypoint, so a test sees which rows are kept."""


def _write_three_keypoints(database_path: Path) -> dict[str, int]:
    """`THREE_KEYPOINTS` and `THREE_DESCRIPTORS` on every image; returns image ids by name."""
    with pycolmap.Database.open(database_path) as database:
        ids: dict[str, int] = {image.name: int(image.image_id) for image in database.read_all_images()}
        for image_id in ids.values():
            database.write_keypoints(image_id, THREE_KEYPOINTS)
            database.write_descriptors(image_id, pycolmap.FeatureDescriptors(data=THREE_DESCRIPTORS, type=pycolmap.FeatureExtractorType.ALIKED_N16ROT))
    return ids


def test_a_database_without_a_wide_camera_keeps_its_mask_source_unchanged(tmp_path: Path) -> None:
    database_path: Path = _two_camera_database(tmp_path)
    user_masks: DictMasks = DictMasks({"wide/1000": np.ones((HEIGHT, WIDTH), dtype=bool)})

    assert with_valid_angle_masks(database_path, None) is None
    assert with_valid_angle_masks(database_path, user_masks) is user_masks


def test_the_valid_angle_mask_joins_a_user_mask_on_the_cameras_it_covers(tmp_path: Path) -> None:
    database_path: Path = _fisheye_and_pinhole_database(tmp_path)
    people: Bool[ndarray, "100 100"] = np.zeros((100, 100), dtype=bool)
    people[50, 50] = True

    guarded: MaskSource | None = with_valid_angle_masks(database_path, DictMasks({"fish/1000": people}))
    unmasked: MaskSource | None = with_valid_angle_masks(database_path, None)

    assert guarded is not None and unmasked is not None
    fish: KeypointMask | None = guarded.mask("fish/1000", 100, 100)
    assert fish is not None and fish[50, 50] and fish[0, 0] and not fish[50, 60]
    only_angle: KeypointMask | None = unmasked.mask("fish/1000", 100, 100)
    assert only_angle is not None and only_angle[0, 0] and not only_angle[50, 50]
    assert guarded.mask("cube/1000", WIDTH, HEIGHT) is None


def test_dropping_masked_keypoints_from_a_database_keeps_the_rest_in_order(tmp_path: Path) -> None:
    """The `pycolmap` backend writes its own keypoints (Nx6, uint8 descriptor blobs): the guard runs after it."""
    database_path: Path = _fisheye_and_pinhole_database(tmp_path)
    ids: dict[str, int] = _write_three_keypoints(database_path)

    guarded: MaskSource | None = with_valid_angle_masks(database_path, None)
    assert guarded is not None
    drop_masked_keypoints(database_path, guarded)

    with pycolmap.Database.open(database_path) as database:
        fish_keypoints: ndarray = database.read_keypoints(ids["fish/1000"])
        fish_descriptors = database.read_descriptors(ids["fish/1000"])
        cube_keypoints: ndarray = database.read_keypoints(ids["cube/1000"])
    np.testing.assert_array_equal(fish_keypoints, THREE_KEYPOINTS[[0, 2]])
    np.testing.assert_array_equal(fish_descriptors.data, THREE_DESCRIPTORS[[0, 2]])
    assert fish_descriptors.type == pycolmap.FeatureExtractorType.ALIKED_N16ROT
    np.testing.assert_array_equal(cube_keypoints, THREE_KEYPOINTS)


def test_a_content_mask_drops_pycolmap_keypoints_by_the_pixel_they_fall_in(tmp_path: Path) -> None:
    """COLMAP keypoints put pixel centres at +0.5: pixel c covers [c, c + 1), so 29.7 is in pixel 29 and 39.7 in 39."""
    database_path: Path = _two_camera_database(tmp_path)
    people: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    people[10:20, 30:40] = True
    keypoints: Float32[ndarray, "4 6"] = np.zeros((4, 6), dtype=np.float32)
    keypoints[:, :2] = [[29.7, 15.0], [30.3, 15.0], [39.7, 19.7], [40.3, 15.0]]
    descriptors: UInt8[ndarray, "4 8"] = np.repeat(np.arange(4, dtype=np.uint8)[:, None], 8, axis=1)
    with pycolmap.Database.open(database_path) as database:
        image_id: int = next(int(image.image_id) for image in database.read_all_images() if image.name == "wide/1000")
        database.write_keypoints(image_id, keypoints)
        database.write_descriptors(image_id, pycolmap.FeatureDescriptors(data=descriptors, type=pycolmap.FeatureExtractorType.ALIKED_N16ROT))

    drop_masked_keypoints(database_path, DictMasks({"wide/1000": people}))

    with pycolmap.Database.open(database_path) as database:
        np.testing.assert_array_equal(database.read_keypoints(image_id), keypoints[[0, 3]])
        np.testing.assert_array_equal(database.read_descriptors(image_id).data, descriptors[[0, 3]])


def test_the_valid_angle_mask_reads_pycolmap_keypoints_at_the_nearest_pixel_centre(tmp_path: Path) -> None:
    """`valid_angle_mask` samples the camera at integer COLMAP coordinates, so its lookup rounds, unlike a content mask's."""
    database_path: Path = _fisheye_and_pinhole_database(tmp_path)
    guarded: MaskSource | None = with_valid_angle_masks(database_path, DictMasks({"fish/1000": np.zeros((100, 100), dtype=bool)}))
    assert guarded is not None
    angle: KeypointMask | None = guarded.mask("fish/1000", 100, 100)
    assert angle is not None
    # The last dropped column left of the centre on the middle row: rounding 0.7 past it reads the kept column next to it.
    edge: int = int(np.flatnonzero(angle[50, :50])[-1])
    keypoints: Float32[ndarray, "1 6"] = np.zeros((1, 6), dtype=np.float32)
    keypoints[0, :2] = [edge + 0.7, 50.0]
    with pycolmap.Database.open(database_path) as database:
        image_id: int = next(int(image.image_id) for image in database.read_all_images() if image.name == "fish/1000")
        database.write_keypoints(image_id, keypoints)
        database.write_descriptors(image_id, pycolmap.FeatureDescriptors(data=np.zeros((1, 8), dtype=np.uint8), type=pycolmap.FeatureExtractorType.ALIKED_N16ROT))

    drop_masked_keypoints(database_path, guarded)

    assert read_keypoints(database_path, image_id).tolist() == keypoints[:, :2].tolist()


def test_pycolmap_extraction_applies_the_valid_angle_guard_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from colsfm import features

    database_path: Path = _fisheye_and_pinhole_database(tmp_path)
    written: dict[str, int] = {}
    monkeypatch.setattr(features.pycolmap, "extract_features", lambda *args, **kwargs: written.update(_write_three_keypoints(database_path)))

    features.extract_features(database_path, tmp_path, ["fish/1000", "cube/1000"], features.FeatureOptions(device="cpu"))

    assert read_keypoints(database_path, written["fish/1000"]).tolist() == [[49.5, 49.5], [59.5, 49.5]]
    assert len(read_keypoints(database_path, written["cube/1000"])) == 3


def test_raco_extraction_receives_the_valid_angle_masks_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tensorrt", reason="colsfm.features_raco imports tensorrt at module scope")
    from colsfm import features, features_raco, frames

    seen: list[MaskSource | None] = []

    def fake_extract_raco(*args: object, masks: MaskSource | None = None, **kwargs: object) -> tuple[dict[int, int], float]:
        seen.append(masks)
        return {}, 0.0

    monkeypatch.setattr(features_raco, "extract_raco", fake_extract_raco)
    for database_path in (_fisheye_and_pinhole_database(tmp_path), _two_camera_database(tmp_path)):
        features.extract_features(database_path, tmp_path, [], features.FeatureOptions(backend="raco"), frames=frames.ImageFolderFrames(tmp_path))

    assert seen[1] is None
    first: MaskSource | None = seen[0]
    assert first is not None and first.mask("cube/1000", WIDTH, HEIGHT) is None
    fish: KeypointMask | None = first.mask("fish/1000", 100, 100)
    assert fish is not None and fish[0, 0] and not fish[49, 49]


def test_pycolmap_extraction_applies_the_input_masks_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Masks the input names (a catalog's people masks) drop keypoints after the `pycolmap` backend, as after `raco`."""
    from colsfm import features

    database_path: Path = _fisheye_and_pinhole_database(tmp_path)
    corner: Bool[ndarray, "48 64"] = np.zeros((HEIGHT, WIDTH), dtype=bool)
    corner[0, 0] = True
    monkeypatch.setattr(features, "open_mask_source", lambda manifest: DictMasks({"cube/1000": corner}))
    written: dict[str, int] = {}
    monkeypatch.setattr(features.pycolmap, "extract_features", lambda *args, **kwargs: written.update(_write_three_keypoints(database_path)))

    features.extract_features(database_path, tmp_path, ["fish/1000", "cube/1000"], features.FeatureOptions(device="cpu"))

    assert read_keypoints(database_path, written["cube/1000"]).tolist() == [[49.5, 49.5], [59.5, 49.5]]
    assert read_keypoints(database_path, written["fish/1000"]).tolist() == [[49.5, 49.5], [59.5, 49.5]]
