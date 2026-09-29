"""Label parity with what the catalog ships: our FK + lens models against the stored landmarks and projections."""

import os

import numpy as np
import pyarrow as pa
import pytest
from beartype.roar import BeartypeException

rr = pytest.importorskip("rerun", reason="needs rerun-sdk with the catalog extra")

from handtrack.data.catalog import (  # noqa: E402
    CATALOG_URL,
    HOT3D_QUEST3,
    SHOW3D,
    TIMELINE,
    UMETRACK,
    HandTimeline,
    SegmentInfo,
    layout_for,
    list_segments,
    point_rows,
    read_hand_timeline,
    read_rig,
    read_statics,
)
from handtrack.data.segment_labels import SegmentLabels, segment_labels  # noqa: E402

pytestmark = pytest.mark.integration

COCO_SLOTS: dict[int, int] = {91: 5, 93: 6, 94: 7, 95: 0, 96: 8, 97: 9, 98: 10, 99: 1, 100: 11, 101: 12, 102: 13, 103: 2, 104: 14, 105: 15, 106: 16, 107: 3, 108: 17, 109: 18, 110: 19, 111: 4}
"""COCO-133 left-hand slot -> our landmark (right hand: slot + 21). Slot 92 is the wrist/thumb-CMC midpoint; the palm centre (20) has none."""
UMETRACK_SEGMENTS: tuple[str, ...] = ("umetrack__real__hand_hand__training__user_00__recording_01", "umetrack__synthetic__separate_hand__training__user_38__recording_12")
SHOW3D_SEGMENT: str = "show3d__ERI327__milk_shake_7a09"
HOT3D_SEGMENT: str = "hot3d-quest3__P0002_c7164ba4"
HOT3D_CATALOG_URL: str = os.environ.get("HANDTRACK_HOT3D_CATALOG_URL", CATALOG_URL)
"""HOT3D is not on the shared catalog; point this at a server that registers ``dataforge-hot3d-quest3``."""


def _dataset(name: str, url: str = CATALOG_URL):
    try:
        return rr.catalog.CatalogClient(url).get_dataset(name)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {url} dataset {name} unreachable: {error}")


def _info(dataset, name: str, segment: str) -> SegmentInfo:
    infos: dict[str, SegmentInfo] = {info.segment_id: info for info in list_segments(dataset, name)}
    if segment not in infos:
        pytest.skip(f"{segment} is not registered in {name}")
    return infos[segment]


def _labels(dataset, info: SegmentInfo) -> tuple[HandTimeline, SegmentLabels]:
    statics: pa.Table = read_statics(dataset, info)
    rig, letterboxes = read_rig(statics, info)
    timeline: HandTimeline = read_hand_timeline(dataset, info, statics)
    rows: np.ndarray = np.arange(len(timeline.video_time_ns), dtype=np.int64)
    return timeline, segment_labels(timeline, rig, letterboxes, rows, layout_for(info.dataset).pose_gated)


def _slots() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(COCO slot, hand, landmark) for the 2 x 20 slots that are landmarks."""
    slots: list[tuple[int, int, int]] = [(slot + 21 * hand, hand, landmark) for hand in (0, 1) for slot, landmark in COCO_SLOTS.items()]
    return np.array([s[0] for s in slots]), np.array([s[1] for s in slots]), np.array([s[2] for s in slots])


@pytest.mark.parametrize(("name", "segment"), [*((UMETRACK, segment) for segment in UMETRACK_SEGMENTS), (HOT3D_QUEST3, HOT3D_SEGMENT)])
def test_fisheye_reprojection_matches_coco133_uv_projected(name: str, segment: str) -> None:
    """Our Fisheye62 against dataforge's projections. HOT3D ships FISHEYE624 with zero thin-prism terms on Quest 3, so dropping them is exact."""
    dataset = _dataset(name, HOT3D_CATALOG_URL if name == HOT3D_QUEST3 else CATALOG_URL)
    info: SegmentInfo = _info(dataset, name, segment)
    timeline, labels = _labels(dataset, info)
    cameras: tuple[str, ...] = layout_for(name).cameras
    columns: list[str] = [f"{camera}/pinhole/coco133_uv_projected:Points2D:positions" for camera in cameras]
    table: pa.Table = (
        dataset.filter_segments(segment)
        .filter_contents([f"{camera}/pinhole/coco133_uv_projected" for camera in cameras])
        .reader(index=TIMELINE)
        .select(TIMELINE, *columns)
        .to_arrow_table()
    )
    np.testing.assert_array_equal(np.asarray(table[TIMELINE].combine_chunks().to_numpy(zero_copy_only=False)).view(np.int64), timeline.video_time_ns)
    slot, hand, landmark = _slots()
    errors: list[np.ndarray] = []
    for camera, column in enumerate(columns):
        stored: np.ndarray = point_rows(table[column], 133, 2)[:, slot]
        ours: np.ndarray = labels.projection.pixels[:, camera].numpy()[:, hand, landmark]
        finite: np.ndarray = np.isfinite(stored).all(-1) & np.isfinite(ours).all(-1)
        errors.append(np.linalg.norm(stored[finite] - ours[finite], axis=-1))
    error: np.ndarray = np.concatenate(errors)
    assert error.size > 1000, "too few stored projections to compare"
    assert np.median(error) < 1e-3 and np.percentile(error, 99) < 1e-2, (np.median(error), np.percentile(error, 99), error.max())


def test_show3d_skinned_landmarks_match_coco133_xyz() -> None:
    dataset = _dataset(SHOW3D)
    info: SegmentInfo = _info(dataset, SHOW3D, SHOW3D_SEGMENT)
    timeline, labels = _labels(dataset, info)
    column: str = "/world/gt/coco133_xyz:Points3D:positions"
    table: pa.Table = dataset.filter_segments(SHOW3D_SEGMENT).filter_contents(["/world/gt/coco133_xyz"]).reader(index=TIMELINE).select(TIMELINE, column).to_arrow_table()
    times: np.ndarray = np.asarray(table[TIMELINE].combine_chunks().to_numpy(zero_copy_only=False)).view(np.int64)
    rows: np.ndarray = np.searchsorted(timeline.video_time_ns, times)
    assert (timeline.video_time_ns[rows] == times).all()
    stored: np.ndarray = point_rows(table[column], 133, 3)
    slot, hand, landmark = _slots()
    ours: np.ndarray = labels.landmarks.numpy()[rows]
    shipped: np.ndarray = stored[:, slot]
    skinned: np.ndarray = ours[:, hand, landmark]
    finite: np.ndarray = np.isfinite(shipped).all(-1)
    assert finite.sum() > 10_000 and np.isfinite(skinned[finite]).all()
    difference_mm: np.ndarray = np.linalg.norm(shipped[finite] - skinned[finite], axis=-1) * 1000.0
    assert difference_mm.max() < 0.01, difference_mm.max()
    # Slot 92 (113 right) is the wrist / thumb-CMC midpoint, not a landmark.
    midpoint: np.ndarray = (ours[:, :, 5] + ours[:, :, 6]) * 0.5
    for side, thumb_base in ((0, 92), (1, 113)):
        present: np.ndarray = np.isfinite(stored[:, thumb_base]).all(-1)
        np.testing.assert_allclose(stored[present, thumb_base], midpoint[present, side], atol=1e-5)
