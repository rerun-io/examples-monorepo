"""The mask layer reader: latest mask at or before a frame, read through the layer's scale."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry

# An in-process catalog server reads the layer.
pytestmark = pytest.mark.integration

SEGMENT: str = "masks-test"
CAM0: str = "/world/rig_00/cam_00/pinhole/people_mask"
CAM1: str = "/world/rig_00/cam_01/pinhole/people_mask"


def _write_layer(path: Path, masks: dict[str, list[tuple[int, UInt8[ndarray, "h w"]]]], scales: dict[str, list[float]]) -> None:
    """One mask layer: per entity, (video_time ns, mask) rows and an optional static scale."""
    rec = rr.RecordingStream("dataforge", recording_id=SEGMENT)
    rec.save(path)
    for entity, scale in scales.items():
        rec.log(entity, rr.Transform3D(scale=scale), static=True)
    for entity, rows in masks.items():
        for t_ns, mask in rows:
            rec.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
            rec.log(entity, rr.SegmentationImage(mask))
    rec.flush()


@contextmanager
def _dataset(path: Path) -> Iterator[DatasetEntry]:
    with rr.server.Server(datasets={"masks": [str(path)]}) as server:
        yield server.client().get_dataset("masks")


def _mask(height: int, width: int, value: int) -> UInt8[ndarray, "h w"]:
    return np.full((height, width), value, dtype=np.uint8)


def test_latest_mask_at_or_before_each_frame(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks, MaskCursor

    path: Path = tmp_path / "masks.rrd"
    _write_layer(path, {CAM0: [(100, _mask(8, 8, 1)), (300, _mask(8, 8, 2))], CAM1: [(200, _mask(8, 8, 3))]}, {})
    with _dataset(path) as dataset:
        masks = MaskCursor(FrameMasks(dataset, SEGMENT, [CAM0, CAM1], [(8, 8), (8, 8)]))
        before = masks.at([50, 50])
        assert before[0] is None and before[1] is None
        at_100 = masks.at([100, 100])
        assert at_100[0] is not None and int(at_100[0][0, 0]) == 1 and at_100[1] is None
        at_250 = masks.at([250, 250])
        assert at_250[0] is not None and at_250[1] is not None
        assert int(at_250[0][0, 0]) == 1 and int(at_250[1][0, 0]) == 3
        at_300 = masks.at([300, 300])
        assert at_300[0] is not None and int(at_300[0][0, 0]) == 2


def test_a_window_holds_the_mask_from_before_it(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks, MaskCursor

    path: Path = tmp_path / "masks.rrd"
    _write_layer(path, {CAM0: [(100, _mask(8, 8, 1)), (900, _mask(8, 8, 2))]}, {})
    with _dataset(path) as dataset:
        masks = MaskCursor(FrameMasks(dataset, SEGMENT, [CAM0], [(8, 8)]), window_ns=200)
        held = masks.at([600])[0]
        assert held is not None and int(held[0, 0]) == 1


def test_native_resolution_masks_carry_their_scale(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks, MaskCursor

    path: Path = tmp_path / "masks.rrd"
    # A 4x5 mask (w x h) over a 20x30 image: scale 5 and 6.
    _write_layer(path, {CAM0: [(100, _mask(5, 4, 1))]}, {CAM0: [5.0, 6.0, 1.0]})
    with _dataset(path) as dataset:
        masks = MaskCursor(FrameMasks(dataset, SEGMENT, [CAM0], [(20, 30)]))
        held = masks.at([100])[0]
        assert held is not None and held.shape == (5, 4)


def test_a_cursor_holds_a_short_window_and_reads_on_as_frames_leave_it(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks, MaskCursor

    path: Path = tmp_path / "masks.rrd"
    # 100 masks, one every 100 ns, each filled with its own index.
    _write_layer(path, {CAM0: [(100 * index, _mask(8, 8, index)) for index in range(100)]}, {})
    with _dataset(path) as dataset:
        masks = MaskCursor(FrameMasks(dataset, SEGMENT, [CAM0], [(8, 8)]), window_ns=1_000)
        for t_ns in range(0, 10_000, 50):
            held = masks.at([t_ns])[0]
            assert held is not None and int(held[0, 0]) == t_ns // 100
            assert len(masks.window_masks[0]) <= 12, (t_ns, len(masks.window_masks[0]))


def test_two_cursors_over_one_layer_keep_their_own_masks(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks, MaskCursor

    path: Path = tmp_path / "masks.rrd"
    _write_layer(path, {CAM0: [(100, _mask(8, 8, 1)), (900, _mask(8, 8, 2))]}, {})
    with _dataset(path) as dataset:
        layer = FrameMasks(dataset, SEGMENT, [CAM0], [(8, 8)])
        first, second = MaskCursor(layer, window_ns=200), MaskCursor(layer, window_ns=200)
        held_first = first.at([150])[0]
        held_second = second.at([950])[0]
        held_again = first.at([160])[0]
        assert held_first is not None and int(held_first[0, 0]) == 1
        assert held_second is not None and int(held_second[0, 0]) == 2
        assert held_again is not None and int(held_again[0, 0]) == 1


def test_the_stamps_are_read_window_by_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog sends whole chunks, pixels included, even for a stamps-only select, so the stamps come in time windows."""
    from slam_rs import frame_masks
    from slam_rs.frame_masks import FrameMasks

    monkeypatch.setattr(frame_masks, "STAMP_WINDOW_NS", 250)
    path: Path = tmp_path / "masks.rrd"
    stamps: list[int] = [100 * index + 7 for index in range(40)]
    _write_layer(path, {CAM0: [(t_ns, _mask(8, 8, 1)) for t_ns in stamps]}, {})
    with _dataset(path) as dataset:
        assert FrameMasks(dataset, SEGMENT, [CAM0], [(8, 8)]).stamps[0].tolist() == stamps


def test_a_mask_that_does_not_scale_onto_its_image_is_refused(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks

    path: Path = tmp_path / "masks.rrd"
    # No transform = scale 1, so a 4x4 mask must cover a 4x4 image, not 20x20.
    _write_layer(path, {CAM0: [(100, _mask(4, 4, 1))]}, {})
    with _dataset(path) as dataset, pytest.raises(ValueError, match="does not cover"):
        FrameMasks(dataset, SEGMENT, [CAM0], [(20, 20)])


def test_a_camera_without_masks_is_refused(tmp_path: Path) -> None:
    from slam_rs.frame_masks import FrameMasks

    path: Path = tmp_path / "masks.rrd"
    _write_layer(path, {CAM0: [(100, _mask(8, 8, 1))]}, {})
    with _dataset(path) as dataset, pytest.raises(ValueError, match="no SegmentationImage"):
        FrameMasks(dataset, SEGMENT, [CAM0, CAM1], [(8, 8), (8, 8)])
