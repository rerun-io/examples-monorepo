"""Mask-layer union: pooling onto the coarsest grid, shared stamps, and the written scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_layer(path: Path, masks: dict[int, np.ndarray], scale: float | None) -> None:
    import rerun as rr

    recording: rr.RecordingStream = rr.RecordingStream("dataforge", recording_id="clip")
    recording.save(path)
    entity: str = "/world/rig_00/cam_00/pinhole/people_mask"
    if scale is not None:
        recording.log(entity, rr.Transform3D(scale=[scale, scale, 1.0]), static=True)
    for stamp, mask in masks.items():
        recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
        recording.log(entity, rr.SegmentationImage(mask.astype(np.uint8)))
    recording.flush()
    recording.disconnect()


def test_pool_any_marks_a_cell_when_any_pixel_is_set() -> None:
    from posekit.apis.mask_union import pool_any

    mask: np.ndarray = np.zeros((4, 6), dtype=bool)
    mask[1, 4] = True
    np.testing.assert_array_equal(pool_any(mask, 2), [[False, False, True], [False, False, False]])
    np.testing.assert_array_equal(pool_any(np.stack([mask, ~mask]), 2), [pool_any(mask, 2), pool_any(~mask, 2)])
    with pytest.raises(ValueError, match="does not pool"):
        pool_any(mask, 4)


def test_union_on_the_coarsest_grid_at_shared_stamps(tmp_path: Path) -> None:
    import rerun.chunk as rrc
    from simplecv.rerun_log_utils import RerunTyroConfig

    from posekit.apis.mask_union import MaskUnionConfig, layer_masks, layer_statics, main

    fine: dict[int, np.ndarray] = {stamp: np.zeros((8, 8), dtype=bool) for stamp in (0, 10, 20)}
    fine[20][7, 7] = True  # one image pixel -> the coarse cell (3, 3)
    coarse: dict[int, np.ndarray] = {stamp: np.zeros((4, 4), dtype=bool) for stamp in (0, 20)}
    coarse[0][0, 1] = True
    _write_layer(tmp_path / "fine.rrd", fine, None)
    _write_layer(tmp_path / "coarse.rrd", coarse, 2.0)
    out: Path = tmp_path / "union.rrd"
    main(
        MaskUnionConfig(
            layers=(tmp_path / "fine.rrd", tmp_path / "coarse.rrd"),
            rr_config=RerunTyroConfig(save=out, headless=True, recording_id="clip", application_id="dataforge"),
        )
    )
    import rerun as rr

    recording = rr.get_data_recording()
    assert recording is not None
    recording.disconnect()
    store = rrc.RrdReader(out).store()
    entity: str = "/world/rig_00/cam_00/pinhole/people_mask"
    np.testing.assert_array_equal(layer_statics(store)[entity], [2.0, 2.0])
    union: dict[int, np.ndarray] = dict(layer_masks(store, entity, "video_time", np.array([2.0, 2.0]), np.array([2.0, 2.0])))
    assert sorted(union) == [0, 20]
    np.testing.assert_array_equal(np.argwhere(union[0]), [[0, 1]])
    np.testing.assert_array_equal(np.argwhere(union[20]), [[3, 3]])
