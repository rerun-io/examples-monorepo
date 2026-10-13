"""exo-calib Types: serde round-trips and the Rerun mapping structure (unit lane: no catalog)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict

from exo_calib.types import Keypoints3dTracks, Kp2dOverlay, MultiViewKeypoints, RefinementDiagnostics, RigAlignment, RigLayout

ROOT = "/world/exocalib/kp2d"


@st.composite
def records(draw: st.DrawFn) -> MultiViewKeypoints:
    v: int = draw(st.integers(1, 4))
    floats = st.floats(-1e4, 1e4, width=32) | st.just(float("nan"))
    return MultiViewKeypoints(
        frame_index=draw(st.integers(0, 10**6)),
        views=tuple(f"rig_{i:02d}/cam_00" for i in range(v)),
        cam_t_ns=draw(arrays(np.int64, (v,), elements=st.integers(0, 2**40))),
        kp_xy=draw(arrays(np.float32, (v, 133, 2), elements=floats)),
        conf=draw(arrays(np.float32, (v, 133), elements=st.floats(0.0, 1.0, width=32))),
        bbox_xyxy=draw(arrays(np.float32, (v, 4), elements=floats)),
        box_source=tuple(draw(st.sampled_from(["yolox", "tracked", "none"])) for _ in range(v)),
        crop_origin_xy=draw(arrays(np.float32, (v, 2), elements=floats)),
        crop_size_wh=draw(arrays(np.float32, (v, 2), elements=floats)),
        crop_input_wh=(192, 256),
        video_wh=(1280, 720),
    )


@settings(max_examples=40, deadline=None)
@given(record=records())
def test_keypoints_serde_roundtrip_keeps_dtype_and_shape(record: MultiViewKeypoints) -> None:
    back: MultiViewKeypoints = from_dict(MultiViewKeypoints, to_dict(record))
    for name in ("cam_t_ns", "kp_xy", "conf", "bbox_xyxy", "crop_origin_xy", "crop_size_wh"):
        np.testing.assert_array_equal(getattr(back, name), getattr(record, name))
        assert getattr(back, name).dtype == getattr(record, name).dtype
    assert (back.views, back.box_source, back.frame_index, back.crop_input_wh) == (record.views, record.box_source, record.frame_index, record.crop_input_wh)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(SerdeError):
        from_dict(RigLayout, {"exo_camera_names": [], "ego_camera_names": [], "calibrated_camera_names": [], "extra": 1})
    diag = RefinementDiagnostics(1.5, [1.0, 0.9], [10, 12], 1.02, [1.0, 0.99])
    assert from_dict(RefinementDiagnostics, to_dict(diag)) == diag


@settings(max_examples=40, deadline=None)
@given(record=records())
def test_keypoints_to_rerun_writes_the_record_under_the_root_and_boxes_under_each_pinhole(record: MultiViewKeypoints) -> None:
    logs = record.to_rerun(ROOT)
    expected: list[str] = []
    for view in record.views:
        expected += [f"{ROOT}/{view}", f"/world/{view}/pinhole/exocalib_kp2d_bbox", f"/world/{view}/pinhole/exocalib_kp2d_crop"]
    assert [log.path for log in logs] == expected and not any(log.static for log in logs)
    assert MultiViewKeypoints.contents(ROOT) == [f"{ROOT}/**"]


@settings(max_examples=40, deadline=None)
@given(xyz=arrays(np.float64, (2, 5, 3), elements=st.floats(-5.0, 5.0) | st.just(float("nan"))))
def test_kp3d_logs_each_rig_variant_that_has_a_finite_point(xyz: np.ndarray) -> None:
    tracks = Keypoints3dTracks(np.arange(5, dtype=np.int64), xyz, np.ones((2, 5)))
    paths: list[str] = [log.path for log in tracks.to_rerun("/world/exocalib/kp3d")]
    expected = [f"/world/exocalib/kp3d_{v}" for k, v in enumerate(("init", "refined")) if np.isfinite(xyz[k]).all(axis=1).any()]
    assert paths == expected


def test_overlay_skips_views_without_keypoints_and_alignment_writes_both_rigs() -> None:
    raw = np.full((2, 133, 2), np.nan, dtype=np.float32)
    raw[1, :3] = 10.0
    overlay = Kp2dOverlay(("a/cam_00", "b/cam_00"), raw.astype(np.float64), np.ones((2, 133)), raw, np.ones((2, 133), np.float32))
    assert [log.path for log in overlay.to_rerun(ROOT)] == ["/world/b/cam_00/pinhole/exocalib_kp2d", "/world/b/cam_00/pinhole/exocalib_kp2d_rejected"]
    align = RigAlignment(("a",), np.zeros((1, 3)), np.repeat(np.eye(3)[None], 2, 0), np.zeros((2, 3)), np.zeros((2, 1, 3)))
    logs = align.to_rerun("/world")
    assert {log.path for log in logs} == {f"/world/{p}" for p in ("exocalib/init", "exocalib/kp3d_init", "exocalib_error/init", "exocalib/refined", "exocalib/kp3d_refined", "exocalib_error/refined")}
    assert all(log.static for log in logs)
    assert RefinementDiagnostics(1.0, [], [], 1.0, []).to_rerun("/world/exocalib/refined")[0].static
