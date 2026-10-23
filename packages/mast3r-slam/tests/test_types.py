"""Types tier (decision 18): serde round-trips, unknown fields refused, Rerun mapping structure."""

import numpy as np
import pytest
import rerun as rr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict
from serde.toml import from_toml, to_toml

from mast3r_slam.types import CameraView, FrameView, KeyframeView, LogLine, MapUpdate, SlamConfig, TrackedFrame, TrackingConfig

FINITE = st.floats(-100.0, 100.0, allow_nan=False, width=32)


def sim3s(n: int | None = None) -> st.SearchStrategy[np.ndarray]:
    """lietorch Sim3 rows [t, q_xyzw, s] (unit quaternion, positive scale)."""

    def row(t: np.ndarray, q: np.ndarray, s: float) -> np.ndarray:
        return np.concatenate([t, q / np.linalg.norm(q), [s]]).astype(np.float32)

    one = st.builds(row, arrays(np.float32, 3, elements=FINITE), arrays(np.float32, 4, elements=st.floats(0.125, 1.0, width=32)), st.floats(0.01, 10.0))
    return one if n is None else st.lists(one, min_size=n, max_size=n).map(np.stack)


@given(pose=sim3s(), frame_index=st.integers(0, 10**6))
@settings(max_examples=30, deadline=None)
def test_tracked_frame_serde_round_trip_keeps_dtype(pose: np.ndarray, frame_index: int) -> None:
    value = TrackedFrame(frame_index, pose)
    back = from_dict(TrackedFrame, to_dict(value))
    assert back.frame_index == frame_index
    assert back.world_sim3_cam.dtype == np.float32
    np.testing.assert_array_equal(back.world_sim3_cam, pose)


@given(k=st.integers(1, 6), data=st.data())
@settings(max_examples=30, deadline=None)
def test_map_update_serde_round_trip_and_rerun_structure(k: int, data: st.DataObject) -> None:
    poses: np.ndarray = data.draw(sim3s(k))
    edges: np.ndarray = data.draw(arrays(np.int64, (2, data.draw(st.integers(0, 8))), elements=st.integers(0, k - 1)))
    value = MapUpdate(poses, edges[0], edges[1])
    back = from_dict(MapUpdate, to_dict(value))
    np.testing.assert_array_equal(back.world_sim3_cam, poses)
    np.testing.assert_array_equal(np.stack([back.ii, back.jj]), edges)
    assert back.ii.dtype == back.jj.dtype == np.int64
    logs = value.to_rerun("/mast3r_slam/map")
    assert {log.path for log in logs} == set(MapUpdate.contents("/mast3r_slam/map")[: 2 if len(edges[0]) else 1])
    assert not any(log.static for log in logs)
    points = next(log.data for log in logs if isinstance(log.data, rr.Points3D))
    assert points.positions is not None and len(points.positions) == k


def test_tracked_frame_to_rerun_logs_the_rigid_pose_at_the_root() -> None:
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32)
    logs = TrackedFrame(7, pose).to_rerun("/mast3r_slam/cam_00")
    assert [log.path for log in logs] == ["/mast3r_slam/cam_00"] * 2
    transform = next(log.data for log in logs if isinstance(log.data, rr.Transform3D))
    assert transform.translation is not None
    np.testing.assert_allclose(transform.translation.as_arrow_array().to_pylist()[0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize("value_type", [TrackedFrame, MapUpdate])
def test_unknown_fields_are_refused(value_type: type) -> None:
    fields = {"frame_index": 0, "world_sim3_cam": [0.0] * 8} if value_type is TrackedFrame else {"world_sim3_cam": [[0.0] * 8], "ii": [], "jj": []}
    with pytest.raises(SerdeError):
        from_dict(value_type, fields | {"bogus": 1})


@given(huber=st.floats(0.1, 10.0), subsample=st.integers(1, 10), img_size=st.sampled_from([224, 512]), strict=st.booleans())
@settings(max_examples=30, deadline=None)
def test_slam_config_toml_round_trip(huber: float, subsample: int, img_size: int, strict: bool) -> None:
    base = SlamConfig()
    cfg = SlamConfig(subsample=subsample, img_size=img_size, tracking=TrackingConfig(huber=huber), reloc=type(base.reloc)(strict=strict))  # type: ignore[arg-type]
    assert from_toml(SlamConfig, to_toml(cfg)) == cfg


def test_slam_config_refuses_unknown_keys_and_coerces_hand_written_numbers() -> None:
    with pytest.raises(SerdeError):
        from_toml(SlamConfig, "single_thread = true\n")  # the old YAML key: lossless is the run mode now
    with pytest.raises(SerdeError):
        from_toml(SlamConfig, "[tracking]\nhubber = 1.0\n")
    assert from_toml(SlamConfig, "[tracking]\nsigma_dist = 10\n").tracking.sigma_dist == 10.0


def plane_view(index: int, focal: float, h: int, w: int, depth: float, confidence: float = 10.0) -> CameraView:
    """A camera with focal ``focal`` (principal point at the centre) looking at a fronto-parallel plane at ``depth``."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    points = np.stack([(xs - w / 2) / focal * depth, (ys - h / 2) / focal * depth, np.full((h, w), depth, np.float32)], axis=-1)
    pose = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32)
    return CameraView(index, pose, np.full((h, w, 3), 7, np.uint8), points.astype(np.float32), np.full((h, w), confidence, np.float32))


@given(focal=st.floats(10.0, 400.0), depth=st.floats(0.1, 20.0), h=st.integers(8, 40), w=st.integers(8, 40))
@settings(max_examples=30, deadline=None)
def test_a_frustum_has_the_focal_its_point_map_implies(focal: float, depth: float, h: int, w: int) -> None:
    """The CLI's frustum: the focal length that projects the point map onto its pixels, the principal point at the centre,
    a rigid pose (the viewer draws no frustum under a scaled transform)."""
    transform, pinhole = plane_view(0, focal, h, w, depth).frustum("/cam", 0.2)
    assert (transform.path, pinhole.path) == ("/cam", "/cam/pinhole")
    assert isinstance(pinhole.data, rr.Pinhole) and isinstance(transform.data, rr.Transform3D) and pinhole.data.image_from_camera is not None
    k = np.asarray(pinhole.data.image_from_camera.as_arrow_array().to_pylist()[0]).reshape(3, 3).T  # column-major
    np.testing.assert_allclose([k[0, 0], k[1, 1]], [focal, focal], rtol=1e-3)
    np.testing.assert_allclose(k[:2, 2], [w / 2, h / 2])
    assert transform.data.scale is None


def test_the_viewer_values_draw_the_cli_entities() -> None:
    current, last = plane_view(5, 30.0, 12, 16, 2.0), plane_view(1, 30.0, 12, 16, 2.0, confidence=3.0)
    paths = [log.path for log in FrameView(current, last).to_rerun("/s")]
    maps = ["pointmap", "pointmap_depth", "confidence"]
    camera = ["/s/current_camera", "/s/current_camera/pinhole", "/s/current_camera/pinhole/image", *(f"/s/current_camera/pinhole/{m}" for m in maps)]
    assert paths == [*camera, "/s/last_keyframe", *(f"/s/last_keyframe/{m}" for m in maps)]
    assert [log.path for log in FrameView(current, None).to_rerun("/s")] == camera
    logs = KeyframeView(last).to_rerun("/s/keyframes")
    assert [log.path for log in logs][-2:] == ["/s/keyframes/keyframe-1/pointcloud"] * 2
    cloud = logs[-1].data
    assert isinstance(cloud, rr.Points3D) and (cloud.positions is None or len(cloud.positions) == 0)  # confidence 3 < 7: none is confident
    full = KeyframeView(current).to_rerun("/s/keyframes")[-1].data
    assert isinstance(full, rr.Points3D) and full.positions is not None and len(full.positions) == 12 * 16  # every point confident
    assert not any(log.static for log in logs)
    with pytest.raises(NotImplementedError):
        KeyframeView.contents("/s")


@given(text=st.text(min_size=1, max_size=40), level=st.sampled_from(["INFO", "WARN"]))
@settings(max_examples=30, deadline=None)
def test_log_line_serde_round_trip(text: str, level: str) -> None:
    value = LogLine(text, level)
    assert from_dict(LogLine, to_dict(value)) == value
    [log] = value.to_rerun("/s/logs")
    assert log.path == "/s/logs" and isinstance(log.data, rr.TextLog)
