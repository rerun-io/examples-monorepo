"""Types tier (decision 18): serde round-trips, unknown fields refused, Rerun mapping structure, point-map filtering."""

import numpy as np
import pytest
import rerun as rr
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from serde import SerdeError, from_dict, to_dict
from serde.toml import from_toml, to_toml

from mast3r_slam.types import Frame, KeyframeCloud, MapUpdate, SlamConfig, TrackedFrame, TrackingConfig

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
    frames = {f"/mast3r_slam/map/keyframes/{i:04d}" for i in range(k)}  # each keyframe's frame, for its KeyframeCloud
    assert {log.path for log in logs} == set(MapUpdate.contents("/mast3r_slam/map")[: 2 if len(edges[0]) else 1]) | frames
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


def frame_with(points: torch.Tensor, confidence: torch.Tensor) -> Frame:
    frame = Frame(0, torch.zeros(1, 3, 2, 2), torch.tensor([[2, 2]]), torch.tensor([[2, 2]]), torch.zeros(2, 2, 3))
    frame.update_pointmap(points, confidence, TrackingConfig())
    return frame


@given(
    a=arrays(np.float32, (4, 3), elements=FINITE),
    b=arrays(np.float32, (4, 3), elements=FINITE),
    ca=arrays(np.float32, (4, 1), elements=st.floats(0.125, 10.0, width=32)),
    cb=arrays(np.float32, (4, 1), elements=st.floats(0.125, 10.0, width=32)),
)
@settings(max_examples=40, deadline=None)
def test_weighted_pointmap_is_a_confidence_weighted_mean(a: np.ndarray, b: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> None:
    frame = frame_with(torch.from_numpy(a), torch.from_numpy(ca))
    frame.update_pointmap(torch.from_numpy(b), torch.from_numpy(cb), TrackingConfig(filtering_mode="weighted_pointmap"))
    assert frame.X_canon is not None and frame.C is not None
    np.testing.assert_allclose(frame.X_canon.numpy(), (ca * a + cb * b) / (ca + cb), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(frame.C.numpy(), ca + cb, rtol=1e-6)
    assert (frame.N, frame.N_updates) == (2, 2)
    average = frame.get_average_conf()
    assert average is not None
    np.testing.assert_allclose(average.numpy(), (ca + cb) / 2, rtol=1e-6)


def test_recent_replaces_and_first_keeps_only_the_second_observation() -> None:
    ones, twos, threes, conf = torch.ones(4, 3), 2 * torch.ones(4, 3), 3 * torch.ones(4, 3), torch.ones(4, 1)
    recent, first = frame_with(ones, conf), frame_with(ones, conf)
    for points in (twos, threes):
        recent.update_pointmap(points, conf, TrackingConfig(filtering_mode="recent"))
        first.update_pointmap(points, conf, TrackingConfig(filtering_mode="first"))
    assert recent.X_canon is not None and first.X_canon is not None
    assert float(recent.X_canon[0, 0]) == 3.0 and float(first.X_canon[0, 0]) == 2.0


@given(index=st.integers(0, 600), points=arrays(np.float32, st.tuples(st.integers(1, 20), st.just(3)), elements=FINITE), data=st.data())
@settings(max_examples=30, deadline=None)
def test_keyframe_cloud_serde_round_trip_and_rerun_path(index: int, points: np.ndarray, data: st.DataObject) -> None:
    colors: np.ndarray = data.draw(arrays(np.uint8, points.shape))
    value = KeyframeCloud(index, points, colors)
    back = from_dict(KeyframeCloud, to_dict(value))
    np.testing.assert_array_equal(back.points, points)
    np.testing.assert_array_equal(back.colors, colors)
    assert back.points.dtype == np.float32 and back.colors.dtype == np.uint8
    [log] = value.to_rerun("/mast3r_slam/map/keyframes")
    assert log.path == f"/mast3r_slam/map/keyframes/{index:04d}/points" and not log.static
