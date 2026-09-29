"""CPU seams for the reference ladder; no upstream checkout or catalog."""
import numpy as np

from handtrack.reference.geometry import circle_crop


class Fisheye:
    """Analytic Fisheye62 with zero distortion, focal 200, principal (320, 240)."""
    camera_to_world_xf = np.eye(4)
    width = 640
    height = 480

    def world_to_eye(self, points):
        return points

    def window_to_eye(self, pixels):
        uv = (pixels - [320, 240]) / 200
        theta = np.linalg.norm(uv, axis=-1, keepdims=True)
        return np.concatenate((uv * np.sinc(theta / np.pi), np.cos(theta)), axis=-1)


def test_circle_crop_centre_margin_and_mirror():
    camera = Fisheye()
    circle = np.array([320., 240., 40.])
    left = circle_crop(camera, circle, 0.0, 0, 1.0)
    right = circle_crop(camera, circle, 0.0, 1, 1.0)
    boundary = camera.window_to_eye(np.array([[360., 240.], [320., 280.]]))
    projected = boundary @ np.linalg.inv(left.camera_to_world).T[:3, :3]
    pixels = projected[:, :2] / projected[:, 2:] * left.focal
    np.testing.assert_allclose(pixels, [[47.5 * .95, 0], [0, 47.5 * .95]], atol=1e-5)
    np.testing.assert_allclose(left.camera_to_world[:3, 2], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(right.camera_to_world[:, 0], -left.camera_to_world[:, 0], atol=1e-6)


def test_view_ranking_caps_before_sorting():
    from handtrack.reference.geometry import select_views
    assert select_views([.6, .5, .9, .8], .5) == [2, 3]
    assert select_views([18., 19., 21., 20.], 19., strict=False) == [2, 3]


def test_track_end_requires_consecutive_misses_and_resets():
    from handtrack.reference.state import TrackState
    state = TrackState("detnet", 3)
    assert state.choose([0], [], [.9, .1]) == [0]
    state.accept(True)
    assert state.choose([1], [0], [.1, .9]) == [0]
    assert state.choose([1], [0], [.1, .9]) == [0]
    assert state.choose([1], [0], [.1, .9]) == []
    assert not state.tracked
    assert state.choose([1], [], [.1, .9]) == [1]
    state.accept(True)
    assert state.choose([1], [], [.9, .9]) == []
    assert not state.tracked


def test_catalog_conversion_preserves_a6_units_angles_lens_and_absence():
    import pyarrow as pa
    from test_catalog import _statics

    from handtrack.data.catalog import UMETRACK_LAYOUT
    from handtrack.reference.catalog import reference_labels
    statics = _statics(True)
    for camera in UMETRACK_LAYOUT.cameras:
        prefix = f"{camera}/pinhole"
        lens_column = f"{prefix}:simplecv.components.DistortionCoefficients"
        statics = statics.set_column(statics.column_names.index(lens_column), lens_column,
                                    pa.array([[[.1, .01, 0., 0., .02, .03, .001, -.001]]]))
        statics = statics.append_column(f"{prefix}:source_camera_angle_deg", pa.array([[90.]]))
        for name, value in zip(("k1", "k2", "k3", "k4", "p1", "p2"), (.1, .01, 0., 0., .001, -.001), strict=True):
            statics = statics.append_column(f"{prefix}:{name}", pa.array([[value]]))
    rig = UMETRACK_LAYOUT.rig
    columns = {"video_time": [0, 1], f"{rig}:Transform3D:quaternion": [[[0., 0., 0., 1.]]] * 2,
               f"{rig}:Transform3D:translation": [[[1., 0., 0.]]] * 2, f"{rig}:untracked": [[False], [True]]}
    for side in ("left", "right"):
        prefix = f"/world/gt/hands/{side}"
        columns.update({f"{prefix}/confidence:Scalars:scalars": [[1.], [1.]],
                        f"{prefix}/joint_angles:joint_angles": [[[0.] * 22]] * 2,
                        f"{prefix}/wrist:Transform3D:quaternion": [[[0., 0., 0., 1.]]] * 2,
                        f"{prefix}/wrist:Transform3D:translation": [[[0., 0., .4]]] * 2})
    labels = reference_labels(statics, pa.table(columns))
    assert labels.cameras[0].angle == 90.
    np.testing.assert_allclose(labels.cameras[0].coefficients, [.1, .01, 0, 0, .001, -.001, .02, .03])
    np.testing.assert_allclose(labels.camera_to_world[0, 0, :3, 3], [1000, 0, 0])
    np.testing.assert_allclose(labels.camera_to_world[0, 1, :3, 3], [990.19933422, 0, -1.98669331], atol=1e-7)
    assert labels.wrists[0, 0, 2, 3] == 400.
    assert not labels.confidence[1].any()
    assert not labels.camera_to_world[1].any()


def test_fisheye_inverse_and_off_axis_crop():
    from handtrack.reference.geometry import FisheyeRays

    class Distorted(Fisheye):
        f = (200., 200.)
        c = (320., 240.)

        def eye_to_window(self, points):
            length = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
            theta = np.arctan2(length, points[:, 2:])
            uv = points[:, :2] * np.divide(theta, length, out=np.ones_like(theta), where=length > 0)
            r2 = (uv * uv).sum(axis=1, keepdims=True)
            return uv * (1 + .04 * r2) * self.f + self.c

    camera = Distorted()
    rays = FisheyeRays(camera)
    circle = np.array([410., 275., 35.])
    crop = circle_crop(rays, circle, 90.0, 0, 1.0)
    target = rays.window_to_eye(circle[None, :2])[0]
    np.testing.assert_allclose(crop.camera_to_world[:3, 2], target, atol=1e-6)
    theta = np.arange(21) * 2 * np.pi / 21
    pixels = circle[:2] + 30 * np.stack((np.cos(theta), np.sin(theta)), axis=1)
    points = rays.window_to_eye(pixels)
    np.testing.assert_allclose(camera.eye_to_window(points), pixels, atol=1e-5)
    eye = points @ np.linalg.inv(crop.camera_to_world)[:3, :3].T
    projected = eye[:, :2] / eye[:, 2:] * crop.focal + 47.5
    assert ((projected >= 0) & (projected <= 95)).all()


def test_calibration_rejects_testing_and_records_median_iqr():
    import pytest

    from handtrack.reference.results import calibration_summary
    result = calibration_summary(["umetrack__real__hand_hand__training__user_10__recording_00"], "source", [.5, .6, .7, .8], [1., 2., 3., 4.], 1)
    assert result.median == pytest.approx(.65)
    assert result.q75 - result.q25 == pytest.approx(.15)
    assert result.axis_median_deg == 2.5
    with pytest.raises(ValueError, match="validation"):
        calibration_summary(["umetrack__real__hand_hand__testing__user_12__recording_13"], "source", [.7], [2.], 1)


def test_restart_checks_identity_modes_and_pools_by_sample_count(tmp_path):
    from serde.json import from_json, to_json
    from test_reference_statistics import frame_stream

    from handtrack.apis.reference_eval import Config, SegmentResult, publish_summary, verified_result
    from handtrack.data.catalog import UMETRACK, SegmentInfo
    from handtrack.eval.segment import PositionScore
    from handtrack.reference.results import ReferenceMetrics, Summary, save_frames
    infos = tuple(SegmentInfo(UMETRACK, f"s{i}", "real", "hand_hand", "testing", "user_12", 4, 30) for i in range(2))
    config = Config(output=tmp_path, modes=("gt_pose",))
    for index, info in enumerate(infos):
        count = index + 1
        metric = ReferenceMetrics(info.segment_id, "gt_pose", "none", "identity", 4,
                                  PositionScore(10. * count, 2. * count, 1., 21 * count, 21 * count),
                                  4, count, 1, count / 4, 0, None, None)
        stream = frame_stream(4)
        stream.error_mm[:] = np.nan
        stream.posed[:] = False
        stream.gt_valid[:] = False
        stream.gt_valid[:, 0] = True
        stream.error_mm[:count, 0, 0] = 10. * count
        stream.posed[:count, 0, 0] = True
        stream.posed[0, 0, 1] = True
        npz = tmp_path / f"{info.segment_id}.npz"
        digest = save_frames(stream, npz)
        result = SegmentResult("identity", info.segment_id, [metric], npz.name, digest)
        path = tmp_path / f"{info.segment_id}.json"
        path.write_text(to_json(result))
        assert verified_result(path, "wrong", info.segment_id, [("gt_pose", "geometry")]) is None
        assert verified_result(path, "identity", info.segment_id, [("detnet", "geometry")]) is None
        if index == 0:
            assert not publish_summary(config, infos, "identity")
            assert not (tmp_path / "summary.json").exists()
    assert publish_summary(config, infos, "identity")
    summary = from_json(Summary, (tmp_path / "summary.json").read_text())
    all_score = summary.metrics[0]
    assert all_score.position.mkpe_mm == 50 / 3
    assert all_score.coverage == 3 / 8
    assert all_score.false_poses == 2
    assert all_score.robust.median_mm == 20.
    assert all_score.robust.p90_mm == 20.
    assert all_score.robust.below_20 == 1 / 8
    assert all_score.robust.below_50 == 3 / 8
    assert all_score.robust.wild == 0.
    (tmp_path / "s0.npz").write_bytes(b"corrupt")
    assert verified_result(tmp_path / "s0.json", "identity", "s0", [("gt_pose", "geometry")]) is None
    (tmp_path / "s0.json").write_text("{")
    assert verified_result(tmp_path / "s0.json", "identity", "s0", [("gt_pose", "geometry")]) is None


def test_ladder_fake_stage_acquires_tracks_ends_and_clears_memory(monkeypatch):
    import torch
    from test_segment_labels import _timeline

    from handtrack.data.catalog import UMETRACK, SegmentInfo
    from handtrack.data.segment_labels import segment_labels
    from handtrack.geometry.camera import CameraRig
    from handtrack.geometry.letterbox import letterbox_for
    from handtrack.models.detnet import Detections
    from handtrack.pipeline import SegmentData
    from handtrack.reference.catalog import ReferenceLabels
    from handtrack.reference.run import run_ladder
    from handtrack.reference.upstream import Pose, PoseStage
    from handtrack.tracker import DetNetDetector

    timeline = _timeline()
    rig = CameraRig(tuple(f"cam{i}" for i in range(4)), torch.tensor([[640., 480.]] * 4), torch.eye(4).repeat(4, 1, 1),
                    torch.tensor([[240., 240.]] * 4), torch.tensor([[320., 240.]] * 4), None)
    letterboxes = (letterbox_for(640, 480),) * 4
    labels = segment_labels(timeline, rig, letterboxes, np.arange(6, dtype=np.int64), False)
    info = SegmentInfo(UMETRACK, "fake", "real", "hand_hand", "testing", "user_12", 6, 30)
    data = SegmentData(info, rig, letterboxes, timeline, labels, ())
    reference = ReferenceLabels([], np.tile(np.eye(4), (6, 4, 1, 1)), np.tile(np.eye(4), (6, 2, 1, 1)),
                                np.zeros((6, 2, 22)), np.zeros((6, 2)), np.ones(6, dtype=bool), timeline.video_time_ns, timeline.hand_model)
    monkeypatch.setattr("handtrack.reference.run.net_frames", lambda *_: iter([torch.zeros((6, 4, 480, 640), dtype=torch.uint8)]))

    class FakeDetector(DetNetDetector):
        def __init__(self):
            pass

        def __call__(self, images, frame, cameras):
            probability = torch.zeros((4, 2))
            probability[0, 0] = .9 if frame in (0, 4, 5) else .1
            circle = torch.tensor([320., 240., 30.]).repeat(4, 2, 1)
            return Detections(circle, probability, probability > .5, torch.zeros((4, 2, 4)))

    class FakeCamera:
        f = (200., 200.)
        c = (320., 240.)
        width = 640
        height = 480
        camera_to_world_xf = np.eye(4)
        def world_to_eye(self, points):
            return points
        def eye_to_window(self, points):
            return points[:, :2]

    class FakeStage(PoseStage):
        def __init__(self):
            self.row = 0
            self.history = False
            self.memory_inputs = []
            self.received = []
            self.prediction = Pose(np.zeros(22), np.eye(4))

        def set_frame(self, row):
            self.row = row

        def ground_truth(self, row):
            return {}

        def gt_circles(self, poses):
            return np.full((4, 2, 3), np.nan), np.zeros((4, 2))

        def circle_crops(self, circles, scores, scale, gt=False):
            return {0: {0: FakeCamera()}} if scores[0, 0] > .5 else {}

        def pose_crops(self, poses):
            if poses:
                assert poses[0] is self.prediction  # Previous prediction, never a GT pose.
            return {0: {1: FakeCamera()}} if poses and self.row != 5 else {}

        def track(self, images, crops):
            self.memory_inputs.append(self.history)
            self.received.append(list(crops.get(0, {})))
            self.history = bool(crops)
            return {0: self.prediction} if crops else {}

        def landmarks(self, pose, hand):
            return np.zeros((21, 3), dtype=np.float32)

    # DetNet scores are absent on tracked camera 1, so three consecutive misses end frame 3.
    stage = FakeStage()
    metrics = run_ladder(data, reference, {("track", "detnet"): stage}, FakeDetector(), 6, 1.0, 3, "identity", torch.device("cpu"))
    assert stage.received == [[0], [1], [1], [], [0], []]
    assert stage.memory_inputs == [False, True, True, True, False, True]
    assert metrics.metrics[0].false_poses == 4
    assert metrics.metrics[0].coverage is None
    geometry = FakeStage()
    run_ladder(data, reference, {("track", "geometry"): geometry}, FakeDetector(), 6, 1.0, 3, "identity", torch.device("cpu"))
    assert geometry.received == [[0], [1], [1], [1], [1], []]
    np.testing.assert_array_equal(metrics.streams.posed[:, 0, 0], [True, True, True, False, True, False])
    assert np.isnan(metrics.streams.error_mm).all()
    assert metrics.streams.selected[1, 0, 0, 1]
    assert metrics.streams.detnet_presence[1, 0, 0] == np.float64(np.float32(.1))

    from dataclasses import replace

    class ScoredStage(FakeStage):
        truth = Pose(np.zeros(22), np.eye(4))

        def ground_truth(self, row):
            return {0: self.truth}

        def gt_circles(self, poses):
            circles = np.full((4, 2, 3), np.nan)
            circles[:, 0] = [317., 236., 28.]
            return circles, np.array([[19., 0.], [18., 0.], [0., 0.], [21., 0.]])

        def landmarks(self, pose, hand):
            points = np.zeros((21, 3), dtype=np.float32)
            if pose is self.prediction:
                points[:, 0] = 10.
            return points

    confidence = np.zeros((6, 2))
    confidence[:, 0] = 1.
    scored = run_ladder(data, replace(reference, confidence=confidence),
        {("detnet", "geometry"): ScoredStage()}, FakeDetector(), 6, 1.0, 3, "identity", torch.device("cpu"))
    np.testing.assert_allclose(scored.streams.error_mm[[0, 4, 5], 0, 0], 10., atol=1e-6)
    assert np.isnan(scored.streams.error_mm[[1, 2, 3], 0, 0]).all()
    assert np.isnan(scored.streams.error_mm[:, 0, 1]).all()
    assert scored.metrics[0].robust.below_20 == .5
    assert scored.metrics[0].cameras[0].selected == 3
    assert scored.metrics[0].cameras[0].selected_visible == 3
    assert scored.metrics[0].cameras[0].centre_mean_px == 5.
    assert scored.metrics[0].cameras[0].radius_mean_px == 2.
    assert scored.metrics[0].cameras[1].error_pairs == 0



def test_known_pose_landmarks_fit_inside_circle_crop():
    import torch

    from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
    from handtrack.labels.circles import enclosing_circles

    model = generic_hand_model()
    pose = HandPose(torch.eye(3), torch.tensor([.03, .02, .5]), torch.zeros(22))
    points = landmarks(model, pose, Side.LEFT).numpy().astype(np.float64)
    length = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
    theta = np.arctan2(length, points[:, 2:])
    pixels = points[:, :2] * theta / length * 200 + [320, 240]
    circle = enclosing_circles(pixels.astype(np.float32), np.ones(21, dtype=bool)).astype(np.float64)
    crop = circle_crop(Fisheye(), circle, 0.0, 0, 1.0)
    eye = points @ np.linalg.inv(crop.camera_to_world)[:3, :3].T
    projected = eye[:, :2] / eye[:, 2:] * crop.focal + 47.5
    assert ((projected >= 0) & (projected <= 95)).all()
