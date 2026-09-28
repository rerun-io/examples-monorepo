"""The per-segment record round-trips, boxes survive the camera/net-frame maps, and segment scores pool exactly."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float32
from serde.json import from_json, to_json
from torch import Tensor

from handtrack.apis.run_pipeline import Networks, RunConfig, RunRecord, file_sha256
from handtrack.eval.metrics import DetectionMetrics, detection_metrics
from handtrack.eval.segment import DetectionScore, GroundTruth, PositionScore, combine_detections, combine_positions, detection_scores, position_score
from handtrack.fit.pose_fit import FitConfig
from handtrack.geometry.letterbox import letterbox_for
from handtrack.pipeline import camera_boxes, net_circles
from handtrack.results import ARRAY_KEYS, SegmentTrack, TrackMetadata, load_track, save_track
from handtrack.tracker import TrackerConfig


def _track(frames: int) -> SegmentTrack:
    rng: np.random.Generator = np.random.default_rng(0)
    return SegmentTrack(
        meta=TrackMetadata(
            segment="umetrack__real__hand_hand__testing__user_00__recording_00",
            detnet_sha256="a" * 64,
            keynet_sha256="b" * 64,
            hand_mode="unknown",
            hand_scale=1.07,
            timings_s={"fit": 1.5},
        ),
        video_time_ns=np.arange(frames, dtype=np.int64) * 33_333_333,
        frame_index=np.arange(frames, dtype=np.int64),
        tracked=rng.random((frames, 2)) > 0.5,
        rotation=rng.random((frames, 2, 3, 3), dtype=np.float32),
        translation=rng.random((frames, 2, 3), dtype=np.float32),
        joint_angles=rng.random((frames, 2, 22), dtype=np.float32),
        landmarks=rng.random((frames, 2, 21, 3), dtype=np.float32),
        box=rng.random((frames, 4, 2, 4), dtype=np.float32),
        box_source=rng.integers(0, 3, (frames, 4, 2)).astype(np.int8),
        keypoints_2d=rng.random((frames, 4, 2, 21, 2), dtype=np.float32),
        presence=rng.random((frames, 4, 2), dtype=np.float32),
        detnet_camera=rng.integers(-1, 4, frames).astype(np.int8),
        detnet_presence=rng.random((frames, 2), dtype=np.float32),
        fit_energy=rng.random((frames, 2), dtype=np.float32),
    )


def test_segment_track_round_trips(tmp_path: Path) -> None:
    track: SegmentTrack = _track(5)
    loaded: SegmentTrack = load_track(save_track(track, tmp_path))
    assert loaded.meta == replace(track.meta, track_sha256=file_sha256(tmp_path / f"{track.meta.segment}.npz"))
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(getattr(loaded, key), getattr(track, key))
        assert getattr(loaded, key).dtype == getattr(track, key).dtype


def test_boxes_survive_camera_and_net_frame_maps() -> None:
    circle: Float32[Tensor, "1 2 2 3"] = torch.tensor([[[[100.0, 200.0, 30.0], [400.0, 250.0, 60.0]], [[10.0, 20.0, 8.0], [600.0, 400.0, 12.0]]]])
    umetrack = (letterbox_for(636, 480), letterbox_for(636, 480))
    boxes: Float32[Tensor, "1 2 2 4"] = camera_boxes(umetrack, circle)
    # UmeTrack real: the net frame is the camera image padded 2 px on the left.
    torch.testing.assert_close(boxes[0, 0, 0], torch.tensor([68.0, 170.0, 128.0, 230.0]))
    torch.testing.assert_close(net_circles(umetrack, boxes), circle)
    show3d = (letterbox_for(1024, 1280), letterbox_for(1024, 1280))
    torch.testing.assert_close(net_circles(show3d, camera_boxes(show3d, circle)), circle, atol=1e-3, rtol=0.0)


def test_position_scores_count_what_they_average_and_pool_exactly() -> None:
    target: Float32[Tensor, "4 1 21 3"] = torch.zeros((4, 1, 21, 3))
    predicted: Float32[Tensor, "4 1 21 3"] = target.clone()
    predicted[..., 0] = 0.002  # 2 mm everywhere
    scored: torch.Tensor = torch.tensor([[True], [True], [True], [False]])
    score: PositionScore = position_score(predicted, target, scored)
    assert score.keypoints == 63 and score.accelerations == 21
    assert score.mkpe_mm is not None and abs(score.mkpe_mm - 2.0) < 1e-5
    pooled: PositionScore = combine_positions([PositionScore(2.0, 1.0, 0.5, 63, 21), PositionScore(6.0, None, None, 21, 0)])
    assert pooled.mkpe_mm == 3.0 and pooled.mka_mm == 1.0 and pooled.mka_gt_mm == 0.5 and (pooled.keypoints, pooled.accelerations) == (84, 21)


def test_detection_counts_sum_per_camera() -> None:
    first: list[DetectionScore] = [DetectionScore(0, 3, 4, 5, 0.75, 0.6), DetectionScore(1, 0, 0, 0, None, None)]
    second: list[DetectionScore] = [DetectionScore(0, 1, 1, 3, 1.0, 1 / 3), DetectionScore(1, 2, 2, 2, 1.0, 1.0)]
    assert combine_detections([first, second]) == [DetectionScore(0, 4, 5, 8, 0.8, 0.5), DetectionScore(1, 2, 2, 2, 1.0, 1.0)]


def test_crop_containment_variant_is_the_rule_at_scale_one_and_forgives_a_small_offset() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(1)
    n: int = 64
    points: Float32[Tensor, "n 21 2"] = torch.tensor([320.0, 240.0]) + 40.0 * torch.randn((n, 21, 2), generator=generator)
    in_front: torch.Tensor = torch.rand((n, 21), generator=generator) > 0.05
    centre: Float32[Tensor, "n 2"] = points.mean(dim=1) + 3.0 * torch.randn((n, 2), generator=generator)
    half: Float32[Tensor, "n 1"] = 60.0 + 20.0 * torch.rand((n, 1), generator=generator)
    boxes: Float32[Tensor, "n 4"] = torch.cat([centre - half, centre + half], dim=-1)
    probability: Float32[Tensor, "n"] = torch.rand(n, generator=generator)
    camera: torch.Tensor = torch.randint(0, 4, (n,), generator=generator)
    hand: torch.Tensor = torch.randint(0, 2, (n,), generator=generator)
    eligible: torch.Tensor = in_front.sum(-1) >= 17
    rule: DetectionMetrics = detection_metrics(boxes, probability, points, in_front, camera, hand, eligible)
    assert detection_metrics(boxes, probability, points, in_front, camera, hand, eligible, containment_scale=1.0) == rule
    # One hand: 21 keypoints on a circle of radius 50 around (300, 200); the tight box is shifted 2 px right.
    angles: Float32[Tensor, "21"] = torch.linspace(0.0, 2 * torch.pi, 22)[:21]
    ring: Float32[Tensor, "1 21 2"] = (torch.tensor([300.0, 200.0]) + 50.0 * torch.stack([angles.cos(), angles.sin()], dim=-1))[None]
    shifted: Float32[Tensor, "1 4"] = torch.tensor([[252.0, 150.0, 352.0, 250.0]])
    args = (shifted, torch.tensor([0.9]), ring, torch.ones((1, 21), dtype=torch.bool), torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    assert detection_metrics(*args).total.true_positive == 0
    assert detection_metrics(*args, containment_scale=1.2).total.true_positive == 1


def test_segment_detection_uses_native_visibility_for_both_containment_scales() -> None:
    # Both hands fit in the net frame; only the second has 17 native-image points.
    points: Float32[Tensor, "1 1 2 21 2"] = torch.full((1, 1, 2, 21, 2), 100.0)
    points[..., 0, 0] = 90.0
    points[..., 1, 0] = 110.0
    truth: GroundTruth = GroundTruth(
        landmarks=torch.zeros((1, 2, 21, 3)), valid=torch.ones((1, 2), dtype=torch.bool),
        net_xy=points, in_front=torch.ones((1, 1, 2, 21), dtype=torch.bool),
        inside=torch.tensor([[[16, 17]]]), image_valid=torch.ones((1, 1), dtype=torch.bool),
    )
    circle: Float32[Tensor, "1 1 2 3"] = torch.tensor([[[[100.0, 100.0, 10.0], [100.0, 100.0, 10.0]]]])
    for scale in (1.0, 1.2):
        assert detection_scores(truth, circle, torch.ones((1, 1, 2)), 1, scale) == [DetectionScore(0, 1, 2, 1, 0.5, 1.0)]


def test_run_record_preserves_the_whole_tracker_config(tmp_path: Path) -> None:
    tracker: TrackerConfig = TrackerConfig(
        detnet_threshold=0.7, presence_threshold=0.6, max_views=1, extrapolate=False,
        refine_shift=0.3, max_reach_m=0.8, min_keypoint_confidence=0.1,
        fit=FitConfig(max_iterations=7, init_iterations=23, dist_weight=0.08, rotation_hypotheses=4),
    )
    record: RunRecord = RunRecord.from_config(RunConfig(tracker=tracker), Networks(None, None, "oracle", "oracle"))
    path: Path = tmp_path / "config.json"
    path.write_text(to_json(record))
    assert json.loads(path.read_text())["tracker"] == asdict(tracker)
    assert from_json(RunRecord, path.read_text()).tracker == tracker


@pytest.mark.parametrize("key, value", [
    ("tracked", np.zeros((5, 2), dtype=np.float32)),
    ("tracked", np.zeros((1, 2), dtype=np.bool_)),
    ("box", np.zeros((5, 3, 2, 4), dtype=np.float32)),
    ("landmarks", np.zeros((5, 2, 20, 3), dtype=np.float32)),
    ("translation", np.zeros((5, 1, 3), dtype=np.float32)),
    ("frame_index", np.zeros(5, dtype=np.int32)),
])
def test_load_track_rejects_malformed_arrays_with_source(tmp_path: Path, key: str, value: np.ndarray) -> None:
    path = save_track(_track(5), tmp_path)
    with np.load(path) as source:
        arrays = dict(source)
    arrays[key] = value
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=path.name):
        load_track(path)


def test_segment_scoring_excludes_unknown_rows_but_counts_confidence_zero() -> None:
    from test_segment_labels import FRAMES, _rig, _timeline

    from handtrack.data.segment_labels import segment_labels
    from handtrack.eval.segment import score_track

    timeline = _timeline()
    timeline.headset_valid[1] = False
    timeline.confidence[3, 0] = torch.nan
    timeline.confidence[4, 0] = 0.0
    camera = _rig()
    rig = replace(camera, names=tuple(f"cam{i}" for i in range(4)),
                  image_size=camera.image_size.repeat(4, 1), cam_from_rig=camera.cam_from_rig.repeat(4, 1, 1),
                  focal=camera.focal.repeat(4, 1), principal=camera.principal.repeat(4, 1))
    letterboxes = (letterbox_for(640, 480),) * 4
    labels = segment_labels(timeline, rig, letterboxes, np.arange(FRAMES, dtype=np.int64), show3d=False)
    track = replace(_track(FRAMES), tracked=np.ones((FRAMES, 2), dtype=np.bool_))
    hands = score_track(track, labels, letterboxes)[1]
    assert hands[0].tracking.acquire_frames == [0, 0]
    assert hands[0].tracking.drop_frames == []
    assert hands[0].tracking.tracked_without_hand == 1
    assert hands[0].tracking.tracked_absent == 1
    assert hands[1].tracking.tracked_without_hand == 4
    assert hands[1].tracking.tracked_absent == 4


def test_load_track_names_a_truncated_archive(tmp_path: Path) -> None:
    path: Path = save_track(_track(2), tmp_path)
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ValueError, match=path.name):
        load_track(path)
