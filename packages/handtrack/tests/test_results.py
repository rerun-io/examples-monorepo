"""The per-segment record round-trips, boxes survive the camera/net-frame maps, and segment scores pool exactly."""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32
from torch import Tensor

from handtrack.eval.metrics import DetectionMetrics, detection_metrics
from handtrack.eval.segment import DetectionScore, PositionScore, combine_detections, combine_positions, detection_metrics_in_box, position_score
from handtrack.geometry.letterbox import letterbox_for
from handtrack.pipeline import camera_boxes, net_circles
from handtrack.results import ARRAY_KEYS, SegmentTrack, TrackMetadata, load_track, save_track


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
    assert loaded.meta == track.meta
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
    rule: DetectionMetrics = detection_metrics(boxes, probability, points, in_front, camera, hand)
    assert detection_metrics_in_box(boxes, probability, points, in_front, camera, hand, 1.0) == rule
    # One hand: 21 keypoints on a circle of radius 50 around (300, 200); the tight box is shifted 2 px right.
    angles: Float32[Tensor, "21"] = torch.linspace(0.0, 2 * torch.pi, 22)[:21]
    ring: Float32[Tensor, "1 21 2"] = (torch.tensor([300.0, 200.0]) + 50.0 * torch.stack([angles.cos(), angles.sin()], dim=-1))[None]
    shifted: Float32[Tensor, "1 4"] = torch.tensor([[252.0, 150.0, 352.0, 250.0]])
    args = (shifted, torch.tensor([0.9]), ring, torch.ones((1, 21), dtype=torch.bool), torch.tensor([0]), torch.tensor([0]))
    assert detection_metrics(*args).total.true_positive == 0
    assert detection_metrics_in_box(*args, 1.2).total.true_positive == 1
