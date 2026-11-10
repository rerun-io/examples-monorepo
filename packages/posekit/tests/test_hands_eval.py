"""Protocol logic of the hands evaluation on a synthetic segment with fake models (no catalog, no GPU)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("warp", reason="the eval imports EgoHand; it needs the posekit-hands Pixi environment")

from simplecv.catalog_video import CatalogVideo

from posekit.apis.hands_eval import (
    EndToEndCell,
    HandsEvalConfig,
    detnet_turns,
    error_quantiles,
    evaluate_segment,
    failed_bands,
    joint_box,
    within_accuracy_band,
)
from posekit.cameras import CalibratedCamera
from posekit.eval.hand_gt import GtDataset, HandGtSegment
from posekit.models.base import HandDetector, TopDownHandPose3d
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, Keypoints3d
from posekit.skeletons import HAND_21


def _segment() -> HandGtSegment:
    """One frame, a left hand around x=100 and a right hand around x=300, all joints visible."""
    rng = np.random.default_rng(0)
    xyz = np.zeros((1, 2, 21, 3), np.float32)
    uv = np.zeros((1, 2, 21, 2), np.float32)
    for side, x in ((0, 100.0), (1, 300.0)):
        uv[0, side] = np.stack([x + rng.uniform(-20, 20, 21), 200 + rng.uniform(-20, 20, 21)], axis=1)
        xyz[0, side] = np.concatenate([(uv[0, side] - 200) / 500, np.full((21, 1), 0.5)], axis=1)
    valid = np.ones((1, 2, 21), bool)
    valid[:, :, 1] = False
    camera = CalibratedCamera(np.array([[500.0, 0, 200], [0, 500, 200], [0, 0, 1]]), np.zeros(0), 400, 400, lens="PINHOLE")
    return HandGtSegment("seg", camera, CatalogVideo(np.zeros(1, np.int64), [], [True], "av1"), np.zeros(1, np.int64), np.zeros(1, np.int64), xyz, uv, valid)


class TruthPose(TopDownHandPose3d):
    """Returns the ground truth of each requested side: zero error by construction."""

    skeleton = HAND_21

    def __init__(self, segment: HandGtSegment) -> None:
        self.segment = segment

    def __call__(self, frames_rgb, detections, camera=None) -> HandPoses3d:
        rows = [(int(f), ("left", "right").index(s)) for f, s in zip(detections.boxes.frame_indices.tolist(), detections.sides, strict=True)]
        xyz = torch.from_numpy(np.stack([self.segment.xyz_camera[f, s] for f, s in rows]))
        xy = torch.from_numpy(np.stack([self.segment.uv[f, s] for f, s in rows]))
        return HandPoses3d(
            keypoints=Keypoints3d(xy, xyz[..., 2] - xyz[:, :1, 2], torch.ones((len(rows), 21)), detections.boxes.frame_indices, HAND_21, (0,), xyz),
            sides=detections.sides,
        )


class FixedDetector(HandDetector):
    def __init__(self, detections: HandDetections) -> None:
        self.detections = detections

    def __call__(self, frames_rgb) -> HandDetections:
        return self.detections


def test_gt_boxes_score_every_visible_hand() -> None:
    segment = _segment()
    count, any_side, scored, _ = evaluate_segment(
        segment, torch.zeros((1, 400, 400, 3), dtype=torch.uint8), FixedDetector(HandDetections.empty("cpu")), TruthPose(segment), "gt-boxes", HandsEvalConfig(), True
    )
    assert (count, any_side, len(scored)) == (2, 2, 2)
    assert all(hand.errors.mpjpe_mm == pytest.approx(0.0, abs=1e-4) and hand.errors.epe_2d_px == pytest.approx(0.0, abs=1e-4) for hand in scored)


def test_detector_recall_needs_the_right_side_and_overlap() -> None:
    """Left hand boxed but called right (side error), right hand missed: recall 0, side-agnostic recall 1/2."""
    segment = _segment()
    left_box = joint_box(segment.uv[0, 0], segment.valid[0, 0], 1.0)
    far_box = np.array([0.0, 0.0, 10.0, 10.0], np.float32)
    detections = HandDetections(BoxDetections(torch.from_numpy(np.stack([left_box, far_box])), torch.ones(2), torch.zeros(2, dtype=torch.long)), ("right", "right"))
    count, any_side, scored, _ = evaluate_segment(
        segment, torch.zeros((1, 400, 400, 3), dtype=torch.uint8), FixedDetector(detections), TruthPose(segment), "detector", HandsEvalConfig(), True
    )
    assert (count, any_side, len(scored)) == (2, 1, 0)


def test_evaluation_cli_selects_every_egohand_stack() -> None:
    import tyro

    config = tyro.cli(HandsEvalConfig, args=["--models", "egohand", "egohand-fast", "egohand-fast-bf16"])
    assert config.models == ("egohand", "egohand-fast", "egohand-fast-bf16")


def test_timing_and_bands_need_the_end_to_end_protocol() -> None:
    with pytest.raises(ValueError, match="end-to-end"):
        HandsEvalConfig(timing=True)
    assert HandsEvalConfig(protocols=("end-to-end",), timing=True).timing


def _row(dataset: GtDataset) -> EndToEndCell:
    return EndToEndCell("egohand-fast", dataset, 10, 1.0, 10, 9, 0.9, 0.8, 1.0, 10.0, 30.0, 20.0, 50.0, 15.0, 40.0, None, None, None)


def test_bands_pass_within_five_percent() -> None:
    reference = [_row("dataforge-umetrack"), _row("gsi-six-view")]
    assert failed_bands(reference, [replace(_row("dataforge-umetrack"), recall=0.9 * 0.96, epe_2d_p95_px=30.0 * 1.04), _row("gsi-six-view")]) == []


def test_bands_name_each_failure_and_a_missing_reference() -> None:
    reference = [_row("dataforge-umetrack")]
    candidate = [replace(_row("dataforge-umetrack"), mean_iou=0.7, absolute_3d_median_mm=None), _row("gsi-six-view")]
    assert failed_bands(reference, candidate) == ["dataforge-umetrack: mean_iou", "dataforge-umetrack: absolute_3d_median_mm", "gsi-six-view: no reference"]


def test_lower_is_better_band_includes_five_percent() -> None:
    assert within_accuracy_band(100.0, 105.0, higher_is_better=False)
    assert not within_accuracy_band(100.0, 105.01, higher_is_better=False)


def test_higher_is_better_band_includes_five_percent() -> None:
    assert within_accuracy_band(0.8, 0.76, higher_is_better=True)
    assert not within_accuracy_band(0.8, 0.759, higher_is_better=True)


def test_missing_metrics_cannot_pass() -> None:
    assert not within_accuracy_band(None, 1.0, higher_is_better=True)
    assert not within_accuracy_band(1.0, None, higher_is_better=False)
    assert not within_accuracy_band(1.0, float("nan"), higher_is_better=False)


def test_errors_report_median_and_p95_without_discarding_failures() -> None:
    assert error_quantiles([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx((3.0, 4.8))
    assert error_quantiles([]) == (None, None)
    with pytest.raises(ValueError, match="finite"):
        error_quantiles([1.0, float("nan")])


def test_a_detector_only_model_scores_recall_and_no_pose() -> None:
    """DetNet-F has no pose stage: its found hands count for recall and carry no pose error."""
    segment = _segment()
    left_box = joint_box(segment.uv[0, 0], segment.valid[0, 0], 1.0)
    detections = HandDetections(BoxDetections(torch.from_numpy(left_box[None]), torch.ones(1), torch.zeros(1, dtype=torch.long)), ("left",))
    count, any_side, scored, _ = evaluate_segment(
        segment, torch.zeros((1, 400, 400, 3), dtype=torch.uint8), FixedDetector(detections), None, "detector", HandsEvalConfig(), False
    )
    assert (count, any_side, len(scored)) == (2, 1, 1)
    assert np.isnan(scored[0].errors.mpjpe_mm) and np.isnan(scored[0].errors.epe_2d_px)


def test_detnet_models_take_their_own_input_convention() -> None:
    import tyro

    from posekit.apis.hands_eval import DETNET_TURNS

    config = tyro.cli(HandsEvalConfig, args=["--models", "detnet", "detnet-legacy", "--detnet-onnx", "/x/new.onnx", "--detnet-legacy-onnx", "/x/old.onnx"])
    assert config.models == ("detnet", "detnet-legacy")
    # Upright: UmeTrack's front camera is mounted a quarter turn; legacy: Quest 3 portrait frames were turned once.
    assert DETNET_TURNS["detnet"] == {"dataforge-hot3d-quest3": 0, "dataforge-umetrack": 3}
    assert DETNET_TURNS["detnet-legacy"] == {"dataforge-hot3d-quest3": 1, "dataforge-umetrack": 0}


def test_detnet_turns_subtract_what_the_conversion_already_turned() -> None:
    # UmeTrack as recorded (stored 0): the upright model turns its front camera 3 times, the legacy model not at all.
    assert (detnet_turns("detnet", "dataforge-umetrack", 0), detnet_turns("detnet-legacy", "dataforge-umetrack", 0)) == (3, 0)
    # After the upright re-convert (stored 3): the upright model needs nothing; the legacy model turns the frame back as recorded.
    assert (detnet_turns("detnet", "dataforge-umetrack", 3), detnet_turns("detnet-legacy", "dataforge-umetrack", 3)) == (0, 1)
    assert detnet_turns("detnet-legacy", "dataforge-hot3d-quest3", 0) == 1
    assert detnet_turns("detnet", "gsi-six-view", 0) == 0, "a subset the table does not list"
