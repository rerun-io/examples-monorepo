"""Protocol logic of the hands evaluation on a synthetic segment with fake models (no catalog, no GPU)."""

import numpy as np
import pytest
import torch

pytest.importorskip("warp", reason="the eval imports EgoHand; it needs the posekit-hands Pixi environment")

from simplecv.catalog_video import CatalogVideo

from posekit.apis.hands_eval import HandsEvalConfig, evaluate_segment, joint_box
from posekit.cameras import CalibratedCamera
from posekit.eval.hand_gt import HandGtSegment
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


def test_evaluation_cli_selects_fast_and_reference_egohand() -> None:
    import tyro

    config = tyro.cli(HandsEvalConfig, args=["--models", "egohand", "egohand-fast"])
    assert config.models == ("egohand", "egohand-fast")


def test_tools_name_egohand_stacks_like_the_cli() -> None:
    """hands_eval's stack names are the registry's; reference stacks keep their colour, fast ones theirs, and names round-trip."""
    from serde import to_dict
    from serde.json import from_json, to_json

    from posekit.apis.hands_eval import EvalCell, eval_model
    from posekit.eval.hand_metrics import Summary

    assert eval_model("wilor").left_class == 2 and not eval_model("wilor").metric_root
    assert eval_model("egohand-selfcap").left_class == eval_model("egohand").left_class == 4
    assert eval_model("egohand-fast").left_class == eval_model("egohand-catalog").left_class == 6
    cell = EvalCell("egohand-catalog", "dataforge-umetrack", "detector", 1, 1, 1.0, 1.0, {"mpjpe_mm": Summary(12.5, None, 1)})
    assert from_json(EvalCell, to_json(cell)) == cell
    assert to_dict(cell)["metrics"] == {"mpjpe_mm": {"mean": 12.5, "p90": None, "count": 1}}
