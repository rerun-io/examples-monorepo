"""Stage bookkeeping of the hands speed benchmark."""

import pytest

from posekit.apis.hands_bench import StageTimes


def test_stage_summary_reports_median_and_mean_per_frame() -> None:
    times = StageTimes()
    for total, detect in ((10.0, 2.0), (30.0, 4.0), (20.0, 3.0)):
        times.add("total", total)
        times.add("detection", detect)
    summary = times.summary()
    assert summary["total"].median_ms == pytest.approx(20.0)
    assert summary["total"].mean_ms == pytest.approx(20.0)
    assert summary["detection"].median_ms == pytest.approx(3.0)
    assert summary["detection"].frames == 3



def test_one_loop_times_any_stack_after_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pose without stage profiling reports total, detection and pose; warm-up frames are not counted."""
    import numpy as np
    import torch

    from posekit.apis import hands_bench
    from posekit.cameras import CalibratedCamera
    from posekit.models.base import HandDetector, TopDownHandPose3d
    from posekit.predictions import HandDetections, HandPoses3d, Keypoints3d
    from posekit.skeletons import HAND_21, KeypointSkeleton

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    camera = CalibratedCamera(np.eye(3), np.zeros(14), 4, 4)
    seen: list[CalibratedCamera | None] = []

    class NoHands(HandDetector):
        def __call__(self, frames_rgb: torch.Tensor) -> HandDetections:
            return HandDetections.empty(frames_rgb.device)

    class EmptyPose(TopDownHandPose3d):
        skeleton: KeypointSkeleton = HAND_21

        def __call__(self, frames_rgb: torch.Tensor, detections: HandDetections, camera: CalibratedCamera | None = None) -> HandPoses3d:
            seen.append(camera)
            return HandPoses3d(keypoints=Keypoints3d.empty(HAND_21, frames_rgb.device, root_indices=(0,)), sides=(), vertices=None, faces=None)

    frames = torch.zeros((3, 4, 4, 3), dtype=torch.uint8)
    times, hands = hands_bench.bench_stack(frames, camera, NoHands(), EmptyPose(), warmup=2, repeats=2)
    assert list(times.values) == ["total", "detection", "pose"]
    assert len(times.values["total"]) == 6 and hands == 0.0
    assert seen == [camera] * 8
