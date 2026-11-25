"""FullCircle masks vs the masks FullCircle released with its dataset (scene ``room3``, equirect omni frames).

Asset: one scene of huggingface.co/datasets/youlenda/FullCircle extracted (``<scene>/omni/images/frame_*.png`` and
``<scene>/omni/masks/frame_*_mask.png``), named by ``POSEKIT_FULLCIRCLE_SCENE``. FullCircle captures at 5 fps; the
Insta360 stage tracks at 3 Hz, so the scene is taken at half rate (every ``STRIDE``-th frame) with the default re-prompt
cadence. On room3 at half rate (179 frames) the full run scores a mean IoU of 0.940 (median 0.956; all 357 frames
with ``reprompt_every=6``: 0.945, median 0.957); this test runs the first 40 of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2", reason="OpenCV reads the FullCircle PNGs")

pytestmark = pytest.mark.golden

FRAMES: int = 40
STRIDE: int = 2
HEIGHT, WIDTH = 720, 1440


def test_fullcircle_masks_match_released_masks() -> None:
    from posekit.models.fullcircle import FullCircleConfig, FullCircleMasker, SphereSource, fill_nearest, operator_directions, view_up
    from posekit.ops.sphere import EquirectCamera

    scene_env: str | None = os.environ.get("POSEKIT_FULLCIRCLE_SCENE")
    if scene_env is None or not torch.cuda.is_available():
        pytest.skip("needs POSEKIT_FULLCIRCLE_SCENE (an extracted FullCircle scene, e.g. room3) and a GPU")
    scene: Path = Path(scene_env)
    paths: list[Path] = sorted((scene / "omni/images").glob("frame_*.png"))[::STRIDE][:FRAMES]
    if not paths:
        pytest.skip(f"no omni frames under {scene}")
    frames: list[np.ndarray] = [
        cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for p in paths
    ]
    masker: FullCircleMasker = FullCircleMasker(
        FullCircleConfig(work_size=WIDTH, dilate_px=0.0), [SphereSource(EquirectCamera(WIDTH, HEIGHT), np.eye(3), (HEIGHT, WIDTH), (HEIGHT, WIDTH))]
    )
    evidence = masker.detect([[torch.from_numpy(f).cuda()] for f in frames], [np.eye(3)] * len(frames))
    config: FullCircleConfig = masker.config
    directions = operator_directions([e.detections for e in evidence], config.direction_gate_deg, config.lost_after)
    filled: list[np.ndarray] = fill_nearest(directions)
    up: np.ndarray = view_up(filled)
    ious: list[float] = []
    for path, frame, direction in zip(paths, frames, filled, strict=True):
        ours: np.ndarray = masker.track([torch.from_numpy(frame).cuda()], direction, up).source_masks[0].cpu().numpy()
        released: np.ndarray = cv2.imread(str(scene / "omni/masks" / f"{path.stem}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
        theirs: np.ndarray = cv2.resize(released.astype(np.uint8), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST).astype(bool)
        ious.append(float(np.logical_and(ours, theirs).sum() / max(np.logical_or(ours, theirs).sum(), 1)))
    assert all(d is not None for d in directions)
    assert float(np.mean(ious)) > 0.93
    assert float(np.percentile(ious, 10)) > 0.9
