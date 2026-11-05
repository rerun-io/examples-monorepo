"""mv-api's WiLoR hand refinement against the pre-port wilor_nano.hand_keypoints output on the same boxes."""

from pathlib import Path

import numpy as np
import pytest
import torch
from rerun.chunk import RrdReader

pytestmark = pytest.mark.golden

GOLDEN: Path = Path("/mnt/nas/ssd-8tb/posekit/goldens/hands-v1/wilor_mvapi_golden.rrd")
CLIP: Path = Path(__file__).resolve().parents[2] / "posekit" / "assets" / "examples" / "selfcap_oak_left.mp4"


def test_refined_hand_keypoints_match_preport() -> None:
    """Same crops as before the port (integer centre, float image); fp16 drift across environments only."""
    from torchcodec.decoders import VideoDecoder

    from mv_api.wilor_refine import WilorKeypointRefiner

    if not GOLDEN.is_file():
        pytest.skip(f"missing pre-port golden: {GOLDEN}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    batch = next(chunk for chunk in RrdReader(GOLDEN).stream() if chunk.entity_path == "/golden/wilor_mvapi_oak").to_record_batch()
    golden = {
        name: np.asarray(batch.column(name)[0].values).reshape(tuple(np.asarray(batch.column(f"{name}_shape")[0].values)))
        for name in ("frame_indices", "sides", "xyxy", "keypoints_2d")
    }
    frames = VideoDecoder(CLIP, dimension_order="NHWC", device="cuda").get_frames_in_range(0, 100).data.cpu().numpy()
    refiner = WilorKeypointRefiner()
    actual = np.stack(
        [
            refiner(frames[frame], box.astype(np.float32), "right" if side == 1 else "left")
            for frame, side, box in zip(golden["frame_indices"], golden["sides"], golden["xyxy"], strict=True)
        ]
    )
    error = np.linalg.norm(actual - golden["keypoints_2d"], axis=-1)
    assert float(np.median(error)) <= 0.4, f"median {np.median(error):.3f} px"
    assert float(np.quantile(error, 0.99)) <= 1.5, f"p99 {np.quantile(error, 0.99):.3f} px"
