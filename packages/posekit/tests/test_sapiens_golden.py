"""Vendored Sapiens2 against the pre-port outputs (main 2100af2bc, packages/sapiens2-pose) on the same person boxes.

The references live on the NAS (README beside them). Inputs: the 16 example images, one call each, and 32 frames of
the hands clip in one call (dynamic batches). The stored YOLOX boxes are fed back, so both runs see identical crops;
keypoints are compared where the reference is confident (score > 0.3).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from rerun.chunk import RrdReader
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image

from posekit.models.sapiens import SapiensPoseConfig
from posekit.predictions import BoxDetections
from posekit.runtimes import TensorRtBackendConfig, TorchBackendConfig

pytestmark = pytest.mark.golden

GOLDEN: Path = Path("/mnt/nas/ssd-8tb/posekit/goldens/sapiens2-v1/sapiens2_golden.rrd")
ASSETS: Path = Path(__file__).resolve().parents[1] / "assets"


def read_golden(group: str) -> dict[str, np.ndarray]:
    """One ``/golden/<group>`` entity of the reference .rrd as named arrays."""
    if not GOLDEN.is_file():
        pytest.skip(f"missing pre-port golden: {GOLDEN}")
    for chunk in RrdReader(GOLDEN).stream():
        if chunk.entity_path == f"/golden/{group}":
            batch = chunk.to_record_batch()
            return {
                column: np.asarray(batch.column(column)[0].values).reshape(tuple(np.asarray(batch.column(f"{column}_shape")[0].values)))
                for column in batch.schema.names
                if not column.endswith("_shape") and not column.startswith("rerun.")
            }
    raise KeyError(group)


# Bands per path: (median px, p99 px, max |score difference|). On the capture machine every path is bit-identical
# (fresh exports and engines). The torch path runs the same weights and kernels; the TensorRT paths rebuild their
# engines from equivalent ONNX exports, so only tactic choice on another GPU or TensorRT build may move them.
PATHS: dict[str, tuple[SapiensPoseConfig, float, float, float]] = {
    "bf16_torch": (SapiensPoseConfig(variant="bf16-1024", backend=TorchBackendConfig(autocast="bf16")), 1e-3, 1e-3, 1e-3),
    "bf16_trt": (SapiensPoseConfig(variant="bf16-1024", backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4)), 0.1, 1.0, 0.02),
    "nvfp4_trt": (SapiensPoseConfig(), 0.1, 1.0, 0.02),
}


def assert_matches(actual_xy: np.ndarray, actual_scores: np.ndarray, expected: dict[str, np.ndarray], bands: tuple[float, float, float]) -> None:
    """Same NaN (unmapped joint) pattern; confident keypoints and all scores within the path's bands."""
    median_px, p99_px, score_tolerance = bands
    assert actual_xy.shape == expected["xy"].shape
    np.testing.assert_array_equal(np.isnan(actual_xy), np.isnan(expected["xy"]))
    confident: np.ndarray = expected["scores"] > 0.3
    error_px: np.ndarray = np.linalg.norm(actual_xy - expected["xy"], axis=-1)[confident]
    assert float(np.median(error_px)) <= median_px, f"median {np.median(error_px):.4f} px > {median_px}"
    assert float(np.quantile(error_px, 0.99)) <= p99_px, f"p99 {np.quantile(error_px, 0.99):.4f} px > {p99_px}"
    assert float(np.abs(actual_scores - expected["scores"]).max()) <= score_tolerance


def as_detections(xyxy: np.ndarray, scores: np.ndarray, frame_indices: np.ndarray) -> BoxDetections:
    # torch.tensor copies: the golden's arrays are read-only Arrow buffers.
    return BoxDetections(torch.tensor(xyxy, device="cuda"), torch.tensor(scores, device="cuda"), torch.tensor(frame_indices, device="cuda"))


@pytest.mark.parametrize("name", PATHS)
def test_sapiens_matches_preport(name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    config: SapiensPoseConfig = PATHS[name][0]
    boxes: dict[str, np.ndarray] = read_golden("boxes")
    video_boxes: dict[str, np.ndarray] = read_golden("video_boxes")
    names: list[str] = bytes(read_golden("images")["names"]).decode().split("\n")
    model = config.setup()
    xy: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for index, image_name in enumerate(names):
            image = decode_image(str(ASSETS / "images" / image_name), mode=ImageReadMode.RGB)
            keep: np.ndarray = boxes["image_ids"] == index
            keypoints = model(
                image.permute(1, 2, 0).unsqueeze(0).contiguous().cuda(),
                as_detections(boxes["xyxy"][keep], boxes["scores"][keep], np.zeros(int(keep.sum()), np.int64)),
            )
            xy.append(keypoints.xy.cpu().numpy())
            scores.append(keypoints.scores.cpu().numpy())
        frames = VideoDecoder(ASSETS / "examples" / "hands_video.mp4", dimension_order="NHWC", device="cuda").get_frames_in_range(0, 32).data
        video = model(frames.contiguous(), as_detections(video_boxes["xyxy"], video_boxes["scores"], video_boxes["frame_indices"]))
    assert_matches(np.concatenate(xy), np.concatenate(scores), read_golden(name), PATHS[name][1:])
    assert_matches(video.xy.cpu().numpy(), video.scores.cpu().numpy(), read_golden(f"{name}_video"), PATHS[name][1:])
