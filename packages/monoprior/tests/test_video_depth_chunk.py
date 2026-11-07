"""process_video_chunk preprocesses only the frames that reach the model; the model input stays bit-identical."""

from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from torch import Tensor, nn

from monopriors.models.relative_depth.video_depth_anything import VideoDepthAnythingPredictor
from monopriors.third_party.video_depth_anything.video_depth import INFER_LEN, KEYFRAMES, OVERLAP

H, W = 6, 8


class _MeanDepth(nn.Module):
    """Stand-in for VideoDepthAnything: one depth map per frame, the channel mean."""

    def forward(self, x: Float32[Tensor, "1 t 3 h w"]) -> Float32[Tensor, "1 t h w"]:
        return x.mean(dim=2)


def _counting_transform(calls: list[int]) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    def transform(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append(1)
        image: Float32[np.ndarray, "h w 3"] = sample["image"]
        return {"image": np.ascontiguousarray(((image - 0.5) / 0.25).transpose(2, 0, 1)).astype(np.float32)}

    return transform


def _predictor() -> VideoDepthAnythingPredictor:
    predictor: VideoDepthAnythingPredictor = VideoDepthAnythingPredictor.__new__(VideoDepthAnythingPredictor)
    predictor.model = _MeanDepth()
    predictor.device = "cpu"
    return predictor


def _frames(n: int) -> list[UInt8[np.ndarray, "h w 3"]]:
    rng: np.random.Generator = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(n)]


def test_later_chunk_skips_overwritten_frames_and_keeps_input_bit_identical() -> None:
    frames: list[UInt8[np.ndarray, "h w 3"]] = _frames(2 * INFER_LEN)
    predictor: VideoDepthAnythingPredictor = _predictor()
    calls: list[int] = []
    transform = _counting_transform(calls)

    _, first_input = predictor.process_video_chunk(frames, 0, transform, "cpu", True, None, H, W)
    assert len(calls) == INFER_LEN

    calls.clear()
    frame_id: int = INFER_LEN - OVERLAP
    _, second_input = predictor.process_video_chunk(frames, frame_id, transform, "cpu", True, first_input, H, W)
    assert len(calls) == INFER_LEN - OVERLAP

    # Reference: what the model saw before the change (every frame preprocessed, then the first OVERLAP overwritten).
    expected: Float32[Tensor, "1 t 3 h w"] = torch.stack(
        [torch.from_numpy(transform({"image": frames[frame_id + i].astype(np.float32) / 255.0})["image"]) for i in range(INFER_LEN)]
    )[None]
    expected[:, :OVERLAP] = first_input[:, KEYFRAMES]
    assert second_input.shape == expected.shape
    assert torch.equal(second_input, expected)
