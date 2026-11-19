"""SAM 3's bf16 switch: the model runs under bfloat16 autocast only when asked, and post-processing sees FP32
(a stand-in model and processor on the CPU; no weights)."""

import pytest
import torch
from torch import Tensor


class FakeOutput(dict):
    """The item access transformers' ModelOutput offers."""


class FakeModel:
    def __init__(self) -> None:
        self.autocast: list[bool] = []

    def __call__(self, pixel_values: Tensor, input_ids: Tensor, attention_mask: Tensor) -> FakeOutput:
        self.autocast.append(torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16)
        logits: Tensor = torch.ones(pixel_values.shape[0], 1, 4) @ torch.ones(4, 4)  # a matmul: bf16 under autocast
        return FakeOutput(pred_logits=logits, pred_masks=torch.zeros(pixel_values.shape[0], 1, 8, 8))


class FakeTokenizer:
    def __call__(self, texts: list[str], return_tensors: str) -> "FakeTokens":
        return FakeTokens()


class FakeTokens(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.ones(1, 3, dtype=torch.int64), attention_mask=torch.ones(1, 3, dtype=torch.int64))

    def to(self, device: torch.device) -> "FakeTokens":
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.dtypes: list[torch.dtype] = []

    def post_process_instance_segmentation(self, outputs: FakeOutput, **_: object) -> list[dict[str, Tensor]]:
        self.dtypes.append(outputs["pred_logits"].dtype)
        return [{"scores": torch.zeros(0), "boxes": torch.zeros(0, 4), "masks": torch.zeros(0, 8, 8)}]


@pytest.mark.parametrize("bf16", [False, True])
def test_sam3_runs_under_bf16_autocast_only_when_asked(bf16: bool) -> None:
    from posekit.models.base import SegmentationPrompts
    from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig

    segmenter: Sam3Segmenter = object.__new__(Sam3Segmenter)
    segmenter.config = Sam3SegmenterConfig(device="cpu", bf16=bf16)
    segmenter.model = FakeModel()
    segmenter.processor = FakeProcessor()
    segmenter.input_hw = (16, 16)

    segmenter(torch.zeros(1, 8, 8, 3, dtype=torch.uint8), SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text="person"))

    assert segmenter.model.autocast == [bf16]
    assert segmenter.processor.dtypes == [torch.float32]


def test_people_masks_ask_sam3_for_bf16_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from posekit.apis.people_masks import MaskOptions, PeopleMasker
    from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig

    configs: list[Sam3SegmenterConfig] = []

    def setup(config: Sam3SegmenterConfig) -> Sam3Segmenter:
        configs.append(config)
        return object.__new__(Sam3Segmenter)

    monkeypatch.setattr(Sam3SegmenterConfig, "setup", setup)

    PeopleMasker(MaskOptions(), "cpu")
    PeopleMasker(MaskOptions(bf16=False), "cpu")

    assert [c.bf16 for c in configs] == [True, False]
    assert Sam3SegmenterConfig().bf16 is False  # other SAM 3 users keep FP32
