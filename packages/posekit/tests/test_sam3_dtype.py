"""SAM 3's torch backend: the model runs under the asked autocast dtype (none for float32), post-processing sees FP32,
and the text encoder runs once per prompt (a stand-in model and processor on the CPU; no weights)."""

from typing import Any, Literal

import pytest
import torch
from torch import Tensor


class FakeOutput(dict):
    """The item access transformers' ModelOutput offers."""


class FakeTextOutput:
    def __init__(self, pooler_output: Tensor) -> None:
        self.pooler_output = pooler_output


class FakeModel:
    def __init__(self) -> None:
        self.autocast: list[torch.dtype | None] = []
        self.text_batches: list[int] = []
        self.text_calls: int = 0

    def get_text_features(self, input_ids: Tensor, attention_mask: Tensor) -> FakeTextOutput:
        self.text_calls += 1
        return FakeTextOutput(torch.ones(1, input_ids.shape[1], 4))

    def __call__(self, pixel_values: Tensor, text_embeds: Any, attention_mask: Tensor) -> FakeOutput:
        self.autocast.append(torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else None)
        self.text_batches.append(int(text_embeds.pooler_output.shape[0]))
        logits: Tensor = torch.ones(pixel_values.shape[0], 1, 4) @ torch.ones(4, 4)  # a matmul: bf16 under autocast
        return FakeOutput(pred_logits=logits, pred_masks=torch.zeros(pixel_values.shape[0], 1, 8, 8))


class FakeTokenizer:
    def __call__(self, *_: object, **__: object) -> "FakeTokens":
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


def fake_segmenter(dtype: Literal["float32", "bfloat16", "float16"]) -> tuple[Any, FakeModel, FakeProcessor]:
    from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig

    model: FakeModel = FakeModel()
    processor: FakeProcessor = FakeProcessor()
    segmenter: Any = object.__new__(Sam3Segmenter)  # the stand-ins are not Sam3Model / Sam3Processor
    segmenter.config = Sam3SegmenterConfig(device="cpu", dtype=dtype)
    segmenter.model = model
    segmenter.processor = processor
    segmenter.input_hw = (16, 16)
    segmenter.text_features = {}
    segmenter.trunk = None
    return segmenter, model, processor


@pytest.mark.parametrize(("dtype", "autocast"), [("float32", None), ("bfloat16", torch.bfloat16), ("float16", torch.float16)])
def test_sam3_runs_under_the_asked_autocast_dtype(dtype: Literal["float32", "bfloat16", "float16"], autocast: torch.dtype | None) -> None:
    from posekit.models.base import SegmentationPrompts

    segmenter, model, processor = fake_segmenter(dtype)

    segmenter(torch.zeros(1, 8, 8, 3, dtype=torch.uint8), SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text="person"))

    assert model.autocast == [autocast]
    assert processor.dtypes == [torch.float32]


def test_sam3_encodes_a_prompt_once_and_expands_it_to_the_batch() -> None:
    from posekit.models.base import SegmentationPrompts

    segmenter, model, _ = fake_segmenter("float16")
    prompts = SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text="person")

    segmenter(torch.zeros(2, 8, 8, 3, dtype=torch.uint8), prompts)
    segmenter(torch.zeros(3, 8, 8, 3, dtype=torch.uint8), prompts)

    assert model.text_calls == 1
    assert model.text_batches == [2, 3]


def test_people_masks_ask_sam3_for_tensorrt_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from posekit.apis.people_masks import MaskOptions, PeopleMasker
    from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig

    configs: list[Sam3SegmenterConfig] = []

    def setup(config: Sam3SegmenterConfig) -> Sam3Segmenter:
        configs.append(config)
        return object.__new__(Sam3Segmenter)

    monkeypatch.setattr(Sam3SegmenterConfig, "setup", setup)

    PeopleMasker(MaskOptions(), "cpu")
    PeopleMasker(MaskOptions(dtype="float32", backend="torch"), "cpu")

    assert [(c.backend, c.dtype) for c in configs] == [("tensorrt", "float16"), ("torch", "float32")]
    assert (Sam3SegmenterConfig().backend, Sam3SegmenterConfig().dtype) == ("torch", "float32")  # other SAM 3 users keep eager FP32
