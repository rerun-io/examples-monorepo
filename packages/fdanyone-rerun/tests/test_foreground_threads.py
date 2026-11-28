"""Threaded BiRefNet pre/post-processing gives the masks of the released serial loop (CPU, a stand-in model)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from fdanyone import foreground


class _Model(torch.nn.Module):
    """A deterministic per-pixel 'segmentation': the BiRefNet output contract (a list whose last item is logits)."""

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [images[:, :1].float() * 3.0 - images[:, 1:2].float()]

    def half(self) -> "_Model":
        return self


def _released(frames: tuple[np.ndarray, ...], batch_size: int) -> np.ndarray:
    transform = transforms.Compose(
        [transforms.Resize((64, 48)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    output: list[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        images = [Image.fromarray(frame, mode="RGB") for frame in frames[start : start + batch_size]]
        inputs = torch.stack([transform(image) for image in images]).to(dtype=torch.float16)
        predictions = _Model()(inputs)[-1].sigmoid().cpu()
        for image, prediction in zip(images, predictions, strict=True):
            output.append(np.asarray(to_pil_image(prediction).resize(image.size).convert("L"), dtype=np.uint8).copy())
    return np.stack(output)


def test_threaded_masks_match_the_serial_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transformers

    monkeypatch.setattr(transformers.AutoModelForImageSegmentation, "from_pretrained", lambda *_a, **_k: _Model())
    monkeypatch.setattr(foreground, "FOREGROUND", type("F", (), {"image_size": (64, 48), "batch_size": 4})())
    generator = np.random.default_rng(0)
    frames = tuple(generator.integers(0, 256, (150, 90, 3), dtype=np.uint8) for _ in range(11))
    masks = foreground.predict_foreground_masks(frames, tmp_path, "cpu", batch_size=4)
    np.testing.assert_array_equal(masks, _released(frames, 4))
