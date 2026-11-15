"""Pinned BiRefNet inference over the canonical source clip."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image

from fdanyone.config import FOREGROUND


def predict_foreground_masks(
    frames: tuple[np.ndarray, ...],
    model_path: str | Path,
    device: str,
    *,
    batch_size: int = FOREGROUND.batch_size,
) -> np.ndarray:
    """Return full-raster 8-bit foreground masks for the canonical clip."""

    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from transformers import AutoModelForImageSegmentation

    if not frames:
        raise ValueError("Foreground inference requires at least one frame.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    shape = frames[0].shape
    if any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames):
        raise ValueError("Foreground frames must share one RGB uint8 raster.")

    model = AutoModelForImageSegmentation.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.eval().half().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(FOREGROUND.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    output: list[np.ndarray] = []
    try:
        for start in range(0, len(frames), batch_size):
            images = [Image.fromarray(frame, mode="RGB") for frame in frames[start : start + batch_size]]
            inputs = torch.stack([transform(image) for image in images]).to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                predictions = model(inputs)[-1].sigmoid().cpu()
            for image, prediction in zip(images, predictions, strict=True):
                mask = to_pil_image(prediction).resize(image.size).convert("L")
                output.append(np.asarray(mask, dtype=np.uint8).copy())
            del inputs, predictions
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(output)
