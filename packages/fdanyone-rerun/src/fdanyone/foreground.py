"""Pinned BiRefNet inference over the canonical source clip."""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from fdanyone.config import FOREGROUND

FOREGROUND_CPU_THREADS: int = 8
"""Threads for BiRefNet's per-frame PIL pre- and post-processing."""


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

    def prepare(frame: np.ndarray):
        return transform(Image.fromarray(frame, mode="RGB"))

    def finish(prediction) -> np.ndarray:
        mask = to_pil_image(prediction).resize((shape[1], shape[0])).convert("L")
        return np.asarray(mask, dtype=np.uint8).copy()

    # PIL resizes without the GIL: the 4K -> 1024 px inputs and the 1024 px -> 4K masks (~80 ms a frame each) run on
    # threads, ahead of and behind the GPU batches, with the same per-frame operations.
    output: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=FOREGROUND_CPU_THREADS, thread_name_prefix="foreground") as pool:
        prepared = pool.map(prepare, frames)
        finished = []
        try:
            for start in range(0, len(frames), batch_size):
                batch = [next(prepared) for _ in range(min(batch_size, len(frames) - start))]
                inputs = torch.stack(batch).to(device=device, dtype=torch.float16)
                with torch.inference_mode():
                    predictions = model(inputs)[-1].sigmoid().cpu()
                finished.extend(pool.submit(finish, prediction) for prediction in predictions)
                del inputs, predictions
            output = [future.result() for future in finished]
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return np.stack(output)
