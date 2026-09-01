"""Shared batching and tensor preparation for Polycam completion tools.

Every completion predictor sees the raw prompt. PromptDA min/max-normalizes it
without a mask, and the student was trained to match that behavior.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched, chain, islice
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Float32, UInt8
from simplecv.data.polycam import PolycamData, PolycamDataset
from torch import Tensor
from tqdm import tqdm


@dataclass(frozen=True, slots=True)
class PolycamBatchPlan:
    """Peeked first batch and one-shot bounded iterator for one capture."""

    first_batch: tuple[PolycamData, ...]
    """First batch, exposed for predictor sizing before iteration."""
    total_batches: int
    """Ceiling-divided batch count shown by the progress bar."""
    _remaining_batches: object
    _description: str

    def __iter__(self) -> Iterator[tuple[PolycamData, ...]]:
        """Yield all bounded batches, including ``first_batch``, with progress."""
        remaining_batches = cast(Iterable[tuple[PolycamData, ...]], self._remaining_batches)
        yield from tqdm(
            chain((self.first_batch,), remaining_batches),
            desc=self._description,
            total=self.total_batches,
        )


@dataclass(frozen=True, slots=True)
class PolycamTensorBatch:
    """GPU inputs shared by all completion predictors."""

    rgb_bhw3: UInt8[Tensor, "b h w 3"]
    """Raw RGB images on CUDA."""
    prompt_bhw: Float32[Tensor, "b 192 256"]
    """Raw Polycam depth prompt in metres on CUDA."""


def prepare_polycam_batches(
    dataset: PolycamDataset,
    *,
    batch_size: int,
    max_frames: int | None,
    capture_path: Path,
    description: str,
) -> PolycamBatchPlan:
    """Peek and progress-wrap batches while enforcing an exact frame budget."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    frame_count: int = min(max_frames if max_frames is not None else len(dataset), len(dataset))
    batch_iterator = batched(islice(dataset, frame_count), batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batch_iterator, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {capture_path} contains no frames.")

    total_batches: int = -(-frame_count // batch_size)
    return PolycamBatchPlan(
        first_batch=first_batch,
        total_batches=total_batches,
        _remaining_batches=batch_iterator,
        _description=description,
    )


def stack_polycam_batch(batch: tuple[PolycamData, ...]) -> PolycamTensorBatch:
    """Stack one batch into raw CUDA RGB and metric-depth prompt tensors."""
    prompt_m_bhw: Float32[np.ndarray, "b 192 256"] = np.stack([data.original_depth_hw for data in batch]).astype(np.float32)
    prompt_m_bhw /= 1000.0
    rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch])).cuda()
    prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(prompt_m_bhw).cuda()
    return PolycamTensorBatch(rgb_bhw3=rgb_bhw3, prompt_bhw=prompt_bhw)


__all__ = ("PolycamBatchPlan", "PolycamTensorBatch", "prepare_polycam_batches", "stack_polycam_batch")
