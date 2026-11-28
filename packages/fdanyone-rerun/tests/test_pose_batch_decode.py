"""Skeleton videos of one pose batch decode side by side but land in camera order (CPU, fake decoder and encoder)."""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from fdanyone.model import conditioning as pose_conditioning


@dataclass(frozen=True)
class _Skeleton:
    path: Path


class _Conditioning:
    def decode_skeleton_frames(self, skeleton: _Skeleton) -> list[torch.Tensor]:
        time.sleep(random.uniform(0.0, 0.05))  # finish out of order
        return [torch.full((2, 2, 3), int(skeleton.path.stem), dtype=torch.uint8) for _ in range(4)]

    def prepare_skeleton(self, frames: list[torch.Tensor], _skeleton: _Skeleton) -> torch.Tensor:
        return torch.stack(frames).permute(3, 0, 1, 2).float()


class _Encoder:
    temporal_prefix = 1


@dataclass
class _Builder:
    stored: torch.Tensor | None = None

    def store(self, _job: object, encoded: torch.Tensor) -> None:
        self.stored = encoded


def test_batch_rows_follow_camera_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pose_conditioning, "_encode_pose_batch", lambda _encoder, batch: batch.clone())
    skeletons = tuple(_Skeleton(Path(f"{index}.mp4")) for index in range(5))
    builder = _Builder()
    # The fakes stand in for the job's collaborators by duck typing; the job only calls the methods they define.
    job = pose_conditioning._PoseEncodingJob(
        feature_indices=(0, 1, 2, 3, 4), skeletons=cast(Any, skeletons), batch_size=6, packed_views=1, builder=cast(Any, builder)
    )

    pose_conditioning._execute_pose_job(job=job, pose_encoder=_Encoder(), conditioning=cast(Any, _Conditioning()), device="cpu")

    batch = builder.stored
    assert batch is not None and batch.shape == (6, 3, 5, 2, 2) and batch.dtype == torch.bfloat16
    for index in range(5):
        assert torch.all(batch[index] == index)
    assert torch.all(batch[5] == -1)  # the packed null slot keeps its fill, prefix included
