"""Skeleton videos of one pose batch decode side by side but land in camera order (CPU, fake decoder and encoder)."""

import random
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from fdanyone.model import conditioning as pose_conditioning


@dataclass(frozen=True)
class _Skeleton:
    path: Path


class _Conditioning:
    def load_skeleton_tensor(self, skeletons: list[_Skeleton]) -> torch.Tensor:
        (skeleton,) = skeletons
        time.sleep(random.uniform(0.0, 0.05))  # finish out of order
        value = float(skeleton.path.stem)
        return torch.full((1, 3, 4, 2, 2), value, dtype=torch.float32)


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
    job = pose_conditioning._PoseEncodingJob(
        feature_indices=(0, 1, 2, 3, 4), skeletons=skeletons, batch_size=6, packed_views=1, builder=builder
    )

    pose_conditioning._execute_pose_job(job=job, pose_encoder=_Encoder(), conditioning=_Conditioning(), device="cpu")

    batch = builder.stored
    assert batch is not None and batch.shape == (6, 3, 5, 2, 2) and batch.dtype == torch.bfloat16
    for index in range(5):
        assert torch.all(batch[index] == index)
    assert torch.all(batch[5] == -1)  # the packed null slot keeps its fill, prefix included
