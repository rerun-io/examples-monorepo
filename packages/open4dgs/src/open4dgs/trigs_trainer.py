"""Open4DGS's TRiGS trainer: the train-step speed logic, on the seams of recipes/open4dgs-trigs-src.

The recipe keeps only fixes and hooks (patch 0004: host-sync fixes inside ``train_step`` + the ``_const_tensor`` seam;
0005: ``_render`` split into ``_render_inputs`` / ``_rasterize_inputs`` + the ``_camera_group`` seam; 0006: a camera list
in ``_rasterize_inputs``). The logic lives here, one section per lever (this version: lever 1):

1. Host syncs: ``HostSizedCameraConfig`` (Python-int sizes, no ``.item()`` per camera) and ``Open4dgsTrainer._const_tensor``
   (the time and background scalars built once per value instead of a blocking host-to-device copy per call).

Renders equal the per-camera path bit for bit; parameter gradients differ only in float summation order.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor
from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer

CONST_CACHE_LIMIT: int = 4096
"""Entries of the constant-tensor cache before it is cleared (times are per frame: a clip has a few hundred)."""

# ---------------------------------------------------------------------------------------------------------------------
# Lever 1: host syncs
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class HostSizedCameraConfig(CameraConfig):
    """``CameraConfig`` whose ``width_for`` / ``height_for`` return Python ints read once, not ``.item()`` per call
    (a host sync per camera when the sizes are CUDA tensors)."""

    _sizes: list[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sizes = list(zip(self.widths.tolist(), self.heights.tolist(), strict=True))

    def width_for(self, cam_idx: int) -> int:
        return self._sizes[cam_idx][0]

    def height_for(self, cam_idx: int) -> int:
        return self._sizes[cam_idx][1]


class Open4dgsTrainer(TrigsTrainer):
    """``TrigsTrainer`` with the host-sync lever on the recipe's ``_const_tensor`` seam."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # pyrefly: ignore[bad-argument-type]
        self._const_cache: dict[tuple[object, str, torch.dtype], Tensor] = {}

    def _const_tensor(self, value: object, device: torch.device | str, dtype: torch.dtype) -> Tensor:
        """Lever 1: ``torch.tensor(value)`` on ``device``, built once per (value, device, dtype); never modify it in place."""
        key = (value, str(device), dtype)
        out = self._const_cache.get(key)
        if out is None:
            if len(self._const_cache) >= CONST_CACHE_LIMIT:
                self._const_cache.clear()
            out = self._const_cache[key] = torch.tensor(value, device=device, dtype=dtype)
        return out
