from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, UInt8

from monopriors.depth_utils import disparity_to_depth, estimate_intrinsics
from monopriors.third_party.zipdepth.architecture import ZipDepth
from monopriors.third_party.zipdepth.predictor import DepthInference

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction

ZIPDEPTH_HF_REPO = "pablovela5620/zipdepth"
# Pinned Hub commit: both released checkpoints, MIT license, model card (sha256-verified against the fork).
ZIPDEPTH_HF_REVISION = "0efcd7353cb92af191a99a594d29e66d903c7f3d"


def download_zipdepth_checkpoint(npu: bool = False) -> Path:
    """Released ZipDepth weights; ``npu`` selects the unfold-free upsampling head."""
    filename = "zipdepth_base_npu.pth" if npu else "zipdepth_base.pth"
    return Path(hf_hub_download(repo_id=ZIPDEPTH_HF_REPO, filename=filename, repo_type="model", revision=ZIPDEPTH_HF_REVISION))


class ZipDepthPredictor(BaseRelativePredictor[ZipDepth]):
    """ZipDepth: ~6 M-parameter zero-shot relative inverse depth (ECCV 2026).

    ``checkpoint`` defaults to the released weights on the Hub; pass a local ``.pth`` (e.g. a
    ``final_model.pth`` written by ``packages/zipdepth`` training — DDP/compile key prefixes are
    stripped on load). Output is relative inverse depth (disparity up to scale) at input resolution.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        checkpoint: Path | None = None,
        input_size: int = 384,
        npu: bool = False,
    ) -> None:
        super().__init__()
        print("Loading ZipDepth model...")
        start = timer()
        if checkpoint is None:
            checkpoint = download_zipdepth_checkpoint(npu=npu)
        # DepthInference owns the network plus its device-bound preprocessing buffers
        # (resize -> BGR2RGB -> [0, 1]; mean/std live inside the model).
        self._inference = DepthInference(
            checkpoint_path=str(checkpoint),
            variant="base",
            device=device,
            input_size=input_size,
            warmup_iters=0,
            upsample_unfold=not npu,
        )
        print(f"ZipDepth model loaded. Time: {timer() - start:.2f}s")

    @property
    def model(self) -> ZipDepth:  # type: ignore[override]
        return self._inference.model

    def set_model_device(self, device: Literal["cpu", "cuda"] = "cuda") -> None:
        # Move the network *and* retarget the runtime's staging buffers, which are lazily
        # reallocated on the new device by DepthInference._ensure_buffers.
        self._inference.model.to(device)
        self._inference.device = device
        self._inference._resize_buf_shape = None

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        # ZipDepth preprocessing expects BGR (cv2.imread convention)
        bgr: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        disparity: Float32[np.ndarray, "h w"] = self._inference.infer_image(bgr)

        K_33_f32: Float32[np.ndarray, "3 3"] = (
            estimate_intrinsics(rgb.shape[0], rgb.shape[1]) if K_33 is None else np.asarray(K_33, dtype=np.float32)
        )
        return RelativeDepthPrediction(
            disparity=disparity,
            depth=disparity_to_depth(disparity, focal_length=int(K_33_f32[0, 0])),
            confidence=np.ones_like(disparity),
            K_33=K_33_f32,
        )
