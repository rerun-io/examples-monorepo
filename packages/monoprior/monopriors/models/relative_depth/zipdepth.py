from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, UInt8

from monopriors.depth_utils import disparity_to_depth, estimate_intrinsics
from monopriors.third_party.zipdepth.architecture import ZipDepth, create_model
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes

from .base_relative_depth import BaseRelativePredictor, RelativeDepthPrediction

ZIPDEPTH_HF_REPO = "pablovela5620/zipdepth"
# Pinned Hub commit: both released checkpoints, MIT license, model card (sha256-verified against the fork).
ZIPDEPTH_HF_REVISION = "0efcd7353cb92af191a99a594d29e66d903c7f3d"


def download_zipdepth_checkpoint(npu: bool = False) -> Path:
    """Released ZipDepth weights; ``npu`` selects the unfold-free upsampling head."""
    filename = "zipdepth_base_npu.pth" if npu else "zipdepth_base.pth"
    return Path(hf_hub_download(repo_id=ZIPDEPTH_HF_REPO, filename=filename, repo_type="model", revision=ZIPDEPTH_HF_REVISION))


def load_zipdepth(checkpoint: Path, npu: bool = False) -> ZipDepth:
    """Build the network from a released or training checkpoint and fuse it for inference.

    Accepts a bare state dict or a trainer checkpoint (``model_state_dict`` + optimizer/scheduler);
    DDP / torch.compile key prefixes are stripped. Loading is strict: the checkpoint must be a
    complete ZipDepth-base.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = strip_state_dict_prefixes(ckpt.get("model_state_dict", ckpt))
    model = create_model(variant="base", upsample_unfold=not npu)
    model.load_state_dict(state_dict, strict=True)
    return model.fuse_for_inference()  # RepVGG re-parameterisation; the fused graph is what the paper benchmarks


class ZipDepthPredictor(BaseRelativePredictor[ZipDepth]):
    """ZipDepth: ~6 M-parameter zero-shot relative inverse depth (ECCV 2026).

    Trained at 384 px; the shorter image side is resized to ``input_size`` (rounded to a multiple
    of 32, aspect kept), the network normalizes internally, and the output is bilinearly resized
    back to the input resolution. Output is relative inverse depth (disparity up to scale).
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
        self.input_size = input_size
        self.model: ZipDepth = load_zipdepth(checkpoint or download_zipdepth_checkpoint(npu=npu), npu=npu).to(device).eval()
        print(f"ZipDepth model loaded. Time: {timer() - start:.2f}s")

    def _network_size(self, h: int, w: int) -> tuple[int, int]:
        scale = self.input_size / min(h, w)
        return tuple(max(32, round(side * scale / 32) * 32) for side in (h, w))  # type: ignore[return-value]

    @torch.no_grad()
    def infer_disparity(self, rgb: UInt8[np.ndarray, "h w 3"]) -> Float32[np.ndarray, "h w"]:
        h, w = rgb.shape[:2]
        net_h, net_w = self._network_size(h, w)
        device = next(self.model.parameters()).device
        resized: UInt8[np.ndarray, "nh nw 3"] = cv2.resize(rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        x: Float32[torch.Tensor, "1 3 nh nw"] = torch.from_numpy(resized).to(device).permute(2, 0, 1)[None].float().div_(255.0)
        disparity: Float32[torch.Tensor, "1 1 nh nw"] = self.model(x).reshape(1, 1, net_h, net_w)
        disparity = F.interpolate(disparity, (h, w), mode="bilinear", align_corners=True)
        return disparity[0, 0].float().cpu().numpy()

    def __call__(
        self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None
    ) -> RelativeDepthPrediction:
        disparity = self.infer_disparity(rgb)
        K_33_f32: Float32[np.ndarray, "3 3"] = (
            estimate_intrinsics(rgb.shape[0], rgb.shape[1]) if K_33 is None else np.asarray(K_33, dtype=np.float32)
        )
        return RelativeDepthPrediction(
            disparity=disparity,
            depth=disparity_to_depth(disparity, focal_length=int(K_33_f32[0, 0])),
            K_33=K_33_f32,
            confidence=None,
        )
