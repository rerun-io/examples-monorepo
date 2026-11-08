"""The tracker's MASt3R forward as a bf16 TensorRT engine.

Tracking runs one forward per frame: encode the new frame, decode it against the keyframe's cached features, and
run both heads. :class:`TrackingForward` is that forward with a fixed image size, exported once to ONNX in bf16
(heads included) and built by trtkit into a machine-local engine. At 224 px on an RTX 5090 the engine replays as a
CUDA graph in 5 ms, against 21 ms eager fp32 and 11 ms for ``torch.compile``; the outputs match fp32 to cosine
0.9995 or better.
"""

from pathlib import Path

import torch
from jaxtyping import Float, Int
from mast3r.model import AsymmetricMASt3R
from torch import Tensor
from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine, export_onnx
from trtkit.artifacts import build_once

TRACKING_INPUT_NAMES: list[str] = ["img_i", "feat_j"]
TRACKING_OUTPUT_NAMES: list[str] = ["feat_i", "X", "C", "D", "Q"]
DEFAULT_ONNX_DIR: Path = Path("checkpoints/onnx")
"""Beside the MASt3R weights the ONNX graph is exported from; the engines go to trtkit's machine-local cache."""


class TrackingForward(torch.nn.Module):
    """Encode frame i, decode it against keyframe j, run both heads; one fixed image size.

    Both frames share the patch grid of that size, so the positions are a buffer instead of inputs. The heads are
    called directly with ``(h, w)``: their landscape wrapper checks ``true_shape`` with a data-dependent
    ``allclose`` that ``torch.export`` cannot trace.
    """

    def __init__(self, model: AsymmetricMASt3R, img_hw: tuple[int, int]) -> None:
        super().__init__()
        self.model: AsymmetricMASt3R = model
        self.img_hw: tuple[int, int] = img_hw
        true_shape: Int[Tensor, "1 2"] = torch.tensor([img_hw], device=next(model.parameters()).device)
        with torch.inference_mode():
            _, pos = model.patch_embed(torch.zeros(1, 3, *img_hw, device=true_shape.device), true_shape=true_shape)
        self.register_buffer("true_shape", true_shape)
        self.register_buffer("pos", pos.clone())

    def forward(
        self, img_i: Float[Tensor, "1 3 h w"], feat_j: Float[Tensor, "1 n_patches feat_dim"]
    ) -> tuple[
        Float[Tensor, "1 n_patches feat_dim"],
        Float[Tensor, "2 h w 3"],
        Float[Tensor, "2 h w"],
        Float[Tensor, "2 h w d"],
        Float[Tensor, "2 h w"],
    ]:
        feat_i, _, _ = self.model._encode_image(img_i, self.true_shape)
        dec_i, dec_j = self.model._decoder(feat_i, self.pos, feat_j, self.pos)
        res_ii: dict[str, Tensor] = self.model.downstream_head1(list(dec_i), self.img_hw)
        res_ji: dict[str, Tensor] = self.model.downstream_head2(list(dec_j), self.img_hw)
        return (
            feat_i,
            torch.cat([res_ii["pts3d"], res_ji["pts3d"]]),
            torch.cat([res_ii["conf"], res_ji["conf"]]),
            torch.cat([res_ii["desc"], res_ji["desc"]]),
            torch.cat([res_ii["desc_conf"], res_ji["desc_conf"]]),
        )


def load_tracking_engine(model: AsymmetricMASt3R, img_hw: tuple[int, int], onnx_dir: Path = DEFAULT_ONNX_DIR) -> tuple[TensorRtRuntime, Tensor]:
    """Export (once), build (once per machine) and load the bf16 tracking engine for one image size.

    The first call on a machine exports the ONNX graph (about 45 s) and builds the engine (about 40 s); later calls
    load the cached engine in well under a second.

    Args:
        model: The MASt3R model, on the GPU.
        img_hw: Height and width of every tracked frame, in pixels.
        onnx_dir: Where the ONNX graph is cached.

    Returns:
        The runtime, replaying as a CUDA graph, and the patch positions it decodes with (``1 n_patches 2``, int64).
    """
    forward: TrackingForward = TrackingForward(model, img_hw).eval()
    onnx_path: Path = onnx_dir / f"mast3r_track_{img_hw[0]}x{img_hw[1]}_bf16.onnx"
    img_i: Float[Tensor, "1 3 h w"] = torch.zeros(1, 3, *img_hw, device=forward.pos.device)
    feat_j: Float[Tensor, "1 n_patches feat_dim"] = torch.zeros(1, forward.pos.shape[1], model.enc_embed_dim, device=forward.pos.device)
    build_once(
        onnx_path,
        lambda: export_onnx(
            forward,
            (img_i, feat_j),
            onnx_path,
            input_names=TRACKING_INPUT_NAMES,
            output_names=TRACKING_OUTPUT_NAMES,
            compute_dtype=torch.bfloat16,
        ),
    )
    engine_path: Path = ensure_engine(onnx_path, TrtBuildConfig(max_batch_size=1, opt_batch_size=1))
    return TensorRtRuntime(engine_path, use_cuda_graph=True), forward.pos
