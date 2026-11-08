from collections.abc import Callable
from typing import Literal

import einops
import mast3r.utils.path_to_dust3r  # noqa
import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, Int
from mast3r.model import AsymmetricMASt3R
from models.pos_embed import RoPE2D
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops import conventions
from torch import Tensor

import mast3r_slam.matching as matching
from mast3r_slam.config import config
from mast3r_slam.frame import Frame
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.retrieval_database import RetrievalDatabase


class FixedLengthRoPE2D(RoPE2D):  # pyrefly: ignore  # croco aliases RoPE2D to cuRoPE2D when that extension is built
    """croco's pytorch RoPE2D with its cos/sin table sized once, for the largest patch grid the model takes.

    The stock module sizes the table from ``int(positions.max())`` on every call: a GPU->CPU sync per query and
    key in every attention layer, about 190 per tracked frame, each of which stops the CPU from queueing kernels
    ahead. Rows of the table do not depend on its length, so the output is the same.
    """

    def __init__(self, freq: float, F0: float, max_positions: int) -> None:
        super().__init__(freq, F0)
        self.max_positions: int = max_positions

    def forward(self, tokens: Float[Tensor, "b h n d"], positions: Int[Tensor, "b n 2"]) -> Float[Tensor, "b h n d"]:
        cos, sin = self.get_cos_sin(tokens.size(3) // 2, self.max_positions, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


def load_mast3r(path: str | None = None, device: str = "cuda") -> AsymmetricMASt3R:
    """Load a pretrained MASt3R model from a checkpoint file.

    Args:
        path: Path to the model checkpoint. Falls back to the default
            ViTLarge checkpoint under ``checkpoints/`` when ``None``.
        device: Torch device string to move the model onto.

    Returns:
        The loaded ``AsymmetricMASt3R`` model in eval mode on ``device``.
    """
    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" if path is None else path
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    # Without croco's compiled cuRoPE2D the model falls back to the pytorch RoPE2D; swap in the sync-free one.
    if type(model.rope) is RoPE2D:
        max_positions: int = int(max(model.patch_embed.img_size)) // int(model.patch_embed.patch_size[0])  # pyrefly: ignore
        rope: FixedLengthRoPE2D = FixedLengthRoPE2D(model.rope.base, model.rope.F0, max_positions)
        for module in model.modules():
            if getattr(module, "rope", None) is not None:
                module.rope = rope
    return model


def load_retriever(
    mast3r_model: AsymmetricMASt3R,
    retriever_path: str | None = None,
    device: str = "cuda",
) -> RetrievalDatabase:
    """Load an image retrieval database backed by a MASt3R backbone.

    Args:
        mast3r_model: The MASt3R model whose encoder is used for feature
            extraction inside the retrieval pipeline.
        retriever_path: Path to the retrieval checkpoint. Falls back to the
            default training-free retrieval checkpoint when ``None``.
        device: Torch device string for the retrieval model.

    Returns:
        A ``RetrievalDatabase`` ready for incremental keyframe insertion and
        nearest-neighbor queries.
    """
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode()
def decoder(
    model: AsymmetricMASt3R,
    feat1: Float[Tensor, "1 n_patches feat_dim"],
    feat2: Float[Tensor, "1 n_patches feat_dim"],
    pos1: Int[Tensor, "1 n_patches 2"],
    pos2: Int[Tensor, "1 n_patches 2"],
    shape1: Int[Tensor, "1 2"],
    shape2: Int[Tensor, "1 2"],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Run the MASt3R decoder on a pair of encoded feature maps.

    Decodes the cross-attended features and applies the downstream prediction
    heads (3D point maps, confidence, descriptors) with AMP disabled for
    numerical stability.

    Args:
        model: The MASt3R model providing ``_decoder`` and ``_downstream_head``.
        feat1: Encoded feature tokens for image 1.
        feat2: Encoded feature tokens for image 2.
        pos1: Positional encodings for image 1 patches.
        pos2: Positional encodings for image 2 patches.
        shape1: True (height, width) of image 1.
        shape2: True (height, width) of image 2.

    Returns:
        A pair of result dicts ``(res1, res2)``, each containing keys
        ``"pts3d"``, ``"conf"``, ``"desc"``, and ``"desc_conf"``.
    """
    return _decode_heads(model, feat1, feat2, pos1, pos2, shape1, shape2)


def _decode_heads(
    model: AsymmetricMASt3R,
    feat1: Float[Tensor, "1 n_patches feat_dim"],
    feat2: Float[Tensor, "1 n_patches feat_dim"],
    pos1: Int[Tensor, "1 n_patches 2"],
    pos2: Int[Tensor, "1 n_patches 2"],
    shape1: Int[Tensor, "1 2"],
    shape2: Int[Tensor, "1 2"],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """:func:`decoder` without its ``inference_mode`` wrapper, so ``torch.compile`` can trace it."""
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def _stack_results(res: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(X, C, D, Q)``, each stacked over ``res`` along a new first dim."""
    return (
        torch.stack([r["pts3d"][0] for r in res]),
        torch.stack([r["conf"][0] for r in res]),
        torch.stack([r["desc"][0] for r in res]),
        torch.stack([r["desc_conf"][0] for r in res]),
    )


_COMPILED_TRACKING_FORWARD: dict[int, Callable[..., tuple[Tensor, ...]]] = {}
"""Compiled encode + decode for tracking, keyed by ``id(model)``; see :func:`compile_tracking_forward`."""


def compile_tracking_forward(model: AsymmetricMASt3R) -> None:
    """Run the tracker's forward (encode the new frame, decode it against the keyframe) as CUDA graphs.

    ``torch.compile(mode="reduce-overhead")`` fuses the small kernels (RoPE among them) and replays the whole
    forward as CUDA graphs, removing the launch overhead that bounds it: 21 -> 11 ms at 224 px on an RTX 5090.
    The first tracked frame pays the compile (about 30 s cold, 14 s with a warm inductor cache), so this is for
    long runs. Outputs match the eager forward to cosine 0.999998.
    """

    def encode_decode(
        img_i: Float[Tensor, "1 3 h w"],
        shape_i: Int[Tensor, "1 2"],
        feat_j: Float[Tensor, "1 n_patches feat_dim"],
        pos_j: Int[Tensor, "1 n_patches 2"],
        shape_j: Int[Tensor, "1 2"],
    ) -> tuple[Tensor, ...]:
        feat_i, pos_i, _ = model._encode_image(img_i, shape_i)
        res11, res21 = _decode_heads(model, feat_i, feat_j, pos_i, pos_j, shape_i, shape_j)
        return (feat_i, pos_i, *_stack_results([res11, res21]))

    _COMPILED_TRACKING_FORWARD[id(model)] = torch.compile(encode_decode, mode="reduce-overhead", dynamic=False)


def downsample(
    X: Float[Tensor, "*batch h w 3"],
    C: Float[Tensor, "*batch h w"],
    D: Float[Tensor, "*batch h w d"],
    Q: Float[Tensor, "*batch h w"],
) -> tuple[
    Float[Tensor, "*batch h2 w2 3"],
    Float[Tensor, "*batch h2 w2"],
    Float[Tensor, "*batch h2 w2 d"],
    Float[Tensor, "*batch h2 w2"],
]:
    """Spatially downsample point maps, confidence, and descriptors.

    Uses the ``config["dataset"]["img_downsample"]`` factor for stride-based
    subsampling. When the factor is 1, the tensors are returned unchanged.

    Args:
        X: 3D point map with trailing shape ``(..., H, W, 3)``.
        C: Confidence map with trailing shape ``(..., H, W)``.
        D: Descriptor map with trailing shape ``(..., H, W, d)``.
        Q: Descriptor confidence map with trailing shape ``(..., H, W)``.

    Returns:
        A tuple ``(X, C, D, Q)`` downsampled by the configured factor.
    """
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode()
def mast3r_decode_symmetric_batch(
    model: AsymmetricMASt3R,
    feat_i: Float[Tensor, "b n_patches feat_dim"],
    pos_i: Int[Tensor, "b n_patches 2"],
    feat_j: Float[Tensor, "b n_patches feat_dim"],
    pos_j: Int[Tensor, "b n_patches 2"],
    shape_i: list[Int[Tensor, "1 2"]],
    shape_j: list[Int[Tensor, "1 2"]],
) -> tuple[
    Float[Tensor, "4 b h w 3"],
    Float[Tensor, "4 b h w"],
    Float[Tensor, "4 b h w d"],
    Float[Tensor, "4 b h w"],
]:
    """Batched symmetric decode over a batch of pre-encoded feature pairs.

    Iterates over the batch dimension and decodes each pair symmetrically,
    stacking results into ``(4, B, h, w, ...)`` tensors. The leading dimension
    of 4 corresponds to ``[res_ii, res_ji, res_jj, res_ij]``.

    Args:
        model: The MASt3R model.
        feat_i: Batch of encoded features for frames i.
        pos_i: Batch of positional encodings for frames i.
        feat_j: Batch of encoded features for frames j.
        pos_j: Batch of positional encodings for frames j.
        shape_i: True image shapes for frames i.
        shape_j: True image shapes for frames j.

    Returns:
        A tuple ``(X, C, D, Q)`` each with shape ``(4, B, h, w, ...)``.
    """
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res],
            strict=False,
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode()
def mast3r_inference_mono(model: AsymmetricMASt3R, frame: Frame) -> tuple[Float[Tensor, "hw 3"], Float[Tensor, "hw 1"]]:
    """Run monocular self-inference on a single frame.

    Decodes the frame against itself (symmetric self-pair) to obtain a
    canonical 3D point map and confidence for frame initialization.

    Args:
        model: The MASt3R model.
        frame: The input frame (features cached on ``frame.feat``).

    Returns:
        A tuple ``(Xii, Cii)`` where ``Xii`` is the self-predicted 3D point
        map of shape ``(1, h*w, 3)`` and ``Cii`` is the per-point confidence
        of shape ``(1, h*w, 1)``.
    """
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.rgb_tensor, frame.img_true_shape)

    assert frame.pos is not None
    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res],
        strict=False,
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(
    model: AsymmetricMASt3R,
    feat_i: Float[Tensor, "b n_patches feat_dim"],
    pos_i: Int[Tensor, "b n_patches 2"],
    feat_j: Float[Tensor, "b n_patches feat_dim"],
    pos_j: Int[Tensor, "b n_patches 2"],
    shape_i: list[Int[Tensor, "1 2"]],
    shape_j: list[Int[Tensor, "1 2"]],
) -> tuple[
    Int[Tensor, "b hw"],
    Int[Tensor, "b hw"],
    Bool[Tensor, "b hw 1"],
    Bool[Tensor, "b hw 1"],
    Float[Tensor, "b hw 1"],
    Float[Tensor, "b hw 1"],
    Float[Tensor, "b hw 1"],
    Float[Tensor, "b hw 1"],
]:
    """Symmetric matching between a batch of frame pairs.

    Decodes both directions symmetrically and computes dense correspondences
    using iterative projection matching. Returns bidirectional match indices,
    validity masks, and descriptor confidences.

    Args:
        model: The MASt3R model.
        feat_i: Encoded features for frames i.
        pos_i: Positional encodings for frames i.
        feat_j: Encoded features for frames j.
        pos_j: Positional encodings for frames j.
        shape_i: True image shapes for frames i.
        shape_j: True image shapes for frames j.

    Returns:
        An 8-tuple of ``(idx_i2j, idx_j2i, valid_match_j, valid_match_i,
        Qii, Qjj, Qji, Qij)`` where ``idx_*`` are linear pixel-index
        correspondences, ``valid_match_*`` are boolean validity masks, and
        ``Q*`` are per-pixel descriptor confidences.
    """
    X, C, D, Q = mast3r_decode_symmetric_batch(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j)

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode()
def mast3r_asymmetric_inference(
    model: AsymmetricMASt3R, frame_i: Frame, frame_j: Frame
) -> tuple[
    Float[Tensor, "2 h w 3"],
    Float[Tensor, "2 h w"],
    Float[Tensor, "2 h w d"],
    Float[Tensor, "2 h w"],
]:
    """Run asymmetric two-frame MASt3R inference (i predicts j only).

    Encodes both frames (lazily) and decodes only in the i->j direction,
    producing two sets of outputs: self-prediction for i and cross-prediction
    of j in i's coordinate frame.

    Args:
        model: The MASt3R model.
        frame_i: Reference frame (features cached on ``frame_i.feat``).
        frame_j: Target frame (features cached on ``frame_j.feat``).

    Returns:
        A tuple ``(X, C, D, Q)`` each of shape ``(2, h, w, ...)``, where
        dim 0 corresponds to ``[res_ii, res_ji]``.
    """
    compiled: Callable[..., tuple[Tensor, ...]] | None = _COMPILED_TRACKING_FORWARD.get(id(model))
    if compiled is not None and frame_i.feat is None and frame_j.feat is not None:
        torch.compiler.cudagraph_mark_step_begin()
        # Clone: the next replay overwrites the graph's output buffers.
        outputs: list[Tensor] = [t.clone() for t in compiled(frame_i.rgb_tensor, frame_i.img_true_shape, frame_j.feat, frame_j.pos, frame_j.img_true_shape)]
        frame_i.feat, frame_i.pos = outputs[0], outputs[1]
        return downsample(outputs[2], outputs[3], outputs[4], outputs[5])
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(frame_i.rgb_tensor, frame_i.img_true_shape)
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(frame_j.rgb_tensor, frame_j.img_true_shape)

    assert frame_i.pos is not None
    assert frame_j.pos is not None
    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    X, C, D, Q = _stack_results([res11, res21])
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


def mast3r_match_asymmetric(
    model: AsymmetricMASt3R,
    frame_i: Frame,
    frame_j: Frame,
    idx_i2j_init: Int[Tensor, "b hw"] | None = None,
) -> tuple[
    Int[Tensor, "b hw"],
    Bool[Tensor, "b hw 1"],
    Float[Tensor, "hw 3"],
    Float[Tensor, "hw 1"],
    Float[Tensor, "hw 1"],
    Float[Tensor, "hw 3"],
    Float[Tensor, "hw 1"],
    Float[Tensor, "hw 1"],
]:
    """Asymmetric matching from frame i to frame j.

    Runs asymmetric inference and computes dense correspondences using
    iterative projection matching. Returns match indices, validity masks,
    and flattened point maps / confidences for downstream pose estimation.

    Args:
        model: The MASt3R model.
        frame_i: Reference frame.
        frame_j: Target frame.
        idx_i2j_init: Optional initial correspondence guess (linear pixel
            indices) to warm-start the iterative matcher.

    Returns:
        An 8-tuple ``(idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji)``
        where ``idx_i2j`` are i-to-j correspondences, ``valid_match_j`` is a
        validity mask, and the remaining tensors are flattened ``(b, h*w, ...)``
        point maps, confidences, and descriptor confidences.
    """
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init)

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def estimate_focal_knowing_depth(pts3d_hw3: Float32[ndarray, "h w 3"], cx: float, cy: float) -> float:
    """Focal length (pixels, square pixels) that best reprojects a pointmap onto its pixel grid.

    Weiszfeld's iteratively re-weighted least squares on ``|pixel - f * (x, y) / z|``, started from the L2 closed
    form (DUSt3R's ``estimate_focal_knowing_depth``, ``focal_mode="weiszfeld"``). NumPy, single-threaded: inside a
    run, torch's CPU thread pool contends with tracking and the same math took 50-175 ms per call instead of a few.

    Args:
        pts3d_hw3: 3D points in the camera frame, one per pixel.
        cx: Principal point x, pixels.
        cy: Principal point y, pixels.
    """
    height: int = pts3d_hw3.shape[0]
    width: int = pts3d_hw3.shape[1]
    u_hw: Float32[ndarray, "h w"]
    v_hw: Float32[ndarray, "h w"]
    u_hw, v_hw = np.meshgrid(np.arange(width, dtype=np.float32) - cx, np.arange(height, dtype=np.float32) - cy, indexing="xy")
    pixels: Float32[ndarray, "n 2"] = np.stack([u_hw.ravel(), v_hw.ravel()], axis=-1)
    points: Float32[ndarray, "n 3"] = pts3d_hw3.reshape(-1, 3).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy_over_z: Float32[ndarray, "n 2"] = np.nan_to_num(points[:, :2] / points[:, 2:3], nan=0.0, posinf=0.0, neginf=0.0)
    dot_xy_px: Float32[ndarray, "n"] = (xy_over_z * pixels).sum(axis=-1)
    dot_xy_xy: Float32[ndarray, "n"] = np.square(xy_over_z).sum(axis=-1)
    focal: float = float(dot_xy_px.mean() / dot_xy_xy.mean())
    for _ in range(10):
        weights: Float32[ndarray, "n"] = 1.0 / np.maximum(np.linalg.norm(pixels - focal * xy_over_z, axis=-1), 1e-8)
        focal = float((weights * dot_xy_px).mean() / (weights * dot_xy_xy).mean())
    return focal


def frame_to_intri(frame: Frame, camera_conventions: Literal["RDF", "RUB"] = "RDF") -> Intrinsics:
    """Estimate camera intrinsics from a Frame's 3D point map.

    Args:
        frame: A Frame with valid ``X_canon`` and ``img_shape``.
        camera_conventions: Output camera axis convention for the returned intrinsics.

    Returns:
        An ``Intrinsics`` object in the requested convention.
    """
    H: int = int(frame.img_shape.squeeze()[0].item())
    W: int = int(frame.img_shape.squeeze()[1].item())

    assert frame.X_canon is not None
    pts3d: Float32[ndarray, "H W 3"] = frame.X_canon.detach().cpu().numpy().reshape(H, W, 3)
    focal: float = estimate_focal_knowing_depth(pts3d, cx=W / 2, cy=H / 2)

    return Intrinsics.from_focal_principal_point(
        camera_conventions=camera_conventions,
        fl_x=focal,
        fl_y=focal,
        cx=W / 2,
        cy=H / 2,
        height=H,
        width=W,
    )


def frame_to_extrinsics(frame: Frame, camera_conventions: Literal["RDF", "RUB"] = "RDF") -> Extrinsics:
    """Convert a Frame's lietorch Sim3 pose to simplecv Extrinsics in the requested convention.

    Args:
        frame: A Frame with a valid ``world_sim3_cam`` pose.
        camera_conventions: Output camera axis convention for the returned extrinsics.

    Returns:
        An ``Extrinsics`` in the requested convention.
    """
    se3 = as_SE3(frame.world_sim3_cam.cpu())
    mat4x4_cv: Float32[ndarray, "4 4"] = se3.matrix().numpy().astype(np.float32)[0]
    if camera_conventions == "RDF":
        mat4x4_out: Float32[ndarray, "4 4"] = mat4x4_cv
    else:
        mat4x4_out = conventions.convert_pose(
            mat4x4_cv, src_convention=conventions.CC.CV, dst_convention=conventions.CC.GL
        )
    return Extrinsics(
        world_R_cam=mat4x4_out[:3, :3],
        world_t_cam=mat4x4_out[:3, 3],
    )


def frame_to_pinhole(frame: Frame, camera_conventions: Literal["RDF", "RUB"] = "RDF") -> PinholeParameters:
    """Convert a Frame into a simplecv PinholeParameters.

    Estimates focal length from the frame's 3D point map, converts the
    lietorch Sim3 pose to the requested camera convention, and
    packages everything into a ``PinholeParameters`` for use with
    ``simplecv.rerun_log_utils.log_pinhole``.

    Args:
        frame: A Frame with a valid ``X_canon`` point map and ``world_sim3_cam`` pose.
        camera_conventions: Output camera axis convention for the returned pinhole.

    Returns:
        A ``PinholeParameters`` in the requested convention.
    """
    extrinsics: Extrinsics = frame_to_extrinsics(frame, camera_conventions=camera_conventions)
    intrinsics: Intrinsics = frame_to_intri(frame, camera_conventions=camera_conventions)
    return PinholeParameters(
        name=f"frame-{frame.frame_id}",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
