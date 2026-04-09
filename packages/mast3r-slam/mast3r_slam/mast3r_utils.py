from typing import Any, Literal

import einops
import mast3r.utils.path_to_dust3r  # noqa
import numpy as np
import torch
from jaxtyping import Bool, Float, Float32, Int
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops import conventions
from torch import Tensor

import mast3r_slam.matching as matching
from mast3r_slam.config import config
from mast3r_slam.frame import Frame
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.retrieval_database import RetrievalDatabase


def load_mast3r(
    path: str | None = None,
    device: str = "cuda",
    img_size: int | tuple[int, int] = 512,
) -> AsymmetricMASt3R:
    """Load a pretrained MASt3R model from a checkpoint file.

    Args:
        path: Path to the model checkpoint. Falls back to the default
            ViTLarge checkpoint under ``checkpoints/`` when ``None``.
        device: Torch device string to move the model onto.
        img_size: Input image size expected by the MASt3R head. The installed
            upstream package currently expects a `(height, width)` tuple during
            `from_pretrained(...)` construction.

    Returns:
        The loaded ``AsymmetricMASt3R`` model in eval mode on ``device``.
    """
    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" if path is None else path
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    model = AsymmetricMASt3R.from_pretrained(weights_path, img_size=img_size).to(device)
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
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


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
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res],
        strict=False,
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
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


def xy_grid(
    W: int,
    H: int,
    device: torch.device | str | None = None,
    origin: tuple[int, int] = (0, 0),
    unsqueeze: int | None = None,
    cat_dim: int | None = -1,
    homogeneous: bool = False,
    **arange_kw: Any,
) -> Float[Tensor, "h w 2"] | Float[ndarray, "h w 2"] | Any:
    """Generate a 2D pixel coordinate grid.

    Produces an ``(H, W, 2)`` grid where ``grid[j, i, 0] = i + origin[0]``
    and ``grid[j, i, 1] = j + origin[1]``. When ``homogeneous=True``, the
    last dimension becomes 3 with an appended ones channel.

    Uses numpy when ``device`` is ``None``, otherwise uses torch on the
    specified device.

    Args:
        W: Grid width in pixels.
        H: Grid height in pixels.
        device: Torch device for the output tensor. ``None`` produces a
            numpy array instead.
        origin: ``(x_origin, y_origin)`` offset added to all coordinates.
        unsqueeze: If not ``None``, unsqueeze each coordinate tensor along
            this dimension before stacking.
        cat_dim: Dimension along which to stack the coordinate arrays.
            ``None`` returns the raw tuple of grids.
        homogeneous: When ``True``, append an all-ones channel.
        **arange_kw: Extra keyword arguments forwarded to ``np.arange`` or
            ``torch.arange``.

    Returns:
        A stacked coordinate grid as a numpy array or torch tensor with
        shape ``(H, W, 2)`` (or ``(H, W, 3)`` when ``homogeneous``).
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones  # pyrefly: ignore
    else:
        # torch
        def arange(*a, **kw):  # pyrefly: ignore
            return torch.arange(*a, device=device, **kw)

        meshgrid, stack = torch.meshgrid, torch.stack  # pyrefly: ignore

        def ones(*a):  # pyrefly: ignore
            return torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin, strict=False)]
    grid = meshgrid(tw, th, indexing="xy")  # pyrefly: ignore
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))  # pyrefly: ignore
    if cat_dim is not None:
        grid = stack(grid, cat_dim)  # pyrefly: ignore
    return grid


def estimate_focal_knowing_depth(
    pts3d: Float32[Tensor, "B H W 3"],
    pp: Float[Tensor, "... 2"],
    focal_mode: Literal["median", "weiszfeld"] = "median",
    min_focal: float = 0.0,
    max_focal: float = np.inf,
) -> Float[Tensor, "B"]:
    """Estimate camera focal length from 3D points with known absolute depth.

    Uses either a median-based or Weiszfeld iterative re-weighted least-squares
    estimator to robustly recover the focal length that best explains the
    observed 3D-to-2D reprojection geometry.

    Args:
        pts3d: Predicted 3D point maps of shape ``(B, H, W, 3)``.
        pp: Principal point ``(cx, cy)`` for each batch element.
        focal_mode: Estimation strategy -- ``"median"`` for a direct robust
            median or ``"weiszfeld"`` for IRLS refinement.
        min_focal: Minimum focal length as a multiple of the default baseline
            focal (derived from image size and 60-degree FOV).
        max_focal: Maximum focal length as a multiple of the default baseline
            focal.

    Returns:
        Estimated focal lengths of shape ``(B,)``, clamped to
        ``[min_focal, max_focal]`` scaled by the baseline focal.
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(  # pyrefly: ignore
        -1, 1, 2
    )  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for _iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
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

    pp: Float32[Tensor, "2"] = torch.tensor((W / 2, H / 2))
    assert frame.X_canon is not None
    pts3d: Float32[Tensor, "H W 3"] = frame.X_canon.clone().cpu().reshape(H, W, 3)
    focal: float = float(estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld"))

    return Intrinsics.from_focal_principal_point(
        camera_conventions=camera_conventions,
        fl_x=focal,
        fl_y=focal,
        cx=float(pp[0].item()),
        cy=float(pp[1].item()),
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
