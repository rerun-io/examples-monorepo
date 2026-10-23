"""MASt3R (Model layer): encoder, decoder and heads (point maps, confidences, descriptors) as the SLAM calls them."""

import einops
import torch
from jaxtyping import Float, Int
from mast3r.model import AsymmetricMASt3R
from torch import Tensor

from mast3r_slam.types import Frame


def load_mast3r(device: str = "cuda") -> AsymmetricMASt3R:
    """The pretrained ViT-Large checkpoint under ``checkpoints/`` (relative to the working directory) on ``device``."""
    return AsymmetricMASt3R.from_pretrained("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").to(device)


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
    """The decoder and the heads on a pair of encoded images: ``pts3d``, ``conf``, ``desc``, ``desc_conf`` per image (heads without AMP)."""
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


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
    """Both directions of every encoded pair: ``(X, C, D, Q)``, dim 0 = ``[res_ii, res_ji, res_jj, res_ij]``."""
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
    return X, C, D, Q


@torch.inference_mode()
def mast3r_inference_mono(model: AsymmetricMASt3R, frame: Frame) -> tuple[Float[Tensor, "hw 3"], Float[Tensor, "hw 1"]]:
    """The frame decoded against itself (encoded once, cached on ``frame.feat``): its canonical point map and confidence."""
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

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


@torch.inference_mode()
def mast3r_asymmetric_inference(
    model: AsymmetricMASt3R, frame_i: Frame, frame_j: Frame
) -> tuple[
    Float[Tensor, "2 h w 3"],
    Float[Tensor, "2 h w"],
    Float[Tensor, "2 h w d"],
    Float[Tensor, "2 h w"],
]:
    """The pair decoded i -> j only (each frame encoded once, cached on ``frame.feat``): ``(X, C, D, Q)``, dim 0 = ``[res_ii, res_ji]``."""
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
    return X, C, D, Q


