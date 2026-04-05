import torch
import torch.nn.functional as F
from jaxtyping import Float


def img_gradient(
    img: Float[torch.Tensor, "b c h w"],
) -> tuple[Float[torch.Tensor, "b c h w"], Float[torch.Tensor, "b c h w"]]:
    """Compute horizontal and vertical Scharr-like image gradients.

    Returns (gx, gy) where gx is the horizontal gradient and gy is the vertical gradient,
    each with the same shape as the input.
    """
    device: torch.device = img.device
    dtype: torch.dtype = img.dtype
    b: int
    c: int
    h: int
    w: int
    b, c, h, w = img.shape

    gx_kernel: Float[torch.Tensor, "c 1 3 3"] = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    ).repeat(c, 1, 1, 1)

    gy_kernel: Float[torch.Tensor, "c 1 3 3"] = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    ).repeat(c, 1, 1, 1)

    padded: Float[torch.Tensor, "b c h_pad w_pad"] = F.pad(img, (1, 1, 1, 1), mode="reflect")

    gx: Float[torch.Tensor, "b c h w"] = F.conv2d(padded, gx_kernel, groups=c)
    gy: Float[torch.Tensor, "b c h w"] = F.conv2d(padded, gy_kernel, groups=c)

    return gx, gy
