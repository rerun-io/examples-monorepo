"""Exact polyphase head for the 768x576 Sapiens2 export."""

from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn


class FusedPolyphase(nn.Module):
    """Exact ConvTranspose2d(k=4, s=2, p=1) using one Conv2d(Cin, 4*Cout, k=2, pad=1) on x (no shifted input copies, FLOPs x (h+1)(w+1)/hw).

    Phase (a,b) of the output = y[:, (2a+b)*Cout:(2a+b+1)*Cout, a:a+h, b:b+w].
    """

    def __init__(self, deconv: nn.ConvTranspose2d) -> None:
        super().__init__()
        if deconv.kernel_size != (4, 4) or deconv.stride != (2, 2) or deconv.padding != (1, 1) or deconv.bias is not None:
            raise ValueError("Fast head requires bias-free ConvTranspose2d(kernel=4, stride=2, padding=1).")
        weight: torch.Tensor = deconv.weight.detach()
        taps: list[list[int]] = [[3, 1], [2, 0]]
        w: torch.Tensor = torch.cat(
            [weight[:, :, taps[a], :][:, :, :, taps[b]].permute(1, 0, 2, 3) for a in range(2) for b in range(2)], 0
        ).contiguous()
        self.conv = nn.Conv2d(w.shape[1], w.shape[0], 2, padding=1, bias=False).to(w.device, w.dtype)
        self.conv.weight.data = w
        self.cout: int = int(weight.shape[1])

    def phases(self, x: Float[torch.Tensor, "n c h w"]) -> Float[torch.Tensor, "n out_c 4 h w"]:
        """[n, Cout, 4, h, w], phase index 2a+b."""
        n: int | torch.SymInt = x.shape[0]
        h: int | torch.SymInt = x.shape[2]
        wd: int | torch.SymInt = x.shape[3]
        y: torch.Tensor = self.conv(x).view(n, 4, self.cout, h + 1, wd + 1)
        return torch.stack([y[:, 2 * a + b, :, a : a + h, b : b + wd] for a in range(2) for b in range(2)], 2)

    def forward(self, x: Float[torch.Tensor, "n c h w"]) -> Float[torch.Tensor, "n out_c out_h out_w"]:
        n: int | torch.SymInt = x.shape[0]
        h: int | torch.SymInt = x.shape[2]
        wd: int | torch.SymInt = x.shape[3]
        y: torch.Tensor = self.phases(x).view(n, self.cout, 2, 2, h, wd)
        return y.permute(0, 1, 4, 2, 5, 3).reshape(n, self.cout, 2 * h, 2 * wd)


class DeferredHead(nn.Module):
    """Head with fused polyphase deconvs; the 2nd deconv's pixel interleave is deferred to the 65-ch conv_pose output.

    Exact: InstanceNorm statistics are permutation invariant, SiLU and the 1x1 convs act per position.
    """

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        dl: nn.Sequential = cast(nn.Sequential, head.deconv_layers)
        self.d1 = FusedPolyphase(cast(nn.ConvTranspose2d, dl[0]))
        self.n1: nn.Module = dl[1]
        self.d2 = FusedPolyphase(cast(nn.ConvTranspose2d, dl[3]))
        self.n2: nn.Module = dl[4]
        self.conv_layers: nn.Module = cast(nn.Module, head.conv_layers)
        self.conv_pose: nn.Module = cast(nn.Module, head.conv_pose)

    def forward(self, x: Float[torch.Tensor, "n c h w"]) -> Float[torch.Tensor, "n k out_h out_w"]:
        x = F.silu(self.n1(self.d1(x)))
        n: int | torch.SymInt = x.shape[0]
        h: int | torch.SymInt = x.shape[2]
        w: int | torch.SymInt = x.shape[3]
        p: torch.Tensor = self.d2.phases(x).reshape(n, self.d2.cout, 4 * h, w)  # phase-major rows
        p = self.conv_pose(self.conv_layers(F.silu(self.n2(p))))
        c: int | torch.SymInt = p.shape[1]
        return p.view(n, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(n, c, 2 * h, 2 * w)
