from collections.abc import Callable

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import tqdm
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure

from vistadream.ops.gs.basic import Frame, Gaussian_Frame, Gaussian_Scene


class RGB_Loss:
    def __init__(self, w_lpips: int | float = 0.2, w_ssim: int | float = 0.2) -> None:
        self.rgb_loss: Callable[..., Tensor] = F.smooth_l1_loss
        self.ssim: StructuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
        self.w_ssim: int | float = w_ssim
        self.w_lpips: int | float = w_lpips

    def __call__(self, pr, gt, valid_mask=None):
        pr = torch.nan_to_num(pr)
        gt = torch.nan_to_num(gt)
        if len(pr.shape) < 3:
            pr = pr[:, :, None].repeat(1, 1, 3)
        if len(gt.shape) < 3:
            gt = gt[:, :, None].repeat(1, 1, 3)
        pr_valid = pr[valid_mask] if valid_mask is not None else pr.reshape(-1, pr.shape[-1])
        gt_valid = gt[valid_mask] if valid_mask is not None else gt.reshape(-1, gt.shape[-1])
        l_rgb = self.rgb_loss(pr_valid, gt_valid)
        l_ssim = 1.0 - self.ssim(pr[None].permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))
        return l_rgb + self.w_ssim * l_ssim


class GS_Train_Tool:
    """
    Frames and well-trained gaussians are kept, refine the trainable gaussians
    The supervision comes from the Frames of GS_Scene
    """

    def __init__(self, GS: Gaussian_Scene, iters: int = 100) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # hyperparameters for prune, densify, and update
        self.lr_factor: float = 1.00
        self.lr_update: float = 0.99
        # learning rate
        self.rgb_lr: float = 0.0005
        self.xyz_lr: float = 0.0001
        self.scale_lr: float = 0.005
        self.opacity_lr: float = 0.05
        self.rotation_lr: float = 0.001
        # GSs for training
        self.GS: Gaussian_Scene = GS
        # hyperparameters for training
        self.iters: int = iters
        self._init_optimizer()
        self.rgb_lossfunc = RGB_Loss(w_lpips=0)

    def _init_optimizer(self):
        self.optimize_frames: list[Gaussian_Frame] = [gf for gf in self.GS.gaussian_frames if gf.rgb.requires_grad]
        # following https://github.com/pointrix-project/msplat
        self.optimizer = torch.optim.Adam(
            [
                {"params": [gf.xyz for gf in self.optimize_frames], "lr": self.xyz_lr},
                {"params": [gf.rgb for gf in self.optimize_frames], "lr": self.rgb_lr},
                {"params": [gf.scale for gf in self.optimize_frames], "lr": self.scale_lr},
                {"params": [gf.opacity for gf in self.optimize_frames], "lr": self.opacity_lr},
                {"params": [gf.rotation for gf in self.optimize_frames], "lr": self.rotation_lr},
            ]
        )

    def _render(self, frame: Frame):
        rgb, dpt, alpha = self.GS._render_RGBD(frame)
        return rgb, dpt, alpha

    def _to_cuda(self, tensor: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(tensor.astype(np.float32)).to("cuda")
        return tensor

    def __call__(self, target_frames: list[Frame] | None = None, log: bool = False) -> Gaussian_Scene:
        target_frames: list[Frame] = self.GS.frames if target_frames is None else target_frames
        for it in tqdm.tqdm(range(self.iters)):
            frame_idx: int = np.random.randint(0, len(target_frames))
            frame: Frame = target_frames[frame_idx]
            render_rgb, _, _ = self._render(frame)
            loss_rgb = self.rgb_lossfunc(render_rgb, self._to_cuda(frame.rgb), valid_mask=frame.inpaint)
            # optimization
            loss = loss_rgb
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if log:
                rr.set_time("time", sequence=it + 1)
                # log gaussians
                xyz: Float[ndarray, "n_splats 3"] = (
                    torch.cat([gf.xyz.reshape(-1, 3) for gf in self.GS.gaussian_frames], dim=0)
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                )
                scale: Float[ndarray, "n_splats 3"] = (
                    torch.cat([gf.scale.reshape(-1, 3) for gf in self.GS.gaussian_frames], dim=0)
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                )
                rgb: Float[torch.Tensor, "n_splats 3"] = torch.sigmoid(
                    torch.cat([gf.rgb.reshape(-1, 3) for gf in self.GS.gaussian_frames], dim=0)
                )
                # log the gaussian scene
                rr.log(
                    "world/gaussian_scene",
                    rr.Points3D(
                        positions=xyz,
                        radii=scale,
                        colors=rgb.clone().detach().cpu().numpy(),
                    ),
                )

        refined_scene: Gaussian_Scene = self.GS
        for gf in refined_scene.gaussian_frames:
            gf._require_grad(False)
        return refined_scene
