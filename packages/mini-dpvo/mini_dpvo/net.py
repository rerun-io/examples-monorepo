
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from lietorch import SE3
from torch import Tensor

from . import altcorr, fastba
from . import projective_ops as pops
from .ba import BA
from .blocks import GatedResidual, GradientClip, SoftAgg
from .extractor import BasicEncoder4
from .utils import *

autocast = torch.cuda.amp.autocast

DIM: int = 384

class Update(nn.Module):
    def __init__(self, p: int) -> None:
        super().__init__()

        self.c1: nn.Sequential = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2: nn.Sequential = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.norm: nn.LayerNorm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk: SoftAgg = SoftAgg(DIM)
        self.agg_ij: SoftAgg = SoftAgg(DIM)

        self.gru: nn.Sequential = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr: nn.Sequential = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d: nn.Sequential = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w: nn.Sequential = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(
        self,
        net: Float[Tensor, "1 n_edges dim"],
        inp: Float[Tensor, "1 n_edges dim"],
        corr: Float[Tensor, "1 n_edges corr_feat"],
        flow: None,
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        kk: Int[Tensor, "n_edges"],
    ) -> tuple[Float[Tensor, "1 n_edges dim"], tuple[Float[Tensor, "1 n_edges 2"], Float[Tensor, "1 n_edges 2"], None]]:
        """ update operator """

        net: Float[Tensor, "1 n_edges dim"] = net + inp + self.corr(corr)
        net = self.norm(net)

        ix: Int[Tensor, "n_edges"]
        jx: Int[Tensor, "n_edges"]
        ix, jx = fastba.neighbors(kk, jj)
        mask_ix: Float[Tensor, "1 n_edges 1"] = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx: Float[Tensor, "1 n_edges 1"] = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        delta: Float[Tensor, "1 n_edges 2"] = self.d(net)
        weight: Float[Tensor, "1 n_edges 2"] = self.w(net)

        return net, (delta, weight, None)


class Patchifier(nn.Module):
    def __init__(self, patch_size: int = 3) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.fnet: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet: BasicEncoder4 = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images: Float[Tensor, "1 n 3 h w"]) -> Float[Tensor, "1 n h4 w4"]:
        gray: Float[Tensor, "1 n h w"] = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx: Float[Tensor, "1 n h_m1 w_m1"] = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy: Float[Tensor, "1 n h_m1 w_m1"] = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g: Float[Tensor, "1 n h_m1 w_m1"] = torch.sqrt(dx**2 + dy**2)
        g: Float[Tensor, "1 n h4 w4"] = F.avg_pool2d(g, 4, 4)
        return g

    def forward(
        self,
        images: Float[Tensor, "1 n 3 h w"],
        patches_per_image: int = 80,
        disps: Float[Tensor, "1 n h w"] | None = None,
        gradient_bias: bool = False,
        return_color: bool = False,
    ) -> tuple[Float[Tensor, "1 n 128 h4 w4"], Float[Tensor, "1 num_patches 128 p p"], Float[Tensor, "1 num_patches dim 1 1"], Float[Tensor, "1 num_patches 3 p p"], Int[Tensor, "num_patches"]] | tuple[Float[Tensor, "1 n 128 h4 w4"], Float[Tensor, "1 num_patches 128 p p"], Float[Tensor, "1 num_patches dim 1 1"], Float[Tensor, "1 num_patches 3 p p"], Int[Tensor, "num_patches"], Float[Tensor, "1 num_patches 3"]]:
        """ extract patches from input images """
        fmap: Float[Tensor, "1 n 128 h4 w4"] = self.fnet(images) / 4.0
        imap: Float[Tensor, "1 n dim h4 w4"] = self.inet(images) / 4.0

        b: int
        n: int
        c: int
        h: int
        w: int
        b, n, c, h, w = fmap.shape
        P: int = self.patch_size

        # bias patch selection towards regions with high gradient
        if gradient_bias:
            g: Float[Tensor, "1 n h4 w4"] = self.__image_gradient(images)
            x: Int[Tensor, "n candidates"] = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y: Int[Tensor, "n candidates"] = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords: Float[Tensor, "n candidates 2"] = torch.stack([x, y], dim=-1).float()
            g: Float[Tensor, "n candidates"] = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)

            ix: Int[Tensor, "n candidates"] = torch.argsort(g, dim=1)
            x: Int[Tensor, "n patches_per_image"] = torch.gather(x, 1, ix[:, -patches_per_image:])
            y: Int[Tensor, "n patches_per_image"] = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            x: Int[Tensor, "n patches_per_image"] = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y: Int[Tensor, "n patches_per_image"] = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        coords: Float[Tensor, "n patches_per_image 2"] = torch.stack([x, y], dim=-1).float()
        imap: Float[Tensor, "1 num_patches dim 1 1"] = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap: Float[Tensor, "1 num_patches 128 p p"] = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        clr: Float[Tensor, "1 num_patches 3"]
        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps: Float[Tensor, "1 n h w"] = torch.ones(b, n, h, w, device="cuda")

        grid: Float[Tensor, "1 n 3 h w"]
        _index: Float[Tensor, "1 n 1 h w"]
        grid, _index = coords_grid_with_index(disps, device=fmap.device)
        patches: Float[Tensor, "1 num_patches 3 p p"] = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index: Int[Tensor, "n 1"] = torch.arange(n, device="cuda").view(n, 1)
        index: Int[Tensor, "num_patches"] = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(
        self,
        fmap: Float[Tensor, "1 n 128 h4 w4"],
        gmap: Float[Tensor, "1 num_patches 128 p p"],
        radius: int = 3,
        dropout: float = 0.2,
        levels: list[int] = [1, 4],
    ) -> None:
        self.dropout: float = dropout
        self.radius: int = radius
        self.levels: list[int] = levels

        self.gmap: Float[Tensor, "1 num_patches 128 p p"] = gmap
        self.pyramid: list[Float[Tensor, "1 n 128 h_l w_l"]] = pyramidify(fmap, lvls=levels)

    def __call__(
        self,
        ii: Int[Tensor, "n_edges"],
        jj: Int[Tensor, "n_edges"],
        coords: Float[Tensor, "1 n_edges 2 p p"],
    ) -> Float[Tensor, "1 n_edges corr_feat"]:
        corrs: list[Float[Tensor, "..."]] = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer: bool = False) -> None:
        super().__init__()
        self.P: int = 3
        self.patchify: Patchifier = Patchifier(self.P)
        self.update: Update = Update(self.P)

        self.DIM: int = DIM
        self.RES: int = 4


    @autocast(enabled=False)
    def forward(
        self,
        images: Float[Tensor, "1 n_frames 3 h w"],
        poses: SE3,
        disps: Float[Tensor, "1 n_frames h w"],
        intrinsics: Float[Tensor, "1 n_frames 4"],
        M: int = 1024,
        STEPS: int = 12,
        P: int = 1,
        structure_only: bool = False,
        rescale: bool = False,
    ) -> list[tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], SE3, SE3, Tensor]]:
        """ Estimates SE3 or Sim3 between pair of frames """

        images: Float[Tensor, "1 n_frames 3 h w"] = 2 * (images / 255.0) - 0.5
        intrinsics: Float[Tensor, "1 n_frames 4"] = intrinsics / 4.0
        disps: Float[Tensor, "1 n_frames h4 w4"] = disps[:, :, 1::4, 1::4].float()

        fmap: Float[Tensor, "1 n_frames 128 h4 w4"]
        gmap: Float[Tensor, "1 num_patches 128 p p"]
        imap: Float[Tensor, "1 num_patches dim 1 1"]
        patches: Float[Tensor, "1 num_patches 3 p p"]
        ix: Int[Tensor, "num_patches"]
        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn: CorrBlock = CorrBlock(fmap, gmap)

        b: int
        N: int
        c: int
        h: int
        w: int
        b, N, c, h, w = fmap.shape
        p: int = self.P

        patches_gt: Float[Tensor, "1 num_patches 3 p p"] = patches.clone()
        Ps: SE3 = poses

        d: Float[Tensor, "1 num_patches"] = patches[..., 2, p//2, p//2]
        patches: Float[Tensor, "1 num_patches 3 p p"] = set_depth(patches, torch.rand_like(d))

        kk: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii: Int[Tensor, "n_edges"] = ix[kk]

        imap: Float[Tensor, "1 num_patches dim"] = imap.view(b, -1, DIM)
        net: Float[Tensor, "1 n_edges dim"] = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        Gs: SE3 = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj: list[tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], SE3, SE3, Tensor]] = []
        bounds: list[int] = [-64, -64, w + 64, h + 64]

        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n: int = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1: Int[Tensor, "new_edges1"]
                jj1: Int[Tensor, "new_edges1"]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2: Int[Tensor, "new_edges2"]
                jj2: Int[Tensor, "new_edges2"]
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1: Float[Tensor, "1 new_edges dim"] = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k: Tensor = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords: Float[Tensor, "1 n_edges p p 2"] = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1: Float[Tensor, "1 n_edges 2 p p"] = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr: Float[Tensor, "1 n_edges corr_feat"] = corr_fn(kk, jj, coords1)
            delta: Float[Tensor, "1 n_edges 2"]
            weight: Float[Tensor, "1 n_edges 2"]
            _damping: None
            net, (delta, weight, _damping) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda: float = 1e-4
            target: Float[Tensor, "1 n_edges 2"] = coords[...,p//2,p//2,:] + delta

            ep: int = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk,
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl: Tensor = torch.as_tensor(0)
            dij: Int[Tensor, "n_edges"] = (ii - jj).abs()
            k: Tensor = (dij > 0) & (dij <= 2)

            coords: Float[Tensor, "..."] = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt: Float[Tensor, "..."]
            valid: Float[Tensor, "..."]
            _jacobians: tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]
            coords_gt, valid, _jacobians = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj
