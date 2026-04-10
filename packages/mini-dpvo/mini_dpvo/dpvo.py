from collections import OrderedDict
from collections.abc import Generator

import lietorch
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32, Float64, Int, UInt8
from lietorch import SE3
from torch import Tensor
from yacs.config import CfgNode

from . import altcorr, fastba
from . import projective_ops as pops
from .net import VONet
from .utils import Timer, flatmeshgrid

autocast = torch.cuda.amp.autocast
Id: SE3 = SE3.Identity(1, device="cuda")


class DPVO:
    def __init__(self, cfg: CfgNode, network: str | VONet, ht: int = 480, wd: int = 640) -> None:
        self.cfg: CfgNode = cfg
        self.load_weights(network)
        self.is_initialized: bool = False
        self.enable_timing: bool = False

        self.n: int = 0  # number of frames
        self.m: int = 0  # number of patches
        self.M: int = self.cfg.PATCHES_PER_FRAME
        self.N: int = self.cfg.BUFFER_SIZE

        self.ht: int = ht  # image height
        self.wd: int = wd  # image width

        DIM: int = self.DIM
        RES: int = self.RES

        ### state attributes ###
        self.tlist: list[int] = []
        self.counter: int = 0

        # dummy image for visualization
        self.image_: UInt8[Tensor, "ht wd 3"] = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_: Float64[Tensor, "N"] = torch.zeros(self.N, dtype=torch.float64, device="cuda")
        self.poses_: Float32[Tensor, "N 7"] = torch.zeros(self.N, 7, dtype=torch.float32, device="cuda")
        self.patches_: Float32[Tensor, "N M 3 P P"] = torch.zeros(
            self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda"
        )
        self.intrinsics_: Float32[Tensor, "N 4"] = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_: Float32[Tensor, "NM 3"] = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_: UInt8[Tensor, "N M 3"] = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_: Int[Tensor, "N M"] = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_: Int[Tensor, "N"] = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem: int = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs: dict[str, object] = {"device": "cuda", "dtype": torch.half}
            kwargs: dict[str, object] = self.kwargs
        else:
            self.kwargs: dict[str, object] = {"device": "cuda", "dtype": torch.float}
            kwargs: dict[str, object] = self.kwargs

        self.imap_: Float[Tensor, "mem M DIM"] = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_: Float[Tensor, "mem M 128 P P"] = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht: int = ht // RES
        wd: int = wd // RES

        self.fmap1_: Float[Tensor, "1 mem 128 h4 w4"] = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_: Float[Tensor, "1 mem 128 h16 w16"] = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid: tuple[Float[Tensor, "1 mem 128 h4 w4"], Float[Tensor, "1 mem 128 h16 w16"]] = (self.fmap1_, self.fmap2_)

        self.net: Float[Tensor, "1 n_edges DIM"] = torch.zeros(1, 0, DIM, **kwargs)
        self.ii: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk: Int[Tensor, "n_edges"] = torch.as_tensor([], dtype=torch.long, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0

        # store relative poses for removed frames
        self.delta: dict[int, tuple[int, SE3]] = {}

    def load_weights(self, network: str | VONet) -> None:
        # load network from checkpoint file
        if isinstance(network, str):
            state_dict: dict[str, Tensor] = torch.load(network)
            new_state_dict: OrderedDict[str, Tensor] = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace("module.", "")] = v

            self.network: VONet = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network: VONet = network

        # steal network attributes
        self.DIM: int = self.network.DIM
        self.RES: int = self.network.RES
        self.P: int = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()

    @property
    def poses(self) -> Float32[Tensor, "1 N 7"]:
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self) -> Float32[Tensor, "1 NM 3 3 3"]:
        return self.patches_.view(1, self.N * self.M, 3, 3, 3)

    @property
    def intrinsics(self) -> Float32[Tensor, "1 N 4"]:
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self) -> Int[Tensor, "NM"]:
        return self.index_.view(-1)

    @property
    def imap(self) -> Float[Tensor, "1 mem_times_M DIM"]:
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self) -> Float[Tensor, "1 mem_times_M 128 3 3"]:
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t: int) -> SE3:
        if t in self.traj:
            return SE3(self.traj[t])

        t0: int
        dP: SE3
        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self) -> tuple[Float[np.ndarray, "n_frames 7"], Float64[np.ndarray, "n_frames"]]:
        """interpolate missing poses"""
        print("Terminating...")
        self.traj: dict[int, Float32[Tensor, "7"]] = {}
        for i in range(self.n):
            current_t: int = self.tstamps_[i].item()
            self.traj[current_t] = self.poses_[i]

        poses: list[SE3] = [self.get_pose(t) for t in range(self.counter)]
        poses: SE3 = lietorch.stack(poses, dim=0)
        poses: Float[np.ndarray, "n_frames 7"] = poses.inv().data.cpu().numpy()
        tstamps: Float64[np.ndarray, "n_frames"] = np.array(self.tlist, dtype=np.float64)
        print("Done!")

        return poses, tstamps

    def corr(self, coords: Float[Tensor, "1 n_edges 2 P P"], indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None) -> Float[Tensor, "1 n_edges corr_feat"]:
        """local correlation volume"""
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1: Int[Tensor, "n_edges"] = ii % (self.M * self.mem)
        jj1: Int[Tensor, "n_edges"] = jj % (self.mem)
        corr1: Float[Tensor, "1 n_edges corr1"] = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2: Float[Tensor, "1 n_edges corr2"] = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies: tuple[Int[Tensor, "n_edges"], Int[Tensor, "n_edges"], Int[Tensor, "n_edges"]] | None = None) -> Float[Tensor, "1 n_edges 2 P P"]:
        """reproject patch k from i -> j"""
        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        kk: Int[Tensor, "n_edges"]
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords: Float[Tensor, "1 n_edges P P 2"] = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii: Int[Tensor, "n_new"], jj: Int[Tensor, "n_new"]) -> None:
        self.jj: Int[Tensor, "n_edges"] = torch.cat([self.jj, jj])
        self.kk: Int[Tensor, "n_edges"] = torch.cat([self.kk, ii])
        self.ii: Int[Tensor, "n_edges"] = torch.cat([self.ii, self.ix[ii]])

        net: Float[Tensor, "1 n_new DIM"] = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net: Float[Tensor, "1 n_edges DIM"] = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m: Int[Tensor, "n_edges"]) -> None:
        self.ii: Int[Tensor, "n_remaining"] = self.ii[~m]
        self.jj: Int[Tensor, "n_remaining"] = self.jj[~m]
        self.kk: Int[Tensor, "n_remaining"] = self.kk[~m]
        self.net: Float[Tensor, "1 n_remaining DIM"] = self.net[:, ~m]

    def motion_probe(self) -> Float[Tensor, ""]:
        """kinda hacky way to ensure enough motion for initialization"""
        kk: Int[Tensor, "M"] = torch.arange(self.m - self.M, self.m, device="cuda")
        jj: Int[Tensor, "M"] = self.n * torch.ones_like(kk)
        ii: Int[Tensor, "M"] = self.ix[kk]

        net: Float[Tensor, "1 M DIM"] = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords: Float[Tensor, "1 M 2 P P"] = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr: Float[Tensor, "1 M corr_feat"] = self.corr(coords, indicies=(kk, jj))
            ctx: Float[Tensor, "1 M DIM"] = self.imap[:, kk % (self.M * self.mem)]
            net: Float[Tensor, "1 M DIM"]
            delta: Float[Tensor, "1 M 2"]
            weight: Float[Tensor, "1 M 2"]
            net, (delta, weight, _) = self.network.update(
                net, ctx, corr, None, ii, jj, kk
            )

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i: int, j: int) -> float:
        k: Int[Tensor, "n_edges"] = (self.ii == i) & (self.jj == j)
        ii: Int[Tensor, "n_matched"] = self.ii[k]
        jj: Int[Tensor, "n_matched"] = self.jj[k]
        kk: Int[Tensor, "n_matched"] = self.kk[k]

        flow: Float[Tensor, "..."] = pops.flow_mag(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5
        )
        return flow.mean().item()

    def keyframe(self) -> None:
        i: int = self.n - self.cfg.KEYFRAME_INDEX - 1
        j: int = self.n - self.cfg.KEYFRAME_INDEX + 1
        m: float = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k: int = self.n - self.cfg.KEYFRAME_INDEX
            t0: float = self.tstamps_[k - 1].item()
            t1: float = self.tstamps_[k].item()

            dP: SE3 = SE3(self.poses_[k]) * SE3(self.poses_[k - 1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove: Int[Tensor, "n_edges"] = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n - 1):
                self.tstamps_[i] = self.tstamps_[i + 1]
                self.colors_[i] = self.colors_[i + 1]
                self.poses_[i] = self.poses_[i + 1]
                self.patches_[i] = self.patches_[i + 1]
                self.intrinsics_[i] = self.intrinsics_[i + 1]

                self.imap_[i % self.mem] = self.imap_[(i + 1) % self.mem]
                self.gmap_[i % self.mem] = self.gmap_[(i + 1) % self.mem]
                self.fmap1_[0, i % self.mem] = self.fmap1_[0, (i + 1) % self.mem]
                self.fmap2_[0, i % self.mem] = self.fmap2_[0, (i + 1) % self.mem]

            self.n -= 1
            self.m -= self.M

        to_remove: Int[Tensor, "n_edges"] = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self) -> None:
        with Timer("other", enabled=self.enable_timing):
            coords: Float[Tensor, "1 n_edges 2 P P"] = self.reproject()

            with autocast(enabled=True):
                corr: Float[Tensor, "1 n_edges corr_feat"] = self.corr(coords)
                ctx: Float[Tensor, "1 n_edges DIM"] = self.imap[:, self.kk % (self.M * self.mem)]
                delta: Float[Tensor, "1 n_edges 2"]
                weight: Float[Tensor, "1 n_edges 2"]
                self.net, (delta, weight, _) = self.network.update(
                    self.net, ctx, corr, None, self.ii, self.jj, self.kk
                )

            lmbda: Float[Tensor, "1"] = torch.as_tensor([1e-4], device="cuda")
            weight: Float[Tensor, "1 n_edges 2"] = weight.float()
            target: Float[Tensor, "1 n_edges 2"] = coords[..., self.P // 2, self.P // 2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            t0: int = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0: int = max(t0, 1)

            try:
                fastba.BA(
                    self.poses,
                    self.patches,
                    self.intrinsics,
                    target,
                    weight,
                    lmbda,
                    self.ii,
                    self.jj,
                    self.kk,
                    t0,
                    self.n,
                    2,
                )
            except:
                print("Warning BA failed...")

            points: Float[Tensor, "1 m P P 4"] = pops.point_cloud(
                SE3(self.poses),
                self.patches[:, : self.m],
                self.intrinsics,
                self.ix[: self.m],
            )
            points: Float[Tensor, "m 3"] = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
            self.points_[: len(points)] = points[:]

    def __edges_all(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_forw(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        r: int = self.cfg.PATCH_LIFETIME
        t0: int = self.M * max((self.n - r), 0)
        t1: int = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"),
            indexing="ij",
        )

    def __edges_back(self) -> Generator[Int[Tensor, "n_edges"], None, None]:
        r: int = self.cfg.PATCH_LIFETIME
        t0: int = self.M * max((self.n - 1), 0)
        t1: int = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n - r, 0), self.n, device="cuda"),
            indexing="ij",
        )

    def __call__(self, tstamp: int, image: Float[Tensor, "3 ht wd"], intrinsics: Float[Tensor, "4"]) -> None:
        """track new frame"""

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"'
            )

        image: Float[Tensor, "1 1 3 ht wd"] = 2 * (image[None, None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap: Float[Tensor, "1 1 128 h4 w4"]
            gmap: Float[Tensor, "1 M 128 P P"]
            imap: Float[Tensor, "1 M DIM 1 1"]
            patches: Float[Tensor, "1 M 3 P P"]
            clr: Float[Tensor, "1 M 3"]
            fmap, gmap, imap, patches, _, clr = self.network.patchify(
                image,
                patches_per_image=self.cfg.PATCHES_PER_FRAME,
                gradient_bias=self.cfg.GRADIENT_BIAS,
                return_color=True,
            )

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr: Float[Tensor, "M 3"] = (clr[0, :, [2, 1, 0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == "DAMPED_LINEAR":
                P1: SE3 = SE3(self.poses_[self.n - 1])
                P2: SE3 = SE3(self.poses_[self.n - 2])

                xi: Float[Tensor, "6"] = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec: Float[Tensor, "7"] = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec: Float[Tensor, "7"] = self.poses[self.n - 1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s: Float[Tensor, ""] = torch.median(self.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()
