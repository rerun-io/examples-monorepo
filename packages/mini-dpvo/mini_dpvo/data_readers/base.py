import os
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torch.utils.data as data
from jaxtyping import Float32, Float64, UInt8

from .augmentation import RGBDAugmentor
from .rgbd_utils import *


class RGBDDataset(data.Dataset):
    def __init__(
        self,
        name: str,
        datapath: str,
        n_frames: int = 4,
        crop_size: list[int] = [480, 640],
        fmin: float = 10.0,
        fmax: float = 75.0,
        aug: bool = True,
        sample: bool = True,
    ) -> None:
        """ Base class for RGBD dataset """
        self.aug: RGBDAugmentor | bool | None = None
        self.root: str = datapath
        self.name: str = name

        self.aug = aug
        self.sample: bool = sample

        self.n_frames: int = n_frames
        self.fmin: float = fmin # exclude very easy examples
        self.fmax: float = fmax # exclude very hard examples

        if self.aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path: str = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        self.scene_info: dict = \
            pickle.load(open('datasets/TartanAir.pickle', 'rb'))[0]

        self._build_dataset_index()

    def _build_dataset_index(self) -> None:
        self.dataset_index: list[tuple[str, int]] = []
        for scene in self.scene_info:
            if not self.__class__.is_test_scene(scene):
                graph: dict = self.scene_info[scene]['graph']
                for i in graph:
                    if i < len(graph) - 65:
                        self.dataset_index.append((scene, i))
            else:
                print(f"Reserving {scene} for validation")

    @staticmethod
    def image_read(image_file: str) -> UInt8[np.ndarray, "h w 3"]:
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file: str) -> Float64[np.ndarray, "h w"]:
        return np.load(depth_file)

    def build_frame_graph(
        self,
        poses: list[Float64[np.ndarray, "7"]] | Float64[np.ndarray, "n 7"],
        depths: list[str],
        intrinsics: list[Float64[np.ndarray, "4"]] | Float64[np.ndarray, "n 4"],
        f: int = 16,
        max_flow: int = 256,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn: str) -> Float64[np.ndarray, "h_sub w_sub"]:
            depth: Float64[np.ndarray, "h w"] = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f

        disps: Float64[np.ndarray, "n h_sub w_sub"] = np.stack(list(map(read_disp, depths)), 0)
        d: Float32[np.ndarray, "n n"] = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index: int) -> tuple[Float32[torch.Tensor, "n 3 h w"], Float32[torch.Tensor, "n 7"], Float32[torch.Tensor, "n h w"], Float32[torch.Tensor, "n 4"]]:
        """ return training video """

        index = index % len(self.dataset_index)
        scene_id: str
        ix: int
        scene_id, ix = self.dataset_index[index]

        frame_graph: dict = self.scene_info[scene_id]['graph']
        images_list: list[str] = self.scene_info[scene_id]['images']
        depths_list: list[str] = self.scene_info[scene_id]['depths']
        poses_list: list = self.scene_info[scene_id]['poses']
        intrinsics_list: list = self.scene_info[scene_id]['intrinsics']

        # stride = np.random.choice([1,2,3])

        d: float = np.random.uniform(self.fmin, self.fmax)
        s: int = 1

        inds: list[int] = [ ix ]

        while len(inds) < self.n_frames:
            # get other frames within flow threshold

            if self.sample:
                k: np.ndarray = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames: np.ndarray = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif ix + 1 < len(images_list):
                    ix = ix + 1

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

            else:
                i: np.ndarray = frame_graph[ix][0].copy()
                g: np.ndarray = frame_graph[ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= ix] = -1
                else:
                    g[i >= ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    ix = i[np.argmax(g)]
                else:
                    if ix + s >= len(images_list) or ix + s < 0:
                        s *= -1

                    ix = ix + s

            inds += [ ix ]


        images: list[UInt8[np.ndarray, "h w 3"]] = []
        depths: list[Float64[np.ndarray, "h w"]] = []
        poses: list = []
        intrinsics: list = []
        for i in inds:
            images.append(self.__class__.image_read(images_list[i]))
            depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            intrinsics.append(intrinsics_list[i])

        images_np: Float32[np.ndarray, "n h w 3"] = np.stack(images).astype(np.float32)
        depths_np: Float32[np.ndarray, "n h w"] = np.stack(depths).astype(np.float32)
        poses_np: Float32[np.ndarray, "n 7"] = np.stack(poses).astype(np.float32)
        intrinsics_np: Float32[np.ndarray, "n 4"] = np.stack(intrinsics).astype(np.float32)

        images_t: Float32[torch.Tensor, "n 3 h w"] = torch.from_numpy(images_np).float()
        images_t = images_t.permute(0, 3, 1, 2)

        disps: Float32[torch.Tensor, "n h w"] = torch.from_numpy(1.0 / depths_np)
        poses_t: Float32[torch.Tensor, "n 7"] = torch.from_numpy(poses_np)
        intrinsics_t: Float32[torch.Tensor, "n 4"] = torch.from_numpy(intrinsics_np)

        if self.aug:
            images_t, poses_t, disps, intrinsics_t = \
                self.aug(images_t, poses_t, disps, intrinsics_t)

        # normalize depth
        s_norm: Float32[torch.Tensor, ""] = .7 * torch.quantile(disps, .98)
        disps = disps / s_norm
        poses_t[...,:3] *= s_norm

        return images_t, poses_t, disps, intrinsics_t

    def __len__(self) -> int:
        return len(self.dataset_index)

    def __imul__(self, x: int) -> "RGBDDataset":
        self.dataset_index *= x
        return self
