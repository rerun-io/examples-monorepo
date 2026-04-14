"""Base RGBD dataset class for DPVO training.

Provides frame-graph-based sampling logic and data augmentation that is
shared across all concrete dataset implementations (e.g. TartanAir).
"""

import os
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torch.utils.data as data
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

from .augmentation import RGBDAugmentor
from .rgbd_utils import compute_distance_matrix_flow


class RGBDDataset(data.Dataset):
    """Abstract base :class:`~torch.utils.data.Dataset` for RGBD video clips.

    Subclasses must implement:

    - :meth:`is_test_scene` -- classify a scene as train or validation.
    - :meth:`image_read` / :meth:`depth_read` -- load a single frame/depth.

    The base class handles frame-graph construction, difficulty-based
    sampling, data augmentation, and depth normalisation.
    """

    def __init__(
        self,
        name: str,
        datapath: str,
        n_frames: int = 4,
        crop_size: list[int] | None = None,
        fmin: float = 10.0,
        fmax: float = 75.0,
        aug: bool = True,
        sample: bool = True,
    ) -> None:
        """Initialise the RGBD dataset.

        Args:
            name: Human-readable dataset name.
            datapath: Root directory containing the dataset files.
            n_frames: Number of frames per training clip.
            crop_size: ``[height, width]`` for augmentation centre-crop.
            fmin: Minimum optical-flow distance for frame sampling
                (excludes trivially easy pairs).
            fmax: Maximum optical-flow distance for frame sampling
                (excludes overly hard pairs).
            aug: Whether to enable data augmentation.
            sample: If ``True``, sample frames stochastically within the
                flow-distance window; otherwise use a deterministic greedy
                strategy.
        """
        if crop_size is None:
            crop_size = [480, 640]
        self.aug: RGBDAugmentor | bool | None = None
        self.root: str = datapath
        self.name: str = name

        self.aug = aug
        self.sample: bool = sample

        self.n_frames: int = n_frames
        self.fmin: float = fmin  # exclude very easy examples
        self.fmax: float = fmax  # exclude very hard examples

        if self.aug:
            self.aug = RGBDAugmentor(crop_size=crop_size)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path: str = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        self.scene_info: dict = \
            pickle.load(open('datasets/TartanAir.pickle', 'rb'))[0]  # noqa: SIM115

        self._build_dataset_index()

    @staticmethod
    def is_test_scene(scene: str) -> bool:
        """Check whether ``scene`` belongs to the validation/test split.

        Subclasses should override this method.

        Args:
            scene: Scene path string.

        Returns:
            ``True`` if the scene is a test scene.
        """
        return False

    def _build_dataset_index(self) -> None:
        """Build a flat list of ``(scene_id, start_frame)`` training indices.

        Iterates over all scenes in :attr:`scene_info`, skipping test scenes
        (as determined by :meth:`is_test_scene`). Each non-test scene
        contributes one entry per valid starting frame.
        """
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
    def image_read(image_file: str) -> UInt8[ndarray, "h w 3"]:
        """Read an image from disk via OpenCV (BGR channel order).

        Args:
            image_file: Absolute path to the image file.

        Returns:
            The loaded image as a uint8 array with shape ``(h, w, 3)``.
        """
        img = cv2.imread(image_file)
        assert img is not None, f"Failed to read image: {image_file}"
        return img

    @staticmethod
    def depth_read(depth_file: str) -> Float64[ndarray, "h w"]:
        """Read a depth map from a ``.npy`` file.

        Args:
            depth_file: Absolute path to the ``.npy`` depth file.

        Returns:
            The depth map as a float64 array with shape ``(h, w)``.
        """
        return np.load(depth_file)

    def build_frame_graph(
        self,
        poses: list[Float64[ndarray, "7"]] | Float64[ndarray, "n 7"],
        depths: list[str],
        intrinsics: list[Float64[ndarray, "4"]] | Float64[ndarray, "n 4"],
        f: int = 16,
        max_flow: int = 256,
    ) -> dict[int, tuple[ndarray, ndarray]]:
        """Build a co-visibility frame graph based on optical-flow distance.

        For every frame *i*, the graph stores the set of frames *j* whose
        induced optical flow magnitude is below ``max_flow``. This is used
        later to sample training clips with appropriate difficulty.

        Args:
            poses: Per-frame poses ``[tx, ty, tz, qx, qy, qz, qw]``.
            depths: List of depth file paths (one per frame).
            intrinsics: Per-frame camera intrinsics ``[fx, fy, cx, cy]``.
            f: Spatial down-sampling factor for disparity computation.
            max_flow: Maximum allowed flow magnitude for an edge.

        Returns:
            A dict mapping each frame index to a ``(neighbours, distances)``
            tuple of 1-D arrays.
        """

        def read_disp(fn: str) -> Float64[ndarray, "h_sub w_sub"]:
            """Read a depth file and convert to down-sampled inverse depth."""
            depth: Float64[ndarray, "h w"] = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f

        disps: Float64[ndarray, "n h_sub w_sub"] = np.stack(list(map(read_disp, depths)), 0)
        d: Float32[ndarray, "n n"] = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph: dict[int, tuple[ndarray, ndarray]] = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph

    def __getitem__(self, index: int) -> tuple[Float32[Tensor, "n 3 h w"], Float32[Tensor, "n 7"], Float32[Tensor, "n h w"], Float32[Tensor, "n 4"]]:
        """Return a training clip of ``n_frames`` frames.

        Sampling walks the frame graph starting from the indexed anchor,
        preferring forward-in-time frames within the ``[fmin, fmax]``
        flow-distance window. The returned disparity is normalised so that
        the 98th-percentile value is 0.7, and the pose translations are
        scaled by the same factor.

        Args:
            index: Dataset index (automatically wrapped modulo dataset size).

        Returns:
            Tuple of ``(images, poses, disparities, intrinsics)`` tensors.
        """
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
                k: ndarray = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                frames: ndarray = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = int(np.random.choice(frames[frames > ix]))

                elif ix + 1 < len(images_list):
                    ix = ix + 1

                elif np.count_nonzero(frames):
                    ix = int(np.random.choice(frames))

            else:
                i: ndarray = frame_graph[ix][0].copy()
                g: ndarray = frame_graph[ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= ix] = -1
                else:
                    g[i >= ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    ix = int(i[np.argmax(g)])
                else:
                    if ix + s >= len(images_list) or ix + s < 0:
                        s *= -1

                    ix = ix + s

            inds += [ ix ]


        images: list[UInt8[ndarray, "h w 3"]] = []
        depths: list[Float64[ndarray, "h w"]] = []
        poses: list = []
        intrinsics: list = []
        for frame_idx in inds:
            images.append(self.__class__.image_read(images_list[int(frame_idx)]))
            depths.append(self.__class__.depth_read(depths_list[frame_idx]))
            poses.append(poses_list[frame_idx])
            intrinsics.append(intrinsics_list[frame_idx])

        images_np: Float32[ndarray, "n h w 3"] = np.stack(images).astype(np.float32)
        depths_np: Float32[ndarray, "n h w"] = np.stack(depths).astype(np.float32)
        poses_np: Float32[ndarray, "n 7"] = np.stack(poses).astype(np.float32)
        intrinsics_np: Float32[ndarray, "n 4"] = np.stack(intrinsics).astype(np.float32)

        images_t: Float32[Tensor, "n 3 h w"] = torch.from_numpy(images_np).float()
        images_t = images_t.permute(0, 3, 1, 2)

        disps: Float32[Tensor, "n h w"] = torch.from_numpy(1.0 / depths_np)
        poses_t: Float32[Tensor, "n 7"] = torch.from_numpy(poses_np)
        intrinsics_t: Float32[Tensor, "n 4"] = torch.from_numpy(intrinsics_np)

        if isinstance(self.aug, RGBDAugmentor):
            images_t, poses_t, disps, intrinsics_t = \
                self.aug(images_t, poses_t, disps, intrinsics_t)

        # normalize depth
        s_norm: Float32[Tensor, ""] = .7 * torch.quantile(disps, .98)
        disps = disps / s_norm
        poses_t[...,:3] *= s_norm

        return images_t, poses_t, disps, intrinsics_t

    def __len__(self) -> int:
        """Return the total number of training clips available."""
        return len(self.dataset_index)

    def __imul__(self, x: int) -> "RGBDDataset":
        """Repeat the dataset index ``x`` times (for oversampling).

        Args:
            x: Repetition multiplier.

        Returns:
            ``self`` with the expanded index.
        """
        self.dataset_index *= x
        return self
