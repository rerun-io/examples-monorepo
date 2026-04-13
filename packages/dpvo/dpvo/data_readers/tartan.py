"""TartanAir dataset loader for DPVO training.

Provides :class:`TartanAir`, a concrete :class:`RGBDDataset` implementation
for the `TartanAir <https://theairlab.org/tartanair-dataset/>`_ benchmark.
Depths are loaded from ``.npy`` files and divided by
:attr:`TartanAir.DEPTH_SCALE`; poses are read from ``pose_left.txt`` and
re-ordered from the TartanAir NED convention to the DPVO convention.
"""

import glob
import os.path as osp

import cv2
import numpy as np
from jaxtyping import Float64, UInt8

from .base import RGBDDataset

# Hard-coded list of scene paths reserved for the validation / test split.
test_split: list[str] = [
    "abandonedfactory/abandonedfactory/Easy/P011",
    "abandonedfactory/abandonedfactory/Hard/P011",
    "abandonedfactory_night/abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/abandonedfactory_night/Hard/P014",
    "amusement/amusement/Easy/P008",
    "amusement/amusement/Hard/P007",
    "carwelding/carwelding/Easy/P007",
    "endofworld/endofworld/Easy/P009",
    "gascola/gascola/Easy/P008",
    "gascola/gascola/Hard/P009",
    "hospital/hospital/Easy/P036",
    "hospital/hospital/Hard/P049",
    "japanesealley/japanesealley/Easy/P007",
    "japanesealley/japanesealley/Hard/P005",
    "neighborhood/neighborhood/Easy/P021",
    "neighborhood/neighborhood/Hard/P017",
    "ocean/ocean/Easy/P013",
    "ocean/ocean/Hard/P009",
    "office2/office2/Easy/P011",
    "office2/office2/Hard/P010",
    "office/office/Hard/P007",
    "oldtown/oldtown/Easy/P007",
    "oldtown/oldtown/Hard/P008",
    "seasidetown/seasidetown/Easy/P009",
    "seasonsforest/seasonsforest/Easy/P011",
    "seasonsforest/seasonsforest/Hard/P006",
    "seasonsforest_winter/seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/seasonsforest_winter/Hard/P018",
    "soulcity/soulcity/Easy/P012",
    "soulcity/soulcity/Hard/P009",
    "westerndesert/westerndesert/Easy/P013",
    "westerndesert/westerndesert/Hard/P007",
]


class TartanAir(RGBDDataset):
    """TartanAir dataset for DPVO visual-odometry training.

    Reads left-camera RGB images, depth maps, and poses from the TartanAir
    directory layout.  Depths are divided by :attr:`DEPTH_SCALE` to
    balance the relative magnitude of rotational and translational
    components in the training loss.
    """

    DEPTH_SCALE: float = 5.0
    """Global depth scaling factor applied to both depths and translations."""

    def __init__(self, mode: str = 'training', **kwargs: object) -> None:
        """Initialise the TartanAir dataset.

        Args:
            mode: Dataset split identifier (currently unused beyond storage).
            **kwargs: Forwarded to :class:`RGBDDataset` (``datapath``,
                ``n_frames``, ``crop_size``, etc.).
        """
        self.mode: str = mode
        self.n_frames: int = 2
        super().__init__(name='TartanAir', **kwargs)

    @staticmethod
    def is_test_scene(scene: str) -> bool:
        """Check whether ``scene`` belongs to the validation/test split.

        Args:
            scene: Scene path string (may be absolute or relative).

        Returns:
            ``True`` if the scene matches any entry in :data:`test_split`.
        """
        return any(x in scene for x in test_split)

    def _build_dataset(self) -> dict[str, dict[str, object]]:
        """Scan the dataset root and build per-scene metadata dicts.

        For each scene directory the method collects image paths, depth
        paths, poses (axis-reordered and depth-scaled), fixed intrinsics,
        and a co-visibility frame graph.

        Returns:
            A dict mapping scene path strings to metadata dicts with keys
            ``images``, ``depths``, ``poses``, ``intrinsics``, ``graph``.
        """
        from tqdm import tqdm
        print("Building TartanAir dataset")

        scene_info: dict[str, dict[str, object]] = {}
        scenes: list[str] = glob.glob(osp.join(self.root, '*/*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images: list[str] = sorted(glob.glob(osp.join(scene, 'image_left/*.png')))
            depths: list[str] = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))

            if len(images) != len(depths):
                continue

            poses: Float64[np.ndarray, "n 7"] = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics: list[Float64[np.ndarray, "4"]] = [TartanAir.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph: dict[int, tuple[np.ndarray, np.ndarray]] = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read() -> Float64[np.ndarray, "4"]:
        """Return the fixed TartanAir intrinsics ``[fx, fy, cx, cy]``.

        All TartanAir scenes share the same 640x480 pinhole camera with
        focal length 320 and principal point at (320, 240).

        Returns:
            Intrinsic vector ``[320, 320, 320, 240]``.
        """
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file: str) -> UInt8[np.ndarray, "h w 3"]:
        """Read an image from disk via OpenCV (BGR channel order).

        Args:
            image_file: Absolute path to the image file.

        Returns:
            The loaded image as a uint8 array with shape ``(h, w, 3)``.
        """
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file: str) -> Float64[np.ndarray, "h w"]:
        """Read and scale a TartanAir depth map from a ``.npy`` file.

        The raw depth is divided by :attr:`DEPTH_SCALE`. ``NaN`` and
        ``inf`` values are replaced with 1.0.

        Args:
            depth_file: Absolute path to the ``.npy`` depth file.

        Returns:
            Scaled depth map as a float64 array with shape ``(h, w)``.
        """
        depth: Float64[np.ndarray, "h w"] = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth
