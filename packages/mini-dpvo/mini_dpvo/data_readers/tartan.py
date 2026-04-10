
import glob
import os.path as osp

import cv2
import numpy as np
from jaxtyping import Float64, UInt8

from .base import RGBDDataset

# cur_path = osp.dirname(osp.abspath(__file__))
# test_split = osp.join(cur_path, 'tartan_test.txt')
# test_split = open(test_split).read().split()


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

    # scale depths to balance rot & trans
    DEPTH_SCALE: float = 5.0

    def __init__(self, mode: str = 'training', **kwargs: object) -> None:
        self.mode: str = mode
        self.n_frames: int = 2
        super().__init__(name='TartanAir', **kwargs)

    @staticmethod
    def is_test_scene(scene: str) -> bool:
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self) -> dict[str, dict[str, object]]:
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
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file: str) -> UInt8[np.ndarray, "h w 3"]:
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file: str) -> Float64[np.ndarray, "h w"]:
        depth: Float64[np.ndarray, "h w"] = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth
