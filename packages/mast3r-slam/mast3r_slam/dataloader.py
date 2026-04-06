import pathlib
import re

import cv2
import numpy as np
import torch
from jaxtyping import Float32, UInt8
from natsort import natsorted

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
from typing import Literal

import yaml

from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import resize_img

HAS_TORCHCODEC: bool = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    HAS_TORCHCODEC = False


class MonocularDataset(torch.utils.data.Dataset):
    """Base class for monocular image datasets.

    Subclasses populate ``rgb_files`` and ``timestamps`` and optionally set
    ``camera_intrinsics`` for calibrated operation.
    """

    def __init__(
        self,
        dtype: type = np.float32,
        img_size: Literal[224, 512] = 512,
    ) -> None:
        self.dtype: type = dtype
        self.rgb_files: list = []
        self.timestamps: list = []
        self.img_size: Literal[224, 512] = img_size
        self.camera_intrinsics = None
        self.use_calibration: bool = config["use_calib"]
        self.save_results: bool = True

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.rgb_files)

    def __getitem__(self, index: int) -> tuple[float | str, np.ndarray]:
        """Return (timestamp, image) for the given index.

        Args:
            index: Frame index.

        Returns:
            A tuple of (timestamp, normalised float image).
        """
        # Call get_image before timestamp for realsense camera
        img: np.ndarray = self.get_image(index)
        timestamp = self.get_timestamp(index)
        return timestamp, img

    def get_timestamp(self, idx: int) -> float | str:
        """Return the timestamp for a given frame index.

        Args:
            idx: Frame index.

        Returns:
            Timestamp value (float or string depending on dataset).
        """
        return self.timestamps[idx]

    def read_img(self, idx: int) -> UInt8[np.ndarray, "h w 3"]:
        """Read the raw RGB image at the given index.

        Args:
            idx: Frame index.

        Returns:
            RGB uint8 image of shape (h, w, 3).
        """
        img_raw = cv2.imread(self.rgb_files[idx])
        assert img_raw is not None, f"Failed to read image: {self.rgb_files[idx]}"
        img: UInt8[np.ndarray, "h w 3"] = img_raw
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx: int) -> Float32[np.ndarray, "h w 3"]:
        """Return the preprocessed float image (optionally undistorted) at the given index.

        Args:
            idx: Frame index.

        Returns:
            Float32 image in [0, 1] range.
        """
        raw_img: np.ndarray = self.read_img(idx)
        if self.use_calibration:
            assert self.camera_intrinsics is not None
            raw_img = self.camera_intrinsics.remap(raw_img)
        return raw_img.astype(self.dtype) / 255.0

    def get_img_shape(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the processed and raw image shapes.

        Returns:
            A tuple of ((processed_h, processed_w), (raw_h, raw_w)).
        """
        raw_img: np.ndarray = self.read_img(0)
        raw_img_shape: tuple[int, int] = raw_img.shape[:2]
        resized = resize_img(raw_img, self.img_size)
        assert isinstance(resized, dict)
        # 3XHxW, HxWx3 -> HxW, HxW
        processed_shape: torch.Size = resized["img"][0].shape[1:]
        return (int(processed_shape[0]), int(processed_shape[1])), raw_img_shape

    def subsample(self, subsample: int) -> None:
        """Subsample the dataset by keeping every ``subsample``-th frame.

        Args:
            subsample: Step size for subsampling.
        """
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self) -> bool:
        """Return whether camera intrinsics are available.

        Returns:
            True if ``camera_intrinsics`` is not None.
        """
        return self.camera_intrinsics is not None


class TUMDataset(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512]) -> None:
        super().__init__(img_size=img_size)
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        rgb_list: pathlib.Path = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.str_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = list(tstamp_rgb[:, 0])

        match = re.search(r"freiburg(\d+)", dataset_path)
        assert match is not None, f"Could not find freiburg index in path: {dataset_path}"
        idx: int = int(match.group(1))
        if idx == 1:
            calib = np.array(
                [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
            )
        elif idx == 2:
            calib = np.array(
                [520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
            )
        elif idx == 3:
            calib = np.array([535.4, 539.2, 320.1, 247.6])
        else:
            raise ValueError(f"Unknown TUM freiburg index: {idx}")
        W: int = 640
        H: int = 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)


class EurocDataset(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512] = 512) -> None:
        super().__init__(img_size=img_size)
        # For Euroc dataset, the distortion is too much to handle for MASt3R.
        # So we always undistort the images, but the calibration will not be used for any later optimization unless specified.
        self.use_calibration: bool = True
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        rgb_list: pathlib.Path = self.dataset_path / "mav0/cam0/data.csv"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=",", dtype=np.str_, skiprows=0)
        self.rgb_files = [
            self.dataset_path / "mav0/cam0/data" / f for f in tstamp_rgb[:, 1]
        ]
        self.timestamps = list(tstamp_rgb[:, 0])
        with open(self.dataset_path / "mav0/cam0/sensor.yaml") as f:
            self.cam0 = yaml.load(f, Loader=yaml.FullLoader)
        W: int
        H: int
        W, H = self.cam0["resolution"]
        intrinsics = self.cam0["intrinsics"]
        distortion = np.array(self.cam0["distortion_coefficients"])
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intrinsics, *distortion], always_undistort=True
        )

    def read_img(self, idx: int) -> UInt8[np.ndarray, "h w 3"]:
        img_raw = cv2.imread(self.rgb_files[idx], cv2.IMREAD_GRAYSCALE)
        assert img_raw is not None, f"Failed to read image: {self.rgb_files[idx]}"
        img = img_raw
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ETH3DDataset(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512] = 512) -> None:
        super().__init__(img_size=img_size)
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        rgb_list: pathlib.Path = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.str_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = list(tstamp_rgb[:, 0])
        calibration = np.loadtxt(
            self.dataset_path / "calibration.txt",
            delimiter=" ",
            dtype=np.float32,
            skiprows=0,
        )
        _, (H, W) = self.get_img_shape()
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calibration)


class SevenScenesDataset(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512] = 512) -> None:
        super().__init__(img_size=img_size)
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(
            list((self.dataset_path / "seq-01").glob("*.color.png"))
        )
        self.timestamps = list(np.arange(0, len(self.rgb_files)).astype(self.dtype))
        fx: float = 585.0
        fy: float = 585.0
        cx: float = 320.0
        cy: float = 240.0
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, 640, 480, [fx, fy, cx, cy]
        )


class RealsenseDataset(MonocularDataset):
    def __init__(self, img_size: Literal[224, 512] = 512) -> None:
        super().__init__(img_size=img_size)
        assert rs is not None, "pyrealsense2 is required for RealsenseDataset"
        self.dataset_path = None
        self.pipeline = rs.pipeline()
        # self.h, self.w = 720, 1280
        self.h: int = 480
        self.w: int = 640
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 30
        )
        self.profile = self.pipeline.start(self.rs_config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        # self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        # self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.save_results: bool = False

        if self.use_calibration:
            rgb_intrinsics = self.rgb_profile.get_intrinsics()
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size,
                self.w,
                self.h,
                [
                    rgb_intrinsics.fx,
                    rgb_intrinsics.fy,
                    rgb_intrinsics.ppx,
                    rgb_intrinsics.ppy,
                ],
            )

    def __len__(self) -> int:
        return 999999

    def get_timestamp(self, idx: int) -> float:
        return self.timestamps[idx]

    def read_img(self, idx: int) -> UInt8[np.ndarray, "h w 3"]:
        frameset = self.pipeline.wait_for_frames()
        timestamp: float = frameset.get_timestamp()
        timestamp /= 1000
        self.timestamps.append(timestamp)

        rgb_frame = frameset.get_color_frame()
        img: UInt8[np.ndarray, "h w 3"] = np.asanyarray(rgb_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img


class Webcam(MonocularDataset):
    def __init__(self, img_size: Literal[224, 512] = 512) -> None:
        super().__init__(img_size=img_size)
        self.use_calibration: bool = False
        self.dataset_path = None
        # load webcam using opencv
        self.cap = cv2.VideoCapture(-1)
        self.save_results: bool = False

    def __len__(self) -> int:
        return 999999

    def get_timestamp(self, idx: int) -> float:
        return self.timestamps[idx]

    def read_img(self, idx: int) -> UInt8[np.ndarray, "h w 3"]:
        ret: bool
        img: UInt8[np.ndarray, "h w 3"]
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512]) -> None:
        super().__init__(img_size=img_size)
        self.use_calibration: bool = False
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps: float = self.decoder.metadata.average_fps
            self.total_frames: int = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride: int = config["dataset"]["subsample"]

    def __len__(self) -> int:
        return self.total_frames // self.stride

    def read_img(self, idx: int) -> np.ndarray:
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]  # c,h,w
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret: bool
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        timestamp: float = idx / self.fps
        self.timestamps.append(timestamp)
        return img


class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path: str, img_size: Literal[224, 512]) -> None:
        super().__init__(img_size=img_size)
        self.use_calibration: bool = False
        self.dataset_path: pathlib.Path = pathlib.Path(dataset_path)
        # Support both PNG and JPG/JPEG files
        png_files: list[pathlib.Path] = list((self.dataset_path).glob("*.png"))
        jpg_files: list[pathlib.Path] = list((self.dataset_path).glob("*.jpg"))
        jpeg_files: list[pathlib.Path] = list((self.dataset_path).glob("*.jpeg"))
        self.rgb_files = natsorted(png_files + jpg_files + jpeg_files)
        self.timestamps = list(np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0)


class Intrinsics:
    """Camera intrinsics including undistortion maps and frame-level calibration.

    Stores the original and optimal camera matrices, distortion coefficients,
    and precomputed undistortion remap tables.  Also computes the intrinsic
    matrix in the MASt3R frame coordinate system.
    """

    def __init__(
        self,
        img_size: Literal[224, 512],
        W: int,
        H: int,
        K_orig: Float32[np.ndarray, "3 3"],
        K: Float32[np.ndarray, "3 3"],
        distortion: Float32[np.ndarray, "n_distortion"],
        mapx: Float32[np.ndarray, "H W"] | None,
        mapy: Float32[np.ndarray, "H W"] | None,
    ) -> None:
        self.img_size: Literal[224, 512] = img_size
        """Target image size for MASt3R (224 or 512)."""
        self.W: int = W
        """Original image width in pixels."""
        self.H: int = H
        """Original image height in pixels."""
        self.K_orig: Float32[np.ndarray, "3 3"] = K_orig
        """Original 3x3 camera intrinsic matrix."""
        self.K: Float32[np.ndarray, "3 3"] = K
        """Optimal 3x3 camera intrinsic matrix after undistortion."""
        self.distortion: Float32[np.ndarray, "n_distortion"] = distortion
        """Distortion coefficients."""
        self.mapx: Float32[np.ndarray, "H W"] | None = mapx
        """Horizontal undistortion remap table."""
        self.mapy: Float32[np.ndarray, "H W"] | None = mapy
        """Vertical undistortion remap table."""
        resize_result = resize_img(
            np.zeros((H, W, 3)), self.img_size, return_transformation=True
        )
        assert isinstance(resize_result, tuple)
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_result
        self.K_frame: Float32[np.ndarray, "3 3"] = self.K.copy()
        """Intrinsic matrix transformed to the MASt3R frame coordinate system."""
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img: UInt8[np.ndarray, "H W 3"]) -> UInt8[np.ndarray, "H W 3"]:
        """Apply undistortion remap to an image.

        Args:
            img: Input distorted image.

        Returns:
            Undistorted image.
        """
        assert self.mapx is not None
        assert self.mapy is not None
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(
        img_size: Literal[224, 512],
        W: int,
        H: int,
        calib: list | np.ndarray,
        always_undistort: bool = False,
    ) -> "Intrinsics | None":
        """Construct an Intrinsics instance from a calibration vector.

        Args:
            img_size: Target image size for MASt3R.
            W: Original image width.
            H: Original image height.
            calib: Calibration parameters [fx, fy, cx, cy, ...distortion].
            always_undistort: Force undistortion even when ``use_calib`` is off.

        Returns:
            An Intrinsics instance, or None if calibration is not active and
            ``always_undistort`` is False.
        """
        if not config["use_calib"] and not always_undistort:
            return None
        fx: float = calib[0]
        fy: float = calib[1]
        cx: float = calib[2]
        cy: float = calib[3]
        distortion: Float32[np.ndarray, "n_distortion"] = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K: Float32[np.ndarray, "3 3"] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt: Float32[np.ndarray, "3 3"] = K.copy()
        mapx: Float32[np.ndarray, "H W"] | None = None
        mapy: Float32[np.ndarray, "H W"] | None = None
        center: bool = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3, dtype=np.float64), K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


def load_dataset(
    dataset_path: str,
    img_size: Literal[224, 512] = 512,
) -> MonocularDataset:
    """Load a dataset by auto-detecting its type from the path.

    Supports TUM, EuRoC, ETH3D, 7-Scenes, RealSense, webcam, MP4/AVI/MOV
    video files, and directories of RGB image files.

    Args:
        dataset_path: Filesystem path or special keyword (e.g. "realsense").
        img_size: Target image size for MASt3R (224 or 512).

    Returns:
        A MonocularDataset subclass instance.
    """
    split_dataset_type: list[str] = dataset_path.split("/")
    if "tum" in split_dataset_type:
        return TUMDataset(dataset_path, img_size)
    if "euroc" in split_dataset_type:
        return EurocDataset(dataset_path, img_size)
    if "eth3d" in split_dataset_type:
        return ETH3DDataset(dataset_path, img_size)
    if "7-scenes" in split_dataset_type:
        return SevenScenesDataset(dataset_path, img_size)
    if "realsense" in split_dataset_type:
        return RealsenseDataset(img_size)
    if "webcam" in split_dataset_type:
        return Webcam(img_size)

    ext: str = split_dataset_type[-1].split(".")[-1]
    if ext in ["mp4", "avi", "MOV", "mov"]:
        return MP4Dataset(dataset_path, img_size)
    return RGBFiles(dataset_path, img_size)
