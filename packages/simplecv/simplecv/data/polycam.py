import zipfile
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import batched, chain, islice
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from jaxtyping import Float32, UInt8, UInt16
from serde import serde
from serde.json import from_json
from torch import Tensor
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.ops import conventions


class DepthConfidenceLevel(IntEnum):
    LOW = 0
    MEDIUM = 54
    HIGH = 255


@serde
class PolycamCameraData:
    """
    serde class to load camera data from json
    """

    blur_score: float
    cx: float
    cy: float
    fx: float
    fy: float
    height: int
    manual_keyframe: bool
    t_00: float
    t_01: float
    t_02: float
    t_03: float
    t_10: float
    t_11: float
    t_12: float
    t_13: float
    t_20: float
    t_21: float
    t_22: float
    t_23: float
    timestamp: int
    width: int
    neighbors: list[int] = field(default_factory=list)  # default value is an empty list

    def __post_init__(self) -> None:
        self.world_T_cam_44: Float32[np.ndarray, "4 4"] = (
            self.calculate_world_t_cam_44()
        )

    def calculate_world_t_cam_44(self) -> Float32[np.ndarray, "4 4"]:
        # in opengl convention
        world_T_cam_44_gl: Float32[np.ndarray, "4 4"] = np.array(
            [
                [self.t_00, self.t_01, self.t_02, self.t_03],
                [self.t_10, self.t_11, self.t_12, self.t_13],
                [self.t_20, self.t_21, self.t_22, self.t_23],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # convert to opencv convention
        world_T_cam_44_cv: Float32[np.ndarray, "4 4"] = conventions.convert_pose(
            world_T_cam_44_gl,
            src_convention=conventions.CC.GL,
            dst_convention=conventions.CC.CV,
        )
        return world_T_cam_44_cv


@dataclass
class PolycamData:
    rgb_hw3: UInt8[np.ndarray, "h w 3"]
    """rgb_hw3: RGB image array with shape (height, width, 3) and uint8 values"""
    depth_hw: UInt16[np.ndarray, "h w"]
    """depth_hw: Depth map array with same shape as image (height, width) and uint16 values"""
    confidence_hw: UInt8[np.ndarray, "h w"]
    """confidence_hw: Confidence map array with same shape as image (height, width) and uint8 values"""
    original_depth_hw: UInt16[np.ndarray, "192 256"]
    """original_depth_hw: Original depth map array with fixed shape (192, 256) and uint16 values"""
    original_confidence_hw: UInt8[np.ndarray, "192 256"]
    """original_confidence_hw: Original confidence map array with fixed shape (192, 256) and uint8 values"""
    pinhole_params: PinholeParameters
    """pinhole_params: Camera pinhole parameters"""


@dataclass
class PolycamDataset:
    keyframe_dir: Path
    cameras_paths: list[Path] = field(default_factory=list)
    images_paths: list[Path] = field(default_factory=list)
    confidence_paths: list[Path] = field(default_factory=list)
    depth_paths: list[Path] = field(default_factory=list)
    using_corrected: bool = False

    def load_data(self) -> None:
        self.cameras_path: list[Path] = sorted(
            (self.keyframe_dir / "cameras").glob("*.json")
        )
        self.images_path: list[Path] = sorted(
            (self.keyframe_dir / "images").glob("*.jpg")
        )
        self.confidence_path: list[Path] = sorted(
            (self.keyframe_dir / "confidence").glob("*.png")
        )
        self.depth_path: list[Path] = sorted(
            (self.keyframe_dir / "depth").glob("*.png")
        )
        if (self.keyframe_dir / "corrected_cameras").exists() and (
            self.keyframe_dir / "corrected_images"
        ).exists():
            self.using_corrected = True
            self.cameras_path: list[Path] = sorted(
                (self.keyframe_dir / "corrected_cameras").glob("*.json")
            )
            self.images_path: list[Path] = sorted(
                (self.keyframe_dir / "corrected_images").glob("*.jpg")
            )

        # make sure the number of files are the same
        assert (
            len(self.cameras_path)
            == len(self.images_path)
            == len(self.confidence_path)
            == len(self.depth_path)
        ), "Number of files do not match"
        # make sure some files are present
        assert len(self.cameras_path) > 0, "No camera files found"

    def __post_init__(self) -> None:
        self.load_data()

    def load_camera_params(self, camera_path: Path) -> PinholeParameters:
        with camera_path.open("r") as f:
            camera_data_json: str = f.read()

        raw_camera_data: PolycamCameraData = from_json(
            PolycamCameraData, camera_data_json
        )

        pinhole_params = PinholeParameters(
            name="perspective-camera",
            extrinsics=Extrinsics(
                world_R_cam=raw_camera_data.world_T_cam_44[:3, :3],
                world_t_cam=raw_camera_data.world_T_cam_44[:3, 3],
            ),
            intrinsics=Intrinsics(
                camera_conventions="RDF",
                fl_x=raw_camera_data.fx,
                fl_y=raw_camera_data.fy,
                cx=raw_camera_data.cx,
                cy=raw_camera_data.cy,
                height=raw_camera_data.height,
                width=raw_camera_data.width,
            ),
        )
        return pinhole_params

    def __len__(self) -> int:
        return len(self.cameras_path)

    def __iter__(self) -> Generator[PolycamData, Any, None]:
        for camera_path, image_path, confidence_path, depth_path in zip(
            self.cameras_path,
            self.images_path,
            self.confidence_path,
            self.depth_path,
            strict=True,
        ):
            # load camera parameters
            pinhole_params: PinholeParameters = self.load_camera_params(camera_path)

            # load image
            bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.imread(str(image_path))
            rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(
                bgr_hw3, cv2.COLOR_BGR2RGB
            )

            # load depth
            original_depth: UInt16[np.ndarray, "192 256"] = cv2.imread(
                str(depth_path), cv2.IMREAD_ANYDEPTH
            )
            # upscale to image size
            depth_hw: UInt16[np.ndarray, "h w"] = cv2.resize(
                original_depth,
                (pinhole_params.intrinsics.width, pinhole_params.intrinsics.height),
                interpolation=cv2.INTER_LINEAR,
            )

            # load confidence
            original_confidence: UInt8[np.ndarray, "192 256"] = cv2.imread(
                str(confidence_path), cv2.IMREAD_GRAYSCALE
            )
            # upscale to image size
            confidence_hw: UInt8[np.ndarray, "h w"] = cv2.resize(
                original_confidence,
                (pinhole_params.intrinsics.width, pinhole_params.intrinsics.height),
                interpolation=cv2.INTER_NEAREST,  # to make sure no new values are introduced
            )

            yield PolycamData(
                rgb_hw3=rgb_hw3,
                depth_hw=depth_hw,
                confidence_hw=confidence_hw,
                original_depth_hw=original_depth,
                original_confidence_hw=original_confidence,
                pinhole_params=pinhole_params,
            )


@dataclass(frozen=True, slots=True)
class PolycamBatchPlan:
    """Peeked first batch and one-shot bounded iterator for one capture."""

    first_batch: tuple[PolycamData, ...]
    """First batch, exposed for predictor sizing before iteration."""
    total_batches: int
    """Ceiling-divided batch count shown by the progress bar."""
    _remaining_batches: Iterator[tuple[PolycamData, ...]]
    """Unconsumed batches after the peeked first batch."""
    _description: str
    """Progress-bar label."""

    def __iter__(self) -> Iterator[tuple[PolycamData, ...]]:
        """Yield all bounded batches, including ``first_batch``, with progress."""
        yield from tqdm(
            chain((self.first_batch,), self._remaining_batches),
            desc=self._description,
            total=self.total_batches,
        )


def prepare_polycam_batches(
    dataset: PolycamDataset,
    *,
    batch_size: int,
    max_frames: int | None,
    capture_path: Path,
    description: str,
) -> PolycamBatchPlan:
    """Peek and progress-wrap batches while enforcing an exact frame budget."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    frame_count: int = min(max_frames if max_frames is not None else len(dataset), len(dataset))
    batch_iterator: Iterator[tuple[PolycamData, ...]] = batched(islice(dataset, frame_count), batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batch_iterator, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {capture_path} contains no frames.")

    total_batches: int = -(-frame_count // batch_size)
    return PolycamBatchPlan(
        first_batch=first_batch,
        total_batches=total_batches,
        _remaining_batches=batch_iterator,
        _description=description,
    )


def stack_polycam_batch(batch: tuple[PolycamData, ...]) -> tuple[UInt8[Tensor, "b h w 3"], Float32[Tensor, "b 192 256"]]:
    """Stack one batch into raw CUDA RGB and metric-depth prompt tensors."""
    prompt_m_bhw: Float32[np.ndarray, "b 192 256"] = np.stack([data.original_depth_hw for data in batch], dtype=np.float32)
    prompt_m_bhw /= 1000.0
    rgb_bhwc: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch])).cuda()
    prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(prompt_m_bhw).cuda()
    return rgb_bhwc, prompt_bhw


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    assert zip_path.suffix == ".zip", "Not a zip file"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def validate_zip(zip_path: Path) -> bool:
    return zip_path.is_file() and zip_path.exists() and zip_path.suffix == ".zip"


def find_keyframes_dir(extract_dir: Path) -> Path | None:
    # Skip __MACOSX metadata cache
    for path in extract_dir.rglob("*"):
        if "__MACOSX" in path.parts:
            continue
        if path.is_dir() and path.name == "keyframes":
            return path
    return None

def load_polycam_data(polycam_zip_or_directory_path: Path) -> PolycamDataset:
    extract_dir: Path = polycam_zip_or_directory_path.with_suffix("")
    zip_path: Path = polycam_zip_or_directory_path

    if validate_zip(zip_path) and (not extract_dir.exists()):
        extract_zip(zip_path, extract_dir=extract_dir)

    keyframes_dir: Path | None = find_keyframes_dir(extract_dir)
    assert keyframes_dir is not None, "Keyframes directory not found"
    return PolycamDataset(keyframe_dir=keyframes_dir)
