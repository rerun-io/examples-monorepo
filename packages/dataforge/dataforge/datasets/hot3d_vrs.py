"""HOT3D VRS streams and the calibration supplied alongside ground truth."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import CameraCalibration, CameraModelType, rotate_camera_calib_cw90deg
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation
from serde import field, serde
from simplecv.camera_parameters import Fisheye62Parameters

from dataforge import aria
from dataforge.datasets.hot3d_source import DEVICES, Device, Hot3dSource, Labels, ManoFrame, MaskSamples, Metadata, PoseRow, UmeFrame, read_labels
from dataforge.datasets.show3d_source import read_json


@serde
@dataclass(frozen=True, slots=True)
class CameraTransform:
    """JSON device-from-camera, solved in the GT device frame."""

    quaternion_wxyz: Float64[ndarray, "4"]
    """Scalar-first rotation."""
    translation_xyz: Float64[ndarray, "3"]
    """Translation in metres."""

    def matrix(self) -> Float64[ndarray, "4 4"]:
        """Return device-from-camera."""
        result: Float64[ndarray, "4 4"] = np.eye(4)
        result[:3, :3] = Rotation.from_quat(self.quaternion_wxyz, scalar_first=True).as_matrix()
        result[:3, 3] = self.translation_xyz
        return result


@serde
@dataclass(frozen=True, slots=True)
class CameraModel:
    """The shipped FISHEYE624 model for either headset."""

    label: str
    """Source camera label."""
    stream_id: aria.AriaStreamId
    """VRS stream identifier."""
    width: int = field(rename="imageWidth")
    """Native width."""
    height: int = field(rename="imageHeight")
    """Native height."""
    projection: str = field(rename="projectionModelType")
    """Must be FISHEYE624."""
    params: list[float] = field(rename="projectionParams")
    """15 single-focal Aria or 16 dual-focal Quest parameters."""
    device_T_camera: CameraTransform = field(rename="T_Device_Camera")
    """Camera pose in the GT device frame."""
    max_solid_angle: float = field(rename="maxSolidAngle")
    """Valid projection cone in radians."""

    def __post_init__(self) -> None:
        if self.projection != "CameraModelType.FISHEYE624" or len(self.params) not in (15, 16):
            raise ValueError(f"{self.label}: expected 15 or 16 FISHEYE624 parameters")

    def calibration(self, *, rotate_cw90: bool) -> CameraCalibration:
        """Keep the full FISHEYE624 model, including thin-prism and field of view."""
        params: list[float] = self.params
        if len(params) == 16:
            if params[0] != params[1]:
                raise ValueError(f"{self.label}: expected fx == fy, got fx={params[0]}, fy={params[1]}")
            params = [params[0], *params[2:]]
        native: CameraCalibration = CameraCalibration(
            self.label,
            CameraModelType.FISHEYE624,
            np.asarray(params),
            SE3.from_matrix(self.device_T_camera.matrix()),
            self.width,
            self.height,
            None,
            self.max_solid_angle,
            "",
        )
        return rotate_camera_calib_cw90deg(native) if rotate_cw90 else native

    def camera(self, *, rotate_cw90: bool) -> Fisheye62Parameters:
        """Convert the library calibration to the logged camera approximation."""
        calibration: CameraCalibration = self.calibration(rotate_cw90=rotate_cw90)
        return aria.fisheye62_from_aria(
            calibration,
            rig_T_cam=np.asarray(calibration.get_transform_device_camera().to_matrix()),
            name=self.label,
        )


@dataclass(frozen=True, slots=True)
class CameraStream:
    """A camera's native clock and the exact frame prefix to encode."""

    model: CameraModel
    """Shipped calibration."""
    times_ns: Int64[ndarray, "n"]
    """Unshifted VRS device capture timestamps."""
    calibration: CameraCalibration
    """Rotated full lens model for derived projections."""
    source_count: int
    """Full stream count before any preview limit."""


@dataclass(frozen=True, slots=True)
class Scene:
    """One opened VRS and native label clocks shared by layer writers."""

    source: Path
    """Read-only sequence directory."""
    device: Device
    """Camera orientation and clock policy."""
    provider: data_provider.VrsDataProvider
    """Provider used serially; never shared between encoding threads."""
    metadata: Metadata
    """GT and episode inventory."""
    cameras: list[CameraStream]
    """RGB first for Aria, left first for Quest."""
    hands: dict[int, UmeFrame]
    """Every retained label row."""
    times_ns: Int64[ndarray, "n"]
    """Fixed device census (labels own their 1 ns offset)."""
    frame_indices: Int64[ndarray, "n"]
    """Nearest primary frameset index, with ties going left."""
    mano: dict[int, ManoFrame]
    """MANO rows keyed by device time."""
    masks: dict[str, dict[str, MaskSamples]]
    """Device-stamped quality arrays."""
    headset: dict[int, PoseRow]
    """Sparse native GT headset poses."""
    objects: dict[str, dict[int, PoseRow]]
    """Sparse native GT object poses."""
    stop_ns: int | None
    """Preview cutoff or None for full native streams."""


def nearest_framesets(primary: Int64[ndarray, "f"], times_ns: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Associate labels with primary frames for the secondary timeline; ties go left."""
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(primary, times_ns), 0, len(primary) - 1)
    left: Int64[ndarray, "n"] = np.maximum(right - 1, 0)
    return np.where(np.abs(times_ns - primary[left]) <= np.abs(times_ns - primary[right]), left, right).astype(np.int64)


def read_scene(source: Hot3dSource, device: Device, frame_limit: int | None = None) -> Scene:
    """Read native cameras and GT; previews include every stream's Nth stamp."""
    provider: data_provider.VrsDataProvider = aria.open_vrs(source.path / "recording.vrs")
    models: list[CameraModel] = read_json(source.path / "camera_models.json", list[CameraModel])
    cameras: list[CameraStream] = []
    primary: Int64[ndarray, "n"] = np.empty(0, dtype=np.int64)
    for stream, _ in DEVICES[device].camera_streams:
        model: CameraModel = next(model for model in models if model.stream_id == stream)
        times: Int64[ndarray, "n"] = aria.frame_timestamps_ns(provider, model.stream_id)
        if not len(times):
            raise ValueError(f"{source.path}/{stream}: empty camera stream")
        if not cameras:
            primary = times
        cameras.append(CameraStream(model, times[:frame_limit], model.calibration(rotate_cw90=True), len(times)))
    stop: int | None = max(int(camera.times_ns[-1]) for camera in cameras) if frame_limit is not None else None
    labels: Labels = read_labels(source, device, primary, stop)
    frames: Int64[ndarray, "n"] = nearest_framesets(cameras[0].times_ns, labels.times_ns)
    return Scene(
        source.path,
        device,
        provider,
        source.metadata,
        cameras,
        labels.hands,
        labels.times_ns,
        frames,
        labels.mano,
        labels.masks,
        labels.headset,
        labels.objects,
        stop,
    )
