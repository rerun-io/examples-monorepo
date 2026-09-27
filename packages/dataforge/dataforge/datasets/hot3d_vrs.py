"""HOT3D VRS streams and the calibration supplied alongside ground truth."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import field, serde
from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.se3 import SE3
from simplecv.sensors.camera import fisheye624

from dataforge import aria
from dataforge.clocks import nearest_framesets
from dataforge.datasets.hot3d_source import DEVICES, Device, Hot3dSource, Labels, ManoFrame, MaskSamples, Metadata, PoseRow, UmeFrame, read_labels
from dataforge.records import read_json
from dataforge.vrs import VrsFile


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

    def calibration(self, *, rotate_cw90: bool) -> Fisheye624Parameters:
        """Keep the full FISHEYE624 model, including thin-prism and field of view.

        The pose goes from the file's quaternion straight to ``SE3``; the SDK took a
        detour through a matrix and an SVD fit that moves the last bits of the rotation.
        """
        params: list[float] = self.params
        if len(params) == 16:
            if params[0] != params[1]:
                raise ValueError(f"{self.label}: expected fx == fy, got fx={params[0]}, fy={params[1]}")
            params = [params[0], *params[2:]]
        native: Fisheye624Parameters = Fisheye624Parameters(
            name=self.label,
            width=self.width,
            height=self.height,
            params=np.asarray(params, dtype=np.float64),
            rig_T_cam=SE3.from_quaternion(self.device_T_camera.quaternion_wxyz, self.device_T_camera.translation_xyz),
            max_solid_angle=self.max_solid_angle,
        )
        return fisheye624.rotate_cw90(native) if rotate_cw90 else native


@dataclass(frozen=True, slots=True)
class CameraStream:
    """A camera's native clock and the exact frame prefix to encode."""

    model: CameraModel
    """Shipped calibration."""
    times_ns: Int64[ndarray, "n"]
    """Unshifted VRS device capture timestamps."""
    calibration: Fisheye624Parameters
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
    vrs: VrsFile
    """``recording.vrs``, its description and record offsets read once."""
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


def read_scene(source: Hot3dSource, device: Device, frame_limit: int | None = None) -> Scene:
    """Read native cameras and GT; previews include every stream's Nth stamp."""
    vrs: VrsFile = VrsFile(source.path / "recording.vrs")
    models: list[CameraModel] = read_json(source.path / "camera_models.json", list[CameraModel])
    cameras: list[CameraStream] = []
    primary: Int64[ndarray, "n"] = np.empty(0, dtype=np.int64)
    for stream, _ in DEVICES[device].camera_streams:
        model: CameraModel = next(model for model in models if model.stream_id == stream)
        times: Int64[ndarray, "n"] = aria.frame_timestamps_ns(vrs, model.stream_id)
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
        vrs,
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
