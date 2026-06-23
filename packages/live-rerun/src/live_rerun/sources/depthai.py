"""DepthAI/OAK source backend: hardware-encoded H.264/H.265 + calibration.

Targets ``depthai==2.27.0.0`` (the 2.x API; 3.x fails to open the OAK-D-W unit
this was developed against). Three streams are encoded on-device — RGB (CAM_A,
IMX378) plus left/right mono (CAM_B/CAM_C, OV9282) — and pulled to the host as
``EncodedFrame`` packets so ``getFrameType()`` gives a correct per-packet
keyframe flag without parsing the bitstream.

IMU is deliberately never enabled: on the target unit even a minimal IMU stream
crashes the device (``INTERNAL_ERROR_CORE`` / ``IMUHalTask``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, cast

import depthai as dai
import numpy as np

from live_rerun.calibration import OakCameraCalib
from live_rerun.rerun_video_logger import Codec
from live_rerun.rig import SensorKind

# (label, board socket, image kind). Ordered reference-first: the LEFT mono
# camera (CAM_B) is the rig reference frame (identity -> cam_00); rgb (CAM_A) and
# right (CAM_C) follow. This order flows through to the calibration list, the
# cam_<index> entity paths, the video paths, and the side-by-side 2D views.
_SENSORS: tuple[tuple[str, dai.CameraBoardSocket, SensorKind], ...] = (
    ("left", dai.CameraBoardSocket.CAM_B, "grayscale"),
    ("rgb", dai.CameraBoardSocket.CAM_A, "rgb"),
    ("right", dai.CameraBoardSocket.CAM_C, "grayscale"),
)
_REFERENCE_LABEL: str = "left"
_SOCKETS: dict[str, dai.CameraBoardSocket] = {label: socket for label, socket, _ in _SENSORS}

_PROFILES: dict[Codec, dai.VideoEncoderProperties.Profile] = {
    "h264": dai.VideoEncoderProperties.Profile.H264_MAIN,
    "h265": dai.VideoEncoderProperties.Profile.H265_MAIN,
}

# label -> (sensor resolution, isp_scale (num, den) | None, encoded width, encoded height).
# The IMX378 has no native 720p mode, so "720p" uses the 1080p sensor downscaled by
# 2/3 via the ISP (1920x1080 -> 1280x720), matching the mono OV9282 720p exactly.
_RGB_RESOLUTIONS: dict[str, tuple[dai.ColorCameraProperties.SensorResolution, tuple[int, int] | None, int, int]] = {
    "720p": (dai.ColorCameraProperties.SensorResolution.THE_1080_P, (2, 3), 1280, 720),
    "1080p": (dai.ColorCameraProperties.SensorResolution.THE_1080_P, None, 1920, 1080),
    "4k": (dai.ColorCameraProperties.SensorResolution.THE_4_K, None, 3840, 2160),
}
_MONO_RESOLUTIONS: dict[str, tuple[dai.MonoCameraProperties.SensorResolution, int, int]] = {
    "720p": (dai.MonoCameraProperties.SensorResolution.THE_720_P, 1280, 720),
    "800p": (dai.MonoCameraProperties.SensorResolution.THE_800_P, 1280, 800),
}


@dataclass
class DepthAiConfig:
    """OAK capture knobs (mirrors the validated prototype)."""

    codec: Codec = "h265"
    """Hardware encoder codec. Both stream straight into Rerun VideoStream."""
    fps: float = 30.0
    """Capture/encode framerate (also the keyframe interval: 1 keyframe/second)."""
    rgb_resolution: Literal["720p", "1080p", "4k"] = "720p"
    """RGB encoded resolution. Default 720p (1280x720) matches the mono cameras."""
    mono_resolution: Literal["720p", "800p"] = "720p"
    """Mono encoded resolution. Default 720p (1280x720) matches RGB."""
    usb2: bool = False
    """Force USB2 mode (the OAK-D-W validated path on macOS was USB2)."""
    queue_size: int = 30
    """Host output-queue depth per stream (non-blocking)."""


@dataclass
class OakFrame:
    """One encoded frame from a single camera."""

    label: str
    sample: bytes
    is_keyframe: bool
    device_time_s: float
    sequence: int


def _encoded_size(config: DepthAiConfig) -> dict[str, tuple[int, int]]:
    _, _, rgb_w, rgb_h = _RGB_RESOLUTIONS[config.rgb_resolution]
    _, mono_w, mono_h = _MONO_RESOLUTIONS[config.mono_resolution]
    return {"rgb": (rgb_w, rgb_h), "left": (mono_w, mono_h), "right": (mono_w, mono_h)}


def _build_pipeline(config: DepthAiConfig) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    profile: dai.VideoEncoderProperties.Profile = _PROFILES[config.codec]
    rgb_res, rgb_isp_scale, _, _ = _RGB_RESOLUTIONS[config.rgb_resolution]
    mono_res, _, _ = _MONO_RESOLUTIONS[config.mono_resolution]

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(_SOCKETS["rgb"])
    cam_rgb.setResolution(rgb_res)
    if rgb_isp_scale is not None:
        cam_rgb.setIspScale(*rgb_isp_scale)  # ISP downscale to the target encoded size
    cam_rgb.setFps(config.fps)

    enc_rgb = pipeline.create(dai.node.VideoEncoder)
    enc_rgb.setDefaultProfilePreset(config.fps, profile)
    cam_rgb.video.link(enc_rgb.input)

    out_rgb = pipeline.create(dai.node.XLinkOut)
    out_rgb.setStreamName("rgb")
    enc_rgb.out.link(out_rgb.input)  # .out -> EncodedFrame (carries getFrameType())

    for label in ("left", "right"):
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(_SOCKETS[label])
        mono.setResolution(mono_res)
        mono.setFps(config.fps)

        enc = pipeline.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(config.fps, profile)
        mono.out.link(enc.input)

        out = pipeline.create(dai.node.XLinkOut)
        out.setStreamName(label)
        enc.out.link(out.input)

    return pipeline


def _read_calibration(device: dai.Device, encoded_size: dict[str, tuple[int, int]]) -> list[OakCameraCalib]:
    calib: dai.CalibrationHandler = device.readCalibration()
    reference_socket: dai.CameraBoardSocket = _SOCKETS[_REFERENCE_LABEL]
    calibs: list[OakCameraCalib] = []
    for label, socket, kind in _SENSORS:
        width, height = encoded_size[label]
        k_matrix = np.asarray(calib.getCameraIntrinsics(socket, width, height), dtype=float)
        distortion = [float(c) for c in calib.getDistortionCoefficients(socket)]
        # The reference camera (left) has the identity pose; the others are relative to it.
        ref_T_cam_cm = (
            None if label == _REFERENCE_LABEL else np.asarray(calib.getCameraExtrinsics(reference_socket, socket, False), dtype=float)
        )
        calibs.append(
            OakCameraCalib(
                label=label,
                width=width,
                height=height,
                k_matrix=k_matrix,
                distortion=distortion,
                kind=kind,
                ref_T_cam_cm=ref_T_cam_cm,
            )
        )
    return calibs


class OakSource:
    """Open an OAK device, expose its calibration, and stream encoded frames.

    Use as a context manager so the device is always released::

        with OakSource(config) as source:
            calibs = source.calibrations
            for frame in source.frames():
                ...
    """

    def __init__(self, config: DepthAiConfig) -> None:
        self.config: DepthAiConfig = config
        self._device: dai.Device | None = None
        self._queues: dict[str, dai.DataOutputQueue] = {}
        self.calibrations: list[OakCameraCalib] = []  # populated on __enter__

    def __enter__(self) -> OakSource:
        pipeline: dai.Pipeline = _build_pipeline(self.config)
        # USB2 (HIGH) was the validated OAK-D-W path on macOS; otherwise let the
        # device negotiate its max speed.
        self._device = dai.Device(pipeline, dai.UsbSpeed.HIGH) if self.config.usb2 else dai.Device(pipeline)
        self.calibrations = _read_calibration(self._device, _encoded_size(self.config))
        # pyrefly only sees depthai's 1-arg getOutputQueue overload; the maxSize/blocking
        # form is real (verified by the hardware test) and baselined in pyrefly-baseline.json.
        self._queues = {label: self._device.getOutputQueue(name=label, maxSize=self.config.queue_size, blocking=False) for label in _SOCKETS}
        return self

    def __exit__(self, *exc: object) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    @property
    def device(self) -> dai.Device:
        if self._device is None:
            raise RuntimeError("OakSource must be used as a context manager before accessing the device.")
        return self._device

    def _to_frame(self, label: str, packet: dai.EncodedFrame) -> OakFrame:
        sequence: int = packet.getSequenceNum()
        device_ts = packet.getTimestampDevice()
        device_time_s: float = device_ts.total_seconds() if device_ts is not None else sequence / self.config.fps
        return OakFrame(
            label=label,
            sample=bytes(packet.getData()),
            is_keyframe=packet.getFrameType() == dai.EncodedFrame.FrameType.I,
            device_time_s=device_time_s,
            sequence=sequence,
        )

    def frames(self) -> Iterator[OakFrame]:
        """Yield encoded frames across all streams until the caller stops iterating."""
        while True:
            any_packet: bool = False
            for label, queue in self._queues.items():
                while queue.has():
                    any_packet = True
                    # The encoder's `.out` queue yields EncodedFrame (typed as the base ADatatype).
                    yield self._to_frame(label, cast(dai.EncodedFrame, queue.get()))
            if not any_packet:
                time.sleep(0.005)
