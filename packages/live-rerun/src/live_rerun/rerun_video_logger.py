"""Generic, sensor-agnostic logging of encoded video into Rerun ``VideoStream``.

This core knows nothing about DepthAI. It logs a stationary multi-camera rig
(intrinsics + extrinsics as pinholes, once, statically) and then forwards each
already-encoded H.264/H.265 sample straight into ``rr.VideoStream`` on a shared
``device_time`` timeline. No decode/encode happens here — the camera's hardware
encoder bytes pass through untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import rerun as rr
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

Codec = Literal["h264", "h265"]

_RR_CODECS: dict[Codec, rr.VideoCodec] = {
    "h264": rr.VideoCodec.H264,
    "h265": rr.VideoCodec.H265,
}

#: Shared duration timeline (seconds) all camera streams are placed on so the
#: viewer can scrub every view together. See the Rerun video concept docs.
DEVICE_TIMELINE: str = "device_time"


class RerunVideoLogger:
    """Logs a stationary encoded-video camera rig to Rerun.

    Entity layout (per the repo idiom, e.g. multiview_calibration / quest3_oakd):
    ``{world}`` (view coordinates) -> ``{rig}/{label}`` (Transform3D) ->
    ``{rig}/{label}/pinhole`` (Pinhole) -> ``{rig}/{label}/pinhole/video``
    (VideoStream). The video must sit under ``pinhole`` so it projects onto the
    camera frustum.
    """

    def __init__(
        self,
        cameras: dict[str, PinholeParameters],
        codec: Codec,
        *,
        world_path: str = "world",
        rig_name: str = "oak",
        image_plane_distance: float = 0.02,
        view_coordinates: rr.components.ViewCoordinates | None = None,
    ) -> None:
        self.cameras: dict[str, PinholeParameters] = cameras
        self.codec: Codec = codec
        self.world_path: str = world_path
        self.rig_path: str = f"{world_path}/{rig_name}"
        self.image_plane_distance: float = image_plane_distance
        # Note: rr.ViewCoordinates.RDF is a *component* instance, not the archetype.
        self.view_coordinates: rr.components.ViewCoordinates = view_coordinates if view_coordinates is not None else rr.ViewCoordinates.RDF
        self.video_paths: dict[str, str] = {label: f"{self.rig_path}/{label}/pinhole/video" for label in cameras}

    def log_static(self) -> None:
        """Log the world frame, per-camera pinholes/extrinsics, and codec — all static."""
        rr.log(self.world_path, self.view_coordinates, static=True)
        codec: rr.VideoCodec = _RR_CODECS[self.codec]
        for label, pinhole in self.cameras.items():
            log_pinhole(
                pinhole,
                cam_log_path=Path(self.rig_path) / label,
                image_plane_distance=self.image_plane_distance,
                static=True,
            )
            rr.log(self.video_paths[label], rr.VideoStream(codec=codec), static=True)

    def log_sample(self, label: str, sample: bytes, *, is_keyframe: bool, device_time_s: float) -> None:
        """Forward one encoded frame to its camera's VideoStream on the shared timeline."""
        rr.set_time(DEVICE_TIMELINE, duration=device_time_s)
        rr.log(self.video_paths[label], rr.VideoStream.from_fields(sample=sample, is_keyframe=is_keyframe))
