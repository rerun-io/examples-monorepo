"""Generic, system-agnostic logging of encoded video into Rerun ``VideoStream``.

This core knows nothing about any specific camera vendor. It logs a multi-sensor
**rig** (per :mod:`live_rerun.rig`) following the COLMAP-aligned entity layout
shared with ``packages/slam-evals`` and then forwards each already-encoded
H.264/H.265 sample straight into ``rr.VideoStream`` on a shared ``device_time``
timeline. No decode/encode happens here — the camera's hardware encoder bytes
pass through untouched.

Entity layout (see ``packages/live-rerun/docs/rig_schema.md``)::

    /world                              ViewCoordinates (static)
      /rig_00                           AnyValues schema metadata (static); NO
                                        transform -> implicit identity now. A
                                        future SLAM pass logs world_T_rig here
                                        *temporally*.
        /cam_00                         Transform3D = rig_T_cam (static)
                                        + AnyValues(name, kind) (static)
          /pinhole                      Pinhole/PinholeWithDistortion (static)
            /video                      VideoStream (encoded samples)

The rig is stationary for now, so the per-sensor transforms/pinholes are logged
once as static and the rig node carries no transform at all (implicit identity);
the per-frame samples are the only time-varying data. Driving the rig node
dynamically (a SLAM trajectory) then moves every sensor rigidly with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import rerun as rr
from simplecv.rerun_log_utils import log_pinhole

from live_rerun.rig import RigCalibration, entity_id

Codec = Literal["h264", "h265"]

_RR_CODECS: dict[Codec, rr.VideoCodec] = {
    "h264": rr.VideoCodec.H264,
    "h265": rr.VideoCodec.H265,
}

#: Shared duration timeline (seconds) all sensor streams are placed on so the
#: viewer can scrub every view together. See the Rerun video concept docs.
DEVICE_TIMELINE: str = "device_time"

#: Schema identifier logged on the rig root so loaders can validate the layout.
SCHEMA_VERSION: str = "live-rerun-rig:v1"

#: Frustum wireframe colour for the reference sensor (the rig origin), so it's
#: trivially distinguishable in a multi-camera view. Matches slam-evals' green.
REFERENCE_CAM_COLOR: tuple[int, int, int, int] = (40, 200, 80, 255)


class RerunVideoLogger:
    """Logs a stationary encoded-video camera rig to Rerun.

    Consumes a generic :class:`~live_rerun.rig.RigCalibration`; the entity paths,
    reference highlight, and per-sensor metadata all follow from it, so this core
    stays vendor-agnostic.
    """

    def __init__(
        self,
        rig: RigCalibration,
        codec: Codec,
        *,
        world_path: str = "world",
        rig_name: str = "rig_00",
        image_plane_distance: float = 0.02,
        view_coordinates: rr.components.ViewCoordinates | None = None,
    ) -> None:
        self.rig: RigCalibration = rig
        self.codec: Codec = codec
        self.world_path: str = world_path
        self.rig_path: str = f"{world_path}/{rig_name}"
        self.image_plane_distance: float = image_plane_distance
        # Note: rr.ViewCoordinates.RDF is a *component* instance, not the archetype.
        self.view_coordinates: rr.components.ViewCoordinates = view_coordinates if view_coordinates is not None else rr.ViewCoordinates.RDF
        # Frames are routed by the sensor's role name (the source yields that label);
        # the cam_<NN> entity path is the canonical identity.
        self.video_paths: dict[str, str] = {cam.name: f"{self.rig_path}/{entity_id('cam', cam.index)}/pinhole/video" for cam in rig.cameras}
        # Keyed by the canonical cam_<NN> id for the blueprint, so 2D panels carry the
        # same names as the 3D entity tree (role label lives in each sensor's metadata).
        self.pinhole_paths: dict[str, str] = {entity_id("cam", cam.index): f"{self.rig_path}/{entity_id('cam', cam.index)}/pinhole" for cam in rig.cameras}

    def log_static(self) -> None:
        """Log the world frame, rig schema, per-sensor pinholes/metadata, codec — all static."""
        rr.log(self.world_path, self.view_coordinates, static=True)
        # The rig node carries NO static transform: it stays at implicit identity
        # while stationary, leaving it free for a future SLAM pass to log
        # ``world_T_rig`` *temporally* here. A static identity transform would
        # shadow that temporal pose and trip Rerun's "static + temporal" warning.
        rr.log(
            self.rig_path,
            rr.AnyValues(
                schema_version=SCHEMA_VERSION,
                reference=entity_id("cam", self.rig.reference_index),
                num_cameras=len(self.rig.cameras),
            ),
            static=True,
        )
        codec: rr.VideoCodec = _RR_CODECS[self.codec]
        for cam in self.rig.cameras:
            cam_path: str = f"{self.rig_path}/{entity_id('cam', cam.index)}"
            # log_pinhole logs the rig_T_cam Transform3D at cam_path and the
            # PinholeWithDistortion at cam_path/pinhole.
            log_pinhole(
                cam.pinhole,
                cam_log_path=Path(cam_path),
                image_plane_distance=self.image_plane_distance,
                static=True,
            )
            if cam.index == self.rig.reference_index:
                # Partial update: tint the reference sensor's frustum wireframe.
                rr.log(
                    f"{cam_path}/pinhole",
                    rr.Pinhole.from_fields(image_plane_distance=self.image_plane_distance, color=REFERENCE_CAM_COLOR),
                    static=True,
                )
            rr.log(cam_path, rr.AnyValues(name=cam.name, kind=cam.kind), static=True)
            rr.log(self.video_paths[cam.name], rr.VideoStream(codec=codec), static=True)

    def log_sample(self, name: str, sample: bytes, *, is_keyframe: bool, device_time_s: float) -> None:
        """Forward one encoded frame to its sensor's VideoStream on the shared timeline."""
        rr.set_time(DEVICE_TIMELINE, duration=device_time_s)
        rr.log(self.video_paths[name], rr.VideoStream.from_fields(sample=sample, is_keyframe=is_keyframe))
