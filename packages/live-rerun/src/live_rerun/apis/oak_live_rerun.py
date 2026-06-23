"""Stream an OAK camera's hardware-encoded H.265/H.264 into Rerun, in realtime.

Default: spawn a viewer and stream live (no file). Pass ``--rr-config.save out.rrd``
to write the recording while also viewing it live (dual-sink). In a shell with no
DISPLAY, pass ``--rr-config.headless`` so the viewer spawn doesn't wedge logging.

    python tools/apps/oak_live_rerun.py --source.codec h265 --rr-config.save out.rrd

Lives in the package (not the ``tools/`` shim) so ``beartype_this_package()``
type-checks ``main`` and the config when running under the dev environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from live_rerun.blueprint import create_blueprint
from live_rerun.calibration import oak_calibration_to_rig
from live_rerun.rerun_video_logger import RerunVideoLogger
from live_rerun.sources.depthai import DepthAiConfig, OakSource


@dataclass
class _ViewerConfig(RerunTyroConfig):
    # Realtime tool: default to a live viewer. Combined with --rr-config.save this
    # fans out to viewer + .rrd simultaneously (see RerunTyroConfig.live).
    live: bool = True


@dataclass
class OakLiveRerunConfig:
    rr_config: _ViewerConfig
    source: DepthAiConfig = field(default_factory=DepthAiConfig)
    seconds: float | None = None
    """Stop after N seconds. Default: run until Ctrl-C."""
    image_plane_distance: float = 0.02
    """How far in front of each camera the image plane is drawn (metres). Small so the
    three frusta don't overlap given their short baselines."""


def main(config: OakLiveRerunConfig) -> None:
    with OakSource(config.source) as source:
        print("device:", source.device.getDeviceInfo(), flush=True)
        print("usb_speed:", source.device.getUsbSpeed(), flush=True)

        rig = oak_calibration_to_rig(source.calibrations)
        logger = RerunVideoLogger(rig, config.source.codec, image_plane_distance=config.image_plane_distance)
        rr.send_blueprint(create_blueprint(logger.pinhole_paths), make_active=True)
        logger.log_static()

        deadline: float | None = None if config.seconds is None else time.monotonic() + config.seconds
        print(f"streaming {config.source.codec} (Ctrl-C to stop)...", flush=True)
        try:
            for frame in source.frames():
                logger.log_sample(
                    frame.label,
                    frame.sample,
                    is_keyframe=frame.is_keyframe,
                    device_time_s=frame.device_time_s,
                )
                if deadline is not None and time.monotonic() >= deadline:
                    break
        except KeyboardInterrupt:
            print("\nstopped.", flush=True)
