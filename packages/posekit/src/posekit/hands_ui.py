"""Gradio app logic: 3D hands in a video (WiLoR or an EgoHand stack), streamed into an embedded Rerun viewer.

The layout lives in :mod:`posekit.hands_app_ui`, the only code ``posekit-hands-app-dev`` hot reloads;
this module holds the loaded pipelines, so they stay loaded while the layout reloads.

One input, a video. The app runs the CLI's code path: each Run builds the same
:class:`~posekit.apis.video_hands.VideoHandsSettings` the CLI parses from the widgets and drains
:func:`~posekit.apis.video_hands.run_video_hands` into a thread-local recording. The camera is a
pinhole whose intrinsics the Advanced accordion fills from an assumed field of view when a video
is chosen; edit them if you know the camera.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import rerun as rr
import torch
from jaxtyping import UInt8
from torch import Tensor

from posekit.apis.video_hands import EXAMPLES, VideoHandsSettings, assumed_pinhole, build_pipeline, pinhole_camera, run_video_hands, video_size
from posekit.cameras import CalibratedCamera
from posekit.models import HAND_DETECTORS, HAND_POSES, HandPose3dPipeline
from posekit.models.egohand_fast import supported_backbone_precision

STACKS: Final[list[str]] = list(HAND_DETECTORS)
"""Every named hand stack, as on the CLI."""
_PIPELINES: dict[str, HandPose3dPipeline] = {}
"""Set-up pipelines by stack name: engines stay loaded for the app's lifetime, so switching stacks costs nothing."""


def default_stack() -> str:
    """The accepted EgoHand stack in the precision this GPU runs: NVFP4 on Blackwell, BF16 on older GPUs."""
    return "egohand-fast" if supported_backbone_precision() == "nvfp4" else "egohand-fast-bf16"


def example_rows() -> list[list[str]]:
    """``[video, stack]`` per example: egocentric clips on the accepted EgoHand stack, and WiLoR's original
    demo clip (a person on stage with a microphone, exocentric) on WiLoR (EgoHand is egocentric).

    The GSI clips are head-camera pinholes (1920x1080); the Gen-HumanEgo clips (1600x1300) were remapped from
    the double-sphere head camera into a 65° pinhole, so the app's assumed camera is their true one.
    """
    return [
        [str(EXAMPLES / "gsi_sewing_left_front.mp4"), default_stack()],
        [str(EXAMPLES / "genhumanego_kettle.mp4"), default_stack()],
        [str(EXAMPLES / "genhumanego_garlic.mp4"), default_stack()],
        [str(EXAMPLES / "gsi_workbench_left_front.mp4"), default_stack()],
        [str(EXAMPLES / "hands_video.mp4"), "wilor"],
    ]


def example_stack(video: str | None, stack: str) -> str:
    """The stack of the example ``video`` is, else ``stack`` unchanged: picking an example also picks its stack."""
    if video is None:
        return stack
    return {Path(example).name: example_stack for example, example_stack in example_rows()}.get(Path(video).name, stack)


def fill_intrinsics(video: str | None) -> tuple[float | None, float | None, float | None, float | None]:
    """``fx, fy, cx, cy`` of the assumed pinhole for the chosen video (the CLI's default for unset intrinsics), to 0.1 px."""
    if video is None:
        return None, None, None, None
    width, height = video_size(Path(video))
    fx, fy, cx, cy = assumed_pinhole(width, height, VideoHandsSettings().hfov_deg)
    return round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1)


def settings_from_inputs(
    video: str, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None
) -> VideoHandsSettings:
    """The widget values as the CLI's settings; the stack name selects the same detector and pose configs as the CLI subcommand."""
    return VideoHandsSettings(
        video_path=Path(video), fx=fx, fy=fy, cx=cx, cy=cy, detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack],
        max_frames=max_frames if max_frames > 0 else None,
    )


def pipeline_for(stack: str) -> HandPose3dPipeline:
    """The pipeline of ``stack``, set up (TensorRT engines built or loaded) on first use and kept."""
    if stack not in _PIPELINES:
        _PIPELINES[stack] = build_pipeline(VideoHandsSettings(detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack]))
    return _PIPELINES[stack]


def warm_up() -> None:
    """Set up every example's stack and run it once on the example's first chunk.

    Setup builds or loads the TensorRT engines; the first call compiles and captures CUDA graphs
    for that frame size (~4 s for ``egohand-fast``). Done at start, the examples run at full speed
    from the first click; an upload of a new frame size pays well under a second once.
    """
    from torchcodec.decoders import VideoDecoder

    warmed: set[tuple[str, int, int]] = set()
    for video, stack in example_rows():
        settings: VideoHandsSettings = settings_from_inputs(video, stack, 0, None, None, None, None)
        decoder = VideoDecoder(settings.video_path, dimension_order="NHWC", device="cuda")
        assert decoder.metadata.width is not None and decoder.metadata.height is not None
        width, height = int(decoder.metadata.width), int(decoder.metadata.height)
        if (stack, width, height) in warmed:  # same graphs as an example already run
            continue
        warmed.add((stack, width, height))
        frames: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(0, settings.frame_batch_size).data.contiguous()
        camera: CalibratedCamera = pinhole_camera(settings, width, height)
        print(f"[posekit] warming up {stack} on {settings.video_path.name}", flush=True)
        with torch.inference_mode():
            pipeline_for(stack)(frames, camera)[1].in_pinhole(camera.intrinsics)


@rr.thread_local_stream("posekit_hands_ui")
def stream_hands(
    video: str | None, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None
) -> Iterator[tuple[bytes | None, str]]:
    """Run the shared video hands core and stream the recording to the viewer after every chunk."""
    stream: rr.BinaryStream = rr.binary_stream()
    if video is None:
        yield stream.read(), "Choose a video first."
        return
    settings: VideoHandsSettings = settings_from_inputs(video, stack, max_frames, fx, fy, cx, cy)
    if stack not in _PIPELINES:
        yield stream.read(), f"Setting up {stack} (its first use builds or loads its TensorRT engines)..."
    pipeline: HandPose3dPipeline = pipeline_for(stack)
    for status in run_video_hands(settings, pipeline):
        yield stream.read(), status


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the hands app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7872
    """Gradio HTTP port."""
