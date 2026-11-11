"""The hands app's pipelines and runs: 3D hands in a video (WiLoR or an EgoHand stack), streamed to a Rerun viewer.

No Gradio here: the widgets, their wiring and the server live in :mod:`posekit.hands_app_ui`, the only code
``posekit-hands-app-dev`` hot reloads; this module holds the loaded pipelines, so they stay loaded while the layout reloads.

One input, a video. The app runs the CLI's code path: each Run builds the same
:class:`~posekit.apis.video_hands.VideoHandsSettings` the CLI parses from the widgets and drains
:func:`~posekit.apis.video_hands.run_video_hands` into a thread-local recording. The camera is a
pinhole whose intrinsics the Advanced accordion fills from an assumed field of view when a video
is chosen; edit them if you know the camera.
"""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import rerun as rr
import torch
from jaxtyping import UInt8
from torch import Tensor
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR

from posekit.apis.video_hands import (
    EXAMPLES,
    VideoHandsProgress,
    VideoHandsSettings,
    assumed_pinhole,
    build_pipeline,
    open_decoder,
    pinhole_camera,
    run_video_hands,
    video_size,
)
from posekit.cameras import CalibratedCamera
from posekit.models import HAND_DETECTORS, HAND_POSES, EgoHandDetectorConfig, EgoHandFastConfig, HandPose3dPipeline
from posekit.models.egohand_fast import supported_backbone_precision
from posekit.predictions import MeshDetail

STACKS: Final[list[str]] = list(HAND_DETECTORS)
"""Every named hand stack, as on the CLI."""
PIPELINES: dict[str, HandPose3dPipeline] = {}
"""Set-up pipelines by stack name: engines stay loaded for the app's lifetime, so switching stacks costs nothing."""
SETUP_NOTES: dict[str, str] = {}
"""What setting each stack up did in this process, for the run log: engines loaded or built, and how long it took."""


def default_stack() -> str:
    """The accepted EgoHand stack in the precision this GPU runs: NVFP4 on Blackwell, BF16 on older GPUs."""
    return "egohand-fast" if supported_backbone_precision() == "nvfp4" else "egohand-fast-bf16"


def example_rows(examples: Path = EXAMPLES) -> list[list[str]]:
    """``[video, stack]`` per example in directory ``examples``: egocentric clips on the accepted EgoHand stack, and
    WiLoR's original demo clip (a person on stage with a microphone, exocentric) on WiLoR (EgoHand is egocentric).

    The GSI clips are head-camera pinholes (1920x1080); the Gen-HumanEgo clips (1600x1300) were remapped from
    the double-sphere head camera into a 65° pinhole, so the app's assumed camera is their true one.
    """
    return [
        [str(examples / "gsi_sewing_left_front.mp4"), default_stack()],
        [str(examples / "genhumanego_kettle.mp4"), default_stack()],
        [str(examples / "genhumanego_garlic.mp4"), default_stack()],
        [str(examples / "gsi_workbench_left_front.mp4"), default_stack()],
        [str(examples / "hands_video.mp4"), "wilor"],
    ]


def fill_intrinsics(video: str | None) -> tuple[float | None, float | None, float | None, float | None]:
    """``fx, fy, cx, cy`` of the assumed pinhole for the chosen video (the CLI's default for unset intrinsics), to 0.1 px."""
    if video is None:
        return None, None, None, None
    width, height = video_size(Path(video))
    fx, fy, cx, cy = assumed_pinhole(width, height, VideoHandsSettings().hfov_deg)
    return round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1)


def settings_from_inputs(
    video: str, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None,
    mesh_detail: MeshDetail,
) -> VideoHandsSettings:
    """The widget values as the CLI's settings; the stack name selects the same detector and pose configs as the CLI subcommand."""
    return VideoHandsSettings(
        video_path=Path(video), fx=fx, fy=fy, cx=cx, cy=cy, detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack],
        max_frames=max_frames if max_frames > 0 else None, mesh_detail=mesh_detail,
    )


def prepare(stack: str) -> None:
    """The part of ``stack``'s setup that needs no GPU: its downloads, and what its configs can build on the CPU.

    An app whose GPU workers are forks of its main process (a ZeroGPU Space) calls this once before it serves: each
    worker inherits the result, and ``pipeline_for`` then only does the GPU part. Any other host may skip it.
    """
    detector, pose = HAND_DETECTORS[stack], HAND_POSES[stack]
    if isinstance(detector, EgoHandDetectorConfig):
        detector.weights_path()
    if isinstance(pose, EgoHandFastConfig):
        pose.prepare()


def pipeline_for(stack: str) -> HandPose3dPipeline:
    """The pipeline of ``stack``, set up (TensorRT engines built or loaded) on first use and kept; ``SETUP_NOTES`` says which."""
    if stack not in PIPELINES:
        start: float = time.perf_counter()
        engines: int = len(list(DEFAULT_TRT_CACHE_DIR.glob("*.engine")))
        PIPELINES[stack] = build_pipeline(VideoHandsSettings(detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack]))
        built: int = len(list(DEFAULT_TRT_CACHE_DIR.glob("*.engine"))) - engines
        did: str = f"Built {built} TensorRT engine{'s' * (built > 1)} for {stack}" if built else f"Loaded {stack}"
        SETUP_NOTES[stack] = f"{did} ({time.perf_counter() - start:.0f} s)"
    return PIPELINES[stack]


def warm_up(examples: list[list[str]] | None = None) -> None:
    """Set up the stack of every ``[video, stack]`` example (None: ``example_rows()``) and run it once on the example's first chunk.

    Setup builds or loads the TensorRT engines; the first call compiles and captures CUDA graphs
    for that frame size (~4 s for ``egohand-fast``). Done at start, the examples run at full speed
    from the first click; an upload of a new frame size pays well under a second once.
    """
    warmed: set[tuple[str, int, int]] = set()
    for video, stack in example_rows() if examples is None else examples:
        settings = VideoHandsSettings(video_path=Path(video), detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack])
        decoder = open_decoder(settings.video_path, "cuda")
        assert decoder.metadata.width is not None and decoder.metadata.height is not None
        width, height = int(decoder.metadata.width), int(decoder.metadata.height)
        if (stack, width, height) in warmed:  # same graphs as an example already run
            continue
        warmed.add((stack, width, height))
        frames: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(0, settings.frame_batch_size).data.contiguous().cuda()
        camera: CalibratedCamera = pinhole_camera(settings, width, height)
        print(f"[posekit] warming up {stack} on {settings.video_path.name}", flush=True)
        with torch.inference_mode():
            pipeline_for(stack)(frames, camera)[1].in_pinhole(camera.intrinsics).at_mesh_detail(settings.mesh_detail)


@rr.thread_local_stream("posekit_hands_ui")
def stream_hands(settings: VideoHandsSettings, stack: str) -> Iterator[tuple[bytes | None, VideoHandsProgress]]:
    """Run the shared video hands core on ``stack``'s pipeline and stream the recording to the viewer after every step.

    Yields:
        The new recording bytes and the run's progress.
    """
    stream: rr.BinaryStream = rr.binary_stream()
    pipeline: HandPose3dPipeline = pipeline_for(stack)
    for progress in run_video_hands(settings, pipeline):
        yield stream.read(), progress
