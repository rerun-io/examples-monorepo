"""Per-frame hand logging of the posekit-video-hands app (no models, rr.log captured)."""

import threading
from typing import Any

import numpy as np
import pytest
import rerun as rr
import torch

from posekit.apis.video_hands import (
    EXAMPLES,
    JOINT_CLASS,
    LINK_COLOR,
    VideoHandsProgress,
    VideoHandsSettings,
    assumed_pinhole,
    describe_camera,
    log_chunk_hands,
    log_hand_annotation_context,
    pinhole_camera,
    run_video_hands,
    send_blueprint,
)
from posekit.models import HandPose3dPipeline
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, Keypoints3d
from posekit.skeletons import HAND_21


def _chunk() -> tuple[HandDetections, HandPoses3d]:
    """Frame 0: a left and a right hand; frame 1: only a left hand."""
    frames = torch.tensor([0, 0, 1])
    sides = ("left", "right", "left")
    detections = HandDetections(BoxDetections(torch.tensor([[0.0, 0, 10, 10], [20, 0, 30, 10], [1, 1, 11, 11]]), torch.ones(3), frames), sides)
    xyz = torch.rand((3, 21, 3)) + torch.tensor([0.0, 0.0, 1.0])
    hands = HandPoses3d(
        keypoints=Keypoints3d(xyz[..., :2] * 100, xyz[..., 2] - xyz[:, :1, 2], torch.ones((3, 21)), frames, HAND_21, (0,), xyz),
        sides=sides,
        vertices=torch.rand((3, 5, 3)),
        faces={"left": torch.tensor([[0, 2, 1]]), "right": torch.tensor([[0, 1, 2]])},
    )
    return detections, hands


def test_hands_go_to_side_slots_and_an_emptied_slot_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, **_: logged.append((path, archetype)))
    monkeypatch.setattr(rr, "set_time", lambda *_, **__: None)
    detections, hands = _chunk()
    live: set[str] = set()
    log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, live_slots=live, meshed_slots=set())

    paths = [path for path, _ in logged]
    # Joints and meshes are logged once in 3D; the video view projects them, so no 2D keypoints.
    assert not any(path.endswith("/keypoints") for path in paths)
    for slot in ("left_0", "right_0"):
        assert f"world/camera/pinhole/boxes/{slot}" in paths
        assert f"world/camera/hands/{slot}/joints" in paths
        assert f"world/camera/hands/{slot}/mesh" in paths
    # Frame 1 has no right hand: its slot is cleared once, the left slot is logged again.
    cleared = [path for path, archetype in logged if isinstance(archetype, rr.Clear)]
    assert sorted(cleared) == sorted(
        ["world/camera/pinhole/boxes/right_0"] + [f"world/camera/hands/right_0/{name}" for name in ("joints", "mesh")]
    )
    assert live == {"left_0"}
    meshes = [archetype for path, archetype in logged if path == "world/camera/hands/right_0/mesh"]
    faces = meshes[0].triangle_indices.as_arrow_array().to_pylist()
    assert faces == [[0, 1, 2]]


def test_mesh_faces_are_logged_once_per_slot_and_frames_carry_only_vertices(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, Any, bool]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, static=False, **_: logged.append((path, archetype, static)))
    monkeypatch.setattr(rr, "set_time", lambda *_, **__: None)
    detections, hands = _chunk()
    meshed: set[str] = set()
    for _ in range(2):  # two chunks
        log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, live_slots=set(), meshed_slots=meshed)
    left = [(archetype, static) for path, archetype, static in logged if path == "world/camera/hands/left_0/mesh"]
    assert [static for _, static in left] == [True, False, False, False, False]
    assert left[0][0].triangle_indices is not None and left[0][0].vertex_positions is None
    assert all(archetype.triangle_indices is None and archetype.vertex_positions is not None for archetype, _ in left[1:])
    assert meshed == {"left_0", "right_0"}


def test_unset_intrinsics_come_from_the_assumed_field_of_view() -> None:
    fx, fy, cx, cy = assumed_pinhole(1920, 1080, 65.0)
    assert fx == fy == pytest.approx(1507.0, abs=0.5) and (cx, cy) == (960.0, 540.0)
    camera = pinhole_camera(VideoHandsSettings(fx=1000.0, cy=500.0), 1920, 1080)
    assert camera.lens == "PINHOLE" and (camera.width, camera.height) == (1920, 1080)
    assert camera.intrinsics.tolist() == [[1000.0, 0.0, 960.0], [0.0, fy, 500.0], [0.0, 0.0, 1.0]]


def test_status_names_the_camera() -> None:
    settings = VideoHandsSettings()
    assert describe_camera(settings, pinhole_camera(settings, 1920, 1080)) == "assumed pinhole, f = 1507 px (65° horizontal FOV)"
    given = VideoHandsSettings(fx=1200.0, fy=1210.0, cx=950.0, cy=530.0)
    assert describe_camera(given, pinhole_camera(given, 1920, 1080)) == "pinhole fx = 1200, fy = 1210, cx = 950, cy = 530 px"


def test_joints_carry_full_confidence_and_show_green(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hand models give no per-joint confidence, so every joint is logged at 1.0: the confidence gradient's green.
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, **_: logged.append((path, archetype)))
    monkeypatch.setattr(rr, "set_time", lambda *_, **__: None)
    detections, hands = _chunk()
    log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, live_slots=set(), meshed_slots=set())

    joints = next(archetype for path, archetype in logged if path == "world/camera/hands/left_0/joints")
    batches = {batch.component_descriptor().component: batch.as_arrow_array().to_pylist() for batch in joints.as_component_batches()}
    assert batches["simplecv.KeypointConfidence3D:confidences"] == [1.0] * 21
    assert {packed >> 8 for packed in batches["Points3D:colors"]} == {0x00FF00}
    assert joints.points3d.class_ids.as_arrow_array().to_pylist() == [JOINT_CLASS]


def test_skeleton_links_have_their_own_colour_and_the_sides_keep_theirs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The links get one contrasting colour; drawn in the side colour they vanish into the mesh of the same colour.
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, **_: logged.append((path, archetype)))
    log_hand_annotation_context(HAND_21)
    context = logged[0][1].context.as_arrow_array().to_pylist()[0]
    by_id = {entry["class_id"]: entry["class_description"] for entry in context}
    assert len(by_id[JOINT_CLASS]["keypoint_connections"]) == len(HAND_21.links)
    assert by_id[JOINT_CLASS]["info"]["color"] >> 8 == (LINK_COLOR[0] << 16) | (LINK_COLOR[1] << 8) | LINK_COLOR[2]
    assert all(not by_id[class_id]["keypoint_connections"] for class_id in by_id if class_id != JOINT_CLASS)


def test_the_viewer_plays_the_video_from_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # A streamed viewer follows the newest data; the blueprint makes it play video_time from 0 while the hands arrive.
    sent: list[Any] = []
    monkeypatch.setattr(rr, "send_blueprint", lambda blueprint, **_: sent.append(blueprint))
    send_blueprint()
    time_panel = sent[0].time_panel
    assert (time_panel.timeline, time_panel.play_state) == ("video_time", "playing")
    assert time_panel.state is None  # the app's panel_states keeps deciding whether the time panel shows
    view_3d = sent[0].root_container.contents[1]
    assert view_3d.properties["EyeControls3D"].spin_speed.as_arrow_array().to_pylist() == [0.1]  # a slow orbit around the hands


def test_hands_from_a_model_that_uses_the_given_camera_stay_as_they_are() -> None:
    # The video core puts every stack's hands in the run's pinhole; only WiLoR (own virtual camera) overrides in_pinhole.
    _, hands = _chunk()
    assert hands.in_pinhole(np.eye(3)) is hands


def test_progress_reads_as_the_status_line() -> None:
    note = "assumed pinhole, f = 1256 px (65° horizontal FOV)"
    assert str(VideoHandsProgress("video", 0, 180, 0, note)) == f"Video loaded, finding hands; {note}"
    assert str(VideoHandsProgress("hands", 16, 180, 32, note)) == f"Frames 16/180, 32 hands; {note}"
    assert str(VideoHandsProgress("done", 180, 180, 360, note)) == f"processed 180 frames, 360 hands; {note}"
    warning: str = "WARNING: hardware video failed, the CPU does this work (slower): decode (NVcuvid unavailable)"
    assert warning in str(VideoHandsProgress("video", 0, 8, 0, note, warning))  # the app's status shows a hardware video fallback


class _NoHands(HandPose3dPipeline):
    """A hand pipeline that finds nothing, so a run exercises only the video decode and the logging."""

    def __init__(self) -> None:
        self.pose = type("Pose", (), {"skeleton": HAND_21})()

    def __call__(self, frames_rgb: torch.Tensor, camera: object = None) -> tuple[HandDetections, HandPoses3d]:
        return HandDetections.empty(frames_rgb.device), HandPoses3d(
            keypoints=Keypoints3d.empty(HAND_21, frames_rgb.device, root_indices=(0,), with_xyz_camera=True), sides=()
        )


@pytest.mark.integration
def test_a_run_resumed_on_new_threads_still_decodes_on_the_gpu(caplog: pytest.LogCaptureFixture) -> None:
    # Gradio resumes a streaming generator on any worker thread; NVDEC needs the CUDA context current there.
    pytest.importorskip("torchcodec", reason="the GPU video decoder comes with the hands env")
    if not torch.cuda.is_available():
        pytest.skip("NVDEC decoding needs a CUDA GPU")
    video = EXAMPLES / "gsi_sewing_left_front.mp4"
    if not video.exists():
        pytest.skip(f"example clip {video.name} missing")
    rr.init("thread-hop-test")
    rr.memory_recording()  # the global recording, visible from every thread; nothing leaves the process
    run = run_video_hands(VideoHandsSettings(video_path=video, max_frames=64), _NoHands())
    steps: list[VideoHandsProgress] = []
    errors: list[BaseException] = []

    def step() -> None:
        try:
            steps.append(next(run))
        except StopIteration:
            pass
        except BaseException as error:  # reported by the assert below
            errors.append(error)

    with caplog.at_level("WARNING", logger="simplecv.video_io"):
        while not errors and not (steps and steps[-1].stage == "done"):
            thread = threading.Thread(target=step)
            thread.start()
            thread.join()
    assert not errors, errors[0]
    assert steps[-1].frames_done == 64
    # open_decoder warns whenever torchcodec decodes on the CPU: it did not, so NVDEC decoded.
    assert not caplog.records and not steps[-1].warning
