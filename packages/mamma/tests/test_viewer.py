"""The capture App's normal view is the CLI's (origin/main ``tools/demo_crossing_arms.py`` -> ``viz/stream_logger.py`` and
``viz/blueprint.py``): main's entities and archetypes under the App's paths, main's blueprint, and the pipeline graph only
as the second tab (unit lane: the App graph with fake models and a fake decoder runs into a saved recording; no GPU, no
catalog)."""

import io
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import av
import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import torch
from dagkit.doc import HEALTH_ROOT
from dagkit.node import Event
from dagkit.runners import RunReport, run_sequential
from dagkit.sources import IterSource, SegmentRef
from dagkit.time import STATIC
from jaxtyping import Int64, UInt8
from mamma_fakes import ENGINE_HW, FakeDetector, FakeEncoder, FakeFitter, FakeLandmarker, FakeMasks, FakeScene
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.sources.catalog_video import VideoSource
from simplecv.types.rig import RigCameras
from torch import Tensor

from mamma.apis import capture
from mamma.apis.blueprint import default_blueprint
from mamma.apis.capture import Config, app_graph, log_file_videos
from mamma.nodes.fit import FitNode
from mamma.nodes.landmarks import LandmarksNode
from mamma.nodes.track import TrackNode, TrackParams
from mamma.ops.window_fit import FitterConfig
from mamma.types import BodyTopology, Landmarks3d, MultiViewLandmarks, MultiViewTracks

MAIN_ENTITIES: dict[str, tuple[str, set[str]]] = {
    "/world": ("/world", {"AnnotationContext", "ViewCoordinates"}),
    "/world/ground": ("/world/ground", {"Mesh3D"}),
    "/world/cameras/{cam}": ("/world/{cam}", {"Transform3D"}),
    "/world/cameras/{cam}/pinhole": ("/world/{cam}/pinhole", {"Pinhole"}),
    # main's /world/cameras/{cam}/pinhole/video is the segment's own /world/{cam}/pinhole/video: the App's recording joins it.
    "/world/cameras/{cam}/pinhole/persons": ("/world/{cam}/pinhole/mamma/persons", {"Boxes2D"}),
    "/world/cameras/{cam}/pinhole/mask": ("/world/{cam}/pinhole/mamma/masks/person_{i}", {"EncodedImage"}),
    "/world/cameras/{cam}/pinhole/landmarks/person_{i}": ("/world/{cam}/pinhole/mamma/landmarks/person_{i}", {"Points2D"}),
    "/world/triangulated/person_{i}": ("/world/mamma/people/person_{i}/landmarks3d", {"Points3D"}),
    "/world/meshes/person_{i}": ("/world/mamma/people/person_{i}/body/mesh", {"Mesh3D"}),
    "/world/joints/person_{i}": ("/world/mamma/people/person_{i}/joints", {"Points3D"}),
    "/metrics/fit/valid_anchors": ("/metrics/fit/person_{i}/valid_anchors", {"Scalars"}),
    "/metrics/fit/floor_contacts": ("/metrics/fit/person_{i}/floor_contacts", {"Scalars"}),
    "/metrics/params/person_{i}/trans": ("/metrics/params/person_{i}/trans", {"Scalars"}),
    "/metrics/params/person_{i}/betas": ("/metrics/params/person_{i}/betas", {"BarChart"}),
}
"""Main's crossing_arms recording (``/world/cameras/<cam>``, ``person_<id>``) -> the App's path and archetypes. Paths moved on
purpose: cameras sit at the catalog layout (``/world/<cam>``), predictions under the ``mamma_fit`` layer roots (one palette
PNG per person, as the layer stores masks), fit health per person. Not restored: main's per-stage ``/metrics/timing``
(the runner's step times are the Stage timings plot) and ``/world/skeleton`` (no pixels in main's view)."""
FRAMES: int = 16


def entities(path: Path) -> dict[str, set[str]]:
    """Entity path (cameras folded to ``{cam}``, persons to ``{i}``) -> archetypes and ``archetype:component`` names."""
    found: dict[str, set[str]] = {}
    for col in rr.chunk.RrdReader(path).store().schema().component_columns():
        if col.archetype and not col.entity_path.startswith("/__"):
            folded: str = re.sub(r"person_\d+", "person_{i}", re.sub(r"rig_\d+/cam_\d+", "{cam}", col.entity_path))
            archetype: str = col.archetype.split(".")[-1]
            found.setdefault(folded, set()).update({archetype, f"{archetype}:{col.component.split(':')[-1]}"})
    return found


class FakeFrames:
    """``FakeScene``'s cameras as a decoder: blank frames at the stored size (twice the engine size), 30 fps."""

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        return np.arange(FRAMES, dtype=np.int64) * 33_333_333

    def frames_at(self, view: int, indices: Sequence[int]) -> UInt8[Tensor, "k 3 h w"]:
        return torch.zeros((len(indices), 3, 2 * ENGINE_HW[0], 2 * ENGINE_HW[1]), dtype=torch.uint8)

    def close(self) -> None:
        pass


def run_with_fakes(config: Config, path: Path) -> None:
    """The App graph on ``FakeScene`` (one person standing for 16 frames) with fake models, its viewer saved at ``path``."""
    scene = FakeScene()
    statics = IterSource({"rig": RigCameras}, lambda: iter([Event(STATIC, "rig", scene.rig)]))
    video = VideoSource(FakeFrames(), scene.rig.names, statics, fenced=True, resize_hw=ENGINE_HW, keep_native=True)  # type: ignore[arg-type]
    nodes = (
        TrackNode(TrackParams(expected_subjects=1), FakeDetector(), FakeEncoder(), FakeMasks()),  # type: ignore[arg-type]
        LandmarksNode(FakeLandmarker(scene)),
        FitNode(FitterConfig(emit_stride=1), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32), lambda K, T: FakeFitter()),  # type: ignore[arg-type,return-value]
    )
    recording = rr.RecordingStream("mamma-test", send_properties=False)
    recording.save(path)
    assert run_sequential(app_graph(config, recording, video, nodes, "x86_64-cuda/quality")).status == "complete"
    recording.flush()
    recording.disconnect()


def test_the_app_viewer_recording_has_the_cli_entities(tmp_path: Path) -> None:
    """Every entity main's CLI drew, with its archetype, at the App's path; the mesh is shaded (vertex normals) and the joints
    are keypoints of their person's class (the bones come from the AnnotationContext); the boxes draw no class label (the
    viewer places a label without the engine-to-video scale). A catalog segment's videos are not copied: the recording joins
    the segment's own, which has them."""
    config = Config(RerunTyroConfig(headless=True))
    run_with_fakes(config, tmp_path / "app.rrd")
    found: dict[str, set[str]] = entities(tmp_path / "app.rrd")
    for main, (path, archetypes) in MAIN_ENTITIES.items():
        assert archetypes <= found.get(path, set()), f"main {main} -> {path}: {found.get(path)} lacks {archetypes}"
    assert {"Mesh3D:vertex_normals", "Mesh3D:albedo_factor"} <= found["/world/mamma/people/person_{i}/body/mesh"]
    assert {"Points3D:class_ids", "Points3D:keypoint_ids"} <= found["/world/mamma/people/person_{i}/joints"]
    assert "Boxes2D:show_labels" in found["/world/{cam}/pinhole/mamma/persons"]
    assert not [path for path in found if path.endswith("/pinhole/video")]


def test_a_capture_directory_logs_its_videos_at_run_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file capture has no catalog segment: the viewer logs the capture's own video files when the run starts."""
    logged: list[tuple[Path, tuple[str, ...], str]] = []
    monkeypatch.setattr(capture, "log_file_videos", lambda recording, root, cameras, timeline: logged.append((root, cameras, timeline)))
    run_with_fakes(Config(RerunTyroConfig(headless=True), video=tmp_path), tmp_path / "file.rrd")
    assert logged == [(tmp_path, FakeScene().rig.names, "video_time")]


@pytest.mark.integration
def test_a_capture_directory_video_is_re_encoded_so_the_viewer_can_seek_it(tmp_path: Path) -> None:
    """The demo files are 4:4:4 HEVC with B-frames and a keyframe every 250 frames, which the viewer stops decoding past the
    second keyframe: the logged video is 4:2:0 without reordering, a keyframe at least every 30 frames, frame i at i / fps."""
    if shutil.which("ffmpeg") is None or not torch.cuda.is_available():
        pytest.skip("needs ffmpeg and a CUDA GPU (NVENC)")
    (tmp_path / "meta").mkdir()
    (tmp_path / "videos").mkdir()
    np.savez(tmp_path / "meta/global.npz", fps=np.asarray(25))
    source: list[str] = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=4", "-c:v", "libx265", "-pix_fmt", "yuv444p"]
    subprocess.run(["ffmpeg", "-loglevel", "error", *source, "-x265-params", "keyint=60:bframes=3:log-level=error", str(tmp_path / "videos/cam.mp4")], check=True)
    recording = rr.RecordingStream("mamma-test", send_properties=False)
    recording.save(tmp_path / "file.rrd")
    log_file_videos(recording, tmp_path, ("cam",), "video_time")
    recording.flush()
    recording.disconnect()
    columns: dict[str, list] = {}
    for chunk in rr.chunk.RrdReader(tmp_path / "file.rrd").stream().to_chunks():
        batch = chunk.to_record_batch()
        columns |= {name: batch.column(name).to_pylist() for name in ("AssetVideo:blob", "video_time") if name in batch.schema.names}
    with av.open(io.BytesIO(bytes(columns["AssetVideo:blob"][0][0])), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.codec_context.pix_fmt == "yuv420p"
        packets = [p for p in container.demux(stream) if p.size]
    keyframes: list[int] = [i for i, packet in enumerate(packets) if packet.is_keyframe]
    assert len(packets) == 100 and keyframes[0] == 0 and max(np.diff([*keyframes, len(packets)])) <= 30
    pts: list[int] = [packet.pts for packet in packets if packet.pts is not None]
    assert len(pts) == len(packets) and pts == sorted(pts)  # decode order is presentation order
    np.testing.assert_array_equal(np.asarray(columns["video_time"], dtype="timedelta64[ns]").astype(np.int64), np.arange(100) * 40_000_000)


def test_each_person_draws_in_its_own_palette_colour() -> None:
    """Masks, 3D landmarks and meshes of two persons differ in colour (main's per-person palette, not one orange)."""
    persons: tuple[int, ...] = (0, 1)
    tracks = MultiViewTracks(
        0, ("rig_00/cam_00",), np.zeros(2, np.int64), np.asarray(persons, np.int64), torch.ones((2, 4, 4), dtype=torch.bool),
        np.ones((2, 4), np.float32), np.ones(2, np.float32), np.ones((1, 2), np.float32),
    )  # fmt: skip
    masks = [log.data.blob.as_arrow_array() for log in tracks.to_rerun("/world") if isinstance(log.data, rr.EncodedImage) and log.data.blob is not None]
    assert len(masks) == 2 and masks[0] != masks[1]  # the same box, a different palette
    cloud = Landmarks3d(0, persons, np.zeros((2, 3, 3), np.float32), np.ones((2, 3), bool))
    colours = [log.data.colors.as_arrow_array() for log in cloud.to_rerun("/p") if isinstance(log.data, rr.Points3D) and log.data.colors is not None]
    assert len(colours) == 2 and colours[0] != colours[1]
    topology = BodyTopology(persons, np.zeros((1, 3), np.int32))
    meshes = [log.data.albedo_factor.as_arrow_array() for log in topology.to_rerun("/p") if isinstance(log.data, rr.Mesh3D) and log.data.albedo_factor is not None]
    assert len(meshes) == 2 and meshes[0] != meshes[1]


def test_dense_2d_landmarks_draw_their_visibility_from_red_to_green() -> None:
    """As main drew them (simplecv's confidence gradient), so they read on top of the person's mask."""
    visibility = torch.tensor([[0.0, 0.5, 1.0]])
    landmarks = MultiViewLandmarks(
        0, ("rig_00/cam_00",), np.zeros(1, np.int64), np.zeros(1, np.int64), torch.zeros((1, 3, 2)), torch.zeros((1, 3)), visibility, torch.zeros((1, 3)),
        np.ones((1, 2), np.float32),
    )  # fmt: skip
    (points,) = [log.data for log in landmarks.to_rerun("/world") if isinstance(log.data, rr.Points2D)]
    assert points.colors is not None and points.colors.as_arrow_array().to_pylist() == [0xFF0000FF, 0xFFFF00FF, 0x00FF00FF]


def tree(item: object) -> object:
    """A blueprint as nested (kind, name, children or origin, active tab)."""
    if isinstance(item, rrb.Blueprint):
        return tree(item.root_container)
    if isinstance(item, rrb.Container):
        return (type(item).__name__, item.name, [tree(c) for c in item.contents], getattr(item, "active_tab", None))
    assert isinstance(item, rrb.View)
    return (type(item).__name__, item.name, str(item.origin))


def test_the_normal_view_is_the_cli_blueprint_and_the_pipeline_tab_comes_second() -> None:
    """Main's layout: the World 3D view (everything under ``/world``: frusta with their video, overlays, meshes) beside the
    camera grid, main's three plots below; the pipeline debug tab second, the normal view active. The World view draws the
    frusta at main's 0.4 m (the rig stores 0.1 m)."""
    cameras: tuple[str, ...] = ("rig_01/cam_00", "rig_03/cam_00", "rig_05/cam_00", "rig_07/cam_00")
    blueprint: rrb.Blueprint = default_blueprint(cameras)
    kind, _, (normal, pipeline), active = tree(blueprint)  # type: ignore[misc]
    assert (kind, active, pipeline[1]) == ("Tabs", 0, "pipeline (debug)")
    grid = ("Grid", "Camera Images", [("Spatial2DView", c, f"/world/{c}/pinhole") for c in cameras], None)
    plots = [
        ("TimeSeriesView", "Stage timings (ms, wall_time)", HEALTH_ROOT.rstrip("/")),
        ("TimeSeriesView", "Fit health", "/metrics/fit"),
        ("TimeSeriesView", "SMPL-X params", "/metrics/params"),
    ]
    assert normal == (
        "Vertical",
        "main",
        [("Horizontal", None, [("Spatial3DView", "World", "/world"), grid], None), ("Horizontal", None, plots, None)],
        None,
    )
    world = blueprint.root_container.contents[0].contents[0].contents[0]  # type: ignore[attr-defined]
    assert world.contents == "$origin/**"  # nothing hidden: the frusta show
    frusta = {path: (v.visualizer_type, v.overrides[0].as_arrow_array().to_pylist()) for path, v in world.visualizer_overrides.items()}
    assert frusta == {f"/world/{c}/pinhole": ("Cameras", [pytest.approx(0.4)]) for c in cameras}
    timings = blueprint.root_container.contents[0].contents[1].contents[0]  # type: ignore[attr-defined]
    assert [str(c) for c in timings.contents] == [f"+ {HEALTH_ROOT}{node}/step_ms" for node in ("track", "landmarks", "fit", "viewer")]


def test_a_segment_run_shows_the_segment_and_a_capture_directory_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_app`` (and the form) open the segment the App names in the viewer before the run writes into its recording."""
    shown: list[SegmentRef | None] = []
    monkeypatch.setattr(capture, "run_app", lambda config, build, segment: shown.append(segment(config)) or RunReport("g", "lossless", "complete", 0.0, 0.0, {}))
    capture.main(Config(RerunTyroConfig(headless=True), SegmentRef("d", "s")))
    capture.main(Config(RerunTyroConfig(headless=True), SegmentRef("d", "s"), video=Path("capture")))
    assert shown == [SegmentRef("d", "s"), None]
