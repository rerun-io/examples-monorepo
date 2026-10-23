"""The App's normal view is the CLI's (origin/main ``tools/mast3r_slam_inference.py``): the same entities and archetypes
under the same names, the same blueprint, and the pipeline graph only as the second tab (unit lane: a synthetic run of
the App's viewer sink, no GPU, no model)."""

import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest
import rerun as rr
import rerun.blueprint as rrb
from dagkit.graph import Graph
from dagkit.node import Emit, RunContext, Tick
from dagkit.runners import RunReport
from dagkit.sources import SegmentRef
from simplecv.rerun_log_utils import RerunTyroConfig

from mast3r_slam.apis import slam as app
from mast3r_slam.apis.slam import ROOT, Config, build
from mast3r_slam.nodes import nerfstudio
from mast3r_slam.nodes.frontend import Frontend
from mast3r_slam.nodes.nerfstudio import NerfstudioSink
from mast3r_slam.nodes.shared import SharedSlam
from mast3r_slam.nodes.viewer import SlamViewer
from mast3r_slam.types import CameraView, FrameView, KeyframeView, LogLine, MapUpdate, TrackedFrame

MAIN_ENTITIES: dict[str, set[str]] = {
    "": {"Transform3D"},
    "/current_camera": {"Transform3D"},
    "/current_camera/pinhole": {"Pinhole"},
    "/current_camera/pinhole/image": {"EncodedImage"},
    "/current_camera/pinhole/pointmap": {"EncodedImage"},
    "/current_camera/pinhole/pointmap_depth": {"EncodedDepthImage"},
    "/current_camera/pinhole/confidence": {"EncodedImage"},
    "/edges": {"LineStrips3D"},
    "/keyframes/keyframe-{i}": {"Transform3D"},
    "/keyframes/keyframe-{i}/pinhole": {"Pinhole"},
    "/keyframes/keyframe-{i}/pinhole/confidence": {"EncodedImage"},
    "/keyframes/keyframe-{i}/pinhole/image": {"EncodedImage"},
    "/keyframes/keyframe-{i}/pinhole/pointmap": {"EncodedImage"},
    "/keyframes/keyframe-{i}/pinhole/pointmap_depth": {"EncodedDepthImage"},
    "/keyframes/keyframe-{i}/pointcloud": {"Points3D"},
    "/last_keyframe": {"EncodedImage"},
    "/last_keyframe/confidence": {"EncodedImage"},
    "/last_keyframe/pointmap": {"EncodedImage"},
    "/last_keyframe/pointmap_depth": {"EncodedDepthImage"},
    "/logs": {"TextLog"},
    "/path": {"LineStrips3D"},
}
"""Main's CLI on TUM fr1_desk (an image input), its ``/world`` read as the App's root and every ``keyframe-<n>`` as one."""
GRAPH_OUTPUTS: tuple[str, ...] = ("/map", "/tracked", "/cam_00")
"""The graph's own outputs (catalog layer, verify), drawn in the pipeline tab only."""
H, W, FOCAL = 24, 32, 20.0


def camera(index: int, z: float = 0.1) -> CameraView:
    """A camera at depth 2 in front of a plane seen with focal ``FOCAL`` (principal point at the image centre)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    points = np.stack([(xs - W / 2) / FOCAL * 2.0, (ys - H / 2) / FOCAL * 2.0, np.full((H, W), 2.0, np.float32)], axis=-1)
    pose = np.array([0.0, 0.0, z * index, 0.0, 0.0, 0.0, 1.0, 1.0 + index], dtype=np.float32)
    return CameraView(index, pose, np.full((H, W, 3), 100 + index, np.uint8), points.astype(np.float32), np.full((H, W), 10.0, np.float32))


def entities(path: Path) -> dict[str, set[str]]:
    """Entity path (``keyframe-<n>`` folded to ``keyframe-{i}``) -> archetypes in a saved recording."""
    found: dict[str, set[str]] = {}
    for col in rr.chunk.RrdReader(path).store().schema().component_columns():
        if col.archetype and not col.entity_path.startswith("/__"):
            found.setdefault(re.sub(r"keyframe-\d+", "keyframe-{i}", col.entity_path), set()).add(col.archetype.split(".")[-1])
    return found


class Discard(Emit):
    def put(self, port: str, t_ns: int, value: object) -> None:
        raise AssertionError("a sink puts nothing")

    def watermark(self, t_ns: int) -> None:
        raise AssertionError("a sink sets no watermark")


def synthetic_run(config: Config, path: Path) -> Graph:
    """Build the App on ``config`` into a recording saved at ``path`` and feed its viewer two frames, two keyframes, two
    map updates, a log line and the final poses, as the graph would."""
    recording = rr.RecordingStream("mast3r-slam-test", send_properties=False)
    recording.save(path)
    graph = build(config, recording)
    assert isinstance(graph, Graph)
    viewer = graph.nodes["viewer"].obj
    assert isinstance(viewer, SlamViewer)
    viewer.reset(RunContext("mast3r-slam", "viewer", "lossless"))
    ticks: list[tuple[int, str, object]] = [
        (0, "view", FrameView(camera(0), None)),
        (0, "keyframe", KeyframeView(camera(0))),
        (0, "map", MapUpdate(camera(0).world_sim3_cam[None], np.zeros(0, np.int64), np.zeros(0, np.int64))),
        (33, "tracked", TrackedFrame(1, camera(1).world_sim3_cam)),
        (33, "view", FrameView(camera(1), camera(0))),
        (33, "keyframe", KeyframeView(camera(1))),
        (33, "map", MapUpdate(np.stack([camera(0).world_sim3_cam, camera(1).world_sim3_cam]), np.array([0]), np.array([1]))),
        (33, "log", LogLine("Database retrieval 1: [0]")),
        (0, "keyframes", TrackedFrame(0, camera(0).world_sim3_cam)),
        (33, "keyframes", TrackedFrame(1, camera(1).world_sim3_cam)),
    ]
    for t_ms, port, value in ticks:
        if port in viewer.inputs:
            viewer.step(Tick(t_ms * 1_000_000, port, value, {}), Discard())
    viewer.flush(Discard())
    recording.disconnect()  # closes the file (its footer)
    return graph


@pytest.fixture
def images(tmp_path: Path) -> Path:
    directory: Path = tmp_path / "rgb"
    directory.mkdir()
    cv2.imwrite(str(directory / "000.png"), np.zeros((480, 640, 3), dtype=np.uint8))
    return directory


def test_the_app_viewer_recording_has_the_cli_entities(images: Path, tmp_path: Path) -> None:
    """Every entity the CLI drew, with its archetype, under the App's root; anything else is a graph output for the
    pipeline tab (and the cloud's Sim3 scale, which keeps each keyframe's points where the map puts them)."""
    synthetic_run(Config(RerunTyroConfig(headless=True), images=images, hardware="x86_64-cuda"), tmp_path / "all.rrd")
    found: dict[str, set[str]] = entities(tmp_path / "all.rrd")
    for suffix, archetypes in MAIN_ENTITIES.items():
        assert archetypes <= found.get(ROOT + suffix, set()), f"{ROOT + suffix}: {found.get(ROOT + suffix)} lacks {archetypes}"
    extra = {p for p in found if p.startswith(ROOT) and p.removeprefix(ROOT) not in MAIN_ENTITIES}
    assert all(p.removeprefix(ROOT).startswith(GRAPH_OUTPUTS) for p in extra), sorted(extra)
    assert found[f"{ROOT}/keyframes/keyframe-{{i}}/pointcloud"] == {"Points3D", "Transform3D"}


def test_the_nerfstudio_export_draws_its_cloud_at_the_end_of_the_run_as_the_cli_did(
    images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main's CLI with ``--ns-save-path`` logged the exported cloud (voxel-downsampled, coloured) at ``/world/final_pointcloud``;
    the App draws it when the Nerfstudio export is on, at the final keyframes' time on the data clock."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0], [0.5, -0.25, 2.0]]))
    cloud.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]))
    monkeypatch.setattr(nerfstudio, "save_kf_to_nerfstudio", lambda path, keyframes: cloud)
    shared = SharedSlam.__new__(SharedSlam)  # the export is faked: no buffers behind it
    shared.keyframes = None  # type: ignore[assignment]
    recording = rr.RecordingStream("mast3r-slam-test", send_properties=False)
    recording.save(tmp_path / "ns.rrd")
    graph = build(Config(RerunTyroConfig(headless=True), images=images, hardware="x86_64-cuda", output_nerfstudio=tmp_path / "ns"), recording)
    sink = graph.nodes["nerfstudio"].obj
    assert isinstance(sink, NerfstudioSink)
    sink.reset(RunContext("mast3r-slam", "nerfstudio", "lossless", resources={"shared": shared}))
    for t_ms in (0, 33):
        sink.step(Tick(t_ms * 1_000_000, "keyframes", TrackedFrame(1, camera(1).world_sim3_cam), {}), Discard())
    sink.flush(Discard())
    recording.disconnect()
    [table] = [c.to_record_batch() for c in rr.chunk.RrdReader(tmp_path / "ns.rrd").stream().to_chunks() if str(c.entity_path) == f"{ROOT}/final_pointcloud"]
    assert table.column("video_time").to_pylist() == [np.timedelta64(33_000_000, "ns")]
    np.testing.assert_allclose(table.column("Points3D:positions").to_pylist()[0], np.asarray(cloud.points))


def test_drawing_only_the_poses_keeps_the_recording_small(images: Path, tmp_path: Path) -> None:
    """``draw="poses"`` (verify's golden): the viewer takes the final keyframe and tracked poses only, and the frontend
    makes no image copies for it."""
    graph = synthetic_run(Config(RerunTyroConfig(headless=True), images=images, hardware="x86_64-cuda", draw="poses"), tmp_path / "poses.rrd")
    assert {p for p in entities(tmp_path / "poses.rrd") if p.startswith(ROOT) and p != f"{ROOT}/logs"} <= {ROOT, f"{ROOT}/cam_00", f"{ROOT}/tracked/cam_00"}
    frontend = graph.nodes["frontend"].obj
    assert isinstance(frontend, Frontend) and not frontend.views


def tree(item: object) -> object:
    """A blueprint as nested (kind, name, origin or children, active tab)."""
    if isinstance(item, rrb.Blueprint):
        return tree(item.root_container)
    if isinstance(item, rrb.Container):
        return (type(item).__name__, item.name, [tree(c) for c in item.contents], getattr(item, "active_tab", None))
    assert isinstance(item, rrb.View)
    return (type(item).__name__, item.name, str(item.origin))


def test_the_normal_view_is_the_cli_blueprint_and_the_pipeline_tab_comes_second(images: Path) -> None:
    graph = build(Config(RerunTyroConfig(headless=True), images=images, hardware="x86_64-cuda"), rr.RecordingStream("test", send_properties=False))
    assert isinstance(graph, Graph)
    viewer = graph.nodes["viewer"].obj
    assert isinstance(viewer, SlamViewer)
    blueprint: rrb.Blueprint = viewer.blueprint(3)
    kind, _, (normal, pipeline), active = tree(blueprint)  # type: ignore[misc]
    assert (kind, active, pipeline[1]) == ("Tabs", 0, "pipeline (debug)")
    current = ("Tabs", "Current", [("Spatial2DView", "Video", f"{ROOT}/current_camera/pinhole/video"), ("Spatial2DView", "Image", f"{ROOT}/current_camera/pinhole/image")], 1)
    last = ("Horizontal", "Last Keyframe", [("Spatial2DView", "Image", f"{ROOT}/last_keyframe"), ("Spatial2DView", "Pointmap", f"{ROOT}/last_keyframe/pointmap")], None)
    side = ("Tabs", None, [("Vertical", "Views", [current, last], None), ("TextLogView", "Logs", f"{ROOT}/logs")], 0)
    assert normal == ("Horizontal", "main", [("Spatial3DView", None, "/"), side], None)
    contents = [str(c) for c in blueprint.root_container.contents[0].contents[0].contents]  # type: ignore[attr-defined]
    hidden = {f"- {ROOT}{p}" for p in ("/map/**", "/tracked/cam_00/**", "/cam_00", "/current_camera/pinhole/video/**", "/current_camera/pinhole/pointmap_depth", "/last_keyframe/pointmap_depth")}
    hidden |= {f"- {ROOT}/keyframes/keyframe-{i}/pinhole/pointmap_depth" for i in range(3)}  # back-projected, the depth would hide the frusta
    assert contents[0] == f"+ {ROOT}/**" and hidden <= set(contents)


def test_the_input_video_lands_under_the_current_camera(images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A video file is logged as it is (main logged the mp4) and the Video pane shows it; a catalog camera logs nothing, its
    Video pane shows the segment's own video (the recording joins the segment's); images and a webcam have none (as on main)."""
    logged: list[str] = []
    monkeypatch.setattr(app, "log_video", lambda source, path, timeline, *, recording: logged.append(f"{source} {path} {timeline}") or np.zeros(1, np.int64))
    video_entity: str = f"{ROOT}/current_camera/pinhole/video"
    clip: Path = tmp_path / "clip.mp4"
    panes: list[str] = []
    for config in (
        Config(RerunTyroConfig(headless=True), video=clip, hardware="x86_64-cuda"),
        Config(RerunTyroConfig(headless=True), segment=SegmentRef("d", "s"), hardware="x86_64-cuda"),
        Config(RerunTyroConfig(headless=True), images=images, hardware="x86_64-cuda"),
    ):
        viewer = build(config, rr.RecordingStream("test", send_properties=False)).nodes["viewer"].obj  # type: ignore[union-attr]
        assert isinstance(viewer, SlamViewer)
        viewer.reset(RunContext("g", "viewer", "lossless"))
        _, _, (normal, _), _ = tree(viewer.blueprint(0))  # type: ignore[misc]
        side = normal[2][1]  # main: (scene, side)
        current = side[2][0][2][0]  # side: (Views, Logs); Views: (Current, Last Keyframe)
        panes.append(current[2][0][2])  # Current: (Video, Image); a view is (kind, name, origin)
    assert logged == [f"{clip} {video_entity} video_time"]
    assert panes == [video_entity, "/world/rig_00/cam_00/pinhole/video", video_entity]


def test_a_segment_run_shows_the_segment_and_other_inputs_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_app`` (and the form) open the segment the App names in the viewer before the run writes into its recording."""
    shown: list[SegmentRef | None] = []
    monkeypatch.setattr(app, "run_app", lambda config, build, segment: shown.append(segment(config)) or RunReport("g", "lossless", "complete", 0.0, 0.0, {}))
    app.main(Config(RerunTyroConfig(headless=True), segment=SegmentRef("d", "s"), hardware="x86_64-cuda"))
    app.main(Config(RerunTyroConfig(headless=True), video=Path("clip.mp4"), segment=SegmentRef("d", "s"), hardware="x86_64-cuda"))
    assert shown == [SegmentRef("d", "s"), None]
