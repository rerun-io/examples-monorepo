"""The App's normal view is main's view of the segment (origin/main: the stage tools wrote catalog layers that the user
opened in the catalog segment under the exocalib .rbl): the segment's video and GT cameras, the same exocalib entities,
the same blueprint, and the pipeline graph only as the second tab (unit lane: the App's viewer sinks fed what their
graphs put, computed by the real nodes on the synthetic ring; the segment reads faked where they enter)."""

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from conftest import FakeScene
from dagkit.graph import Graph
from dagkit.node import Event, RunContext, Tick
from dagkit.sinks import Latest
from dagkit.sources import SegmentRef
from dagkit.testing import Recorder
from dagkit.time import STATIC
from simplecv.rerun_log_utils import RerunTyroConfig
from test_nodes import kp2d_records, refine_inputs, run_segment_node

from exo_calib.apis import calibrate
from exo_calib.apis.calibrate import Config, SegmentViewer, build
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import MetricScale, RigLayout

EGO: tuple[str, ...] = ("rig_04/cam_00", "rig_04/cam_01")
MAIN_LAYER_ENTITIES: dict[str, set[str]] = {
    "/world/exocalib": {"AnnotationContext"},
    "/world/exocalib/init": {"Transform3D", "metric_scale"},
    "/world/exocalib/init/{exo}": {"Transform3D"},
    "/world/exocalib/init/{exo}/pinhole": {"Pinhole"},
    "/world/exocalib/kp3d_init": {"Points3D", "KeypointConfidence3D", "Transform3D"},
    "/world/exocalib/kp3d_refined": {"Points3D", "KeypointConfidence3D", "Transform3D"},
    "/world/exocalib/refined": {"Transform3D", "init_reprojection_px", "ba_reprojection_px", "observation_count_per_view", "metric_rescale_fix", "focal_scale"},
    "/world/exocalib/refined/{exo}": {"Transform3D"},
    "/world/exocalib/refined/{exo}/pinhole": {"Pinhole"},
    "/world/exocalib_error/init": {"LineStrips3D"},
    "/world/exocalib_error/refined": {"LineStrips3D"},
    "/world/{exo}/pinhole/exocalib_kp2d": {"AnnotationContext", "Points2D", "KeypointConfidence2D"},
    "/world/{exo}/pinhole/exocalib_kp2d_bbox": {"Boxes2D", "box_source"},
    "/world/{exo}/pinhole/exocalib_kp2d_crop": {"Boxes2D"},
    "/world/{exo}/pinhole/exocalib_kp2d_raw": {"AnnotationContext", "Points2D", "KeypointConfidence2D"},
    "/world/{exo}/pinhole/exocalib_kp2d_rejected": {"AnnotationContext", "Points2D", "KeypointConfidence2D"},
    "/world/exocalib/kp2d/{exo}": {"frame_index", "bbox_xyxy", "box_source", "crop_origin_xy", "crop_size_wh", "crop_input_wh", "video_wh"},
    "/world/exocalib_report": {"ground_truth_available"},
}
"""main's four layers on the short case (``short.pre.pre-median-bug.rrd``, the pre-port golden) with every exo camera
read as ``{exo}``: archetypes, and the component names of ``AnyValues``. The Stage B record moved from ``…_raw`` (its
``sample_idx`` is ``frame_index``) to ``/world/exocalib/kp2d/{exo}``; main wrote ``/world/exocalib_report`` only without GT."""


def entities(path: Path, exo: tuple[str, ...]) -> dict[str, set[str]]:
    """Entity path (an exo camera folded to ``{exo}``) -> archetypes, and component names where there is no archetype."""
    found: dict[str, set[str]] = {}
    for col in rr.chunk.RrdReader(path).store().schema().component_columns():
        folded: str = re.sub("|".join(map(re.escape, exo)), "{exo}", col.entity_path)
        found.setdefault(folded, set()).add(col.archetype.split(".")[-1] if col.archetype else col.component)
    return found


def hand_on(graph: Graph, values: dict[str, object]) -> None:
    """Fill the graph's ``Latest`` sink as its run would."""
    keep = graph.nodes["keep"].obj
    assert isinstance(keep, Latest)
    keep.values = {port: values[port] for port in keep.inputs}


def draw(graph: Graph, puts: list[tuple[str, int, object]]) -> None:
    """Run the graph's viewer sink on the values its graph puts, in order."""
    viewer = graph.nodes["viewer"].obj
    viewer.reset(RunContext(graph.name, "viewer", "lossless"))
    for port, t_ns, value in puts:
        if port in viewer.inputs:
            viewer.step(Tick(t_ns, port, value, {}), Recorder())
    viewer.flush(Recorder())


def synthetic_run(monkeypatch: pytest.MonkeyPatch, path: Path, draw_mode: Literal["all", "outputs"] = "all") -> list[tuple[str, object]]:
    """The App on the synthetic ring into a recording saved at ``path``: the init graph's viewer draws the Stage A rig and
    scale, pass 2's viewer the Stage B records and the refine and report outputs. Returns the segment reads, in order."""
    scene = FakeScene()
    reads: list[tuple[str, object]] = []
    layout = RigLayout(scene.rig.names, EGO, scene.rig.names)
    monkeypatch.setattr(calibrate, "discover_rig_layout", lambda segment, exo_rigs: layout)
    monkeypatch.setattr(calibrate, "model_factories", lambda *args: {name: (lambda value=value: value) for name, value in scene.resources().items()})
    monkeypatch.setattr(calibrate, "log_segment_videos", lambda recording, segment, entities: reads.append(("videos", dict(entities))))
    monkeypatch.setattr(calibrate, "log_segment_cameras", lambda recording, segment, cameras: reads.append(("cameras", cameras)))
    recording = rr.RecordingStream("exo-calib-test", send_properties=False)
    recording.save(path)
    steps = build(Config(RerunTyroConfig(headless=True), scene.segment, register="off", hardware="x86_64-cuda", draw=draw_mode), recording)
    init: Graph = next(steps)
    assert not reads, "build read the segment: the first viewer reads it when its run starts"
    draw(init, [("init", STATIC, scene.rig), ("metric_scale", STATIC, MetricScale(1.25))])
    hand_on(init, {"init": scene.rig})
    hand_on(next(steps), {"init": scene.rig, "refined": scene.rig})  # pass 1: no viewer
    records = kp2d_records(scene)
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    refined: list[tuple[str, int, object]] = run_segment_node(RefineNode(layout, device="cpu"), refine_inputs(scene, records), ctx)
    outputs: dict[str, object] = {port: value for port, _, value in refined}
    report_inputs = {port: [Event(STATIC, port, value)] for port, value in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig))}
    report: list[tuple[str, int, object]] = run_segment_node(ReportNode(scene.segment, layout), report_inputs, ctx)
    draw(next(steps), [*(("kp2d", t_ns, record) for t_ns, record in records), *refined, *report])
    recording.disconnect()  # closes the file (its footer)
    return reads


def test_the_app_viewer_recording_has_main_s_layer_entities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every entity main's layers drew, with its archetype or AnyValues component, under the same path (the Stage B record
    under its new path); the Stage A metric scale, the grey raw joints and the report status included."""
    synthetic_run(monkeypatch, tmp_path / "app.rrd")
    found: dict[str, set[str]] = entities(tmp_path / "app.rrd", FakeScene().rig.names)
    missing = {path: archetypes - found.get(path, set()) for path, archetypes in MAIN_LAYER_ENTITIES.items() if not archetypes <= found.get(path, set())}
    assert not missing, missing


def test_the_first_viewer_draws_the_segment_when_its_run_starts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The video of every exo and ego camera under its pinhole and the dataset's cameras, once, when the init graph's
    viewer starts (never in ``build``); ``draw="outputs"`` (verify's golden) draws the graph outputs only."""
    exo: tuple[str, ...] = FakeScene().rig.names
    reads = synthetic_run(monkeypatch, tmp_path / "all.rrd")
    assert reads == [("videos", {camera: f"/world/{camera}/pinhole/video" for camera in (*exo, *EGO)}), ("cameras", (*exo, *EGO))]
    assert synthetic_run(monkeypatch, tmp_path / "outputs.rrd", "outputs") == []


def tree(item: object) -> object:
    """A blueprint as nested (kind, name, origin or children, active tab)."""
    if isinstance(item, rrb.Blueprint):
        return tree(item.root_container)
    if isinstance(item, rrb.Container):
        return (type(item).__name__, item.name, [tree(c) for c in item.contents], getattr(item, "active_tab", None))
    assert isinstance(item, rrb.View)
    return (type(item).__name__, item.name, str(item.origin))


def test_the_normal_view_is_main_s_blueprint_and_the_pipeline_tab_comes_second() -> None:
    """main's .rbl (origin/main ``apis/blueprint.py``): the 3D scene of everything beside the ego column over the exo strip,
    the raw joints hidden in the exo panes, an uncalibrated ego camera out of the 3D scene, panels collapsed; then the
    pipeline tab."""
    exo: tuple[str, ...] = FakeScene().rig.names
    layout = RigLayout(exo, EGO, (*exo, EGO[0]))
    blueprint: rrb.Blueprint = calibrate.blueprint(layout, ["exocalib-init", "exocalib-refine-pass1", "exocalib-refine"])
    kind, _, (normal, pipeline), active = tree(blueprint)  # type: ignore[misc]
    assert (kind, active, pipeline[1]) == ("Tabs", 0, "pipeline (debug)") and blueprint.collapse_panels
    ego_column = ("Vertical", None, [("Tabs", None, [("Spatial2DView", name, f"/world/{name}/pinhole")], None) for name in EGO], None)
    exo_strip = ("Horizontal", None, [("Tabs", None, [("Spatial2DView", name, f"/world/{name}/pinhole")], None) for name in exo], None)
    assert normal == ("Vertical", "main", [("Horizontal", None, [("Spatial3DView", "3D View", "/"), ego_column], None), exo_strip], None)
    main_view = blueprint.root_container.contents[0]  # type: ignore[attr-defined]
    top, strip = main_view.contents
    assert (main_view.row_shares, top.column_shares) == ([4, 1], [4, 1])
    scene = top.contents[0]
    assert str(scene.contents) == "/**" and set(scene.visualizer_overrides) == {f"/world/{EGO[1]}/pinhole"}
    for tabs, name in zip(strip.contents, exo, strict=True):
        assert set(tabs.contents[0].visualizer_overrides) == {f"/world/{name}/pinhole/exocalib_kp2d_raw"}


def base_layer(path: Path, segment: str) -> None:
    """A dataforge base layer in small: the root's view coordinates, a static exo rig, a worn ego rig that moves, each
    camera's node, pinhole and video (two AV1 samples), and a dataset label beside them."""
    recording = rr.RecordingStream("base", recording_id=segment, send_properties=False)
    recording.save(path)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for rig, camera, kind in (("rig_00", "rig_00/cam_00", "exo"), ("rig_04", "rig_04/cam_00", "ego")):
        recording.log(f"/world/{rig}", rr.AnyValues(kind=kind), static=True)
        recording.log(f"/world/{camera}", rr.Transform3D(translation=[1.0, 2.0, 3.0], relation=rr.TransformRelation.ChildFromParent), static=True)
        recording.log(f"/world/{camera}/pinhole", rr.Pinhole(focal_length=100.0, width=64, height=48), static=True)
        recording.log(f"/world/{camera}/pinhole/video", rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
        times = np.array([0, 33_000_000], dtype="timedelta64[ns]")
        recording.send_columns(f"/world/{camera}/pinhole/video", indexes=[rr.TimeColumn("video_time", duration=times)], columns=rr.VideoStream.columns(sample=[b"\x01", b"\x02"], is_keyframe=[True, False]))
    recording.send_columns("/world/rig_04", indexes=[rr.TimeColumn("video_time", duration=times)], columns=rr.Transform3D.columns(translation=[[0.0, 0.0, 1.0], [0.0, 0.5, 1.0]]))
    recording.log("/world/gt/coco133_xyz", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.disconnect()


@pytest.mark.integration
def test_the_segment_view_copies_the_dataset_s_cameras_their_video_and_a_worn_rig_s_motion(tmp_path: Path) -> None:
    """The first viewer's reset reads an in-process catalog (integration lane): the base layer's root, rig and camera
    nodes, pinholes and videos come over as they are, static stays static, the worn rig's pose stays temporal, and
    nothing else of the segment (its labels) comes with them."""
    base_layer(tmp_path / "base.rrd", "segment-0")
    recording = rr.RecordingStream("exo-calib-test", send_properties=False)
    recording.save(tmp_path / "app.rrd")
    with rr.server.Server(host="127.0.0.1", datasets={"ds": [str(tmp_path / "base.rrd")]}) as server:
        segment = SegmentRef("ds", "segment-0", catalog_url=server.url())
        SegmentViewer(recording, {}, {}, segment, ("rig_00/cam_00", "rig_04/cam_00")).reset(RunContext("exocalib-init", "viewer", "lossless"))
    recording.disconnect()
    copied: dict[str, set[str]] = {}
    for col in rr.chunk.RrdReader(tmp_path / "app.rrd").store().schema().component_columns():
        copied.setdefault(col.entity_path, set()).add(f"{col.archetype.split('.')[-1] if col.archetype else col.component}{'' if col.is_static else ' (temporal)'}")
    expected: dict[str, set[str]] = {"/": {"ViewCoordinates"}, "/world/rig_04": {"kind", "Transform3D (temporal)"}, "/world/rig_00": {"kind"}}
    for camera in ("rig_00/cam_00", "rig_04/cam_00"):
        expected |= {f"/world/{camera}": {"Transform3D"}, f"/world/{camera}/pinhole": {"Pinhole"}, f"/world/{camera}/pinhole/video": {"VideoStream", "VideoStream (temporal)"}}
    assert copied == expected
