"""Node contracts with fake models on a synthetic 4-camera ring (no GPU, no catalog)."""

import json
from dataclasses import asdict, replace

import cv2
import numpy as np
import rerun as rr
from conftest import FakeModel, FakeScene, Person
from dagkit.node import Event, Fenced, RunContext, Tick
from dagkit.sinks import RerunSink
from dagkit.testing import Recorder, check_node_contract
from dagkit.time import STATIC
from serde.json import to_json
from simplecv.types.rig import RigCameras

from exo_calib.apis.calibrate import PORT_TYPES, ROOTS
from exo_calib.nodes.body import BodyNode
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import OVERLAY_LOG_STRIDE, RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import (
    BaseCalibration,
    BodyShape,
    EvalReport,
    Kp2dOverlay,
    MultiViewKeypoints,
    RefinementDiagnostics,
    ReportStatus,
    RigLayout,
)

FRAMES: range = range(0, 48, 4)


def kp2d_records(scene: FakeScene) -> list[tuple[int, MultiViewKeypoints]]:
    ctx = RunContext("exocalib-refine", "kp2d", "lossless", resources=scene.resources())
    ticks = [Tick(f.t_ns, "frames", Fenced(f, None), {"tracking_rig": scene.rig}) for f in map(scene.frames, FRAMES)]
    return [(t, value) for _, t, value in check_node_contract(Kp2dNode(scene.layout), ticks, ctx)]  # type: ignore[misc]


def test_kp2d_contract_detects_then_tracks_and_keeps_the_projections() -> None:
    scene = FakeScene()
    records = kp2d_records(scene)
    assert [t for t, _ in records] == [scene.frames(i).t_ns for i in FRAMES]
    assert set(records[0][1].box_source) == {"yolox"} and "tracked" in {s for _, r in records[2:] for s in r.box_source}
    for _, record in records:
        np.testing.assert_allclose(record.kp_xy, scene.project(record.frame_index), atol=1e-3)


def test_kp2d_keeps_one_person_when_another_out_scores_it_in_some_views() -> None:
    """A second person 1 m to the side scores above the subject in views 0 and 2. Ticks 1.5 s apart, so the 2 s re-anchor
    detects in every view again; the first frameset and every re-anchor keep the subject in all four views."""
    scene = FakeScene(people=(Person(score=0.5), Person(offset_m=(0.0, 1.0, 0.0), views=(0, 2))))
    ctx = RunContext("exocalib-refine", "kp2d", "lossless", resources=scene.resources())
    ticks = [Tick(1_500_000_000 * k, "frames", Fenced(scene.frames(i), None), {"tracking_rig": scene.rig}) for k, i in enumerate(FRAMES)]
    records = [value for _, _, value in check_node_contract(Kp2dNode(scene.layout), ticks, ctx)]
    assert sum(set(r.box_source) == {"yolox"} for r in records) > 1  # the re-anchors ran
    for record in records:
        np.testing.assert_allclose(record.kp_xy, scene.project(record.frame_index), atol=1e-3)  # type: ignore[union-attr]


def test_kp2d_keeps_detecting_a_view_whose_rig_projection_misses_the_person() -> None:
    """Six cameras; the tracking rig turns view 3 by 12 degrees (a camera Stage A placed badly), so the skeleton the other five
    views agree on projects beside the subject there; the view re-detects every frameset until its projection agrees with the
    detector, and keeps the subject's keypoints."""
    scene = FakeScene(n_views=6)
    turned = scene.rig.cam_T_world.copy()
    turned[3, :3] = cv2.Rodrigues(np.array([0.0, np.deg2rad(12.0), 0.0]))[0] @ turned[3, :3]  # about its own centre
    tracking = RigCameras(scene.rig.names, scene.rig.intrinsics, turned, scene.rig.resolution_wh)
    ctx = RunContext("exocalib-refine", "kp2d", "lossless", resources=scene.resources())
    ticks = [Tick(f.t_ns, "frames", Fenced(f, None), {"tracking_rig": tracking}) for f in map(scene.frames, FRAMES)]
    records = [value for _, _, value in check_node_contract(Kp2dNode(scene.layout), ticks, ctx)]
    for record in records:
        assert record.conf[3].min() > 0.5, (record.frame_index, record.box_source)  # type: ignore[union-attr]
        np.testing.assert_allclose(record.kp_xy[3], scene.project(record.frame_index)[3], atol=1e-3)  # type: ignore[union-attr]


def run_segment_node(node: RefineNode | ReportNode, inputs: dict[str, list[Event]], ctx: RunContext) -> list[tuple[str, int, object]]:
    out = Recorder()
    node.reset(ctx)
    node.run(inputs, out)
    return out.items


def refine_inputs(scene: FakeScene, records: list[tuple[int, MultiViewKeypoints]]) -> dict[str, list[Event]]:
    return {"kp2d": [Event(t, "kp2d", r) for t, r in records], "init": [Event(STATIC, "init", scene.rig)]}


def test_refine_is_deterministic_and_keeps_an_exact_rig_and_report_scores_it() -> None:
    scene = FakeScene()
    inputs = refine_inputs(scene, kp2d_records(scene))
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    runs = [run_segment_node(RefineNode(scene.layout, device="cpu"), inputs, ctx) for _ in range(2)]
    plain = [[(p, t, asdict(v)) for p, t, v in run] for run in runs]  # type: ignore[call-overload]
    np.testing.assert_equal(plain[1], plain[0])
    outputs = {p: v for p, _, v in runs[0]}
    assert {p for p, _, _ in runs[0]} == {"refined", "diag", "kp3d", "kp2d_overlay"}  # every camera has GT: no base calibration
    # exact observations: the metric rig itself comes back (the fake MoGe rescale is 1.00015, i.e. < 1 mm at 4 m)
    np.testing.assert_allclose(outputs["refined"].cam_T_world, scene.rig.cam_T_world, atol=2e-3)
    np.testing.assert_allclose(outputs["refined"].intrinsics, scene.rig.intrinsics, rtol=1e-6)
    report_inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig))}
    report = {p: v for p, _, v in run_segment_node(ReportNode(scene.segment, scene.layout), report_inputs, ctx)}
    assert isinstance(report["eval"], EvalReport) and report["eval"].init is not None and report["eval"].refined is not None
    assert report["eval"].init.se3.rotation_deg.max < 1e-6 and report["eval"].refined.se3.rotation_deg.max < 0.05
    assert report["eval"].refined.se3.translation_cm.max < 0.5 and report["eval"].refined.focal_error_pct.max_abs < 1e-3
    geometry = json.loads(to_json(report["eval"]))["refined_rig_geometry"]  # the eval.json schema of the stage mains
    assert geometry["height_above_floor_m"] == [center[2] for center in geometry["camera_centers_m"]] and len(geometry["pairwise_distance_m"]) == 4


def test_refine_corrects_a_camera_stage_a_turned_15_degrees() -> None:
    """The keypoints, not a fixed bound, decide how far refine turns a camera (Ego-Exo4D georgiatech's rig_03 started 15.7 deg
    off; a 10 deg guard put Stage A's pose back and kept the error)."""
    scene = FakeScene()
    turned = scene.rig.cam_T_world.copy()
    turn = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(15.0)]))[0]
    turned[2, :3] = turn @ turned[2, :3]  # camera 2 rolls 15 deg about its own centre
    inputs = {"kp2d": [Event(t, "kp2d", r) for t, r in kp2d_records(scene)], "init": [Event(STATIC, "init", replace(scene.rig, cam_T_world=turned))]}
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    refined = {p: v for p, _, v in run_segment_node(RefineNode(scene.layout, device="cpu"), inputs, ctx)}["refined"]
    relative = refined.cam_T_world[2, :3, :3] @ refined.cam_T_world[0, :3, :3].T @ (scene.rig.cam_T_world[2, :3, :3] @ scene.rig.cam_T_world[0, :3, :3].T).T  # type: ignore[attr-defined]
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0])) < 0.5


def test_without_gt_for_every_camera_the_report_says_so_and_aligns_nothing() -> None:
    """main's align layer on data without GT: ``ground_truth_available=False`` and no alignment (with GT: True and the
    alignment)."""
    scene = FakeScene()
    ctx = RunContext("exocalib-refine", "report", "lossless")
    diag = RefinementDiagnostics(1.0, [0.5], [10, 10, 10, 10], 1.0, [1.0, 1.0, 1.0, 1.0])
    inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", scene.rig), ("diag", diag), ("init", scene.rig), ("gt", scene.rig))}
    for calibrated, expected in (((), {"eval": None, "status": ReportStatus(False)}), (scene.rig.names, {"eval": None, "status": ReportStatus(True), "align": None})):
        report = {p: v for p, _, v in run_segment_node(ReportNode(scene.segment, RigLayout(scene.rig.names, (), calibrated)), inputs, ctx)}
        assert report.keys() == expected.keys() and report["status"] == expected["status"]


def cleared(value: object) -> bool:
    """A drawing that only clears (the frame after a sample)."""
    return all(isinstance(log.data, rr.Clear) for log in value.to_rerun("/world"))  # type: ignore[attr-defined]


def test_refine_draws_each_camera_at_its_own_sample_time() -> None:
    scene = FakeScene()
    offsets = 250_000 * np.arange(len(scene.rig.names), dtype=np.int64)[::-1]  # the last camera samples first
    records = [(t, replace(r, cam_t_ns=r.cam_t_ns + offsets)) for t, r in kp2d_records(scene)]
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    puts = run_segment_node(RefineNode(scene.layout, device="cpu"), refine_inputs(scene, records), ctx)
    draws = [(t, v) for p, t, v in puts if p == "kp2d_overlay" and not cleared(v)]
    assert all(isinstance(v, Kp2dOverlay) for _, v in draws)
    expected = sorted((int(r.cam_t_ns[i]), i, view) for _, r in records for i, view in enumerate(r.views))
    assert [(t, v.view) for t, v in draws] == [(t, view) for t, _, view in expected]  # type: ignore[union-attr]
    skeleton_times = {int(r.cam_t_ns[i]) for k, (_, r) in enumerate(records) if k % OVERLAY_LOG_STRIDE == 0 for i in range(len(r.views))}
    assert {t for t, v in draws if v.gated_xy is not None} == skeleton_times  # type: ignore[union-attr]


def test_refine_clears_each_drawing_at_its_camera_s_next_video_frame() -> None:
    """A sample's overlay and 3D joints show on that video frame only: at the camera's next frame an empty drawing clears
    them (the fake samples every 4th frame, so the next frame is never a sample)."""
    scene = FakeScene()
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    puts = run_segment_node(RefineNode(scene.layout, device="cpu"), refine_inputs(scene, kp2d_records(scene)), ctx)
    overlays = [(t, v) for p, t, v in puts if p == "kp2d_overlay"]
    assert sorted((t, v.view) for t, v in overlays if cleared(v)) == sorted((1_000_000 * (k + 1), view) for k in FRAMES for view in scene.rig.names)  # type: ignore[attr-defined]
    assert [t for t, _ in overlays] == sorted(t for t, _ in overlays)
    kp3d = [(t, v) for p, t, v in puts if p == "kp3d"]
    assert [t for t, v in kp3d] == [1_000_000 * (k + d) for k in FRAMES for d in (0, 1)]
    assert [cleared(v) for _, v in kp3d] == [False, True] * len(FRAMES)


def test_3d_joints_clear_at_a_next_sample_that_triangulates_none() -> None:
    scene = FakeScene()
    records = [replace(r, frame_index=k, cam_t_ns=np.full(len(r.views), 1_000_000 * k)) for k, (_, r) in enumerate(kp2d_records(scene))]
    records[1] = replace(records[1], conf=np.zeros_like(records[1].conf))  # frame 1: no joint to triangulate
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    puts = run_segment_node(RefineNode(scene.layout, device="cpu"), refine_inputs(scene, [(1_000_000 * k, r) for k, r in enumerate(records)]), ctx)
    kp3d = [(t, cleared(v)) for p, t, v in puts if p == "kp3d"]
    last = len(records) - 1
    assert kp3d == [(0, False), (1_000_000, True)] + [(1_000_000 * k, False) for k in range(2, last + 1)] + [(1_000_000 * (last + 1), True)]


def test_cameras_on_one_clock_that_start_apart_clear_at_their_own_next_frame() -> None:
    """SelfCap's phones share the clock but start apart, so a frameset is frame k of the first camera and frame k + 3v of
    camera v: each camera's drawing still clears at that camera's next frame, the same moment for all of them."""

    class StartApart(FakeScene):
        def times_ns(self, view: int) -> np.ndarray:
            return 1_000_000 * (np.arange(64, dtype=np.int64) - 3 * view)  # camera v started 3v frames earlier

    scene = StartApart()
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    puts = run_segment_node(RefineNode(scene.layout, device="cpu"), refine_inputs(scene, kp2d_records(scene)), ctx)
    clears = sorted((t, v.view) for p, t, v in puts if p == "kp2d_overlay" and cleared(v))  # type: ignore[attr-defined]
    assert clears == sorted((1_000_000 * (k + 1), view) for k in FRAMES for view in scene.rig.names)


def test_a_rerun_sink_takes_every_refine_and_report_output() -> None:
    """Every value the App hands to a RerunSink or LayerSink satisfies dagkit's Type contract (checked in dev mode)."""
    scene = FakeScene()
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    refined = run_segment_node(RefineNode(scene.layout, device="cpu"), refine_inputs(scene, kp2d_records(scene)), ctx)
    outputs = {p: v for p, _, v in refined}
    report_inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig))}
    items = [*refined, *run_segment_node(ReportNode(scene.segment, scene.layout), report_inputs, ctx), ("base_rig", STATIC, BaseCalibration(scene.rig.select(scene.rig.names[:2])))]
    ports = sorted({p for p, _, _ in items} - {"eval"})
    recording = rr.RecordingStream("exo-calib-test", send_properties=False)
    recording.memory_recording()
    sink = RerunSink(recording, {p: PORT_TYPES[p] for p in ports}, {p: ROOTS[p] for p in ports})
    for port, t_ns, value in items:
        if port != "eval":
            sink.step(Tick(t_ns, port, value, {}), Recorder())


def test_kp2d_keeps_its_person_through_a_frameset_without_body_joints_when_another_person_arrives() -> None:
    """A track built from the hands alone still has an identity: when a second person arrives (detected in every view, and
    the person a fresh cross-view vote would pick here), the node keeps the first one."""
    scene = FakeScene()
    intrinsics = scene.rig.intrinsics.copy()
    intrinsics[:, [0, 1], [0, 1]] *= 0.5  # both people and every box inside every image
    scene.rig = replace(scene.rig, intrinsics=intrinsics)

    def hands_only(frames_hwc, boxes):
        result = scene.pose(frames_hwc, boxes)
        result.scores[:, :17] = 0.0
        return result

    ctx = RunContext("exocalib-refine", "kp2d", "lossless", resources=scene.resources() | {"pose": FakeModel(hands_only, crop_spec=scene.crop_spec)})
    node, out = Kp2dNode(scene.layout), Recorder()
    node.reset(ctx)
    node.step(Tick(1_000_000 * 0, "frames", Fenced(scene.frames(0), None), {"tracking_rig": scene.rig}), out)
    scene.people = (Person(score=0.5), Person(offset_m=(1.0746, 0.9509, -0.3334)))
    node.step(Tick(1_000_000 * 4, "frames", Fenced(scene.frames(4), None), {"tracking_rig": scene.rig}), out)

    np.testing.assert_allclose(out.items[-1][2].kp_xy, scene.project(4, person=0), atol=1e-3)  # type: ignore[union-attr]


def test_the_body_is_fitted_once_through_the_refined_rig_and_drawn_on_each_frameset_until_its_next_video_frame() -> None:
    """The fake fitter's "mesh" is the triangulated shoulders and left hip, so it lands on the subject only if the node hands it
    the keypoints of the right framesets in the rig's pinhole pixels."""
    scene = FakeScene()
    ctx = RunContext("exocalib-refine", "body", "lossless", resources=scene.resources())
    inputs = {"kp2d": [Event(t, "kp2d", r) for t, r in kp2d_records(scene)], "rig": [Event(STATIC, "rig", scene.rig)]}

    puts = run_segment_node(BodyNode(scene.layout), inputs, ctx)

    assert [(t, type(v)) for p, t, v in puts if p == "body_shape"] == [(STATIC, BodyShape)]
    bodies = [(t, v) for p, t, v in puts if p == "body"]
    assert [t for t, _ in bodies] == [1_000_000 * (k + d) for k in FRAMES for d in (0, 1)]
    for k, (_, body) in zip(FRAMES, bodies[::2], strict=True):
        np.testing.assert_allclose(body.vertices, scene.skeleton[[5, 6, 11]] + np.array([0.002 * k, 0.0, 0.0]), atol=1e-3)  # type: ignore[attr-defined]
    assert all(len(v.vertices) == 0 for _, v in bodies[1::2])  # type: ignore[attr-defined]


def test_a_body_clears_at_a_next_sample_that_does_not_see_the_body() -> None:
    """Only a sample that fits a body replaces the previous one: a sample on the very next frame that sees no body clears it."""
    scene = FakeScene()
    ctx = RunContext("exocalib-refine", "body", "lossless", resources=scene.resources())
    records = [r for _, r in kp2d_records(scene)[:3]]
    adjacent = [replace(r, frame_index=k, cam_t_ns=np.full(len(r.views), 1_000_000 * k)) for k, r in enumerate(records)]
    adjacent[1] = replace(adjacent[1], conf=np.zeros_like(adjacent[1].conf))  # frame 1 sees no body
    inputs = {"kp2d": [Event(1_000_000 * k, "kp2d", r) for k, r in enumerate(adjacent)], "rig": [Event(STATIC, "rig", scene.rig)]}

    puts = run_segment_node(BodyNode(scene.layout), inputs, ctx)

    bodies = [(t, len(v.vertices) > 0) for p, t, v in puts if p == "body"]  # type: ignore[attr-defined]
    assert bodies == [(0, True), (1_000_000, False), (2_000_000, True), (3_000_000, False)]
