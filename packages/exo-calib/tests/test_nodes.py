"""Node contracts with fake models on a synthetic 4-camera ring (no GPU, no catalog)."""

import json
from dataclasses import asdict, replace

import numpy as np
import rerun as rr
from conftest import FakeScene
from dagkit.node import Event, Fenced, RunContext, Tick
from dagkit.sinks import RerunSink
from dagkit.testing import Recorder, check_node_contract
from dagkit.time import STATIC
from serde.json import to_json

from exo_calib.apis.calibrate import PORT_TYPES, ROOTS
from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import OVERLAY_LOG_STRIDE, RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import BaseCalibration, EvalReport, Kp2dOverlay, MultiViewKeypoints, RigLayout

FRAMES: range = range(0, 48, 4)


def kp2d_records(scene: FakeScene) -> list[tuple[int, MultiViewKeypoints]]:
    ctx = RunContext("exocalib-refine", "kp2d", "lossless", resources=scene.resources())
    ticks = [Tick(f.t_ns, "frames", Fenced(f, None), {"tracking_rig": scene.rig}) for f in map(scene.frames, FRAMES)]
    return [(t, value) for _, t, value in check_node_contract(Kp2dNode(), ticks, ctx)]  # type: ignore[misc]


def test_kp2d_contract_detects_then_tracks_and_keeps_the_projections() -> None:
    scene = FakeScene()
    records = kp2d_records(scene)
    assert [t for t, _ in records] == [scene.frames(i).t_ns for i in FRAMES]
    assert set(records[0][1].box_source) == {"yolox"} and "tracked" in {s for _, r in records[2:] for s in r.box_source}
    for _, record in records:
        np.testing.assert_allclose(record.kp_xy, scene.project(record.frame_index), atol=1e-3)


def run_segment_node(node: RefineNode | ReportNode, inputs: dict[str, list[Event]], ctx: RunContext) -> list[tuple[str, int, object]]:
    out = Recorder()
    node.reset(ctx)
    node.run(inputs, out)
    return out.items


def refine_inputs(scene: FakeScene, records: list[tuple[int, MultiViewKeypoints]]) -> dict[str, list[Event]]:
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    return {"kp2d": [Event(t, "kp2d", r) for t, r in records], "init": [Event(STATIC, "init", scene.rig)], "rig_layout": [Event(STATIC, "rig_layout", layout)]}


def test_refine_is_deterministic_and_keeps_an_exact_rig_and_report_scores_it() -> None:
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    inputs = refine_inputs(scene, kp2d_records(scene))
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    runs = [run_segment_node(RefineNode(device="cpu"), inputs, ctx) for _ in range(2)]
    plain = [[(p, t, asdict(v)) for p, t, v in run] for run in runs]  # type: ignore[call-overload]
    np.testing.assert_equal(plain[1], plain[0])
    outputs = {p: v for p, _, v in runs[0]}
    assert {p for p, _, _ in runs[0]} == {"refined", "diag", "kp3d", "kp2d_overlay"}  # every camera has GT: no base calibration
    # exact observations: the metric rig itself comes back (the fake MoGe rescale is 1.00015, i.e. < 1 mm at 4 m)
    np.testing.assert_allclose(outputs["refined"].cam_T_world, scene.rig.cam_T_world, atol=2e-3)
    np.testing.assert_allclose(outputs["refined"].intrinsics, scene.rig.intrinsics, rtol=1e-6)
    report_inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig), ("rig_layout", layout))}
    report = {p: v for p, _, v in run_segment_node(ReportNode(scene.segment), report_inputs, ctx)}
    assert isinstance(report["eval"], EvalReport) and report["eval"].init is not None and report["eval"].refined is not None
    assert report["eval"].init.se3.rotation_deg.max < 1e-6 and report["eval"].refined.se3.rotation_deg.max < 0.05
    assert report["eval"].refined.se3.translation_cm.max < 0.5 and report["eval"].refined.focal_error_pct.max_abs < 1e-3
    geometry = json.loads(to_json(report["eval"]))["refined_rig_geometry"]  # the eval.json schema of the stage mains
    assert geometry["height_above_floor_m"] == [center[2] for center in geometry["camera_centers_m"]] and len(geometry["pairwise_distance_m"]) == 4


def test_refine_draws_each_camera_at_its_own_sample_time() -> None:
    scene = FakeScene()
    offsets = 250_000 * np.arange(len(scene.rig.names), dtype=np.int64)[::-1]  # the last camera samples first
    records = [(t, replace(r, cam_t_ns=r.cam_t_ns + offsets)) for t, r in kp2d_records(scene)]
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    draws = [(t, v) for p, t, v in run_segment_node(RefineNode(device="cpu"), refine_inputs(scene, records), ctx) if p == "kp2d_overlay"]
    assert all(isinstance(v, Kp2dOverlay) for _, v in draws)
    expected = sorted((int(r.cam_t_ns[i]), i, view) for _, r in records for i, view in enumerate(r.views))
    assert [(t, v.view) for t, v in draws] == [(t, view) for t, _, view in expected]  # type: ignore[union-attr]
    skeleton_times = {int(r.cam_t_ns[i]) for k, (_, r) in enumerate(records) if k % OVERLAY_LOG_STRIDE == 0 for i in range(len(r.views))}
    assert {t for t, v in draws if v.gated_xy is not None} == skeleton_times  # type: ignore[union-attr]


def test_a_rerun_sink_takes_every_refine_and_report_output() -> None:
    """Every value the App hands to a RerunSink or LayerSink satisfies dagkit's Type contract (checked in dev mode)."""
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    refined = run_segment_node(RefineNode(device="cpu"), refine_inputs(scene, kp2d_records(scene)), ctx)
    outputs = {p: v for p, _, v in refined}
    report_inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig), ("rig_layout", layout))}
    items = [*refined, *run_segment_node(ReportNode(scene.segment), report_inputs, ctx), ("base_rig", STATIC, BaseCalibration(scene.rig.select(scene.rig.names[:2])))]
    ports = sorted({p for p, _, _ in items} - {"eval"})
    recording = rr.RecordingStream("exo-calib-test", send_properties=False)
    recording.memory_recording()
    sink = RerunSink(recording, {p: PORT_TYPES[p] for p in ports}, {p: ROOTS[p] for p in ports})
    for port, t_ns, value in items:
        if port != "eval":
            sink.step(Tick(t_ns, port, value, {}), Recorder())
