"""Node contracts with fake models on a synthetic 4-camera ring (no GPU, no catalog)."""

from dataclasses import asdict

import numpy as np
from conftest import FakeScene
from dagkit.node import Event, Fenced, RunContext, Tick
from dagkit.testing import Recorder, check_node_contract
from dagkit.time import STATIC

from exo_calib.nodes.kp2d import Kp2dNode
from exo_calib.nodes.refine import RefineNode
from exo_calib.nodes.report import ReportNode
from exo_calib.types import EvalReport, MultiViewKeypoints, RigLayout

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


def test_refine_is_deterministic_and_keeps_an_exact_rig_and_report_scores_it() -> None:
    scene = FakeScene()
    layout = RigLayout(scene.rig.names, (), scene.rig.names)
    inputs = {"kp2d": [Event(t, "kp2d", r) for t, r in kp2d_records(scene)], "init": [Event(STATIC, "init", scene.rig)], "rig_layout": [Event(STATIC, "rig_layout", layout)]}
    ctx = RunContext("exocalib-refine", "refine", "lossless", resources=scene.resources())
    runs = [run_segment_node(RefineNode(device="cpu"), inputs, ctx) for _ in range(2)]
    plain = [[(p, t, asdict(v)) for p, t, v in run] for run in runs]  # type: ignore[call-overload]
    np.testing.assert_equal(plain[1], plain[0])
    outputs = {p: v for p, _, v in runs[0]}
    assert {p for p, _, _ in runs[0]} == {"refined", "diag", "kp3d", "kp2d_gated"}  # every camera has GT: no base calibration
    report_inputs = {p: [Event(STATIC, p, v)] for p, v in (("refined", outputs["refined"]), ("diag", outputs["diag"]), ("init", scene.rig), ("gt", scene.rig), ("rig_layout", layout))}
    report = {p: v for p, _, v in run_segment_node(ReportNode(scene.segment), report_inputs, ctx)}
    assert isinstance(report["eval"], EvalReport) and report["eval"].init is not None and report["eval"].refined is not None
    assert report["eval"].init.se3.rotation_deg.max < 1e-6 and report["eval"].refined.se3.rotation_deg.max < 0.05
    assert report["eval"].refined.sim3.translation_cm.max < 0.5
