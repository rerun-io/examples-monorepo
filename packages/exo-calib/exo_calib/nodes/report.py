"""Stage E (whole segment): score the Stage A and refined rigs against the segment's GT calibration.

GT enters here and nowhere else. Both rigs are aligned to GT with SE(3) (primary: tests the metric claim) and Sim(3)
(isolates the scale error). Without GT for every exo camera the report still carries the refinement diagnostics and
the GT-free rig geometry, and no alignment is drawn.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from dagkit.node import Emit, Event, PortType, RunContext
from dagkit.sources import SegmentRef
from dagkit.time import STATIC
from simplecv.ops.umeyama import SimilarityTransform
from simplecv.types.rig import RigCameras

from exo_calib.ops.eval import RIG_MODES, align_rigs, camera_centers, rig_geometry, variant_metrics
from exo_calib.types import VARIANTS, EvalReport, RefinementDiagnostics, RigAlignment, RigGeometryReport, RigLayout, VariantMetrics


class ReportNode:
    inputs: Mapping[str, PortType] = {"refined": RigCameras, "diag": RefinementDiagnostics, "init": RigCameras, "gt": RigCameras, "rig_layout": RigLayout}
    outputs: Mapping[str, PortType] = {"align": RigAlignment, "eval": EvalReport}

    def __init__(self, segment: SegmentRef) -> None:
        self.segment: SegmentRef = segment

    def reset(self, ctx: RunContext) -> None:
        pass

    def run(self, inputs: Mapping[str, Sequence[Event]], out: Emit) -> None:
        refined, diag, layout = (inputs[port][-1].value for port in ("refined", "diag", "rig_layout"))
        assert isinstance(refined, RigCameras) and isinstance(diag, RefinementDiagnostics) and isinstance(layout, RigLayout)
        names: tuple[str, ...] = refined.names
        init: RigCameras = inputs["init"][-1].value.select(names)  # type: ignore[union-attr]
        rigs: list[RigCameras] = [init, refined]
        gt_rig: object = inputs["gt"][-1].value if inputs["gt"] else None
        gt: RigCameras | None = None
        if isinstance(gt_rig, RigCameras) and set(names) <= set(layout.calibrated_camera_names) & set(gt_rig.names):
            gt = gt_rig.select(names)
        else:
            print(f"ground truth: unavailable for some of {list(names)}; evaluation and alignment skipped")
        metrics: list[VariantMetrics | None] = [None if gt is None else variant_metrics(rig, gt) for rig in rigs]
        for variant, m in zip(VARIANTS, metrics, strict=True):
            for mode in RIG_MODES if m is not None else ():
                e = getattr(m, mode)
                print(f"{variant:8s} {mode:4s}: trans cm mean {e.translation_cm.mean:6.2f} max {e.translation_cm.max:6.2f} | rot deg mean {e.rotation_deg.mean:5.2f} max {e.rotation_deg.max:5.2f} | scale {e.scale:.4f}")
        geometry = rig_geometry(refined.cam_T_world)
        print(f"refined pairwise camera distances (m): {geometry.pairwise_distance_m.round(3).tolist()}")
        print(f"refined camera heights above fitted floor (m): {geometry.height_above_floor_m.round(3).tolist()}")
        geometry_report = RigGeometryReport(
            geometry.camera_centers_m.round(6).tolist(), geometry.pairwise_distance_m.round(6).tolist(), geometry.height_above_floor_m.round(6).tolist()
        )
        report = EvalReport(self.segment.segment, self.segment.dataset, list(names), metrics[0], metrics[1], diag, geometry_report)
        out.put("eval", STATIC, report)
        if gt is not None:  # each rig gets its own SE(3) pred-world -> GT-world transform, so everything overlays GT
            aligned: list[SimilarityTransform] = [align_rigs(rig.cam_T_world, gt.cam_T_world) for rig in rigs]
            centers = np.stack([a.apply(camera_centers(rig.cam_T_world)) for a, rig in zip(aligned, rigs, strict=True)])
            rotations, translations = np.stack([a.dst_R_src for a in aligned]), np.stack([a.dst_t_src for a in aligned])
            out.put("align", STATIC, RigAlignment(names, camera_centers(gt.cam_T_world), rotations, translations, centers))
