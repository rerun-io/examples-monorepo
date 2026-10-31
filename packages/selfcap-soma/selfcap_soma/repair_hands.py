"""Diagnose and repair one segment's unconstrained hand rotations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from serde import serde
from serde.json import from_json, to_json
from soma.body import SOMALayer

from selfcap_soma.catalog import read_window
from selfcap_soma.fitting import Calibration, FitOptions, fit
from selfcap_soma.hand_diagnostics import Plausibility, measure
from selfcap_soma.hand_pose import read_orientations
from dataforge.quest_joint_poses import read_segment_hand_poses
from rerun.catalog import CatalogClient
from selfcap_soma.output import save


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Diagnosis:
    before: Plausibility
    twist_display_only: Plausibility
    after: Plausibility
    wrist_along_forearm_mm: list[float]
    raw_source: str


def main() -> None:
    root = Path("/tmp/soma-fit/hand-repair-final")
    root.mkdir(exist_ok=True, parents=True)
    assets = Path("/tmp/soma-fit/assets")
    calibration = from_json(Calibration, Path("/tmp/soma-fit/ears/ears-offset-rotation-calibration.json").read_text())
    data = read_window("rerun+http://dgx-spark:51235", "selfcap-post-dev", "selfcap__2025-12-03__d31bf1da__ep001", 0.0, 183.0, 1)
    baseline = fit(data, assets, 400, 80, fixed=calibration)

    def model(twists: bool) -> SOMALayer:
        result = SOMALayer(
            data_root=assets,
            device="cuda",
            identity_model_type="soma",
            mode="torch",
            lod="low",
            enable_procedural_transforms=twists,
            correctives_model_path=None,
        )
        result.requires_grad_(False)
        with torch.no_grad():
            result.prepare_identity(
                torch.tensor(calibration.identity, device="cuda"),
                scale_params=torch.tensor(calibration.bone_scales, device="cuda"),
                global_scale=calibration.scale,
            )
        return result

    plain = model(False)
    before = measure(plain, baseline, data.times)
    print("BEFORE", to_json(before), flush=True)
    procedural = model(True)
    twist_only = measure(procedural, baseline, data.times)
    print("TWIST ONLY", to_json(twist_only), flush=True)
    dataset = CatalogClient("rerun+http://dgx-spark:51235").get_dataset(name="selfcap-post-dev")
    orientations = read_orientations(read_segment_hand_poses(dataset.filter_segments("selfcap__2025-12-03__d31bf1da__ep001")), data.times, data.offset_ms)
    result = fit(
        data,
        assets,
        400,
        80,
        FitOptions(orientation_weight=30.0, hands_only=True, display_twists=True),
        fixed=calibration,
        initial=baseline,
        orientations=orientations,
    )
    after = measure(procedural, result, data.times)
    wrist = []
    for side, elbow, key in ((91, 7, 9), (112, 8, 10)):
        direction = data.exo[:, key] - data.exo[:, elbow]
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        wrist.append(float(np.nanmedian(np.sum((data.hands[:, side] - data.exo[:, key]) * direction, axis=-1)) * 1000))
    (root / "diagnosis.json").write_text(to_json(Diagnosis(before, twist_only, after, wrist, str(raw))))
    save(root, data, result, 4.144, 3)
    # Store only selected-frame geometry for the close-up screenshot producer.
    import rerun as rr

    recording = rr.RecordingStream("hand-repair-comparison")
    recording.save(root / "comparison.rrd")
    recording.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    for label, candidate, mesh_model in [("before", baseline, plain), ("after", result, procedural)]:
        for stamp in [*before.worst_times_s, 39.862079571, 4.144403454]:
            i = int(np.argmin(abs(data.times - stamp)))
            with torch.no_grad():
                posed = mesh_model.pose(
                    torch.tensor(candidate.poses[i : i + 1], device="cuda"),
                    transl=torch.tensor(candidate.translations[i : i + 1], device="cuda"),
                    apply_correctives=False,
                )
            vertices = posed.vertices[0].cpu().numpy()
            faces = mesh_model.faces.cpu().numpy()
            recording.set_time("video_time", duration=stamp)
            for side in ("Left", "Right"):
                center = candidate.joints[i, candidate.names.index(side + "Hand")]
                # Tight wrist/hand crop; only triangles wholly within 20 cm.
                keep = np.linalg.norm(vertices - center, axis=-1) < 0.20
                selected = faces[keep[faces].all(-1)]
                ids = np.unique(selected)
                inverse = np.full(len(vertices), -1)
                inverse[ids] = np.arange(len(ids))
                recording.log(
                    f"/world/{side}/{label}",
                    rr.Mesh3D(vertex_positions=vertices[ids], triangle_indices=inverse[selected], albedo_factor=[0.3, 0.7, 0.85, 1.0]),
                )
    recording.flush()
    recording.disconnect()
    print("AFTER", to_json(after), flush=True)
