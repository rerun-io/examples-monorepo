"""Fit the upper body and hands with frozen, unobserved legs and fixed shared shape."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float64
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import from_json, to_json

from selfcap_soma.catalog import read_window
from selfcap_soma.fitting import Calibration, FitOptions, FitResult, fit
from selfcap_soma.hand_pose import read_orientations
from selfcap_soma.output import save
from selfcap_soma.source_comparison import ASSETS, CATALOG, DATASET, SEGMENT, Metrics, evaluate, load_fit, model_for


@dataclass(frozen=True, slots=True)
class Config:
    output: Path = Path("/tmp/soma-fit/upper-body/published")
    """Local candidate output, before explicit catalog replacement."""
    baseline: Path = Path("/tmp/soma-fit/sapiens-comparison/b-upright")
    """Previously published Sapiens2 ears-plus-rotation baseline."""
    calibration: Path = Path("/tmp/soma-fit/sapiens-comparison/calibration-b-upright.json")
    """Baseline shape and segment-wide head rotation."""
    shared_shape: Path = Path("/tmp/soma-fit/feet-fix/current/calibration.json")
    """Retain the earlier shape fitted with leg observations and limb symmetry."""
    initial_fit: Path | None = None
    """Optional previous pose fit to continue without restarting its pose solution."""
    iterations: int = 600
    """Pose-only refinement iterations with the shared shape fixed."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class UpperBodyReport:
    source_sha256: str
    """Hash of the source 3D observations and confidence values."""
    frames: int
    """Number of fitted video-time frames."""
    start_s: float
    """First fitted video timestamp."""
    end_s: float
    """Last fitted video timestamp."""
    before: Metrics
    """Old baseline re-scored on current source, upper-body points only."""
    after: Metrics
    """Upper-body candidate scored on the same points."""
    leg_support_weight: float
    """Knee/ankle confidence multiplier in the optimizer."""
    shape_seconds: float
    """Shared shape-stage runtime."""
    retained_vertices: int
    """Vertices in the published upper-body crop."""
    retained_faces: int
    """Triangles fully inside the upper-body selection."""
    hip_landmark_weight: float = 1.0
    """COCO hip confidence multiplier."""
    before_torso_jitter_deg_s: float | None = None
    """90th percentile chest angular speed in the baseline."""
    after_torso_jitter_deg_s: float | None = None
    """90th percentile chest angular speed in the candidate."""
    before_pelvis_height_std_mm: float | None = None
    """Standard deviation of baseline pelvis height over the segment."""
    after_pelvis_height_std_mm: float | None = None
    """Standard deviation of candidate pelvis height over the segment."""
    shared_shape_source: str | None = None
    """Earlier fixed shape calibration, rather than a torso-only identity refit."""
    torso_temporal_weight: float = 0.0
    """Applied pelvis/chest angular-velocity-change prior."""


def torso_jitter(result: FitResult, times: Float64[np.ndarray, "n"]) -> float:
    """Chest angular speed between neighbouring frames, with actual timestamps."""
    model = model_for(Calibration(result.identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor))
    with torch.no_grad():
        posed = model.pose(torch.as_tensor(result.poses, device="cuda"), fk_only=True, apply_correctives=False)
    rotations = posed.transforms[:, result.names.index("Chest") + 1, :3, :3].cpu().numpy()
    delta = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    speed = np.degrees(Rotation.from_matrix(delta).magnitude()) / np.diff(times)
    return float(np.percentile(speed, 90))


def main(config: Config) -> None:
    config.output.mkdir(exist_ok=True, parents=True)
    data = read_window(CATALOG, DATASET, SEGMENT, 0.0, 183.0, 1, "sapiens2")
    digest = sha256(data.exo.tobytes() + data.confidence.tobytes()).hexdigest()
    if digest == "53db6232cd64588020b107195a5aedc39908f98b5d3efdc122dcee74e650b2a3":
        raise ValueError("source is still the previous, weak-support Sapiens2 layer")
    print("strict source", digest, "frames", len(data.times), flush=True)
    calibration = from_json(Calibration, config.calibration.read_text())
    baseline = load_fit(config.baseline, calibration, data, with_foot_landmarks=False)
    initial = baseline
    if config.initial_fit is not None:
        initial = load_fit(
            config.initial_fit, from_json(Calibration, (config.initial_fit / "calibration.json").read_text()), data, with_foot_landmarks=False
        )
    calibration = from_json(Calibration, config.shared_shape.read_text())
    options = FitOptions(
        head_mode="ears-rotation",
        head_rotation_weight=0.3,
        use_feet=False,
        foot_weight=0.0,
        leg_support_weight=0.0,
        hip_landmark_weight=0.15,
        freeze_legs=True,
        torso_temporal_weight=0.1,
        shape_components=128,
        identity_prior=0.05,
        scale_prior=2.0,
        learn_bone_scales=True,
        bone_length_weight=2.0,
        limb_symmetry_weight=10.0,
        display_twists=True,
    )
    (config.output / "calibration.json").write_text(to_json(calibration))
    raw = Path("/mnt/nas/datasets/exoego-self-collected/main/cut/2025-12-03/d31bf1da-c23c-4e4b-b88b-eb5c5ad20e3f/episodes/episode-001/quest")
    orientations = read_orientations(raw, data.times, data.offset_ms)
    result = fit(
        data, ASSETS, config.iterations, 1, replace(options, orientation_weight=30.0), fixed=calibration, initial=initial, orientations=orientations
    )
    save(config.output, data, result, 39.862079571, 3, upper_body_only=True)
    if result.upper_body_vertices is None:
        raise ValueError("missing upper-body display selection")
    retained = np.isin(result.faces, result.upper_body_vertices).all(1).sum()
    report = UpperBodyReport(
        digest,
        len(data.times),
        float(data.times[0]),
        float(data.times[-1]),
        evaluate("Previous B, current source", data, baseline, upper_body_only=True),
        evaluate("Upper body and hands", data, result, upper_body_only=True),
        options.leg_support_weight,
        0.0,
        len(result.upper_body_vertices),
        int(retained),
        hip_landmark_weight=options.hip_landmark_weight,
        before_torso_jitter_deg_s=torso_jitter(baseline, data.times),
        after_torso_jitter_deg_s=torso_jitter(result, data.times),
        before_pelvis_height_std_mm=float(np.std(baseline.joints[:, baseline.names.index("Hips"), 1]) * 1000),
        after_pelvis_height_std_mm=float(np.std(result.joints[:, result.names.index("Hips"), 1]) * 1000),
        shared_shape_source=str(config.shared_shape),
        torso_temporal_weight=options.torso_temporal_weight,
    )
    (config.output / "upper-body-report.json").write_text(to_json(report))
    print(to_json(report), flush=True)
