"""Fit and evaluate foot surfaces, gated floor contact and symmetric limbs."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float64
from serde import serde
from serde.json import from_json, to_json

from selfcap_soma.catalog import read_window
from selfcap_soma.feet import contact_plan
from selfcap_soma.fitting import Calibration, FitOptions, FitResult, fit
from selfcap_soma.hand_pose import read_orientations
from selfcap_soma.losses import body_lengths
from selfcap_soma.observations import coco_joint_names
from selfcap_soma.output import Residual, residual, save
from selfcap_soma.shoulders import subset
from selfcap_soma.source_comparison import ASSETS, CATALOG, DATASET, SEGMENT, Metrics, evaluate, load_fit, model_for


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FootMetrics:
    feet: Residual
    """Surface landmarks versus the current exo observations, millimetres."""
    tilt_median_deg: list[float]
    """Left/right sole-up tilt from world up."""
    tilt_p90_deg: list[float]
    """Left/right sole-up tilt 90th percentile."""
    ankle_rotation_median_deg: list[float]
    """Left/right rotation magnitude relative to the model's neutral ankle."""
    lengths_mm: list[float]
    """Left/right upper arm, forearm, thigh and shin, in that order."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FootReport:
    source_sha256: str
    """Hash of source 3D positions and confidences for current/strict comparison."""
    floor_height_m: float | None
    """Robust floor estimate in the calibrated world frame."""
    contact_percent: list[float]
    """Fraction of frames with any active contact landmark, left/right."""
    before: Metrics
    """Published configuration re-evaluated against the current source."""
    after: Metrics
    """The candidate fit, before any catalog replacement."""
    before_feet: FootMetrics
    """Baseline foot geometry and limb proportions."""
    after_feet: FootMetrics
    """Candidate foot geometry and limb proportions."""
    shape_seconds: float
    """Shape-stage optimization and mesh time."""


@dataclass(frozen=True, slots=True)
class Config:
    output: Path = Path("/tmp/soma-fit/feet-fix/current")
    """Candidate output; registration is a separate action after source approval."""
    baseline: Path = Path("/tmp/soma-fit/sapiens-comparison/b-upright")
    """The previously published Sapiens2 ears+rotation fit."""
    calibration: Path = Path("/tmp/soma-fit/sapiens-comparison/calibration-b-upright.json")
    """Baseline shared identity and constant head rotation."""


def foot_metrics(result: FitResult, targets: Float64[np.ndarray, "n 6 3"]) -> FootMetrics:
    """Measure foot surfaces, orientation and symmetric neutral limb lengths."""
    calibration = Calibration(result.identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor)
    model = model_for(calibration)
    with torch.no_grad():
        rest = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
        posed = model.pose(torch.as_tensor(result.poses, device="cuda"), fk_only=True, apply_correctives=False)
    tilts = []
    rotations = []
    for side in ("Left", "Right"):
        index = result.names.index(side + "Foot")
        normals = (posed.transforms[:, index + 1, :3, :3].cpu().numpy() @ rest.transforms[0, index + 1, :3, :3].cpu().numpy().T)[:, :, 1]
        tilts.append(np.degrees(np.arccos(np.clip(normals[:, 1], -1.0, 1.0))))
        rotations.append(np.degrees(np.linalg.norm(result.poses[:, index], axis=-1)))
    mapping = coco_joint_names()
    ids = [result.names.index(mapping[key]) for key in range(5, 17)]
    lengths = body_lengths(rest.joints[:, ids])[0, 3:].cpu().numpy() * 1000
    if result.foot_landmarks is None:
        raise ValueError("foot evaluation requires surface landmarks")
    return FootMetrics(
        residual(result.foot_landmarks.astype(np.float64), targets),
        [float(np.median(x)) for x in tilts],
        [float(np.percentile(x, 90)) for x in tilts],
        [float(np.median(x)) for x in rotations],
        lengths.astype(float).tolist(),
    )


def main(config: Config) -> None:
    config.output.mkdir(exist_ok=True, parents=True)
    data = read_window(CATALOG, DATASET, SEGMENT, 0.0, 183.0, 1, "sapiens2")
    digest = sha256(data.exo.tobytes() + data.confidence.tobytes()).hexdigest()
    calibration = from_json(Calibration, config.calibration.read_text())
    baseline = load_fit(config.baseline, calibration, data)
    contact = contact_plan(data.times, data.exo[:, 17:23], data.confidence[:, 17:23])
    print("source", digest, "floor", contact.floor_height_m, "contact percent", contact.active.reshape(-1, 2, 3).any(-1).mean(0) * 100, flush=True)
    mask = (data.times < 25.0) & (np.arange(len(data.times)) % 8 == 0) & np.isfinite(data.exo[:, [3, 4, 5, 6, 11, 12]]).all((1, 2))
    initial = replace(
        baseline,
        poses=baseline.poses[mask],
        translations=baseline.translations[mask],
        joints=baseline.joints[mask],
        vertices=baseline.vertices[mask],
        ear_landmarks=baseline.ear_landmarks[mask],
        foot_landmarks=baseline.foot_landmarks[mask] if baseline.foot_landmarks is not None else None,
        head_angle_deg=baseline.head_angle_deg[mask],
    )
    options = FitOptions(
        head_mode="ears-rotation",
        head_rotation_weight=0.3,
        use_feet=True,
        floor_height_m=contact.floor_height_m,
        shape_components=128,
        identity_prior=0.05,
        scale_prior=2.0,
        learn_bone_scales=True,
        bone_length_weight=2.0,
        limb_symmetry_weight=10.0,
        display_twists=True,
    )
    shape = fit(subset(data, mask), ASSETS, 600, 300, options, initial=initial)
    calibration = Calibration(shape.identity, shape.scale, shape.head_offset, shape.bone_scales, shape.head_attachment, shape.head_anchor)
    (config.output / "calibration.json").write_text(to_json(calibration))
    raw = Path("/mnt/nas/datasets/exoego-self-collected/main/cut/2025-12-03/d31bf1da-c23c-4e4b-b88b-eb5c5ad20e3f/episodes/episode-001/quest")
    orientations = read_orientations(raw, data.times, data.offset_ms)
    result = fit(data, ASSETS, 600, 80, replace(options, orientation_weight=30.0), fixed=calibration, initial=baseline, orientations=orientations)
    save(config.output, data, result, 39.862079571, 3)
    report = FootReport(
        digest,
        contact.floor_height_m,
        (contact.active.reshape(-1, 2, 3).any(-1).mean(0) * 100).tolist(),
        evaluate("Before foot fix", data, baseline),
        evaluate("Foot and symmetry fix", data, result),
        foot_metrics(baseline, data.exo[:, 17:23]),
        foot_metrics(result, data.exo[:, 17:23]),
        shape.seconds,
    )
    (config.output / "foot-report.json").write_text(to_json(report))
    print(to_json(report), flush=True)
