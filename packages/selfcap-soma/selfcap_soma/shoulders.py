"""Shoulder proportions, signed residuals, and controlled attachment/shape ablations."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from jaxtyping import Bool
from numpy import ndarray
from serde import serde
from serde.json import to_json

from selfcap_soma.catalog import Observations, read_window
from selfcap_soma.evaluate import mapped
from selfcap_soma.fitting import Calibration, FitOptions, FitResult, fit
from selfcap_soma.output import save


def subset(data: Observations, mask: Bool[ndarray, "n"]) -> Observations:
    return replace(
        data,
        times=data.times[mask],
        exo=data.exo[mask],
        confidence=data.confidence[mask],
        hands=data.hands[mask],
        head_positions=data.head_positions[mask],
        head_rotations=data.head_rotations[mask],
        support=data.support[mask] if data.support is not None else None,
    )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Bone:
    name: str
    observed_mm: float
    fitted_mm: float


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Joint:
    name: str
    vertical_mm: float
    lateral_mm: float
    median_mm: float
    p90_mm: float


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Run:
    name: str
    scale: float
    head_offset_m: list[float]
    identity_norm: float
    identity_max: float
    bones: list[Bone]
    joints: list[Joint]
    seconds: float


def describe(name: str, data: Observations, result: FitResult) -> Run:
    points = mapped(result, data)
    rows = []
    for label, pairs in (
        ("shoulder width", [(5, 6)]),
        ("hip width", [(11, 12)]),
        ("upper arm", [(5, 7), (6, 8)]),
        ("forearm", [(7, 9), (8, 10)]),
        ("thigh", [(11, 13), (12, 14)]),
        ("shin", [(13, 15), (14, 16)]),
    ):
        obs = np.concatenate([np.linalg.norm(data.exo[:, a] - data.exo[:, b], axis=-1) for a, b in pairs])
        pred = np.concatenate([np.linalg.norm(points[:, a] - points[:, b], axis=-1) for a, b in pairs])
        valid = np.isfinite(obs) & np.isfinite(pred)
        rows.append(Bone(label, float(np.median(obs[valid]) * 1000), float(np.median(pred[valid]) * 1000)))
    for label, obs, pred in (
        (
            "shoulder to hip midpoint",
            np.linalg.norm(data.exo[:, [5, 6]].mean(1) - data.exo[:, [11, 12]].mean(1), axis=-1),
            np.linalg.norm(points[:, [5, 6]].mean(1) - points[:, [11, 12]].mean(1), axis=-1),
        ),
        (
            "Quest tracked pose to shoulder midpoint",
            np.linalg.norm(data.head_positions - data.exo[:, [5, 6]].mean(1), axis=-1),
            np.linalg.norm(data.head_positions - points[:, [5, 6]].mean(1), axis=-1),
        ),
    ):
        valid = np.isfinite(obs) & np.isfinite(pred)
        rows.append(Bone(label, float(np.median(obs[valid]) * 1000), float(np.median(pred[valid]) * 1000)))
    lateral = data.exo[:, 5] - data.exo[:, 6]
    lateral[:, 1] = 0.0
    lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True)
    joints = []
    for key, label in (
        (3, "left ear"),
        (4, "right ear"),
        (5, "left shoulder"),
        (6, "right shoulder"),
        (11, "left hip"),
        (12, "right hip"),
        (7, "left elbow"),
        (8, "right elbow"),
        (9, "left wrist"),
        (10, "right wrist"),
    ):
        delta = points[:, key] - data.exo[:, key]
        valid = np.isfinite(delta).all(-1) & np.isfinite(lateral).all(-1)
        distance = np.linalg.norm(delta[valid], axis=-1) * 1000
        joints.append(
            Joint(
                label,
                float(np.median(delta[valid, 1]) * 1000),
                float(np.median(np.sum(delta[valid] * lateral[valid], axis=-1)) * 1000),
                float(np.median(distance)),
                float(np.percentile(distance, 90)),
            )
        )
    return Run(
        name,
        result.scale,
        result.head_offset.astype(float).tolist(),
        float(np.linalg.norm(result.identity)),
        float(abs(result.identity).max()),
        rows,
        joints,
        result.seconds,
    )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Report:
    runs: list[Run]
    calibration_frames: int
    train_end_s: float


def main() -> None:
    output = Path("/tmp/soma-fit/ears")
    output.mkdir(parents=True, exist_ok=True)
    data = read_window("rerun+http://dgx-spark:51235", "selfcap-post-dev", "selfcap__2025-12-03__d31bf1da__ep001", 0.3, 50.0, 2)
    reports = []
    baseline = fit(data, Path("/tmp/soma-fit/assets"), 400, 80)
    reports.append(describe("baseline", data, baseline))
    save(output / "baseline", data, baseline, 4.144, 3)
    valid = (data.times < 25.0) & np.isfinite(data.exo[:, [5, 6, 11, 12]]).all((1, 2)) & (data.confidence[:, [5, 6, 11, 12]].min(1) >= 0.7)
    indices = np.flatnonzero(valid)[::4]
    train = np.zeros(len(data.times), dtype=bool)
    train[indices] = True
    for name, options in (
        (
            "ears-offset-rotation",
            FitOptions(
                shape_components=128,
                identity_prior=0.05,
                scale_prior=2.0,
                learn_head_offset=True,
                learn_head_rotation=True,
                use_ears=True,
                learn_bone_scales=True,
                bone_length_weight=2.0,
            ),
        ),
    ):
        calibrated = fit(subset(data, train), Path("/tmp/soma-fit/assets"), 600, 300, options)
        state = Calibration(
            calibrated.identity, calibrated.scale, calibrated.head_offset, calibrated.bone_scales, calibrated.head_attachment, calibrated.head_anchor
        )
        (output / f"{name}-calibration.json").write_text(to_json(state))
        (output / f"{name}-options.json").write_text(to_json(options))
        result = fit(data, Path("/tmp/soma-fit/assets"), 400, 80, options, fixed=calibrated)
        reports.append(describe(name, data, result))
        reports.append(
            describe(
                name + "-after25s",
                subset(data, data.times >= 25.0),
                replace(result, joints=result.joints[data.times >= 25.0], ear_landmarks=result.ear_landmarks[data.times >= 25.0]),
            )
        )
        save(output / name, data, result, 4.144, 3)
        print(to_json(reports[-2]), flush=True)
    report = Report(reports, int(train.sum()), 25.0)
    (output / "diagnosis.json").write_text(to_json(report))
    print(to_json(report), flush=True)
