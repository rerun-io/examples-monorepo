"""Single-segment held-out fitting experiment; no catalog writes."""

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float, Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import to_json

from selfcap_soma.camera_geometry import project, project_ego, triangulate_frames
from selfcap_soma.catalog import Observations, read_window
from selfcap_soma.evaluation import apply_body_offsets, heldout_blocks, learn_body_offsets
from selfcap_soma.evaluation_io import nearest_indices, read_image_tracks, read_views
from selfcap_soma.fitting import FitOptions, FitResult, fit
from selfcap_soma.output import mapped, save


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Score:
    name: str
    unit: Literal["mm", "px"]
    count: int
    fit_median: float | None
    fit_p90: float | None
    comparison_median: float | None
    comparison_p90: float | None


def score(
    name: str, predicted: Float[ndarray, "... d"], comparison: Float[ndarray, "... d"], target: Float[ndarray, "... d"], unit: Literal["mm", "px"]
) -> Score:
    """Score both methods on exactly the same finite point correspondences."""
    valid = np.isfinite(predicted).all(-1) & np.isfinite(comparison).all(-1) & np.isfinite(target).all(-1)
    factor = 1000.0 if unit == "mm" else 1.0
    a = np.linalg.norm(predicted - target, axis=-1)[valid] * factor
    b = np.linalg.norm(comparison - target, axis=-1)[valid] * factor
    return Score(
        name,
        unit,
        int(valid.sum()),
        float(np.median(a)) if a.size else None,
        float(np.percentile(a, 90)) if a.size else None,
        float(np.median(b)) if b.size else None,
        float(np.percentile(b, 90)) if b.size else None,
    )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EvaluationReport:
    segment: str
    frames: int
    heldout_frames: int
    split: str
    body_split_s: float
    scores: list[Score]
    optimizer_seconds: list[float]
    total_seconds: float
    body_offsets_m: list[list[float]]
    limitations: list[str]


@dataclass(frozen=True, slots=True)
class Config:
    output: Path = Path("/tmp/soma-fit/evaluation")
    """New local experiment directory."""
    assets: Path = Path("/tmp/soma-fit/assets")
    """Official model files."""
    egohand: Path = Path("/tmp/egohand/comparison-corrected.rrd")
    """Optional corrected image evidence; source is never changed."""
    iterations: int = 400
    """Fixed optimization budget, not tuned on evaluation labels."""
    calibrated: bool = False
    """Fit shape only on well-observed frames before 25 s, then freeze."""
    options: FitOptions = field(default_factory=FitOptions)
    """Explicit candidate settings; defaults keep the original baseline."""


def main(config: Config) -> None:
    start = perf_counter()
    segment = "selfcap__2025-12-03__d31bf1da__ep001"
    url, dataset_name = "rerun+http://dgx-spark:51235", "selfcap-post-dev"
    data: Observations = read_window(url, dataset_name, segment, 0.3, 50.0, 1)
    views, oak = read_views(CatalogClient(url).get_dataset(dataset_name), segment, data.times)
    calibration_cache: dict[bytes, FitResult] = {}
    calibration_times: list[float] = []

    def solve(observed: Observations) -> FitResult:
        calibrated = None
        if config.calibrated:
            valid = (
                (observed.times < 25.0)
                & np.isfinite(observed.exo[:, [5, 6, 11, 12]]).all((1, 2))
                & (observed.confidence[:, [5, 6, 11, 12]].min(1) >= 0.7)
            )
            selected = np.flatnonzero(valid)[::8]
            key = sha256(observed.exo[selected].tobytes() + observed.hands[selected].tobytes() + observed.confidence[selected].tobytes()).digest()
            if key not in calibration_cache:
                training = replace(
                    observed,
                    times=observed.times[selected],
                    exo=observed.exo[selected],
                    hands=observed.hands[selected],
                    confidence=observed.confidence[selected],
                    head_positions=observed.head_positions[selected],
                    head_rotations=observed.head_rotations[selected],
                )
                calibrated = fit(training, config.assets, 600, 300, config.options)
                calibration_cache[key] = calibrated
                calibration_times.append(calibrated.seconds)
            calibrated = calibration_cache[key]
        return fit(observed, config.assets, config.iterations, 80, config.options, fixed=calibrated)

    heldout: Bool[ndarray, "n"] = heldout_blocks(data.times)
    withheld: Float64[ndarray, "n 133 3"] = data.hands.copy()
    withheld[heldout] = np.nan
    heldout_fit: FitResult = solve(replace(data, hands=withheld))
    points: Float64[ndarray, "n 133 3"] = mapped(heldout_fit, data)
    rows = [score("Quest hands: alternating 2 s held-out blocks", points[heldout, 91:], data.exo[heldout, 91:], data.hands[heldout, 91:], "mm")]
    target_index = np.searchsorted(data.times, 4.144403454, side="right") - 1
    rows.append(
        score("Quest hands: held-out target frame", points[target_index, 91:], data.exo[target_index, 91:], data.hands[target_index, 91:], "mm")
    )
    runtimes = [heldout_fit.seconds]
    config.output.mkdir(parents=True, exist_ok=True)
    save(config.output / "quest-holdout", data, heldout_fit, 4.144403454)

    # The body evaluation target is absent from ALL fitting and initialization
    # on [25,50]. Hand slots use independent Quest measurements there.
    train: Bool[ndarray, "n"] = data.times < 25.0
    if config.options.use_ears:
        ear_input = data.exo.copy()
        ear_input[~train, 3:5] = np.nan
        ear_data = replace(data, exo=ear_input)
        ear_fit = solve(ear_data)
        eye_pin = fit(ear_data, config.assets, config.iterations, 80)
        rows.append(
            score(
                "Ears [25,50]: withheld ears vs original eye pin",
                ear_fit.ear_landmarks[~train],
                eye_pin.ear_landmarks[~train],
                data.exo[~train, 3:5],
                "mm",
            )
        )
        runtimes.extend((ear_fit.seconds, eye_pin.seconds))
    body_input: Float64[ndarray, "n 133 3"] = data.exo.copy()
    body_input[~train, 5:17] = np.nan
    body_fit = solve(replace(data, exo=body_input))
    body_points = mapped(body_fit, data)
    rotations = Rotation.from_rotvec(body_fit.poses[:, 0].astype(np.float64)).as_matrix()
    offsets = learn_body_offsets(body_points[:, 5:17], data.exo[:, 5:17], rotations, train)
    corrected = apply_body_offsets(body_points[:, 5:17], rotations, offsets)
    rows.append(score("Body on [25,50]: fixed offsets vs joint centers", corrected[~train], body_points[~train, 5:17], data.exo[~train, 5:17], "mm"))
    runtimes.append(body_fit.seconds)

    # Leave each scored body landmark out on the test interval while keeping
    # the other body observations. Wrists are excluded: hand wrist slots and
    # Quest hands would otherwise supply the same anatomical point.
    scored_body = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    body_before = np.full((len(data.times), len(scored_body), 3), np.nan)
    body_after = np.full_like(body_before, np.nan)
    body_truth = data.exo[:, scored_body]
    for column, key in enumerate(scored_body):
        leave_joint_out = data.exo.copy()
        leave_joint_out[~train, key] = np.nan
        joint_fit = solve(replace(data, exo=leave_joint_out))
        predicted_joint = mapped(joint_fit, data)[:, key : key + 1]
        joint_rotations = Rotation.from_rotvec(joint_fit.poses[:, 0].astype(np.float64)).as_matrix()
        offset = learn_body_offsets(predicted_joint, data.exo[:, key : key + 1], joint_rotations, train)
        body_before[:, column] = predicted_joint[:, 0]
        body_after[:, column] = apply_body_offsets(predicted_joint, joint_rotations, offset)[:, 0]
        runtimes.append(joint_fit.seconds)
    rows.append(score("Body [25,50]: leave-one-landmark-out offsets vs centers", body_after[~train], body_before[~train], body_truth[~train], "mm"))

    # Exclude the scored phone BEFORE triangulation. Its detections never
    # reach shape, pose, initialization, or temporal regularization targets.
    for index, view in enumerate(views):
        others = [other for i, other in enumerate(views) if i != index]
        triangulation = triangulate_frames(
            [v.camera for v in others], np.stack([v.pixels for v in others], axis=1), np.stack([v.scores for v in others], axis=1)
        )
        triangulated, confidence = triangulation.xyz, triangulation.confidence
        camera_fit = solve(replace(data, exo=triangulated.astype(np.float64), confidence=confidence.astype(np.float64)))
        candidate = mapped(camera_fit, data)
        projected = project(view.camera, candidate.reshape(-1, 3)).reshape(-1, 133, 3)
        raw_projected = project(view.camera, triangulated.reshape(-1, 3).astype(np.float64)).reshape(-1, 133, 3)
        targets = np.where((view.scores >= 0.7)[..., None], view.pixels, np.nan)
        candidate_uv = np.where(projected[:, :, 2:] > 0.001, projected[:, :, :2], np.nan)
        raw_uv = np.where(raw_projected[:, :, 2:] > 0.001, raw_projected[:, :, :2], np.nan)
        for label, keys in (("body", slice(5, 17)), ("hands", slice(91, 133)), ("ears", slice(3, 5))):
            rows.append(score(f"{view.name} leave-camera-out {label}", candidate_uv[:, keys], raw_uv[:, keys], targets[:, keys], "px"))
        runtimes.append(camera_fit.seconds)

    # The OAK detections were never fitting inputs. Match side by independent
    # Quest projection; reject ambiguous center assignments rather than
    # choosing the assignment that improves the fitted score.
    if oak is None:
        fit_uv = np.full((*points.shape[:2], 2), np.nan)
        raw_uv = fit_uv.copy()
        quest_uv = fit_uv.copy()
    else:
        fit_uv = project_ego(oak, data.times, points.astype(np.float32))
        raw_uv = project_ego(oak, data.times, data.exo.astype(np.float32))
        quest_uv = project_ego(oak, data.times, data.hands.astype(np.float32))
    hand_order = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    for track in read_image_tracks(config.egohand):
        idx = nearest_indices(data.times, track.times)
        targets = track.points[:, hand_order] if track.points.shape[1] == 25 else track.points
        predictions = np.full_like(targets, np.nan)
        comparisons = np.full_like(targets, np.nan)
        for row, i in enumerate(idx):
            if abs(data.times[i] - track.times[row]) > 0.022 or not heldout[i]:
                continue
            if not np.isfinite(targets[row]).any():
                continue
            center = np.nanmedian(targets[row], axis=0)
            centers = [np.nanmedian(quest_uv[i, base : base + 21], axis=0) for base in (91, 112)]
            distance = np.linalg.norm(np.asarray(centers) - center, axis=-1)
            if not np.isfinite(distance).all() or abs(distance[0] - distance[1]) < 20.0:
                continue
            base = (91, 112)[int(np.argmin(distance))]
            predictions[row] = fit_uv[i, base : base + 21]
            comparisons[row] = raw_uv[i, base : base + 21]
        rows.append(score(f"OAK {track.name}: Quest-held-out frames", predictions, comparisons, targets, "px"))
    report = EvaluationReport(
        segment,
        len(data.times),
        int(heldout.sum()),
        "[0,2), [4,6), ... withheld from Quest hand fitting",
        25.0,
        rows,
        runtimes + calibration_times,
        perf_counter() - start,
        offsets.tolist(),
        [
            "Fixed calibration and previously selected wearer associations are shared; calibration was not held out.",
            "Phone scores use leave-camera-out triangulation, not the stored all-camera triangulation.",
            "The body comparison scores absent fitting targets on [25,50]; offsets were learned only before 25 s.",
            "Body offsets calibrate observation landmarks. Ear mode uses documented mesh vertices; no foot-surface constraints.",
            "OAK evidence is only the available corrected 2–6 s artifact; only finite common points are scored.",
            "EgoHand/Sapiens side is selected by Quest projection with a 20 px ambiguity margin, independently of either scored method.",
            "No OAK detector confidence was retained in that artifact, so those rows use finite points only.",
        ],
    )
    (config.output / "evaluation.json").write_text(to_json(report))
    print(to_json(report), flush=True)
    recording = rr.RecordingStream("selfcap-soma-heldout-landmarks")
    recording.save(config.output / "body-removed-stress.rrd")
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    for name, array, color in (
        ("centers", body_points[:, 5:17], [255, 160, 40]),
        ("offsets", corrected, [0, 255, 255]),
        ("targets", data.exo[:, 5:17], [240, 80, 230]),
    ):
        recording.log(f"/world/{name}", rr.Points3D.from_fields(colors=color, radii=0.01), static=True)
        recording.send_columns(
            f"/world/{name}",
            indexes=[rr.TimeColumn("video_time", duration=data.times)],
            columns=rr.Points3D.columns(positions=array.reshape(-1, 3)).partition([12] * len(data.times)),
        )
    recording.flush()
    recording.disconnect()
