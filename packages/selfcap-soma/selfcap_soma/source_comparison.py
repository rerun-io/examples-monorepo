"""Single-segment Sapiens2 ears-only versus rotation-only comparison."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from datafusion import col, lit
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import from_json, to_json
from soma.body import SOMALayer

from selfcap_soma.camera_geometry import project, project_ego
from selfcap_soma.catalog import Observations, read_window, seconds
from selfcap_soma.evaluate import mapped
from selfcap_soma.evaluation_io import nearest_indices, read_views
from selfcap_soma.fitting import Calibration, FitOptions, FitResult, fit
from selfcap_soma.hand_diagnostics import Plausibility, measure
from selfcap_soma.hand_pose import read_orientations
from selfcap_soma.landmarks import FOOT_VERTICES
from selfcap_soma.output import Report, Residual, residual, save
from selfcap_soma.shoulders import subset

ROOT = Path("/tmp/soma-fit/sapiens-comparison")
ASSETS = Path("/tmp/soma-fit/assets")
SEGMENT = "selfcap__2025-12-03__d31bf1da__ep001"
CATALOG = "rerun+http://dgx-spark:51235"
DATASET = "selfcap-post-dev"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Pixels:
    median: float | None
    p90: float | None
    count: int


def pixel_error(errors: Float64[ndarray, "..."]) -> Pixels:
    valid = errors[np.isfinite(errors)]
    return Pixels(float(np.median(valid)) if len(valid) else None, float(np.percentile(valid, 90)) if len(valid) else None, len(valid))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CrossView:
    detector: str
    view: str
    pixels: Pixels


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Metrics:
    name: str
    body: Residual
    shoulders: Residual
    ears: Residual
    quest_hands: Residual
    oak_fingers: Pixels
    head_pitch_median_deg: float
    head_pitch_abs_p90_deg: float
    head_pitch_min_deg: float
    head_pitch_max_deg: float
    head_pitch_speed_p90_deg_s: float
    head_pitch_extreme_times_s: list[float]
    cross: list[CrossView]
    plausibility: Plausibility
    seconds: float


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LandmarkDifference:
    coco: int
    sapiens_minus_rtmw_mm: list[float]
    distance: Residual


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Comparison:
    landmarks: list[LandmarkDifference]
    runs: list[Metrics]
    shape_seconds: float
    notes: tuple[str, ...]


def stream(path: Path, entity: str, column: str) -> tuple[Float64[ndarray, "n"], pa.ChunkedArray]:
    """Read and sort an existing fitted stream without re-fitting its baseline."""
    batches = []
    for chunk in RrdReader(path).stream().filter(content=entity).to_chunks():
        batch = chunk.to_record_batch()
        if "video_time" in batch.schema.names and column in batch.schema.names:
            batches.append(pa.Table.from_batches([batch]).select(["video_time", column]))
    if not batches:
        raise ValueError(f"missing {entity}:{column} in {path}")
    table = pa.concat_tables(batches).sort_by([("video_time", "ascending")])
    return seconds(table["video_time"]), table[column]


def model_for(calibration: Calibration) -> SOMALayer:
    model = SOMALayer(
        data_root=ASSETS,
        device="cuda",
        identity_model_type="soma",
        mode="torch",
        lod="low",
        enable_procedural_transforms=True,
        correctives_model_path=None,
    )
    model.requires_grad_(False)
    with torch.no_grad():
        model.prepare_identity(
            torch.tensor(calibration.identity, device="cuda"),
            scale_params=torch.tensor(calibration.bone_scales, device="cuda"),
            global_scale=calibration.scale,
        )
    return model


def load_fit(directory: Path, calibration: Calibration, data: Observations) -> FitResult:
    times, raw = stream(directory / "fit.rrd", "/parameters/poses", "Tensor:data")
    if not np.allclose(times, data.times, atol=1e-8):
        raise ValueError("baseline and observation timestamps differ")
    poses = np.stack([np.asarray(row[0]["buffer"], dtype=np.float32).reshape(77, 3) for row in raw.to_pylist()])
    _, roots = stream(directory / "fit.rrd", "/parameters/root", "Points3D:positions")
    _, joints = stream(directory / "fit.rrd", "/world/fit/joints", "Points3D:positions")
    _, ears = stream(directory / "fit.rrd", "/world/fit/ears", "Points3D:positions")
    model = model_for(calibration)
    if calibration.head_attachment is None:
        raise ValueError("baseline lacks a head attachment")
    report = from_json(Report, (directory / "report.json").read_text())
    translations = np.asarray(roots.to_pylist(), dtype=np.float32).reshape(-1, 3)
    foot_batches = []
    with torch.no_grad():
        for start in range(0, len(poses), 32):
            posed = model.pose(
                torch.tensor(poses[start : start + 32], device="cuda"),
                transl=torch.tensor(translations[start : start + 32], device="cuda"),
                apply_correctives=False,
            )
            foot_batches.append(posed.vertices[:, torch.as_tensor(FOOT_VERTICES, device="cuda")].mean(2).cpu().numpy())
    feet = np.concatenate(foot_batches)
    return FitResult(
        poses,
        np.asarray(roots.to_pylist(), dtype=np.float32).reshape(-1, 3),
        calibration.identity,
        calibration.scale,
        np.asarray(joints.to_pylist(), dtype=np.float32),
        np.empty((len(times), 0, 3), dtype=np.float32),
        model.faces.cpu().numpy(),
        tuple(model.public_joint_names[1:]),
        model.public_joint_parent_ids.cpu().numpy().astype(np.int64),
        report.elapsed_s,
        0,
        np.zeros(len(times)),
        calibration.head_attachment,
        np.asarray(ears.to_pylist(), dtype=np.float32),
        calibration.head_anchor,
        calibration.head_offset,
        calibration.bone_scales,
        True,
        True,
        feet,
    )


def evaluate(name: str, data: Observations, result: FitResult) -> Metrics:
    dataset = CatalogClient(CATALOG).get_dataset(DATASET)
    points = mapped(result, data)
    cross = []
    # Both detectors see exactly the same predicted frames; missing detections stay absent.
    camera = None
    for detector in ("rtmw", "sapiens2"):
        phones, camera = read_views(dataset, SEGMENT, data.times, detector)
        for phone in phones:
            prediction = project(phone.camera, points.reshape(-1, 3)).reshape(-1, 133, 3)
            error = np.linalg.norm(prediction[..., :2] - phone.pixels, axis=-1)
            error[(phone.scores < 0.7) | (prediction[..., 2] <= 0)] = np.nan
            cross.append(CrossView(detector, phone.name, pixel_error(error)))
    assert camera is not None
    view = dataset.filter_segments(SEGMENT)
    columns = view.schema().column_names()
    paths = [
        name.split(":", 1)[0]
        for name in columns
        if name.startswith("/world/rig_04/cam_00/pinhole/") and "/egohand/" in name and name.endswith(":Points2D:positions")
    ]
    if len(paths) != 2:
        raise ValueError(f"expected two EgoHand RGB tracks, got {paths}")
    table = (
        view.filter_contents(paths)
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int(data.times[0] * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int(data.times[-1] * 1e9)))
        .to_arrow_table()
    )
    order25 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    fingers = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
    errors = []
    for path in paths:
        side = 0 if path.endswith("/left") else 1
        key = f"{path}:Points2D:positions"
        rows = table.filter(table[key].is_valid())
        stamps = seconds(rows["video_time"])
        target = np.full((len(stamps), 21, 2), np.nan)
        for i, row in enumerate(rows[key].to_pylist()):
            if len(row) == 25:
                target[i] = np.asarray(row)[order25]
        hand = points[:, 91 + side * 21 : 112 + side * 21]
        world = np.empty((len(stamps), 21, 3), dtype=np.float32)
        for joint in range(21):
            for axis in range(3):
                world[:, joint, axis] = np.interp(stamps, data.times, hand[:, joint, axis])
        errors.extend(np.linalg.norm(project_ego(camera, stamps, world)[:, fingers] - target[:, fingers], axis=-1).ravel())
    calibration = Calibration(result.identity, result.scale, result.head_offset, result.bone_scales, result.head_attachment, result.head_anchor)
    model = model_for(calibration)
    with torch.no_grad():
        neutral = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
        rest = neutral.joints[0].cpu().numpy()
        index = result.names.index("Head")
        right = rest[result.names.index("RightEye")] - rest[result.names.index("LeftEye")]
        up = rest[result.names.index("HeadEnd")] - rest[index]
        forward = np.cross(right, -up)
        forward /= np.linalg.norm(forward)
        local = neutral.transforms[0, index + 1, :3, :3].cpu().numpy().T @ forward
        posed = model.pose(torch.tensor(result.poses, device="cuda"), fk_only=True, apply_correctives=False)
        direction = posed.transforms[:, index + 1, :3, :3].cpu().numpy() @ local
    pitch = np.degrees(np.arctan2(direction[:, 1], np.linalg.norm(direction[:, [0, 2]], axis=1)))
    plausibility = measure(model, result, data.times)
    return Metrics(
        name,
        residual(points[:, 5:17], data.exo[:, 5:17]),
        residual(points[:, 5:7], data.exo[:, 5:7]),
        residual(points[:, 3:5], data.exo[:, 3:5]),
        residual(points[:, 91:], data.hands[:, 91:]),
        pixel_error(np.asarray(errors)),
        float(np.median(pitch)),
        float(np.percentile(abs(pitch), 90)),
        float(pitch.min()),
        float(pitch.max()),
        float(np.percentile(abs(np.diff(pitch) / np.diff(data.times)), 90)),
        [float(data.times[np.argmin(pitch)]), float(data.times[np.argmax(pitch)])],
        cross,
        plausibility,
        result.seconds,
    )


def ear_axis_attachment(data: Observations, model: SOMALayer, mask: Bool[ndarray, "n"]) -> Float32[ndarray, "3 3"]:
    """Fit one Quest/skull rotation from ear axes and a torso-up pitch gauge."""
    with torch.no_grad():
        neutral = model.pose(torch.zeros(1, 77, 3, device="cuda"), fk_only=True, apply_correctives=False)
    names = model.public_joint_names[1:]
    xyz = neutral.joints[0].cpu().numpy()
    right = xyz[names.index("RightEye")] - xyz[names.index("LeftEye")]
    right /= np.linalg.norm(right)
    up = xyz[names.index("HeadEnd")] - xyz[names.index("Head")]
    up -= right * (up @ right)
    up /= np.linalg.norm(up)
    neutral_basis = np.column_stack((right, up, np.cross(right, up)))
    right = data.exo[mask, 4] - data.exo[mask, 3]
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    up = data.exo[mask][:, [5, 6]].mean(1) - data.exo[mask][:, [11, 12]].mean(1)
    up -= right * np.sum(up * right, axis=-1, keepdims=True)
    up /= np.linalg.norm(up, axis=-1, keepdims=True)
    observed = np.stack((right, up, np.cross(right, up)), axis=-1)
    skull = observed @ neutral_basis.T @ neutral.transforms[0, names.index("Head") + 1, :3, :3].cpu().numpy()
    return Rotation.from_matrix(data.head_rotations[mask].transpose(0, 2, 1) @ skull).mean().as_matrix().astype(np.float32)


def main() -> None:
    ROOT.mkdir(exist_ok=True, parents=True)
    data = read_window(CATALOG, DATASET, SEGMENT, 0.0, 183.0, 1, "sapiens2")
    rtmw = read_window(CATALOG, DATASET, SEGMENT, 0.0, 183.0, 1, "rtmw")
    if data.times.shape != rtmw.times.shape or not np.allclose(data.times, rtmw.times, atol=1e-8):
        indices = nearest_indices(data.times, rtmw.times)
        valid = abs(data.times[indices] - rtmw.times) < 0.021
        data = replace(
            rtmw, exo=np.where(valid[:, None, None], data.exo[indices], np.nan), confidence=np.where(valid[:, None], data.confidence[indices], 0.0)
        )
    differences = []
    for key in [3, 4, 5, 6, 11, 12]:
        delta = data.exo[:, key] - rtmw.exo[:, key]
        valid = (data.confidence[:, key] >= 0.7) & (rtmw.confidence[:, key] >= 0.7) & np.isfinite(delta).all(1)
        differences.append(
            LandmarkDifference(key, (np.median(delta[valid], axis=0) * 1000).tolist(), residual(data.exo[valid, key], rtmw.exo[valid, key]))
        )
    options = FitOptions(head_mode="ears", shape_components=128, identity_prior=0.05, scale_prior=2.0, learn_bone_scales=True, bone_length_weight=2.0)
    shape_mask = (data.times < 25.0) & (np.arange(len(data.times)) % 8 == 0) & np.isfinite(data.exo[:, [3, 4, 5, 6, 11, 12]]).all((1, 2))
    shape = fit(subset(data, shape_mask), ASSETS, 600, 300, options)
    calibration = Calibration(shape.identity, shape.scale, np.zeros(3, dtype=np.float32), shape.bone_scales, np.eye(3, dtype=np.float32), "ears")
    (ROOT / "calibration-a.json").write_text(to_json(calibration))
    raw = Path("/mnt/nas/datasets/exoego-self-collected/main/cut/2025-12-03/d31bf1da-c23c-4e4b-b88b-eb5c5ad20e3f/episodes/episode-001/quest")
    orientations = read_orientations(raw, data.times, data.offset_ms)
    base = fit(data, ASSETS, 400, 80, FitOptions(head_mode="ears"), fixed=calibration)
    initial = fit(
        data,
        ASSETS,
        400,
        80,
        FitOptions(head_mode="ears", orientation_weight=30.0, hands_only=True, display_twists=True),
        fixed=calibration,
        initial=base,
        orientations=orientations,
    )
    initial = replace(initial, seconds=initial.seconds + base.seconds)
    del base
    # The ear line determines yaw/roll, but not pitch. Torso-up supplies a
    # documented neutral-pitch gauge for estimating the constant attachment;
    # the soft tracked rotation then supplies relative pitch motion.
    model = model_for(calibration)
    attachment = ear_axis_attachment(data, model, shape_mask)
    calibration_b = replace(calibration, head_attachment=attachment, head_anchor="ears-rotation")
    (ROOT / "calibration-b.json").write_text(to_json(calibration_b))
    b = fit(
        data,
        ASSETS,
        400,
        80,
        FitOptions(head_mode="ears-rotation", head_rotation_weight=0.3, orientation_weight=30.0, display_twists=True),
        fixed=calibration_b,
        initial=initial,
        orientations=orientations,
    )
    b = replace(b, seconds=b.seconds + initial.seconds)
    a = fit(
        data,
        ASSETS,
        400,
        80,
        FitOptions(head_mode="ears", orientation_weight=30.0, display_twists=True),
        fixed=calibration,
        initial=initial,
        orientations=orientations,
    )
    a = replace(a, seconds=a.seconds + initial.seconds)
    save(ROOT / "a", data, a, 39.862079571, 3)
    save(ROOT / "b", data, b, 39.862079571, 3)
    runs = []
    for label, observations, result in [
        (
            "RTMW legacy",
            rtmw,
            load_fit(
                Path("/tmp/soma-fit/hand-repair-final"),
                from_json(Calibration, Path("/tmp/soma-fit/ears/ears-offset-rotation-calibration.json").read_text()),
                rtmw,
            ),
        ),
        ("Sapiens A ears only", data, a),
        ("Sapiens B soft rotation", data, b),
    ]:
        report = evaluate(label, observations, result)
        runs.append(report)
        (ROOT / f"metrics-{len(runs)}.json").write_text(to_json(report))
        print(to_json(report), flush=True)
    (ROOT / "comparison.json").write_text(
        to_json(
            Comparison(
                differences,
                runs,
                shape.seconds,
                (
                    "Own-source 3D residuals are training residuals, not independent accuracy.",
                    "Cross-detector 2D is a consistency check; neither detector is ground truth.",
                    "A and B share Sapiens2 shape; B rotation weight 0.3 and no translation.",
                    "Head pitch is world elevation of anatomical forward, positive up.",
                    "B attachment uses observed ears and torso-up to fix its otherwise unobservable constant pitch gauge.",
                ),
            )
        )
    )
