"""Catalog inputs and held-out transfer checks for the exo candidate solver."""

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
from dataforge.schema import coco133_uv_path
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from serde.json import from_json

from selfcap_curate import catalog_io
from selfcap_curate.calibration_objective import ObjectiveShares, label_weights
from selfcap_curate.exo_calibration import BODY, BodyObservation, Camera, bundle, triangulate
from selfcap_curate.exo_landmarks import Landmarks, read_landmarks
from selfcap_curate.exo_refine import FocalSource, Refinement, RefinementProblem, predict_labels, select_labels, solve
from selfcap_curate.factory import QuestBlock, optical_targets, read_quest_block
from selfcap_curate.keypoints import read_cameras
from selfcap_curate.quest_fit import Correspondences, load_quest_hands, pairs
from selfcap_curate.records import HandTrack, PhoneDetections, Side, TimeSync


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FocalRecord:
    focals: dict[str, float]
    """Fixed focal pixels keyed by full phone identifier."""
    method: str
    """Measurement procedure."""

    def __post_init__(self) -> None:
        if not self.focals or not all(np.isfinite(f) and f > 0.0 for f in self.focals.values()):
            raise ValueError("fixed focal record requires finite positive values")


@dataclass(frozen=True, slots=True)
class Inputs:
    problem: RefinementProblem
    """All-data problem before fold selection."""
    phones: tuple[str, ...]
    """Full phone identifiers in rig order."""
    detections: tuple[PhoneDetections, ...]
    """Sapiens2 observations on capture_time."""
    tracks: dict[Side, HandTrack]
    """Filtered raw Quest hands."""
    sync: TimeSync
    """Frozen alignment record."""
    body: tuple[PhoneDetections, ...]
    """Independent RTMW observations on video_time; empty when absent."""
    phone_offsets_s: tuple[float, ...]
    """Validated offsets from phone capture to video time."""
    phone_to_pose_s: tuple[float, ...]
    """Validated offsets from phone capture to raw Quest time."""
    quest_offset_s: float
    """Validated offset from raw Quest to video time."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Score:
    n: int
    """Number of finite held-out residuals."""
    median: float | None
    """Median pixel distance; null when absent."""
    p90: float | None
    """90th percentile pixel distance; null when absent."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Gate:
    baseline: dict[str, Score]
    """Current calibration, evaluated on the same holdouts."""
    candidate: dict[str, Score]
    """Pooled predictions from two independently fitted folds."""
    folds: tuple[Refinement, ...]
    """Models retained for exact replay of the held-out predictions."""
    passes: bool
    """Every scored metric has no median regression; all five metrics must exist."""
    heldout_labels: dict[str, Score] = field(default_factory=dict)
    """Per-view label scores pooled from the full held-out procedures."""
    deployed_labels: dict[str, Score] = field(default_factory=dict)
    """Final all-frame model label residuals, including each view."""
    policy: str = "alternating label frames; alternating 10s blocks with 200ms guard; RTMW hands stride 3; body stride 6, leave-one-view-out"
    """Deterministic holdout and independent transfer definitions."""
    full_fit_converged: bool | None = None
    """Convergence of the deployed all-frame model; null is unknown and cannot pass."""


def read_rtmw_recording(path: Path, segment: str, rigs: tuple[int, ...]) -> tuple[PhoneDetections, ...]:
    """Read original S1 observations without querying a replaced catalog layer."""
    reader = RrdReader(path)
    store = next((entry for entry in reader.recordings() if entry.recording_id == segment), None)
    if store is None:
        raise ValueError(f'{path}: recording does not match {segment}')
    entities = {coco133_uv_path(rig, 0, pred='rtmw'): rig for rig in rigs}
    batches: dict[int, list[pa.RecordBatch]] = {rig: [] for rig in rigs}
    for chunk in reader.stream(store=store):
        entity = str(chunk.entity_path)
        if not chunk.is_static and entity in entities:
            batch = chunk.to_record_batch()
            if 'Points2D:positions' in batch.schema.names:
                batches[entities[entity]].append(batch)
    result = []
    for rig in rigs:
        if not batches[rig]:
            raise ValueError(f'{path}: missing original RTMW observations for rig_{rig:02d}')
        table = pa.Table.from_batches(batches[rig])
        times = catalog_io.seconds(table['video_time'])
        order = np.argsort(times)
        result.append(PhoneDetections(times[order], catalog_io.floats(table['Points2D:positions'], -1, 133, 2)[order],
                                      catalog_io.floats(table['simplecv.KeypointConfidence2D:confidences'], -1, 133)[order]))
    return tuple(result)


def load_inputs(segment: catalog_io.Segment, *, cameras: tuple[Camera, ...] | None = None,
                quest: QuestBlock | None = None, landmarks: Landmarks | None = None,
                camera_rigs: tuple[int, ...] | None = None, rtmw_rrd: Path | None = None) -> Inputs:
    """Load inputs from one catalog segment, never from the raw capture files."""
    record = landmarks if landmarks is not None else read_landmarks(segment.dataset, segment.id)
    if record is None:
        raise ValueError(f"{segment.id}: no exo_landmarks layer")
    rigs = camera_rigs if camera_rigs is not None else tuple(range(len(segment.layout.phones)))
    local = {rig: index for index, rig in enumerate(rigs)}
    selected = []
    for label in record.labels:
        target = label.phone
        if label.view in local and (target is None or target in local):
            selected.append(replace(label, view=local[label.view], target_id=f'phone{local[target]}' if target is not None else label.target_id))
    record = replace(record, labels=selected)
    sync = catalog_io.read_time_sync(segment.dataset, segment.id)
    if sync.offsets_ms is None or sync.offsets_ms.get("quest_pose") is None:
        raise ValueError("exo refinement requires aligned Quest and phone clocks")
    quest_offset = sync.offsets_ms["quest_pose"]
    assert quest_offset is not None
    source = catalog_io.read_catalog_hands(segment)
    tracks = load_quest_hands(source)
    phones = tuple(segment.layout.phones[rig] for rig in rigs)
    phone_offsets: list[float] = []
    phone_to_pose: list[float] = []
    for phone in phones:
        offset = sync.offsets_ms.get(f"exo/{phone}")
        tau = sync.exo_fit[phone].tau_phone_to_pose_ms
        if offset is None or tau is None:
            raise ValueError(f"exo refinement needs a valid clock for {phone}")
        phone_offsets.append(offset / 1000.0)
        phone_to_pose.append(tau / 1000.0)
    detections_by_phone = catalog_io.read_phone_detections(segment)
    detections = tuple(detections_by_phone[phone] for phone in phones)
    if cameras is None:
        current = read_cameras(segment.dataset, segment.id, phones)
        result = []
        for phone in phones:
            c = current[phone]
            inverse = np.linalg.inv(c.world_T_cam)
            result.append(Camera(np.r_[Rotation.from_matrix(inverse[:3, :3]).as_rotvec(), inverse[:3, 3],
                                       c.K[0, 0], c.K[0, 2], c.K[1, 2], *c.radial]))
        cameras = tuple(result)
    factory = quest if quest is not None else read_quest_block(segment)
    targets = optical_targets(factory)
    record = replace(record, labels=[label for label in record.labels if not label.target_id.startswith("quest") or label.target_id in targets])
    times = source.head_times_s + quest_offset / 1000.0
    stamps = np.array([label.video_time for label in record.labels])
    if np.any((stamps < times[0]) | (stamps > times[-1])):
        raise ValueError("landmarks fall outside the Quest head track")
    rotations = Slerp(times, Rotation.from_quat(source.head_quaternions))(stamps).as_matrix()
    positions = np.stack([np.interp(stamps, times, source.head_positions[:, j]) for j in range(3)], axis=1)
    for i, label in enumerate(record.labels):
        if label.target_id.startswith("quest"):
            positions[i] += rotations[i] @ targets[label.target_id][:3, 3]
    hands = tuple(pairs(detection, tracks, tau) for detection, tau in zip(detections, phone_to_pose, strict=True))
    if rtmw_rrd is not None:
        body = read_rtmw_recording(rtmw_rrd, segment.id, rigs)
        return Inputs(RefinementProblem(cameras, hands, record, positions, rotations), phones, detections, tracks, sync,
                      body, tuple(phone_offsets), tuple(phone_to_pose), quest_offset / 1000.0)
    entities = [coco133_uv_path(rig, 0, pred="rtmw") for rig in rigs]
    table = segment.dataset.filter_segments(segment.id).filter_contents(entities).reader(index="video_time").to_arrow_table()
    body = []
    for entity in entities:
        column = f"{entity}:Points2D:positions"
        if column not in table.column_names:
            break
        rows = table.filter(table[column].is_valid())
        body.append(PhoneDetections(catalog_io.seconds(rows["video_time"]), catalog_io.floats(rows[column], -1, 133, 2),
                                    catalog_io.floats(rows[f"{entity}:simplecv.KeypointConfidence2D:confidences"], -1, 133)))
    return Inputs(RefinementProblem(cameras, hands, record, positions, rotations), phones, detections, tracks, sync,
                  tuple(body) if len(body) == len(phones) else (), tuple(phone_offsets), tuple(phone_to_pose), quest_offset / 1000.0)


def balanced_folds(visible: Bool[ndarray, "n v"]) -> Int64[ndarray, "n"]:
    """Alternate observations and repair coverage so each camera appears in both folds."""
    if np.any(visible.sum(axis=0) < 2):
        raise ValueError("Cross-validation needs at least two usable observations per camera")
    folds = np.arange(len(visible), dtype=np.int64) % 2
    for _ in range(2 * visible.shape[1] + 1):
        counts = np.stack([visible[folds == fold].sum(axis=0) for fold in (0, 1)])
        missing = np.argwhere(counts == 0)
        if not len(missing):
            return folds
        destination, view = missing[0]
        candidates = np.flatnonzero(visible[:, view] & (folds != destination))
        for row in candidates:
            if np.all(counts[1 - destination, visible[row]] > 1):
                folds[row] = destination
                break
        else:
            raise ValueError("Cannot split observations with coverage in both folds")
    raise ValueError("Cannot balance cross-validation coverage")


def hand_frame_folds(inputs: Inputs, view: int) -> Int64[ndarray, "n"]:
    """Assign each usable hand frame to a fold, independently for each camera."""
    detection = inputs.detections[view]
    usable = np.array([len(pairs(PhoneDetections(detection.times_s[i:i+1], detection.xy[i:i+1], detection.scores[i:i+1]),
                                inputs.tracks, inputs.phone_to_pose_s[view]).world) > 0 for i in range(len(detection.times_s))], dtype=bool)
    folds = np.full(len(usable), -1, dtype=np.int64)
    folds[usable] = balanced_folds(np.ones((int(usable.sum()), 1), dtype=bool))
    return folds


def heldout_hands(inputs: Inputs, fold: int, *, train: bool) -> tuple[Correspondences, ...]:
    """Hold out alternating usable hand frames per camera; never train on an empty view."""
    output = []
    for view, (detection, tau) in enumerate(zip(inputs.detections, inputs.phone_to_pose_s, strict=True)):
        assignment = hand_frame_folds(inputs, view)
        select = assignment == (fold if train else 1 - fold)
        output.append(pairs(PhoneDetections(detection.times_s[select], detection.xy[select], detection.scores[select]), inputs.tracks,
                            tau))
    return tuple(output)


def body_frame_folds(inputs: Inputs) -> tuple[list[dict[str, BodyObservation]], Float64[ndarray, "n"], Int64[ndarray, "n"]]:
    """Build shared body observations and preserve each camera in both training folds."""
    body = []
    selected_times = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    # Match the usual sparse calibration frame count, with explicit timestamp matching.
    target_times = np.unique(np.concatenate([d.times_s + offset for d, offset in zip(inputs.detections, inputs.phone_offsets_s, strict=True)]))
    for time in target_times:
        views = {}
        rows = []
        for i, detection in enumerate(inputs.body):
            if not len(detection.times_s):
                continue
            nearest = int(np.argmin(abs(detection.times_s - time)))
            if abs(detection.times_s[nearest] - time) > .025:
                continue
            scores = detection.scores[nearest, list(BODY)].copy()
            scores[:3] = 0.0
            views[str(i)] = BodyObservation(detection.xy[nearest, list(BODY)], scores)
            rows.append((i, nearest))
        key = tuple(rows)
        if len(views) >= 2 and key not in seen:
            seen.add(key)
            body.append(views)
            selected_times.append(float(time))
    visibility = np.array([[str(view) in row and np.count_nonzero((row[str(view)].scores >= .7) & np.isfinite(row[str(view)].xy).all(axis=1)) >= 4
                            for view in range(len(inputs.problem.cameras))] for row in body], dtype=bool).reshape(-1, len(inputs.problem.cameras))
    body_assignment = balanced_folds(visibility)
    return body, np.array(selected_times), body_assignment


def residuals(inputs: Inputs, model: Refinement, fold: int) -> dict[str, list[float]]:
    """Evaluate unseen labels/hands and independent dense RTMW transfer."""
    cameras = model.cameras
    test = select_labels(inputs.problem, 1 - fold)
    pixels = predict_labels(test, cameras, model.head_translation_m)
    distances = np.linalg.norm(pixels - np.array([[label.u, label.v] for label in test.landmarks.labels]), axis=1)
    output = {"headset": [], "phones": [], "sapiens_hands": [], "rtmw_hands": [], "body_transfer": []}
    for label, distance in zip(test.landmarks.labels, distances, strict=True):
        output["headset" if label.target_id.startswith("quest") else "phones"].append(float(distance))
    for camera, hand in zip(cameras, heldout_hands(inputs, fold, train=False), strict=True):
        output["sapiens_hands"].extend(camera.residuals(hand).tolist())
    for view, (camera, detection) in enumerate(zip(cameras, inputs.body, strict=False)):
        times = detection.times_s
        assignment = hand_frame_folds(inputs, view)
        source = inputs.detections[view].times_s + inputs.phone_offsets_s[view]
        usable = assignment >= 0
        nearest = np.abs(times[:, None] - source[usable]).argmin(axis=1)
        select = assignment[usable][nearest] != fold
        indices = np.flatnonzero(select)[::3]
        dense = pairs(PhoneDetections(times[indices], detection.xy[indices], detection.scores[indices]), inputs.tracks,
                      -inputs.quest_offset_s + inputs.sync.confidence.hand_latency_ms / 1000.0)
        output["rtmw_hands"].extend(camera.residuals(dense).tolist())
    if inputs.body:
        _, body_times, body_assignments = body_frame_folds(inputs)
        for time in inputs.body[0].times_s[::6]:
            if body_assignments[int(np.argmin(abs(body_times - time)))] == fold:
                continue
            views = {}
            for v, detection in enumerate(inputs.body):
                j = int(np.argmin(abs(detection.times_s - time)))
                if abs(detection.times_s[j] - time) < 0.025:
                    views[str(v)] = BodyObservation(detection.xy[j, list(BODY)], detection.scores[j, list(BODY)])
            for v, camera in enumerate(cameras):
                others = {key: observation for key, observation in views.items() if int(key) != v}
                if str(v) not in views or len(others) < 2:
                    continue
                points = triangulate({str(j): c for j, c in enumerate(cameras)}, others)
                observation = views[str(v)]
                distance = np.linalg.norm(camera.project(points)[:, :2] - observation.xy, axis=1)
                valid = np.isfinite(points).all(1) & np.isfinite(observation.xy).all(1) & (observation.scores >= 0.7)
                valid[:3] = False
                output["body_transfer"].extend(distance[valid].tolist())
    return output


def polish(inputs: Inputs, problem: RefinementProblem, model: Refinement, fold: int | None, objective: ObjectiveShares | None = None) -> Refinement:
    """Polish poses/body points from the optical solution, retaining its fixed lens anchors.

    Training body observations retain per-camera coverage in each fold.
    Quest points and wearable transforms remain unchanged.
    """
    cameras = {str(i): camera for i, camera in enumerate(model.cameras)}
    hands = {str(i): hand for i, hand in enumerate(problem.hands)}
    body, _, assignments = body_frame_folds(inputs)
    if fold is not None:
        body = [row for row, assigned in zip(body, assignments, strict=True) if assigned == fold]
    targets = np.array([[label.u, label.v] for label in problem.landmarks.labels])
    keys = [(label.view, label.target_id) for label in problem.landmarks.labels]
    repeats = np.array([max(1, round(20 / keys.count(key))) for key in keys])
    weights = 2.0 * np.sqrt([label.confidence for label in problem.landmarks.labels])[:, None]

    anchor_weights = None
    if objective is not None:
        repeats[:] = 1
        weights[:] = 1.0
        anchor_weights = np.repeat(label_weights(problem.landmarks.labels, objective), 2) / 2.0

    def anchors(candidate: dict[str, Camera]) -> Float64[ndarray, "a"]:
        ordered = tuple(candidate[str(i)] for i in range(len(model.cameras)))
        errors = (predict_labels(problem, ordered, model.head_translation_m) - targets) * weights
        return np.repeat(errors, repeats, axis=0).ravel()

    result = bundle(hands, body, cameras, max_nfev=400 if objective is not None else 120, anchors=anchors, objective=objective, anchor_weights=anchor_weights)
    polished = tuple(result.cameras[str(i)] for i in range(len(model.cameras)))
    assert all(before.parameters[6:].tobytes() == after.parameters[6:].tobytes()
               for before, after in zip(model.cameras, polished, strict=True)), "Body polish must preserve every intrinsic"
    return replace(model, cameras=polished, method=model.method + "; fixed-intrinsic hands/body BA with optical anchors",
                   converged=model.converged and result.converged)


def evaluate(inputs: Inputs, *, focal_source: FocalSource, fixed_focals: tuple[float, ...] | None = None,
             focal_record: str | None = None, static_phones: bool = False,
             body_polish: bool = False, objective: ObjectiveShares | None = None) -> tuple[Refinement, Gate]:
    """Fit both holdout folds, report transfer, then fit the deployable all-data model."""
    baseline = Refinement(inputs.problem.cameras, np.zeros(3), inputs.problem.landmarks.digest, "table")
    fit_problem = inputs.problem
    pools: dict[str, dict[str, list[float]]] = {"baseline": {}, "candidate": {}}
    models = []
    label_pools = {}
    for fold in (0, 1):
        problem = replace(select_labels(fit_problem, fold, static_phones=static_phones), hands=heldout_hands(inputs, fold, train=True))
        model = replace(solve(problem, focal_source=focal_source, fixed_focals=fixed_focals, focal_record=focal_record, objective=objective),
                        label_layer_hash=inputs.problem.landmarks.digest)
        if body_polish:
            model = polish(inputs, problem, model, fold, objective)
        models.append(model)
        test_labels = select_labels(fit_problem, 1 - fold)
        predicted = predict_labels(test_labels, model.cameras, model.head_translation_m)
        distances = np.linalg.norm(predicted - np.array([[label.u, label.v] for label in test_labels.landmarks.labels]), axis=1)
        for label, distance in zip(test_labels.landmarks.labels, distances, strict=True):
            kind = 'quest' if label.target_id.startswith('quest') else 'phone'
            for key in (f'{kind}/all', f'{kind}/view_{label.view}'):
                label_pools.setdefault(key, []).append(float(distance))
        for name, fit in (("baseline", baseline), ("candidate", model)):
            for metric, values in residuals(inputs, fit, fold).items():
                pools[name].setdefault(metric, []).extend(values)
    scores: dict[str, dict[str, Score]] = {}
    for name, metrics in pools.items():
        scores[name] = {}
        for metric, values in metrics.items():
            finite = np.asarray(values)[np.isfinite(values)]
            scores[name][metric] = Score(len(finite), float(np.median(finite)) if len(finite) else None,
                                        float(np.percentile(finite, 90)) if len(finite) else None)
    passes = True
    for metric, candidate in scores["candidate"].items():
        before, after = scores["baseline"][metric].median, candidate.median
        if before is None or after is None or after > before:
            passes = False
    final = solve(fit_problem, focal_source=focal_source, fixed_focals=fixed_focals, focal_record=focal_record, objective=objective)
    if body_polish:
        final = polish(inputs, fit_problem, final, None, objective)
    gate = Gate(scores["baseline"], scores["candidate"], tuple(models), passes and final.converged and all(model.converged for model in models),
                policy="alternating usable hand frames per camera; dense hand holdout follows nearest sparse frame; visibility-balanced body frames; held-out headset frames",
                full_fit_converged=final.converged)
    predicted = predict_labels(fit_problem, final.cameras, final.head_translation_m)
    distances = np.linalg.norm(predicted - np.array([[label.u, label.v] for label in fit_problem.landmarks.labels]), axis=1)
    deployed = {}
    for kind in ('quest', 'phone'):
        for view in (-1, *range(len(final.cameras))):
            values = np.array([distance for label, distance in zip(fit_problem.landmarks.labels, distances, strict=True)
                               if label.target_id.startswith(kind) and (view == -1 or label.view == view)])
            deployed[f'{kind}/all' if view == -1 else f'{kind}/view_{view}'] = Score(len(values), float(np.median(values)) if len(values) else None,
                                                                               float(np.percentile(values, 90)) if len(values) else None)
    heldout = {key: Score(len(values), float(np.median(values)), float(np.percentile(values, 90))) for key, values in label_pools.items()}
    gate = replace(gate, deployed_labels=deployed, heldout_labels=heldout)
    if objective is not None:
        gate = replace(gate, policy=gate.policy + "; fixed total headset/phone/hand/body shares; raw-pixel robust loss; final-model label sanity checked")
    if static_phones:
        gate = replace(gate, policy=gate.policy + "; static phone anchors shared in both training folds; phone score is in-sample consistency, not independent validation")
    return final, gate


def read_focals(path: Path | None, phones: tuple[str, ...]) -> tuple[tuple[float, ...] | None, str | None]:
    """Read a typed fixed-focal record in full device order and retain its hash."""
    if path is None:
        return None, None
    text = path.read_text()
    record = from_json(FocalRecord, text)
    return tuple(record.focals[phone] for phone in phones), hashlib.sha256(text.encode()).hexdigest()
