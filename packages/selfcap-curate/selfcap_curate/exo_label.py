"""Bounded optical-landmark packets, Sol annotation, and fail-closed exo gating."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from serde.json import from_json, to_json

from selfcap_curate import catalog_io
from selfcap_curate.calib_layer import CalibSession
from selfcap_curate.calibration_objective import ObjectiveShares
from selfcap_curate.device_vp import read_vp_table
from selfcap_curate.exo_landmarks import Landmark, Landmarks
from selfcap_curate.exo_refine_catalog import Gate, evaluate, load_inputs


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Panel:
    view: int
    """Observer rig index."""
    video_time: float
    """Aligned source frame time in seconds."""
    targets: list[str]
    """Permitted optical target IDs."""
    bounds: tuple[int, int, int, int]
    """Source image crop x0,y0,x1,y1."""
    path: str
    """Annotated PNG path."""
    source: str
    """Unannotated source PNG for local centroid refinement."""
    predicted: tuple[float, float] | None = None
    """Initial projected phone centre, in source pixels."""
    frontality: float | None = None
    """Cosine between target optical axis and its direction to the observing camera."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Packet:
    segment: str
    """Explicit segment ID."""
    phones: tuple[str, ...]
    """Full device IDs in rig order."""
    panels: list[Panel]
    """Source-coordinate annotated crops."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SolLabels:
    labels: list[Landmark]
    """Visible and unambiguous optical marks only."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class GateThresholds:
    optical_min_drop_px: float = 2.0
    """Minimum absolute reduction for each optical target class."""
    optical_min_drop_fraction: float = .1
    """Minimum fractional reduction for each optical target class."""
    sapiens_max_regression_px: float = 5.0
    """Guard against large regression on the original fit's Sapiens samples."""
    deployed_label_tolerance_px: float = 2.0
    """Maximum deployed-label median increase over its corresponding fold score."""
    rtmw_max_regression_px: float = 1.5
    """Primary hand gate: maximum independent RTMW median increase."""
    max_nonconverged_cv_folds: int = 1
    """Allow at most one failed CV fold, subject to every other gate check."""
    require_full_fit_convergence: bool = True
    """Mandatory deployed-model convergence; this safety condition cannot be disabled."""

    def __post_init__(self) -> None:
        if not 0 <= self.max_nonconverged_cv_folds <= 1:
            raise ValueError('gate-v3 permits at most one non-converged CV fold')
        if not self.require_full_fit_convergence:
            raise ValueError('gate-v3 requires deployed full-fit convergence')


DEFAULT_THRESHOLDS = GateThresholds()
DEFAULT_OBJECTIVE = ObjectiveShares()
GATE_VERSION = 'gate-v3'
SOLVER_VERSION: str = 'exo-visible-folds-physical-targets-v4'
LabelFocalSource: TypeAlias = Literal["device-vp", "table"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LabelResult:
    segment: str
    """Explicit segment ID."""
    decision: str
    """accepted, rejected, timeout, or failed; all but accepted use the initial calibration."""
    reason: str
    """Human-readable fallback explanation."""
    labels: int
    """Number of retained annotations."""
    elapsed_s: float
    """Total wall time including packet, annotation and fitting."""
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    """Explicit acceptance thresholds applied to this result."""
    gate: Gate | None = None
    """Held-out results when scoring completed."""
    objective: ObjectiveShares | None = None
    """Count-invariant loss shares used in both folds and the deployed fit."""
    solver_version: str | None = None
    """Exo-only solver contract; older offset-enabled results cannot be resumed."""
    focal_source: str | None = None
    """Table means the initial calibration intrinsics stayed fixed; null for older results."""
    focals: dict[str, float] = field(default_factory=dict)
    """Fixed focal pixels per device."""
    focal_sources: dict[str, str] = field(default_factory=dict)
    """device-vp or initial-calibration per camera; neither is fitted."""
    rtmw_source: str | None = None
    """Explicit original S1 recording path, or null for the live catalog."""
    rtmw_sha256: str | None = None
    """Hash pinning the observation bytes used to fit and score."""
    cached_labels_sha256: str | None = None
    """Hash pinning the input labels in inference-free redecision mode."""
    gate_version: str | None = None
    """Acceptance-policy version; null identifies a legacy decision needing re-scoring."""


def accepts(gate: Gate, thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> bool:
    """Require clear optical improvement, body improvement, and the configured hand regression limits."""
    if gate.full_fit_converged is not True or len(gate.folds) != 2:
        return False
    if sum(not model.converged for model in gate.folds) > thresholds.max_nonconverged_cv_folds:
        return False
    for metric in ('headset', 'phones', 'sapiens_hands', 'rtmw_hands', 'body_transfer'):
        before, after = gate.baseline.get(metric), gate.candidate.get(metric)
        if (before is None or after is None or before.n <= 0 or after.n <= 0 or before.median is None or after.median is None
                or not np.isfinite([before.median, after.median]).all()):
            return False
        if metric in ('headset', 'phones'):
            if after.median > before.median - max(thresholds.optical_min_drop_px, thresholds.optical_min_drop_fraction * before.median):
                return False
        elif metric == 'body_transfer':
            if after.median >= before.median:
                return False
        elif after.median > before.median + (thresholds.sapiens_max_regression_px if metric == 'sapiens_hands' else thresholds.rtmw_max_regression_px):
            return False
    for key, deployed in gate.deployed_labels.items():
        heldout = gate.heldout_labels.get(key)
        if (deployed.median is not None and heldout is not None and heldout.median is not None
                and deployed.median > heldout.median + thresholds.deployed_label_tolerance_px):
            return False
    return True


def grid_crop(image: UInt8[ndarray, "h w 3"], u: float, v: float, size: int, title: str) -> tuple[UInt8[ndarray, "h2 w2 3"], tuple[int, int, int, int]]:
    """Upscale a crop 2x with ticks expressed in original image pixels."""
    h, w = image.shape[:2]
    size = min(size, h, w)
    x = int(np.clip(round(u - size / 2), 0, w - size))
    y = int(np.clip(round(v - size / 2), 0, h - size))
    crop = cv2.resize(image[y:y + size, x:x + size], (size * 2, size * 2), interpolation=cv2.INTER_CUBIC)
    layer = crop.copy()
    for a in range(((x + 19) // 20) * 20, x + size, 20):
        cv2.line(layer, ((a - x) * 2, 0), ((a - x) * 2, size * 2 - 1), (210, 220, 220), 1)
    for b in range(((y + 19) // 20) * 20, y + size, 20):
        cv2.line(layer, (0, (b - y) * 2), (size * 2 - 1, (b - y) * 2), (210, 220, 220), 1)
    crop = cv2.addWeighted(layer, .35, crop, .65, 0)
    cv2.drawMarker(crop, (round((u - x) * 2), round((v - y) * 2)), (170, 180, 160), cv2.MARKER_CROSS, 12, 1)
    result = cv2.copyMakeBorder(crop, 30, 30, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(result, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    for a in range(((x + 39) // 40) * 40, x + size, 40):
        cv2.putText(result, str(a), ((a - x) * 2, size * 2 + 50), 0, .4, (0, 0, 0), 1)
    for b in range(((y + 39) // 40) * 40, y + size, 40):
        cv2.putText(result, str(b), (2, (b - y) * 2 + 30), 0, .4, (180, 80, 20), 1)
    return result, (x, y, x + size, y + size)


def filter_labels(labels: list[Landmark], allowed: dict[tuple[int, float, str], tuple[int, int, int, int]]) -> list[Landmark]:
    """Drop hallucinated targets, weak marks, duplicates and out-of-crop coordinates."""
    result = {}
    for label in labels:
        matches = [key for key in allowed if key[0] == label.view and key[2] == label.target_id and abs(key[1] - label.video_time) < 1e-6]
        key = matches[0] if len(matches) == 1 else (label.view, label.video_time, label.target_id)
        bounds = allowed.get(key)
        if bounds is not None:
            label = replace(label, video_time=key[1])
        if bounds is None or label.confidence < .6:
            continue
        x0, y0, x1, y1 = bounds
        if x0 <= label.u < x1 and y0 <= label.v < y1 and (key not in result or label.confidence > result[key].confidence):
            result[key] = label
    return list(result.values())


def filter_phone_labels(labels: list[Landmark], panels: list[Panel]) -> list[Landmark]:
    """Reject back-facing identities and marks outside the initial phone search radius."""
    result = []
    for label in labels:
        panel = next((p for p in panels if label.target_id in p.targets and abs(p.video_time - label.video_time) < 1e-6), None)
        if panel is None or (panel.frontality is not None and panel.frontality < .3):
            continue
        if panel.predicted is not None and np.linalg.norm(np.array([label.u, label.v]) - panel.predicted) > 240.0:
            continue
        result.append(label)
    return result


def calibrated_rigs(state: CalibSession, phones: tuple[str, ...]) -> tuple[int, ...]:
    """Keep physical rig indices for cameras that have an initial pose."""
    return tuple(i for i, phone in enumerate(phones) if phone in state.exo.phones and state.exo.phones[phone].camera is not None)


def run_codex(args: list[str], prompt: str, env: dict[str, str], path: Path, log_path: Path, timeout_s: float) -> None:
    """Retain stderr and retry a transient model/service failure once."""
    for attempt in range(2):
        attempt_log = log_path if attempt == 0 else log_path.with_suffix('.retry.log')
        try:
            with attempt_log.open('w') as log:
                subprocess.run(args, env=env, input=prompt, text=True, stdout=log, stderr=log, timeout=timeout_s, check=True)
            return
        except subprocess.CalledProcessError as error:
            detail = attempt_log.read_text()[-6000:]
            transient = any(token in detail.lower() for token in ('at capacity', 'rate limit', 'http 429', 'http 503',
                                                                 '429 too many', 'temporarily unavailable', 'service unavailable',
                                                                 'connection reset', 'stream disconnected'))
            if attempt == 0 and transient:
                path.unlink(missing_ok=True)
                continue
            path.unlink(missing_ok=True)
            raise RuntimeError(f'Codex failed; stderr retained in {attempt_log}:\n{detail}') from error


def make_packet(segment: catalog_io.Segment, state: CalibSession, output: Path) -> Packet:
    """Decode a few facing frames per view from the base and initial camera solution."""

    from selfcap_curate.factory import optical_targets
    from selfcap_curate.hands_gpu import decode_frames

    output.mkdir(parents=True, exist_ok=True)
    source = catalog_io.read_catalog_hands(segment)
    sync = catalog_io.read_time_sync(segment.dataset, segment.id)
    if sync.offsets_ms is None or sync.offsets_ms.get('quest_pose') is None:
        raise ValueError('Quest pose clock is unavailable')
    head_times = source.head_times_s + sync.offsets_ms['quest_pose'] / 1000.0
    interp = Slerp(head_times, Rotation.from_quat(source.head_quaternions))
    phones = segment.layout.phones
    cameras = tuple(state.exo.phones[p].camera if p in state.exo.phones else None for p in phones)
    videos = tuple(catalog_io.read_phone_videos(segment).values())
    optical = optical_targets(state.quest)
    if not optical:
        raise ValueError('No resolved physical Quest RGB targets')
    optical_rotation = next(iter(optical.values()))[:3, :3]
    panels = []
    for view, (camera, video, phone) in enumerate(zip(cameras, videos, phones, strict=True)):
        if camera is None:
            continue
        offset = sync.offsets_ms.get(f'exo/{phone}')
        if offset is None:
            raise ValueError(f'Missing clock for {phone}')
        times = video.t_ns * 1e-9 + offset / 1000.0
        candidates = np.flatnonzero((times >= head_times[0]) & (times <= head_times[-1]))[::12]
        if not len(candidates):
            continue
        t = times[candidates]
        positions = np.stack([np.interp(t, head_times, source.head_positions[:, j]) for j in range(3)], axis=1)
        rotations = interp(t).as_matrix()
        towards = camera.world_T_cam[:3, 3] - positions
        towards /= np.linalg.norm(towards, axis=1)[:, None]
        facing = np.einsum('ni,ni->n', rotations @ optical_rotation[:, 2], towards)
        projected = camera.project(positions)
        valid = (projected[:, 2] > 0) & (projected[:, 0] > 40) & (projected[:, 0] < 1880) & (projected[:, 1] > 40) & (projected[:, 1] < 1040)
        picked = []
        for block in np.array_split(np.arange(len(candidates)), 4):
            block = block[valid[block]]
            if len(block):
                picked.append(int(candidates[block[np.argmax(facing[block])]]))
        if not picked:
            # Static phone crops are still needed when no facing headset sample exists.
            picked = [int(candidates[len(candidates) // 2])]
        indices = np.array(sorted(set(picked)), dtype=np.int64)
        fps = 1.0 / float(np.median(np.diff(times)))
        images = decode_frames(video, indices, fps).cpu().numpy()
        for number, (index, image) in enumerate(zip(indices, images, strict=True)):
            stamp = float(times[index])
            source_path = output / f'view{view}-frame{number}.png'
            cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rotation = interp([stamp]).as_matrix()[0]
            position = np.array([np.interp(stamp, head_times, source.head_positions[:, j]) for j in range(3)])
            points = np.stack([position + rotation @ matrix[:3, 3] for matrix in optical.values()])
            uv = camera.project(points)
            center = uv[:, :2].mean(0)
            towards_observer = rotation.T @ (camera.world_T_cam[:3, 3] - position)
            lateral = float(towards_observer[0] / np.linalg.norm(towards_observer))
            # The factory baseline points from left camera 0 to right camera 1.
            baseline = state.quest.rig_T_cam.get("1", np.eye(4))[:3, 3]
            near_right = lateral * (1.0 if baseline[0] >= 0.0 else -1.0) > 0.0
            targets = ['quest1' if near_right else 'quest0'] if abs(lateral) > .6 else ['quest0', 'quest1']
            targets = [target for target in targets if target in optical] if state.quest.mode == "healthy" else list(optical)
            crop, bounds = grid_crop(image, float(center[0]), float(center[1]), 260, f'view={view} time={stamp:.9f} {"/".join(targets)}')
            path = output / f'view{view}-head{number}.png'
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if targets:
                panels.append(Panel(view, stamp, targets, bounds, str(path), str(source_path)))
            if number == 0:
                for target, other in enumerate(cameras):
                    if target == view or other is None:
                        continue
                    pixel = camera.project(other.world_T_cam[:3, 3][None])[0]
                    if pixel[2] <= 0 or not (0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]):
                        continue
                    crop, bounds = grid_crop(image, float(pixel[0]), float(pixel[1]), 540, f'view={view} time={stamp:.9f} phone{target} {phones[target]}')
                    path = output / f'view{view}-phone{target}-{number}.png'
                    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    direction = camera.world_T_cam[:3, 3] - other.world_T_cam[:3, 3]
                    facing = float(other.world_T_cam[:3, 2] @ direction / np.linalg.norm(direction))
                    panels.append(Panel(view, stamp, [f'phone{target}'], bounds, str(path), str(source_path),
                                        (float(pixel[0]), float(pixel[1])), facing))
    packet = Packet(segment.id, phones, panels)
    (output / 'packet.json').write_text(to_json(packet))
    return packet


PROMPT = """Label optical camera apertures in these image crops. Return only the schema JSON.
Use ORIGINAL image pixel coordinates from the tick labels, not crop pixels. Grid lines are 20 original pixels apart; image enlarged 2x. Title gives view, exact video_time and allowed target IDs. Faint crosses are uncertain predictions, NOT labels.
Phones: mark the centre of the physical back-camera lens facing this observer, not the screen, tripod, body centre or reflection. Device IDs identify targets generically; scene placement varies. Skip partly hidden lenses, screens and backs without a clear visible aperture. Omit rather than guess when you cannot confidently identify the correct phone.
Headset: quest0 is the wearer's LEFT front RGB aperture, quest1 the wearer's RIGHT (when frontal, wearer left is image right). Quest 3S has two triangular three-aperture clusters on its white visor: RGB is the large dark outer aperture in each cluster; do not label the small neighboring tracking/IR/depth apertures or the centre of the triangle. Quest 3 has three vertical front pills: RGB cameras are on the two outer pills, not the central depth sensor. Skip occluded, side/back-facing, blurred, or ambiguous apertures. Never infer invisible points by symmetry. Confidence >=0.6 means clear visible lens, <=0.5 is uncertain and should be omitted. Only IDs listed in each panel are permitted. For an oblique view, the listed near-side lens ID was determined from calibrated head orientation: use that ID rather than guessing from image-left/image-right. One mark per visible target. No tools, no prose.
"""


def annotate(packet: Packet, output: Path, timeout_s: float = 180.0) -> Landmarks:
    """Run separate read-only headset/phone calls per view; cache each typed result."""
    from concurrent.futures import ThreadPoolExecutor

    schema = {'type': 'object', 'properties': {'labels': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'view': {'type': 'integer'}, 'video_time': {'type': 'number'}, 'target_id': {'type': 'string'},
        'u': {'type': 'number'}, 'v': {'type': 'number'}, 'confidence': {'type': 'number'}},
        'required': ['view', 'video_time', 'target_id', 'u', 'v', 'confidence'], 'additionalProperties': False}}},
        'required': ['labels'], 'additionalProperties': False}
    schema_path = output / 'schema.json'
    schema_path.write_text(json.dumps(schema))

    def label_panels(view: int, panels: list[Panel], kind: str, context: str = "") -> list[Landmark]:
        if not panels:
            return []
        path = output / f'sol-view{view}-{kind}.json'
        if not path.exists():
            args = ['codex', 'exec', '--ignore-user-config', '--strict-config', '--ephemeral', '--skip-git-repo-check',
                    '-C', str(output), '-s', 'read-only', '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="medium"',
                    '-c', 'service_tier="fast"', '--output-schema', str(schema_path), '-o', str(path)]
            for panel in panels:
                args.extend(['-i', panel.path])
            focus = ("These are PHONE-only crops. Inspect each crop for the actual back-camera lens, including small landscape phones. "
                     "The title supplies its identity; you do not need to recognize the device ID from appearance. "
                     "White stickers or a central clamp on the phone are not lenses. A plainly visible lens should be labelled; "
                     "do not omit it just because the hardware model is unknown. The prediction may be displaced by 150 pixels. "
                     "Keep overlapping crops consistent: never assign the same visible phone to two different target IDs. "
                     "Skip a target if its own lens is hidden or it cannot be distinguished from another phone. "
                     if kind.startswith('phone') else "These are HEADSET-only crops. ")
            prompt = focus + context + ' Each phone panel includes predicted position and frontality cosine. A cosine below 0.3 means a grazing/back-facing camera whose lens is normally not visible; do not substitute another nearby phone. ' + PROMPT + '\nTargets and crop bounds:\n' + to_json(Packet(packet.segment, packet.phones, panels))
            env = os.environ.copy()
            env.pop('CODEX_HOME', None)
            run_codex(args, prompt, env, path, output / f'sol-view{view}-{kind}.log', timeout_s)
        labels = from_json(SolLabels, path.read_text()).labels
        allowed = {(p.view, p.video_time, target): p.bounds for p in panels for target in p.targets}
        return filter_labels(labels, allowed)

    def label_view(view: int) -> list[Landmark]:
        panels = [panel for panel in packet.panels if panel.view == view]
        heads = [panel for panel in panels if panel.targets[0].startswith('quest')]
        phones = [panel for panel in panels if panel.targets[0].startswith('phone')]
        head_labels = label_panels(view, heads, 'heads')
        phone_labels = filter_phone_labels(label_panels(view, phones, 'phones'), phones)
        present = {label.target_id for label in phone_labels}
        missing = [p for p in phones if p.targets[0] not in present and (p.frontality is None or p.frontality >= .3)]
        if missing:
            context = "Already identified phone lenses (do not assign these same phones to a missing ID): " + to_json(SolLabels(phone_labels))
            phone_labels.extend(filter_phone_labels(label_panels(view, missing, 'phone-retry', context), missing))
        return head_labels + phone_labels

    views = sorted({p.view for p in packet.panels})
    with ThreadPoolExecutor(max_workers=4) as pool:
        labels = [label for group in pool.map(label_view, views) for label in group]
    refined = []
    for label in labels:
        panel = next(p for p in packet.panels if p.view == label.view and p.video_time == label.video_time and label.target_id in p.targets)
        image = cv2.imread(panel.source, cv2.IMREAD_GRAYSCALE)
        x, y = round(label.u), round(label.v)
        if image is not None and label.target_id.startswith('quest') and 14 <= x < image.shape[1] - 14 and 14 <= y < image.shape[0] - 14:
            patch = image[y - 13:y + 14, x - 13:x + 14]
            count, _, stats, centres = cv2.connectedComponentsWithStats((patch < 90).astype(np.uint8))
            options = [(float(np.linalg.norm(center - [13, 13])), center) for stat, center in zip(stats[1:count], centres[1:count], strict=True)
                       if 3 <= stat[4] <= 130 and stat[0] > 0 and stat[1] > 0 and stat[0] + stat[2] < 27 and stat[1] + stat[3] < 27]
            if options:
                distance, center = min(options, key=lambda item: item[0])
                if distance <= 5.0:
                    label = replace(label, u=float(x - 13 + center[0]), v=float(y - 13 + center[1]))
        refined.append(label)
    record = Landmarks(refined, 'gpt-6.1-sol medium fast', 'automatic per-view headset and phone passes; all projected phone crops; source-pixel grid; confidence >=0.6; <=5px dark centroid snap; geometry-filtered phones with automatic missing-target retry', datetime.now(UTC).date().isoformat())
    (output / 'landmarks.json').write_text(to_json(record))
    return record


def run_session(segment: catalog_io.Segment, state: CalibSession, output: Path, *, thresholds: GateThresholds = DEFAULT_THRESHOLDS,
                focal_source: LabelFocalSource = "device-vp", objective: ObjectiveShares = DEFAULT_OBJECTIVE,
                cached_labels: Path | None = None, rtmw_rrd: Path | None = None) -> LabelResult:
    """Build, label and gate one session; runner must bound this call or use the CLI."""
    start = time.monotonic()
    if cached_labels is not None and rtmw_rrd is None:
        raise ValueError('cached_labels requires an original S1 rtmw_rrd')
    rtmw_hash = None
    label_hash = None
    if rtmw_rrd is not None:
        with rtmw_rrd.open('rb') as source:
            rtmw_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    if cached_labels is not None:
        label_hash = hashlib.sha256(cached_labels.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    active = calibrated_rigs(state, segment.layout.phones)
    if len(active) < 2:
        for name in ('calib.json', 'refinement.json'):
            (output / name).unlink(missing_ok=True)
        result = LabelResult(segment.id, 'skipped', f'Fewer than two initial exo cameras ({len(active)} calibrated)', 0,
                             time.monotonic() - start, thresholds, objective=objective, solver_version=SOLVER_VERSION,
                             rtmw_source=str(rtmw_rrd.resolve()) if rtmw_rrd else None, rtmw_sha256=rtmw_hash, cached_labels_sha256=label_hash,
                             gate_version=GATE_VERSION)
        (output / 'result.json').write_text(to_json(result))
        return result
    table = read_vp_table()
    fixed_focals = {}
    focal_sources = {}
    initial = {}
    for rig in active:
        phone = segment.layout.phones[rig]
        camera = state.exo.phones[phone].camera
        if camera is None:
            raise ValueError(f'Missing initial camera for {phone}')
        initial[rig] = camera
        measured = table.devices.get(phone) if focal_source == 'device-vp' else None
        fixed_focals[phone] = measured.focal_px if measured is not None else float(camera.parameters[6])
        focal_sources[phone] = 'device-vp' if measured is not None else 'initial-calibration'
    result_path = output / 'result.json'
    if result_path.exists():
        previous = from_json(LabelResult, result_path.read_text())
        if previous.segment != segment.id:
            raise ValueError('Output directory belongs to another segment')
        if (previous.decision in ('accepted', 'rejected') and previous.thresholds == thresholds and previous.focal_source == focal_source
                and previous.focals == fixed_focals and previous.focal_sources == focal_sources and previous.solver_version == SOLVER_VERSION
                and previous.objective == objective and previous.rtmw_sha256 == rtmw_hash and previous.cached_labels_sha256 == label_hash
                and previous.gate_version == GATE_VERSION):
            return previous
    for name in ('calib.json', 'refinement.json'):
        (output / name).unlink(missing_ok=True)
    packet_path = output / 'packet.json'
    if cached_labels is not None:
        packet = Packet(segment.id, segment.layout.phones, [])
        record = from_json(Landmarks, cached_labels.read_text())
        (output / 'landmarks.json').write_text(to_json(record))
    else:
        if packet_path.exists():
            packet = from_json(Packet, packet_path.read_text())
        else:
            segment = replace(segment, hands=catalog_io.read_catalog_hands(segment))  # make_packet and load_inputs share one query
            packet = make_packet(segment, state, output)
        packet = replace(packet, panels=[panel for panel in packet.panels if panel.view in active
                                        and all(not target.startswith('phone') or int(target[5:]) in active for target in panel.targets)])
        record_path = output / 'landmarks.json'
        record = from_json(Landmarks, record_path.read_text()) if record_path.exists() else annotate(packet, output)
    record = replace(record, labels=[label for label in record.labels if label.view in active
                                    and (label.phone is None or label.phone in active)])
    if len(record.labels) < 12 or len({label.view for label in record.labels}) < len(active):
        result = LabelResult(segment.id, 'rejected', 'insufficient visible landmark coverage', len(record.labels), time.monotonic() - start, thresholds)
    else:
        inputs = load_inputs(segment, cameras=tuple(initial[rig] for rig in active), quest=state.quest, landmarks=record, camera_rigs=active, rtmw_rrd=rtmw_rrd)
        record = inputs.problem.landmarks
        truth_bytes = tuple((track.times_s.tobytes(), track.joints.tobytes()) for track in inputs.tracks.values())
        model, gate = evaluate(inputs, focal_source="fixed", fixed_focals=tuple(fixed_focals[packet.phones[rig]] for rig in active),
                               static_phones=True, body_polish=True, objective=objective)
        model = replace(model, camera_rigs=active)
        gate = replace(gate, folds=tuple(replace(fit, camera_rigs=active) for fit in gate.folds))
        assert truth_bytes == tuple((track.times_s.tobytes(), track.joints.tobytes()) for track in inputs.tracks.values()), "Exo fitting must preserve Quest hands"
        (output / 'candidate.json').write_text(to_json(model))
        accepted = accepts(gate, thresholds) and model.converged
        gate = replace(gate, passes=accepted, policy=gate.policy + "; gate-v3: mandatory full-fit convergence; at most one failed CV fold; thresholds in result.json")
        assert not np.any(model.head_translation_m), 'Exo-only refinement must not move the Quest world'
        if accepted:
            (output / 'refinement.json').write_text(to_json(model))
            updated = replace(state, exo_refinements={**state.exo_refinements, segment.id: model})
            assert updated.oak is state.oak and updated.quest is state.quest and updated.stereo is state.stereo, 'Only exo cameras may change'
            (output / 'calib.json').write_text(to_json(updated))
        result = LabelResult(segment.id, 'accepted' if accepted else 'rejected', 'held-out optical/body/hand gate', len(record.labels), time.monotonic() - start, thresholds, gate, objective, SOLVER_VERSION, focal_source,
                             fixed_focals, focal_sources)
    temporary = result_path.with_suffix('.tmp')
    result = replace(result, rtmw_source=str(rtmw_rrd.resolve()) if rtmw_rrd else None,
                     rtmw_sha256=rtmw_hash, cached_labels_sha256=label_hash, gate_version=GATE_VERSION)
    temporary.write_text(to_json(result))
    temporary.replace(result_path)
    return result


def bounded_command(args: list[str], output: Path, segment: str, timeout_s: float, thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> LabelResult:
    """Run an isolated worker with a hard deadline; kill only this worker's process group."""
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (output / 'worker.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'selfcap_curate.apis.exo_label', *args, '--worker'], stdout=log, stderr=log, start_new_session=True)
        try:
            code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -1
    path = output / 'result.json'
    if code == 0 and path.exists():
        return from_json(LabelResult, path.read_text())
    result = LabelResult(segment, 'timeout' if code == -1 else 'failed', 'initial calibration retained; see worker.log', 0,
                         time.monotonic() - start, thresholds, gate_version=GATE_VERSION)
    path.write_text(to_json(result))
    return result


def label_session(segment: str, calib_record: Path, output: Path, *, dataset: str = "selfcap-post-dev",
                  catalog_url: str = "rerun+http://dgx-spark:51235", timeout_s: float = 360.0,
                  thresholds: GateThresholds = DEFAULT_THRESHOLDS, focal_source: LabelFocalSource = "device-vp", objective: ObjectiveShares = DEFAULT_OBJECTIVE,
                  cached_labels: Path | None = None, rtmw_rrd: Path | None = None) -> LabelResult:
    """Runner API: isolate one session, resume its files, and enforce a hard deadline."""
    args = ["--segment", segment, "--calib-record", str(calib_record.resolve()), "--output-dir", str(output.resolve()),
            "--dataset", dataset, "--catalog-url", catalog_url, "--focal-source", focal_source]
    if cached_labels is not None:
        if rtmw_rrd is None:
            raise ValueError('cached_labels requires an original S1 rtmw_rrd')
        args.extend(['--cached-labels', str(cached_labels.resolve())])
    if rtmw_rrd is not None:
        args.extend(['--rtmw-rrd', str(rtmw_rrd.resolve())])
    args.extend(["--thresholds.optical-min-drop-px", str(thresholds.optical_min_drop_px),
                 "--thresholds.optical-min-drop-fraction", str(thresholds.optical_min_drop_fraction),
                 "--thresholds.sapiens-max-regression-px", str(thresholds.sapiens_max_regression_px),
                 "--thresholds.rtmw-max-regression-px", str(thresholds.rtmw_max_regression_px),
                 "--thresholds.max-nonconverged-cv-folds", str(thresholds.max_nonconverged_cv_folds),
                 "--thresholds.deployed-label-tolerance-px", str(thresholds.deployed_label_tolerance_px),
                 "--objective.headset", str(objective.headset), "--objective.phones", str(objective.phones),
                 "--objective.hands", str(objective.hands), "--objective.body", str(objective.body)])
    return bounded_command(args, output, segment, timeout_s, thresholds)
