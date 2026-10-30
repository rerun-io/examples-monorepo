"""Bounded optical-landmark packets, Sol annotation, and fail-closed exo gating."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp
from serde import serde
from serde.json import from_json, to_json

from selfcap_curate import catalog_io
from selfcap_curate.calib_layer import CalibSession
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
    sapiens_max_regression_px: float = 1.0
    """Maximum held-out Sapiens hand median increase."""
    rtmw_max_regression_px: float = 1.5
    """Maximum independent RTMW hand median increase."""


DEFAULT_THRESHOLDS = GateThresholds()


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


def accepts(gate: Gate, thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> bool:
    """Require clear optical improvement, body improvement, and the configured hand regression limits."""
    if any(not model.converged for model in gate.folds):
        return False
    for metric in ('headset', 'phones', 'sapiens_hands', 'rtmw_hands', 'body_transfer'):
        before, after = gate.baseline.get(metric), gate.candidate.get(metric)
        if before is None or after is None or before.n == 0 or after.n == 0 or before.median is None or after.median is None:
            return False
        if metric in ('headset', 'phones'):
            if after.median > before.median - max(thresholds.optical_min_drop_px, thresholds.optical_min_drop_fraction * before.median):
                return False
        elif metric == 'body_transfer':
            if after.median >= before.median:
                return False
        elif after.median > before.median + (thresholds.sapiens_max_regression_px if metric == 'sapiens_hands' else thresholds.rtmw_max_regression_px):
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


def make_packet(segment: catalog_io.Segment, state: CalibSession, output: Path) -> Packet:
    """Decode a few facing frames per view from the base and initial camera solution."""
    from dataforge.schema import video_path
    from simplecv.catalog_video import read_catalog_videos

    from selfcap_curate.hands_gpu import decode_frames

    output.mkdir(parents=True, exist_ok=True)
    source = catalog_io.read_catalog_hands(segment)
    sync = catalog_io.read_time_sync(segment.dataset, segment.id)
    if sync.offsets_ms is None or sync.offsets_ms.get('quest_pose') is None:
        raise ValueError('Quest pose clock is unavailable')
    head_times = source.head_times_s + sync.offsets_ms['quest_pose'] / 1000.0
    interp = Slerp(head_times, Rotation.from_quat(source.head_quaternions))
    phones = segment.layout.phones
    cameras = tuple(state.exo.phones[p].camera for p in phones)
    if any(camera is None for camera in cameras):
        raise ValueError('All exo cameras need an initial pose')
    entities = [video_path(i, 0) for i in range(len(phones))]
    videos = (tuple(segment.videos[e] for e in entities) if segment.videos is not None else
              read_catalog_videos(segment.dataset, segment.id, entities, 'capture_time'))
    panels = []
    for view, (camera, video, phone) in enumerate(zip(cameras, videos, phones, strict=True)):
        assert camera is not None
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
        facing = np.einsum('ni,ni->n', rotations[:, :, 2], towards)
        projected = camera.project(positions)
        valid = (projected[:, 2] > 0) & (projected[:, 0] > 40) & (projected[:, 0] < 1880) & (projected[:, 1] > 40) & (projected[:, 1] < 1040)
        picked = []
        for block in np.array_split(np.arange(len(candidates)), 4):
            block = block[valid[block]]
            if len(block):
                picked.append(int(candidates[block[np.argmax(facing[block])]]))
        if not picked:
            continue
        indices = np.array(sorted(set(picked)), dtype=np.int64)
        fps = 1.0 / float(np.median(np.diff(times)))
        images = decode_frames(video, indices, fps).cpu().numpy()
        for number, (index, image) in enumerate(zip(indices, images, strict=True)):
            stamp = float(times[index])
            source_path = output / f'view{view}-frame{number}.png'
            cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rotation = interp([stamp]).as_matrix()[0]
            position = np.array([np.interp(stamp, head_times, source.head_positions[:, j]) for j in range(3)])
            points = np.stack([position + rotation @ state.quest.rig_T_cam.get(str(s), np.eye(4))[:3, 3] for s in (0, 1)])
            uv = camera.project(points)
            center = uv[:, :2].mean(0)
            towards_observer = rotation.T @ (camera.world_T_cam[:3, 3] - position)
            lateral = float(towards_observer[0] / np.linalg.norm(towards_observer))
            # The factory baseline points from left camera 0 to right camera 1.
            baseline = state.quest.rig_T_cam.get("1", np.eye(4))[:3, 3]
            near_right = lateral * (1.0 if baseline[0] >= 0.0 else -1.0) > 0.0
            targets = ['quest1' if near_right else 'quest0'] if abs(lateral) > .6 else ['quest0', 'quest1']
            crop, bounds = grid_crop(image, float(center[0]), float(center[1]), 260, f'view={view} time={stamp:.9f} {"/".join(targets)}')
            path = output / f'view{view}-head{number}.png'
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            panels.append(Panel(view, stamp, targets, bounds, str(path), str(source_path)))
            if number == 0:
                for target, other in enumerate(cameras):
                    if target == view or other is None:
                        continue
                    toward_observer = camera.world_T_cam[:3, 3] - other.world_T_cam[:3, 3]
                    frontality = float(other.world_T_cam[:3, 2] @ toward_observer / np.linalg.norm(toward_observer))
                    if frontality < .3:
                        continue
                    pixel = camera.project(other.world_T_cam[:3, 3][None])[0]
                    if pixel[2] <= 0 or not (0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]):
                        continue
                    crop, bounds = grid_crop(image, float(pixel[0]), float(pixel[1]), 540, f'view={view} time={stamp:.9f} phone{target} {phones[target]}')
                    path = output / f'view{view}-phone{target}-{number}.png'
                    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    panels.append(Panel(view, stamp, [f'phone{target}'], bounds, str(path), str(source_path)))
    packet = Packet(segment.id, phones, panels)
    (output / 'packet.json').write_text(to_json(packet))
    return packet


PROMPT = """Label optical camera apertures in these image crops. Return only the schema JSON.
Use ORIGINAL image pixel coordinates from the tick labels, not crop pixels. Grid lines are 20 original pixels apart; image enlarged 2x. Title gives view, exact video_time and allowed target IDs. Faint crosses are uncertain predictions, NOT labels.
Phones: mark the centre of the physical back-camera lens facing this observer, not the screen, tripod, body centre or reflection. Device IDs identify targets generically; scene placement varies. Skip partly hidden lenses, screens and backs without a clear visible aperture. Omit rather than guess when you cannot confidently identify the correct phone.
Headset: quest0 is the wearer's LEFT front RGB aperture, quest1 the wearer's RIGHT (when frontal, wearer left is image right). Quest 3S has two triangular three-aperture clusters on its white visor: RGB is the large dark outer aperture in each cluster; do not label the small neighboring tracking/IR/depth apertures or the centre of the triangle. Quest 3 has three vertical front pills: RGB cameras are on the two outer pills, not the central depth sensor. Skip occluded, side/back-facing, blurred, or ambiguous apertures. Never infer invisible points by symmetry. Confidence >=0.6 means clear visible lens, <=0.5 is uncertain and should be omitted. Only IDs listed in each panel are permitted. For an oblique view, the listed near-side lens ID was determined from calibrated head orientation: use that ID rather than guessing from image-left/image-right. One mark per visible target. No tools, no prose.
"""


def annotate(packet: Packet, output: Path, timeout_s: float = 180.0) -> Landmarks:
    """Run one read-only Sol call per view, caching validated per-view JSON."""
    from concurrent.futures import ThreadPoolExecutor

    schema = {'type': 'object', 'properties': {'labels': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'view': {'type': 'integer'}, 'video_time': {'type': 'number'}, 'target_id': {'type': 'string'},
        'u': {'type': 'number'}, 'v': {'type': 'number'}, 'confidence': {'type': 'number'}},
        'required': ['view', 'video_time', 'target_id', 'u', 'v', 'confidence'], 'additionalProperties': False}}},
        'required': ['labels'], 'additionalProperties': False}
    schema_path = output / 'schema.json'
    schema_path.write_text(json.dumps(schema))

    def label_view(view: int) -> list[Landmark]:
        path = output / f'sol-view{view}.json'
        panels = [panel for panel in packet.panels if panel.view == view]
        if not path.exists():
            args = ['codex', 'exec', '--ignore-user-config', '--strict-config', '--ephemeral', '--skip-git-repo-check',
                    '-C', str(output), '-s', 'read-only', '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="medium"',
                    '-c', 'service_tier="fast"', '--output-schema', str(schema_path), '-o', str(path)]
            for panel in panels:
                args.extend(['-i', panel.path])
            prompt = PROMPT + '\nTargets and crop bounds:\n' + to_json(Packet(packet.segment, packet.phones, panels))
            env = os.environ.copy()
            env.pop('CODEX_HOME', None)
            with (output / f'sol-view{view}.log').open('w') as log:
                subprocess.run(args, env=env, input=prompt, text=True, stdout=log, stderr=log, timeout=timeout_s, check=True)
        labels = from_json(SolLabels, path.read_text()).labels
        allowed = {(p.view, p.video_time, target): p.bounds for p in panels for target in p.targets}
        return filter_labels(labels, allowed)

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
    record = Landmarks(refined, 'gpt-6.1-sol medium fast', 'source-pixel grid; visible apertures only; confidence >=0.6; <=5px dark centroid snap', datetime.now(UTC).date().isoformat())
    (output / 'landmarks.json').write_text(to_json(record))
    return record


def run_session(segment: catalog_io.Segment, state: CalibSession, output: Path, *, thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> LabelResult:
    """Build, label and gate one session; runner must bound this call or use the CLI."""
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'result.json'
    if result_path.exists():
        previous = from_json(LabelResult, result_path.read_text())
        if previous.segment != segment.id:
            raise ValueError('Output directory belongs to another segment')
        if previous.decision in ('accepted', 'rejected') and previous.thresholds == thresholds:
            return previous
    packet_path = output / 'packet.json'
    packet = from_json(Packet, packet_path.read_text()) if packet_path.exists() else make_packet(segment, state, output)
    record_path = output / 'landmarks.json'
    record = from_json(Landmarks, record_path.read_text()) if record_path.exists() else annotate(packet, output)
    if len(record.labels) < 12 or len({label.view for label in record.labels}) < len(packet.phones):
        result = LabelResult(segment.id, 'rejected', 'insufficient visible landmark coverage', len(record.labels), time.monotonic() - start, thresholds)
    else:
        cameras = []
        for phone in packet.phones:
            camera = state.exo.phones[phone].camera
            if camera is None:
                raise ValueError(f"Missing initial camera for {phone}")
            cameras.append(camera)
        inputs = load_inputs(segment, cameras=tuple(cameras), quest=state.quest, landmarks=record)
        model, gate = evaluate(inputs, focal_source='fit', static_phones=True, landmark_target_count=20)
        accepted = accepts(gate, thresholds) and model.converged
        gate = replace(gate, passes=accepted, policy=gate.policy + "; configurable optical/body/Sapiens/RTMW acceptance thresholds in result.json")
        if accepted:
            (output / 'refinement.json').write_text(to_json(model))
            (output / 'calib.json').write_text(to_json(replace(state, exo_refinements={**state.exo_refinements, segment.id: model})))
        result = LabelResult(segment.id, 'accepted' if accepted else 'rejected', 'held-out optical/body/hand gate', len(record.labels), time.monotonic() - start, thresholds, gate)
    temporary = result_path.with_suffix('.tmp')
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
    result = LabelResult(segment, 'timeout' if code == -1 else 'failed', 'initial calibration retained; see worker.log', 0, time.monotonic() - start, thresholds)
    path.write_text(to_json(result))
    return result


def label_session(segment: str, calib_record: Path, output: Path, *, dataset: str = "selfcap-post-dev",
                  catalog_url: str = "rerun+http://dgx-spark:51235", timeout_s: float = 360.0,
                  thresholds: GateThresholds = DEFAULT_THRESHOLDS) -> LabelResult:
    """Runner API: isolate one session, resume its files, and enforce a hard deadline."""
    args = ["--segment", segment, "--calib-record", str(calib_record.resolve()), "--output-dir", str(output.resolve()),
            "--dataset", dataset, "--catalog-url", catalog_url]
    args.extend(["--thresholds.optical-min-drop-px", str(thresholds.optical_min_drop_px),
                 "--thresholds.optical-min-drop-fraction", str(thresholds.optical_min_drop_fraction),
                 "--thresholds.sapiens-max-regression-px", str(thresholds.sapiens_max_regression_px),
                 "--thresholds.rtmw-max-regression-px", str(thresholds.rtmw_max_regression_px)])
    return bounded_command(args, output, segment, timeout_s, thresholds)
