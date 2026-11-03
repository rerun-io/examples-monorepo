"""Resumable per-segment SOMA fitting and local layer export."""

import fcntl
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from dataforge.writing import atomic_write
from rerun.catalog import CatalogClient
from serde import SerdeError, serde
from serde.json import from_json, to_json

from selfcap_soma.anthropometry import plausible_ears, usable_ears
from selfcap_soma.assets import SOMA_BODY_FILES, SOMA_X_REPO_ID, SOMA_X_REVISION
from selfcap_soma.catalog import ExoSource, open_segment, read_window, subset
from selfcap_soma.fitting import Calibration, FitOptions, calibrate_shape, fit_pose
from selfcap_soma.hand_pose import HandOrientationSource, read_orientations
from selfcap_soma.output import write_report
from selfcap_soma.overlay import layer_fit, write_layer
from selfcap_soma.shape_frames import InsufficientBodyObservation, select_shape_frames
from selfcap_soma.temporal import group_jitter

PROTECTED_DATASETS: frozenset[str] = frozenset({"selfcap-post-ep1", "dataforge-selfcap"})

VERSION = "soma-batch-v9-direct-layer"
SEGMENT = re.compile(r"selfcap__(\d{4}-\d{2}-\d{2})__([a-f0-9]{8})__ep(\d{3})")


@dataclass(frozen=True, slots=True)
class Config:
    catalog_url: str
    """Catalog holding the source layers."""
    dataset: str
    """Explicit catalog dataset. Only its soma layer is written."""
    segment: list[str] = field(default_factory=list)
    """One or more SelfCap segment IDs."""
    segments_file: Path | None = None
    """Optional newline-separated IDs; blank/comment lines are ignored."""
    exo_source: ExoSource = "rtmw"
    """Triangulation and per-view evidence source."""
    output: Path = Path("/tmp/soma-batch")
    """Persistent local checkpoints, one directory per dataset/segment."""
    assets: Path | None = None
    """Local SOMA-X directory; None takes the pinned download."""
    iterations: int = 600
    """Full-rate pose iterations after freezing shape."""
    body_acceleration_weight: float = 1.0
    """Acceleration on Hips, Spine1/2, Chest, Neck1/2, Left/RightShoulder and Left/RightArm. Gaps over 100 ms do not couple."""
    torso_temporal_weight: float = 1.0
    """Root/chest rotation-velocity-change weight."""
    final_learning_rate: float = 0.001
    """Adam rate decays from 0.025 to this value during the second half of the pose pass."""
    shape_iterations: int = 300
    """Shape-update iterations within the 600-step sparse calibration pass."""
    force: bool = False
    """Ignore matching local checkpoints and refit."""


@serde(deny_unknown_fields=True)
@dataclass(slots=True)
class Timings:
    catalog_s: float = 0.0
    """Catalog metadata and observation materialization."""
    shape_s: float = 0.0
    """Sparse shape fitting and its model/mesh work."""
    fit_s: float = 0.0
    """Full-rate pose fitting, including mesh evaluation."""
    layer_s: float = 0.0
    """Report serialization, mesh normals, projections and layer writing."""
    total_s: float = 0.0
    """Wall time for the most recent invocation, including resumed work."""
    peak_gpu_mib: float = 0.0
    """Maximum allocated CUDA memory across shape and pose stages."""


@serde(deny_unknown_fields=True)
@dataclass(slots=True)
class Checkpoint:
    signature: str
    """Digest of algorithm/settings, input streams, calibration and raw rotations."""
    phase: Literal["new", "shape", "export"]
    """Last completely written stage; incomplete stages are rerun."""
    timings: Timings
    """Per-stage measurements."""
    layer_sha256: str = ""
    """Integrity check for a completed local layer."""
    layer_bytes: int = 0
    """Completed local layer size."""
    hand_orientation_source: HandOrientationSource | None = None
    """Measured Quest orientation availability for each hand."""


def selected_segments(values: list[str], path: Path | None) -> list[str]:
    """Validate explicit IDs before constructing local or remote paths."""
    if path is not None:
        values = [*values, *(line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#"))]
    result = list(dict.fromkeys(values))
    if not result or any(SEGMENT.fullmatch(value) is None for value in result):
        raise ValueError("provide at least one valid selfcap__DATE__HASH__epNNN segment")
    return result


def write_checkpoint(path: Path, state: Checkpoint) -> None:
    """Atomically advance a checkpoint after its artifact is complete."""
    with atomic_write(path) as staged:
        staged.write_text(to_json(state))


def read_checkpoint(path: Path, signature: str) -> Checkpoint:
    """The stored state when it matches; a missing, older-format or other-signature checkpoint starts over."""
    fresh: Checkpoint = Checkpoint(signature, "new", Timings())
    if not path.exists():
        return fresh
    try:
        state: Checkpoint = from_json(Checkpoint, path.read_text())
    except SerdeError:
        return fresh
    return state if state.signature == signature else fresh


def file_sha256(path: Path) -> str:
    """Digest a layer without loading it into memory."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def finish(state_path: Path, segment: str, state: Checkpoint, started: float) -> None:
    """Record and print one segment's outcome, with a greppable line of stage seconds."""
    state.timings.total_s = perf_counter() - started
    write_checkpoint(state_path, state)
    print(segment, to_json(state), flush=True)
    stages: str = " ".join(f"{stage.name}={getattr(state.timings, stage.name):.1f}" for stage in fields(state.timings))
    print(f"SOMA_TIME {segment} {stages}", flush=True)


def run_segment(config: Config, segment: str) -> None:
    """Fit one segment under an exclusive local lock; raise on missing inputs."""
    if config.dataset in PROTECTED_DATASETS:
        raise ValueError(f"publication to protected dataset {config.dataset!r} is disabled")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.dataset):
        raise ValueError("dataset must be a simple name")
    directory = config.output / config.dataset / segment
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(config, segment, directory)


def _run_locked(config: Config, segment: str, directory: Path) -> None:
    started = perf_counter()
    dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    catalog = open_segment(dataset, segment)
    metadata = catalog.view.segment_table().to_arrow_table()
    if len(metadata) != 1:
        raise ValueError(f"{segment}: segment is absent or ambiguous in {config.dataset}")
    end = metadata["video_time:end"][0].as_py().total_seconds()
    data = read_window(catalog, 0.0, float(end + 0.1), 1, config.exo_source)
    hands = data.hand_poses
    if set(hands) != {"left", "right"}:
        raise ValueError(f"{segment}: the base layer has no Quest joint poses for {sorted({'left', 'right'} - set(hands))}; re-ingest it")
    digest = hashlib.sha256()
    digest.update(
        f"{VERSION}:{config.exo_source}:{config.iterations}:{config.shape_iterations}:{config.body_acceleration_weight}:{config.torso_temporal_weight}:{config.final_learning_rate}".encode()
    )
    for array in (data.times, data.exo, data.confidence, data.hands, data.head_positions, data.head_rotations):
        digest.update(array.tobytes())
    if data.support is not None:
        digest.update(data.support.tobytes())
    # Camera/calibration changes invalidate overlays even if 3D inputs stay unchanged.
    calib = [name for name in metadata.column_names if name.startswith("property:calib:")]
    digest.update(metadata.select(calib).combine_chunks().to_batches()[0].serialize().to_pybytes())
    for hand in hands.values():
        for array in (hand.times_ns, hand.positions, hand.orientations):
            if array is not None:
                digest.update(array.tobytes())
    evidence = sorted(name for name in catalog.columns if name.startswith("/world/rig_") and any(
        kind in name for kind in (":Transform3D:", ":Pinhole:", ":Points2D:", ":confidence", ":simplecv.KeypointConfidence2D:", ":simplecv.components.DistortionCoefficients")
    ))
    if evidence:
        entities = sorted({name.split(":", 1)[0] for name in evidence})
        for index in (None, "video_time"):
            reader = catalog.view.filter_contents(entities).reader(index=index)
            available = set(reader.schema().names)
            columns = [name for name in evidence if name in available]
            if not columns:
                continue  # The catalog union can include evidence absent from this segment/index.
            table = reader.select(*([] if index is None else [index]), *columns).to_arrow_table()
            for batch in table.combine_chunks().to_batches():
                digest.update(batch.serialize().to_pybytes())
    if config.assets is None:
        digest.update(f"{SOMA_X_REPO_ID}@{SOMA_X_REVISION}".encode())
    else:
        for name in SOMA_BODY_FILES:
            digest.update(name.encode())
            digest.update(file_sha256(config.assets / name).encode())
    signature = digest.hexdigest()
    state_path = directory / "checkpoint.json"
    state = Checkpoint(signature, "new", Timings()) if config.force else read_checkpoint(state_path, signature)
    state.timings.catalog_s = perf_counter() - started
    layer = directory / f"{segment}.rrd"
    if state.phase == "export" and (not layer.is_file() or file_sha256(layer) != state.layer_sha256):
        raise ValueError(f"{segment}: completed layer is missing/corrupt; rerun with --force")
    options = FitOptions(
        head_mode="ears-rotation",
        head_rotation_weight=0.3,
        learn_head_rotation=True,
        shape_components=128,
        identity_prior=0.05,
        scale_prior=2.0,
        learn_bone_scales=True,
        bone_length_weight=2.0,
        limb_symmetry_weight=10.0,
        display_twists=True,
    )
    if state.phase == "new":
        began = perf_counter()
        ear_pairs = plausible_ears(data.exo)
        confidence = data.confidence.copy()
        confidence[~ear_pairs, 3:5] = 0.0
        surviving = int(usable_ears(data.exo, data.confidence).sum())
        selection = select_shape_frames(data.times, data.exo, confidence, data.support, allow_torso_only=surviving < 3)
        print(f"ear span guard: {surviving}/{len(ear_pairs)} surviving frames; torso/rotation fallback={surviving < 3}", flush=True)
        with atomic_write(directory / "shape-frames.json") as staged:
            staged.write_text(to_json(selection))
        print(f"shape frames: {len(selection.indices)}, whole-segment fallback: {selection.used_fallback}", flush=True)
        mask = np.zeros(len(data.times), dtype=bool)
        mask[selection.indices] = True
        shape = calibrate_shape(subset(data, mask), config.assets, 600, config.shape_iterations, options)
        calibration = Calibration(shape.identity, shape.scale, shape.head_offset, shape.bone_scales, shape.head_attachment, shape.head_anchor)
        with atomic_write(directory / "calibration.json") as staged:
            staged.write_text(to_json(calibration))
        state.timings.shape_s = perf_counter() - began
        state.timings.peak_gpu_mib = shape.peak_bytes / 2**20
        state.phase = "shape"
        write_checkpoint(state_path, state)
    if state.phase == "shape":
        began = perf_counter()
        calibration = from_json(Calibration, (directory / "calibration.json").read_text())
        orientations = read_orientations(hands, data.times, data.offset_ms)
        state.hand_orientation_source = HandOrientationSource(*orientations.sources)
        with atomic_write(directory / "hand-orientation-source.json") as staged:
            staged.write_text(to_json(state.hand_orientation_source))
        print("hand_orientation_source", to_json(state.hand_orientation_source), flush=True)
        result = fit_pose(
            data,
            calibration,
            config.assets,
            config.iterations,
            replace(
                options,
                learn_head_rotation=False,
                freeze_legs=True,
                leg_support_weight=0.0,
                hip_landmark_weight=0.15,
                torso_temporal_weight=config.torso_temporal_weight,
                body_acceleration_weight=config.body_acceleration_weight,
                final_learning_rate=config.final_learning_rate,
                orientation_weight=30.0,
            ),
            orientations=orientations,
        )
        state.timings.fit_s = perf_counter() - began
        state.timings.peak_gpu_mib = max(state.timings.peak_gpu_mib, result.peak_bytes / 2**20)
        began = perf_counter()
        write_report(directory, data, result, upper_body_only=True)
        with atomic_write(directory / "jitter.json") as staged:
            staged.write_text(to_json(group_jitter(result.joints, data.times, result.names)))
        for stale in directory.glob("*.rrd.tmp"):  # a killed writer's; this segment's lock is held
            stale.unlink()
        write_layer(segment, config.dataset, config.catalog_url, config.exo_source, layer, layer_fit(data, result), catalog=catalog)
        state.timings.layer_s = perf_counter() - began
        state.layer_sha256 = file_sha256(layer)
        state.layer_bytes = layer.stat().st_size
        state.phase = "export"
        write_checkpoint(state_path, state)
    finish(state_path, segment, state, started)


def main(config: Config) -> None:
    """Process explicit segments serially; external workers may run different IDs."""
    for segment in selected_segments(config.segment, config.segments_file):
        try:
            run_segment(config, segment)
        except InsufficientBodyObservation as error:
            print(f"SOMA_INPUT_LIMITATION insufficient_body_observation {config.dataset}/{segment}: {error}", file=sys.stderr)
            raise SystemExit(20) from None
        except (ValueError, FileNotFoundError, KeyError, StopIteration, IndexError, subprocess.CalledProcessError, BlockingIOError) as error:
            raise RuntimeError(f"SOMA batch failed for {config.dataset}/{segment}: {error}") from error
