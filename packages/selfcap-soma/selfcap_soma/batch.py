"""Resumable per-segment SOMA fitting and soma-only catalog publication."""

import fcntl
import hashlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pyarrow as pa
from dataforge.quest_joint_poses import read_segment_hand_poses
from dataforge.schema import SOMA_LAYER
from dataforge.writing import atomic_write
from datafusion import col, lit
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer
from rerun.chunk import RrdReader
from serde import SerdeError, serde
from serde.json import from_json, to_json

from selfcap_soma.anthropometry import plausible_ears, usable_ears
from selfcap_soma.catalog import ExoSource, read_window, subset
from selfcap_soma.fitting import Calibration, FitOptions, fit
from selfcap_soma.hand_pose import HandOrientationSource, read_orientations
from selfcap_soma.output import write_report
from selfcap_soma.overlay import SOMA_MESH, layer_fit, mesh_vertex_rows, write_layer
from selfcap_soma.overlay import Config as OverlayConfig
from selfcap_soma.shape_frames import InsufficientBodyObservation, select_shape_frames
from selfcap_soma.temporal import group_jitter

VERSION = "soma-batch-v9-direct-layer"
SEGMENT = re.compile(r"selfcap__(\d{4}-\d{2}-\d{2})__([a-f0-9]{8})__ep(\d{3})")


@dataclass(frozen=True, slots=True)
class Config:
    dataset: str
    """Explicit catalog dataset. Only its soma layer is written."""
    spark_dir: Path | None = None
    """Absolute destination directory on Spark: ship there and register soma. Unset: all output stays local and the caller registers the layer."""
    segment: list[str] = field(default_factory=list)
    """One or more SelfCap segment IDs."""
    segments_file: Path | None = None
    """Optional newline-separated IDs; blank/comment lines are ignored."""
    exo_source: ExoSource = "rtmw"
    """Triangulation and per-view evidence source."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog holding the source layers and destination dataset."""
    spark_host: str = "spark"
    """SSH host or configured alias for the catalog's file host."""
    output: Path = Path("/tmp/soma-batch")
    """Persistent local checkpoints, one directory per dataset/segment."""
    """Raw SelfCap cut tree, needed only for missing catalog hand rotations."""
    assets: Path = Path("/tmp/soma-fit/assets")
    """Downloaded SOMA model files."""
    iterations: int = 600
    """Full-rate pose iterations after freezing shape."""
    body_acceleration_weight: float = 1.0
    """Body-chain acceleration weight; excludes elbows, wrists and fingers."""
    torso_temporal_weight: float = 1.0
    """Root/chest rotation-velocity-change weight."""
    final_learning_rate: float = 0.001
    """Adam learning rate at the end of the pose pass."""
    shape_iterations: int = 300
    """Shape-update iterations within the 600-step sparse calibration pass."""
    native_exo: bool = False
    """Use native exo/Quest views and bake OAK only; requires a native-projection blueprint."""
    cleanup_after_publish: bool = False
    """After verified catalog readback, delete only this segment's known heavy local RRDs; requires spark_dir."""
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
    ship_s: float = 0.0
    """SSH/SCP file transfer."""
    register_s: float = 0.0
    """Catalog soma-layer registration."""
    total_s: float = 0.0
    """Wall time for the most recent invocation, including resumed work."""
    peak_gpu_mib: float = 0.0
    """Maximum allocated CUDA memory across shape and pose stages."""


@serde(deny_unknown_fields=True)
@dataclass(slots=True)
class Checkpoint:
    signature: str
    """Digest of algorithm/settings, input streams, calibration and raw rotations."""
    phase: Literal["new", "shape", "export", "published"]
    """Last completely written stage; incomplete stages are rerun."""
    timings: Timings
    """Per-stage measurements."""
    layer_sha256: str = ""
    """Integrity check for a completed local layer."""
    layer_bytes: int = 0
    """Completed local layer size."""
    hand_orientation_source: HandOrientationSource | None = None
    """Measured Quest orientation availability for each hand."""
    mesh_sample_sha256: str = ""
    """First/last mesh samples retained as a readback integrity check."""
    readback_verified: bool = False
    """Catalog URI, remote digest, mesh times and endpoint geometry were checked."""
    cleaned: bool = False
    """Heavy local artifacts may be absent; resume must verify the published layer."""


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
    temporary = path.with_suffix(".tmp")
    temporary.write_text(to_json(state))
    temporary.replace(path)


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
    """Digest of a layer file, without reading it into memory."""
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
    if config.dataset in ("selfcap-post-ep1", "dataforge-selfcap"):
        raise ValueError(f"publication to protected dataset {config.dataset!r} is disabled")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.dataset) or (config.spark_dir is not None and not config.spark_dir.is_absolute()):
        raise ValueError("dataset must be a simple name and spark-dir must be absolute")
    if config.cleanup_after_publish and config.spark_dir is None:
        raise ValueError("--cleanup-after-publish requires --spark-dir")
    directory = config.output / config.dataset / segment
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(config, segment, directory)


def _run_locked(config: Config, segment: str, directory: Path) -> None:
    started = perf_counter()
    dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    metadata = dataset.filter_segments(segment).segment_table().to_arrow_table()
    if len(metadata) != 1:
        raise ValueError(f"{segment}: segment is absent or ambiguous in {config.dataset}")
    end = metadata["video_time:end"][0].as_py().total_seconds()
    data = read_window(config.catalog_url, config.dataset, segment, 0.0, float(end + 0.1), 1, config.exo_source)
    hands = read_segment_hand_poses(dataset.filter_segments(segment))
    if set(hands) != {"left", "right"}:
        raise ValueError(f"{segment}: the base layer has no Quest joint poses for {sorted({'left', 'right'} - set(hands))}; re-ingest it")
    digest = hashlib.sha256()
    digest.update(
        f"{VERSION}:{config.exo_source}:{config.iterations}:{config.shape_iterations}:{config.body_acceleration_weight}:{config.torso_temporal_weight}:{config.final_learning_rate}".encode()
    )
    for array in (data.times, data.exo, data.confidence, data.hands, data.head_rotations):
        digest.update(array.tobytes())
    if data.support is not None:
        digest.update(data.support.tobytes())
    if config.native_exo:
        digest.update(b":native-exo-v2-ui-radii")
    # Camera/calibration changes invalidate overlays even if 3D inputs stay unchanged.
    calib = [name for name in metadata.column_names if name.startswith("property:calib:")]
    digest.update(metadata.select(calib).combine_chunks().to_batches()[0].serialize().to_pybytes())
    for hand in hands.values():
        for array in (hand.times_ns, hand.positions, hand.orientations):
            if array is not None:
                digest.update(array.tobytes())
    signature = digest.hexdigest()
    state_path = directory / "checkpoint.json"
    state = Checkpoint(signature, "new", Timings()) if config.force else read_checkpoint(state_path, signature)
    state.timings.catalog_s = perf_counter() - started
    layer = directory / f"{segment}.rrd"
    remote: Path | None = None if config.spark_dir is None else config.spark_dir / layer.name
    if state.phase == "published" and state.cleaned:
        if remote is None:
            raise ValueError(f"{segment}: local layer was cleaned after publication; pass --spark-dir to verify it or --force to refit")
        verify_publication(config, segment, remote, state, data.times)
        if config.cleanup_after_publish:
            cleanup_published(config, directory, segment, state)
        finish(state_path, segment, state, started)
        return
    if state.phase in ("export", "published") and (not layer.is_file() or file_sha256(layer) != state.layer_sha256):
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
        (directory / "shape-frames.json").write_text(to_json(selection))
        print(f"shape frames: {len(selection.indices)}, whole-segment fallback: {selection.used_fallback}", flush=True)
        mask = np.zeros(len(data.times), dtype=bool)
        mask[selection.indices] = True
        shape = fit(subset(data, mask), config.assets, 600, config.shape_iterations, options)
        calibration = Calibration(shape.identity, shape.scale, shape.head_offset, shape.bone_scales, shape.head_attachment, shape.head_anchor)
        (directory / "calibration.json").write_text(to_json(calibration))
        state.timings.shape_s = perf_counter() - began
        state.timings.peak_gpu_mib = shape.peak_bytes / 2**20
        state.phase = "shape"
        write_checkpoint(state_path, state)
    if state.phase == "shape":
        began = perf_counter()
        calibration = from_json(Calibration, (directory / "calibration.json").read_text())
        orientations = read_orientations(hands, data.times, data.offset_ms)
        state.hand_orientation_source = HandOrientationSource(*orientations.sources)
        (directory / "hand-orientation-source.json").write_text(to_json(state.hand_orientation_source))
        print("hand_orientation_source", to_json(state.hand_orientation_source), flush=True)
        result = fit(
            data,
            config.assets,
            config.iterations,
            1,
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
            fixed=calibration,
            orientations=orientations,
        )
        state.timings.fit_s = perf_counter() - began
        state.timings.peak_gpu_mib = max(state.timings.peak_gpu_mib, result.peak_bytes / 2**20)
        began = perf_counter()
        write_report(directory, data, result, 39.862079571, upper_body_only=True)
        (directory / "jitter.json").write_text(to_json(group_jitter(result.joints, data.times, result.names)))
        for stale in directory.glob("*.rrd.tmp"):  # a killed writer's; this segment's lock is held
            stale.unlink()
        with atomic_write(layer) as staged:
            write_layer(
                OverlayConfig(output=staged, catalog_url=config.catalog_url, dataset=config.dataset, segment=segment, exo_source=config.exo_source, native_exo=config.native_exo),
                layer_fit(data, result),
            )
        state.timings.layer_s = perf_counter() - began
        state.layer_sha256 = file_sha256(layer)
        state.layer_bytes = layer.stat().st_size
        state.phase = "export"
        write_checkpoint(state_path, state)
    if remote is not None:
        state.timings.ship_s, state.timings.register_s = publish_layer(config, layer, remote)
        state.phase = "published"
        if config.cleanup_after_publish:
            endpoints = np.rint(data.times[[0, -1]] * 1e9).astype(np.int64)
            state.mesh_sample_sha256 = local_mesh_digest(layer, endpoints)
            verify_publication(config, segment, remote, state, data.times)
            cleanup_published(config, directory, segment, state)
    finish(state_path, segment, state, started)


def publish_layer(config: Config, layer: Path, remote: Path) -> tuple[float, float]:
    """Atomically ship one RRD and register only the soma layer with REPLACE."""
    began = perf_counter()
    subprocess.run(["ssh", config.spark_host, f"mkdir -p -- {shlex.quote(str(remote.parent))}"], check=True)
    subprocess.run(["scp", str(layer), f"{config.spark_host}:{shlex.quote(str(remote) + '.upload')}"], check=True)
    subprocess.run(["ssh", config.spark_host, f"mv -- {shlex.quote(str(remote) + '.upload')} {shlex.quote(str(remote))}"], check=True)
    ship_s = perf_counter() - began
    began = perf_counter()
    client = CatalogClient(config.catalog_url)
    client.get_dataset(config.dataset).register([remote.as_uri()], layer_name=SOMA_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    return ship_s, perf_counter() - began


def local_mesh_digest(path: Path, endpoints: Int64[ndarray, "2"]) -> str:
    """Hash the two endpoint mesh samples before local cleanup."""
    rows = []
    component = "Mesh3D:vertex_positions"
    for chunk in RrdReader(path).stream().filter(content=SOMA_MESH).to_chunks():
        batch = chunk.to_record_batch()
        if "video_time" not in batch.schema.names or component not in batch.schema.names:
            continue
        stamps = batch.column("video_time").cast("int64").to_numpy()
        for index in np.flatnonzero(np.isin(stamps, endpoints)):
            rows.append((int(stamps[index]), mesh_vertex_rows(batch.column(component).slice(int(index), 1))[0]))
    if len(rows) != 2:
        raise ValueError("local layer must contain both endpoint mesh samples before cleanup")
    digest = hashlib.sha256()
    for stamp, points in sorted(rows, key=lambda row: row[0]):
        digest.update(np.asarray([stamp], dtype="<i8").tobytes())
        digest.update(points.astype("<f4").tobytes())
    return digest.hexdigest()


def verify_publication(config: Config, segment: str, remote: Path, state: Checkpoint, times: Float64[ndarray, "n"]) -> None:
    """Verify registration, remote file integrity and actual catalog mesh readback."""
    state.readback_verified = False
    result = subprocess.run(["ssh", config.spark_host, f"sha256sum -- {shlex.quote(str(remote))}"], check=True, capture_output=True, text=True)
    if not result.stdout.split() or result.stdout.split()[0] != state.layer_sha256:
        raise ValueError(f"{segment}: published remote layer digest mismatch; cleanup refused")
    client = CatalogClient(config.catalog_url)
    dataset = client.get_dataset(config.dataset)
    view = dataset.filter_segments(segment)
    metadata = view.segment_table().select("rerun_layer_names", "rerun_storage_urls").to_arrow_table()
    if len(metadata) != 1:
        raise ValueError(f"{segment}: missing catalog registration; cleanup refused")
    registered = dict(zip(metadata["rerun_layer_names"][0].as_py(), metadata["rerun_storage_urls"][0].as_py(), strict=True))
    if registered.get(SOMA_LAYER) != remote.as_uri():
        raise ValueError(f"{segment}: catalog soma URI mismatch; cleanup refused")
    component = f"{SOMA_MESH}:Mesh3D:vertex_positions"
    reader = view.filter_contents([SOMA_MESH]).reader(index="video_time")
    stamps = reader.filter(col(component).is_not_null()).select("video_time").to_arrow_table()["video_time"].cast(pa.int64()).to_numpy()
    expected = np.rint(times * 1e9).astype(np.int64)
    if not np.array_equal(np.sort(stamps), expected):
        raise ValueError(f"{segment}: catalog mesh timestamps mismatch; cleanup refused")
    samples = (
        reader.filter((col("video_time").cast(pa.int64()) == lit(int(expected[0]))) | (col("video_time").cast(pa.int64()) == lit(int(expected[-1]))))
        .select("video_time", component)
        .to_arrow_table()
        .sort_by("video_time")
    )
    digest = hashlib.sha256()
    if len(samples) != 2:
        raise ValueError(f"{segment}: catalog endpoint mesh samples missing; cleanup refused")
    for index in range(2):
        stamp = int(samples["video_time"][index].cast(pa.int64()).as_py())
        points = mesh_vertex_rows(samples[component].slice(index, 1))[0]
        if not np.isfinite(points).all():
            raise ValueError(f"{segment}: catalog mesh has nonfinite vertices; cleanup refused")
        digest.update(np.asarray([stamp], dtype="<i8").tobytes())
        digest.update(points.astype("<f4").tobytes())
    if not state.mesh_sample_sha256 or digest.hexdigest() != state.mesh_sample_sha256:
        raise ValueError(f"{segment}: catalog mesh sample digest mismatch; cleanup refused")
    state.readback_verified = True


def cleanup_published(config: Config, directory: Path, segment: str, state: Checkpoint) -> None:
    """Delete only known local RRDs after successful publication and readback."""
    if not config.cleanup_after_publish or state.phase != "published" or not state.readback_verified:
        raise ValueError("cleanup requires opt-in and a verified successful publication")
    if directory.resolve().parent != (config.output / config.dataset).resolve() or directory.name != segment:
        raise ValueError("cleanup directory is outside the segment output directory")
    # fit.rrd: left by a run from before layers were written directly. *.rrd.tmp: a killed writer's stage.
    paths = [directory / "fit.rrd", directory / f"{segment}.rrd", *directory.glob("*.rrd.tmp")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("cleanup refuses symlink artifacts")
    # Persist first: a crash between unlinks resumes through catalog verification.
    state.cleaned = True
    write_checkpoint(directory / "checkpoint.json", state)
    for path in paths:
        path.unlink(missing_ok=True)


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
